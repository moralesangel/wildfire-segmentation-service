"""The HTTP and WebSocket surface.

Frames arrive over a WebSocket rather than as POSTs because the connection is
the backpressure signal: when the service drops a frame it says so on the same
socket, and a client that is sending faster than the service can serve learns
it immediately instead of from a rising latency curve.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from service import metrics, preprocess
from service.batcher import Batcher, QueueFull
from service.config import FRAME_BUDGET_MS, TARGET_FPS, Settings, resolve_threads
from service.engine import load_engine


def _static_dir() -> Path:
    """Where the demo page and its sample clip live.

    Checked in order rather than assumed from __file__: the repo layout puts
    static/ two levels up from this module, but an installed package has no
    such parent, and the Docker image copies it next to the working directory.
    An override exists because both guesses are wrong somewhere.
    """
    if override := os.getenv("WFS_STATIC_DIR"):
        return Path(override)

    candidates = [
        Path(__file__).resolve().parents[2] / "static",  # running from the repo
        Path.cwd() / "static",  # the Docker image's layout
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return candidates[0]


_STATIC = _static_dir()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    threads = resolve_threads(settings.onnx_threads)
    engine = load_engine(settings.model_path, threads=threads)
    batcher = Batcher(engine, settings)
    await batcher.start()

    app.state.settings = settings
    app.state.engine = engine
    app.state.batcher = batcher
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(
    title="Wildfire Segmentation Service",
    summary=f"Segmentation under a stated latency budget: {FRAME_BUDGET_MS:.1f} ms at p95.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness plus the one fact that decides whether output is meaningful.

    A stub engine still answers every request, so `model_loaded` is what tells
    a caller whether it is looking at predictions or at a placeholder.
    """
    engine = app.state.engine
    return {
        "status": "ok",
        "engine": engine.name,
        "model_loaded": engine.name != "stub",
        "target_fps": TARGET_FPS,
        "frame_budget_ms": round(FRAME_BUDGET_MS, 1),
        # Reported because a missing static dir shows up as a 404 on the
        # sample clip, which looks like a broken video rather than a
        # misconfigured path.
        "onnx_threads": resolve_threads(app.state.settings.onnx_threads),
        "static_dir": str(_STATIC),
        "sample_clip": (_STATIC / "sample.mp4").is_file(),
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/segment")
async def segment(websocket: WebSocket) -> None:
    """Stream frames in, get masks out, and hear about the ones that were dropped.

    Every reply carries the latency and whether it met the budget, so the
    client can show the truth rather than the service grading its own homework
    out of sight.
    """
    await websocket.accept()
    settings: Settings = app.state.settings
    batcher: Batcher = app.state.batcher
    engine = app.state.engine
    send_lock = asyncio.Lock()

    async def handle(payload: bytes, started: float) -> None:
        """One frame, start to reply."""
        try:
            decoded = await asyncio.to_thread(preprocess.decode, payload, settings.input_size)
        except Exception as exc:
            async with send_lock:
                await websocket.send_json({"type": "error", "detail": f"decode failed: {exc}"})
            return

        try:
            mask = await batcher.submit(decoded.tensor)
        except QueueFull:
            # Told immediately, while the frame is still current. A client that
            # keeps sending regardless keeps being refused, which is the point:
            # the service sheds load instead of lagging.
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "dropped",
                        "reason": "queue_full",
                        "max_outstanding": settings.max_queue_depth,
                    }
                )
            return

        png = preprocess.encode_mask(mask)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        metrics.frame_latency.observe(elapsed_ms / 1000.0)
        metrics.frames_processed.inc()
        within_budget = elapsed_ms <= FRAME_BUDGET_MS
        if not within_budget:
            metrics.budget_misses.inc()

        async with send_lock:
            await websocket.send_json(
                {
                    "type": "mask",
                    "mask_png_b64": base64.b64encode(png).decode("ascii"),
                    "latency_ms": round(elapsed_ms, 2),
                    "within_budget": within_budget,
                    "engine": engine.name,
                }
            )

    # Frames are handled as tasks rather than awaited in the read loop.
    # Awaiting each mask before reading the next one meant a single connection
    # could never have more than one frame outstanding, so the queue never
    # filled, the batcher never batched, and the drop policy was unreachable
    # code -- all of it passing tests that drove the Batcher directly.
    pending: set[asyncio.Task[None]] = set()
    try:
        while True:
            payload = await websocket.receive_bytes()
            metrics.frames_received.inc()
            task = asyncio.create_task(handle(payload, time.perf_counter()))
            pending.add(task)
            task.add_done_callback(pending.discard)
    except WebSocketDisconnect:
        for task in pending:
            task.cancel()


@app.get("/", response_class=HTMLResponse)
async def index() -> Response:
    """The demo page, or a plain pointer when it has not been built yet."""
    page = _STATIC / "index.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return PlainTextResponse(
        "Wildfire Segmentation Service\n"
        f"Budget: {FRAME_BUDGET_MS:.1f} ms at p95 ({TARGET_FPS} FPS)\n"
        "Endpoints: /health  /metrics  /ws/segment\n"
    )


# Mounted last so it cannot shadow the routes above. StaticFiles serves byte
# ranges, which the sample clip needs: without range support a browser cannot
# seek, and Safari will not play the file at all.
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC), name="static")
