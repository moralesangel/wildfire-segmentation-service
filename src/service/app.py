"""The HTTP and WebSocket surface.

Frames arrive over a WebSocket rather than as POSTs because the connection is
the backpressure signal: when the service drops a frame it says so on the same
socket, and a client that is sending faster than the service can serve learns
it immediately instead of from a rising latency curve.
"""

from __future__ import annotations

import base64
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from service import metrics, preprocess
from service.batcher import Batcher, QueueFull
from service.config import FRAME_BUDGET_MS, TARGET_FPS, Settings
from service.engine import load_engine

_STATIC = Path(__file__).resolve().parents[2] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    engine = load_engine(settings.model_path, threads=settings.onnx_threads)
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

    try:
        while True:
            payload = await websocket.receive_bytes()
            metrics.frames_received.inc()
            started = time.perf_counter()

            try:
                decoded = preprocess.decode(payload, settings.input_size)
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": f"decode failed: {exc}"})
                continue

            try:
                mask = await batcher.submit(decoded.tensor)
            except QueueFull:
                # Told immediately, while the frame is still current. A client
                # that keeps sending regardless will keep being refused, which
                # is the point: the service sheds load instead of lagging.
                await websocket.send_json(
                    {
                        "type": "dropped",
                        "reason": "queue_full",
                        "max_outstanding": settings.max_queue_depth,
                    }
                )
                continue

            png = preprocess.encode_mask(mask)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            metrics.frame_latency.observe(elapsed_ms / 1000.0)
            metrics.frames_processed.inc()
            within_budget = elapsed_ms <= FRAME_BUDGET_MS
            if not within_budget:
                metrics.budget_misses.inc()

            await websocket.send_json(
                {
                    "type": "mask",
                    "mask_png_b64": base64.b64encode(png).decode("ascii"),
                    "latency_ms": round(elapsed_ms, 2),
                    "within_budget": within_budget,
                    "engine": engine.name,
                }
            )
    except WebSocketDisconnect:
        return


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
