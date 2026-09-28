"""The bounded queue and the dynamic batcher.

This is where the service decides what to do when it cannot keep up, and the
answer is: refuse work rather than fall behind. A frame that arrives when the
queue is full is dropped immediately and the client is told, which keeps the
frames that *do* get served fresh.

The alternative -- an unbounded queue -- does not make the service faster. It
converts overload into staleness: every frame is still processed, each one a
little further behind reality, until the operator is watching a fire line as
it looked ten seconds ago. For a live feed that is worse than a gap.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from service import metrics
from service.config import Settings
from service.engine import Engine


class QueueFull(Exception):
    """Raised when a frame arrives and there is no room to hold it."""


@dataclass
class _Job:
    """One frame waiting for inference, and somewhere to put the answer."""

    tensor: np.ndarray
    future: asyncio.Future[np.ndarray]
    queued_at: float = field(default_factory=time.perf_counter)


class Batcher:
    """Collects frames into batches and runs them through the engine.

    One consumer task owns the engine, so inference is serialised: an ONNX
    session is not safe to call concurrently, and two threads fighting over two
    vCPUs would be slower than one anyway.
    """

    def __init__(self, engine: Engine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._in_flight = 0
        """Frames accepted and not yet answered, counting the batch currently
        running through the engine.

        The queue's own `maxsize` is not the limit to enforce. The batcher
        drains the queue into a batch before inference, so at the moment a new
        frame arrives the queue can read empty while a full batch is still in
        the engine -- and the frame is accepted, waits behind that batch, and
        the bound silently becomes `max_queue_depth + max_batch_size`. Counting
        outstanding work instead makes the limit mean what it says."""

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def submit(self, tensor: np.ndarray) -> np.ndarray:
        """Queue one frame and wait for its mask.

        Raises `QueueFull` straight away rather than blocking, so the caller
        can tell the client its frame was dropped while it is still current.
        """
        if self._in_flight >= self._settings.max_queue_depth:
            metrics.frames_dropped.labels(reason="queue_full").inc()
            raise QueueFull(f"{self._in_flight} frames already outstanding; frame dropped")

        loop = asyncio.get_running_loop()
        job = _Job(tensor=tensor, future=loop.create_future())
        self._in_flight += 1
        self._queue.put_nowait(job)
        metrics.queue_depth.set(self._in_flight)

        try:
            return await job.future
        finally:
            self._in_flight -= 1
            metrics.queue_depth.set(self._in_flight)

    async def _collect(self) -> list[_Job]:
        """One batch: the first job, plus whatever arrives within the window.

        The wait is bounded by `batch_wait_ms` rather than by batch size, so a
        quiet service does not hold a lone frame hostage waiting for company
        that is not coming.
        """
        first = await self._queue.get()
        batch = [first]

        deadline = time.perf_counter() + self._settings.batch_wait_ms / 1000.0
        while len(batch) < self._settings.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break

        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            metrics.batch_size.observe(len(batch))

            stacked = np.stack([job.tensor for job in batch])
            started = time.perf_counter()
            try:
                # Off the event loop: ONNX Runtime releases the GIL, but the
                # call still blocks, and blocking here would stall the very
                # websocket reads that feed the queue.
                masks = await loop.run_in_executor(None, self._engine.infer, stacked)
            except Exception as exc:  # the engine failing must not kill the batcher
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue

            metrics.stage_latency.labels(stage="inference").observe(time.perf_counter() - started)

            for job, mask in zip(batch, masks, strict=True):
                if not job.future.done():
                    job.future.set_result(mask)
