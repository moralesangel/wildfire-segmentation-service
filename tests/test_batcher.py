"""The queue's behaviour under load, which is the service's main design claim."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from service.batcher import Batcher, QueueFull
from service.config import Settings
from service.engine import StubEngine


def _frame(size: int = 32) -> np.ndarray:
    return np.zeros((3, size, size), dtype=np.float32)


@pytest.fixture
def settings() -> Settings:
    return Settings(max_queue_depth=2, max_batch_size=4, batch_wait_ms=5.0)


class TestBackpressure:
    """Overload has to cost frames, not freshness."""

    async def test_a_frame_is_served(self, settings: Settings) -> None:
        batcher = Batcher(StubEngine(delay_ms=1.0), settings)
        await batcher.start()
        try:
            mask = await batcher.submit(_frame())
            assert mask.shape == (32, 32)
        finally:
            await batcher.stop()

    async def test_a_full_queue_refuses_rather_than_waits(self, settings: Settings) -> None:
        # A slow engine so the queue fills before anything drains.
        batcher = Batcher(StubEngine(delay_ms=200.0), settings)
        await batcher.start()
        try:
            # One job leaves for inference immediately; the queue bound is 2.
            pending = [asyncio.create_task(batcher.submit(_frame())) for _ in range(4)]
            await asyncio.sleep(0.05)

            with pytest.raises(QueueFull):
                await batcher.submit(_frame())

            for task in pending:
                task.cancel()
        finally:
            await batcher.stop()

    async def test_refusal_is_immediate(self, settings: Settings) -> None:
        """The point of refusing is that the client hears back while the frame
        is still current. A slow rejection would be no better than queueing."""
        batcher = Batcher(StubEngine(delay_ms=200.0), settings)
        await batcher.start()
        try:
            pending = [asyncio.create_task(batcher.submit(_frame())) for _ in range(4)]
            await asyncio.sleep(0.05)

            started = asyncio.get_running_loop().time()
            with pytest.raises(QueueFull):
                await batcher.submit(_frame())
            elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

            assert elapsed_ms < 20, f"rejection took {elapsed_ms:.1f} ms"
            for task in pending:
                task.cancel()
        finally:
            await batcher.stop()


class TestBatching:
    async def test_frames_arriving_together_share_a_batch(self) -> None:
        settings = Settings(max_queue_depth=16, max_batch_size=8, batch_wait_ms=30.0)
        seen: list[int] = []

        class Recording(StubEngine):
            def infer(self, batch: np.ndarray) -> np.ndarray:
                seen.append(len(batch))
                return super().infer(batch)

        batcher = Batcher(Recording(delay_ms=1.0), settings)
        await batcher.start()
        try:
            await asyncio.gather(*(batcher.submit(_frame()) for _ in range(4)))
            assert max(seen) > 1, f"no batch ever formed: {seen}"
        finally:
            await batcher.stop()

    async def test_a_lone_frame_is_not_held_for_company(self) -> None:
        """The batch window is an upper bound, not a wait. A quiet service must
        not add its full window to every frame's latency."""
        settings = Settings(max_queue_depth=8, max_batch_size=8, batch_wait_ms=25.0)
        batcher = Batcher(StubEngine(delay_ms=1.0), settings)
        await batcher.start()
        try:
            loop = asyncio.get_running_loop()
            started = loop.time()
            await batcher.submit(_frame())
            elapsed_ms = (loop.time() - started) * 1000

            # It waits out the window for company that never comes, but must
            # not exceed it by much.
            assert elapsed_ms < 60, f"lone frame took {elapsed_ms:.1f} ms"
        finally:
            await batcher.stop()


class TestEngineFailure:
    async def test_a_failing_engine_does_not_kill_the_batcher(self) -> None:
        settings = Settings(max_queue_depth=8, max_batch_size=2, batch_wait_ms=5.0)

        class Broken(StubEngine):
            def infer(self, batch: np.ndarray) -> np.ndarray:
                raise RuntimeError("engine exploded")

        batcher = Batcher(Broken(), settings)
        await batcher.start()
        try:
            with pytest.raises(RuntimeError, match="engine exploded"):
                await batcher.submit(_frame())

            # Still serving: the failure reached the caller, not the loop.
            with pytest.raises(RuntimeError):
                await batcher.submit(_frame())
        finally:
            await batcher.stop()
