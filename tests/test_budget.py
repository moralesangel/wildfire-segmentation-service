"""The latency budget as a test, not as a paragraph in the README.

These do not assert that the service is fast. A shared CI runner cannot answer
that, and a test that fails because someone else's job was busy teaches people
to ignore the suite. What they pin is that the budget is *stated, measurable
and reported* -- that it is a number the code is held to rather than whatever
the last benchmark happened to produce.

The real timing lives in scripts/benchmark.py, run on a known machine.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service import preprocess
from service.app import app
from service.config import FRAME_BUDGET_MS, TARGET_FPS, Settings


def _jpeg(width: int, height: int) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class TestTheBudgetIsStated:
    def test_it_follows_from_the_target_rate(self) -> None:
        """66.7 ms is 1/15 s. Derived, not chosen to match a measurement."""
        assert pytest.approx(1000 / TARGET_FPS) == FRAME_BUDGET_MS

    def test_the_service_publishes_it(self) -> None:
        with TestClient(app) as client:
            body = client.get("/health").json()
        assert body["frame_budget_ms"] == pytest.approx(FRAME_BUDGET_MS, abs=0.1)
        assert body["target_fps"] == TARGET_FPS

    def test_the_histogram_has_a_bucket_at_the_budget(self) -> None:
        """Without an edge exactly at 66.7 ms, the p95 the service is judged on
        is interpolated across a bucket rather than counted."""
        with TestClient(app) as client:
            assert 'le="0.0667"' in client.get("/metrics").text


class TestEveryStageIsMeasured:
    """A single end-to-end number cannot say which stage spent the time, and
    the answer here turned out to be neither the one that was assumed."""

    def test_decode_and_encode_are_timed_apart(self) -> None:
        preprocess.decode(_jpeg(640, 480), 128)
        preprocess.encode_mask(np.zeros((128, 128), dtype=np.float32))

        with TestClient(app) as client:
            text = client.get("/metrics").text

        assert 'stage="decode"' in text
        assert 'stage="encode"' in text

    def test_inference_is_timed_apart(self) -> None:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/segment") as ws:
                ws.send_bytes(_jpeg(320, 240))
                ws.receive_json()
            assert 'stage="inference"' in client.get("/metrics").text


class TestCostIsNotPaidTwice:
    """Regressions on the two decisions the measurements forced."""

    def test_the_mask_is_not_upscaled_to_the_source_frame(self) -> None:
        """Returning a 4K mask cost 237 ms a frame against 0.3 ms, for pixels
        carrying nothing the 128x128 mask did not already have."""
        mask = np.zeros((128, 128), dtype=np.float32)
        payload = preprocess.encode_mask(mask)

        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.size == (128, 128)
        # A 4K PNG of this mask is ~2.3 MB; the native one is ~16 KB.
        assert len(payload) < 100_000

    def test_decoding_a_4k_frame_does_not_build_the_full_bitmap(self) -> None:
        """Image.draft lets the JPEG decoder skip to a smaller DCT scale while
        reading. Without it the decode alone is 74.7 ms and the budget is gone
        before inference starts.

        Asserted as a ratio against a no-draft decode rather than as an
        absolute time, so it means the same thing on any machine.
        """
        payload = _jpeg(3840, 2160)

        import time

        started = time.perf_counter()
        preprocess.decode(payload, 128)
        with_draft = time.perf_counter() - started

        started = time.perf_counter()
        with Image.open(io.BytesIO(payload)) as img:
            np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
        without_draft = time.perf_counter() - started

        assert with_draft < without_draft, (
            f"draft() bought nothing: {with_draft * 1000:.1f} ms vs {without_draft * 1000:.1f} ms"
        )


class TestQueueBoundMeansWhatItSays:
    def test_the_bound_is_outstanding_frames_not_queued_ones(self) -> None:
        """The batcher drains the queue into a batch before inference, so
        counting the queue lets the real limit become max_queue_depth +
        max_batch_size without the setting ever saying so."""
        import asyncio

        from service.batcher import Batcher, QueueFull
        from service.engine import StubEngine

        async def scenario() -> None:
            settings = Settings(max_queue_depth=2, max_batch_size=8, batch_wait_ms=1.0)
            batcher = Batcher(StubEngine(delay_ms=200.0), settings)
            await batcher.start()
            try:
                frame = np.zeros((3, 32, 32), dtype=np.float32)
                pending = [asyncio.create_task(batcher.submit(frame)) for _ in range(2)]
                await asyncio.sleep(0.05)

                # Both frames are inside the engine now and the queue reads
                # empty, but they are still outstanding, so the third is
                # refused rather than quietly accepted.
                with pytest.raises(QueueFull):
                    await batcher.submit(frame)

                for task in pending:
                    task.cancel()
            finally:
                await batcher.stop()

        asyncio.run(scenario())


class TestThreadResolution:
    """Threads were pinned to 2 for a 2-vCPU host, which wasted a 14-core one."""

    def test_an_explicit_setting_is_respected(self) -> None:
        from service.config import resolve_threads

        assert resolve_threads(4) == 4

    def test_zero_means_decide_from_the_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from service import config

        monkeypatch.setattr(config.os, "cpu_count", lambda: 14)
        assert config.resolve_threads(0) == 7

    def test_it_never_takes_every_core(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # All 14 threads measured 48.6 ms against 14.0 ms at 8: past a point
        # they synchronise over a 128x128 tensor for longer than they compute.
        from service import config

        monkeypatch.setattr(config.os, "cpu_count", lambda: 64)
        assert config.resolve_threads(0) == 8

    def test_a_single_core_host_still_gets_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from service import config

        monkeypatch.setattr(config.os, "cpu_count", lambda: 1)
        assert config.resolve_threads(0) == 1
