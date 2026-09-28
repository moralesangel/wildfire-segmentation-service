"""The HTTP and WebSocket surface, including what it promises about itself."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.app import app
from service.config import FRAME_BUDGET_MS


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _jpeg(width: int = 320, height: int = 240) -> bytes:
    """A frame with something fire-coloured in it, so the stub has signal."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 2] = 40  # dim blue background
    arr[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3, 0] = 230  # red patch
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG")
    return buffer.getvalue()


class TestHealth:
    def test_health_reports_the_budget_it_is_held_to(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["frame_budget_ms"] == pytest.approx(FRAME_BUDGET_MS, abs=0.1)

    def test_health_admits_when_there_is_no_model(self, client: TestClient) -> None:
        """Without a checkpoint the service still serves, so it has to say so.
        A stub mask that passed for a prediction would be the worst outcome."""
        body = client.get("/health").json()
        assert body["model_loaded"] is False
        assert body["engine"] == "stub"


class TestMetrics:
    def test_metrics_are_prometheus_formatted(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "wfs_frame_latency_seconds" in response.text

    def test_the_budget_bucket_exists(self, client: TestClient) -> None:
        """The histogram must have a bucket edge at the budget, or the number
        the service is judged on is an interpolation."""
        text = client.get("/metrics").text
        assert 'le="0.0667"' in text


class TestSegmentStream:
    def test_a_frame_comes_back_as_a_mask(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/segment") as ws:
            ws.send_bytes(_jpeg())
            reply = ws.receive_json()

        assert reply["type"] == "mask"
        assert reply["engine"] == "stub"
        png = base64.b64decode(reply["mask_png_b64"])
        with Image.open(io.BytesIO(png)) as mask:
            assert mask.mode == "L"
            # At the model's resolution, not the source frame's. Upscaling it
            # here cost 237 ms a frame and added no information; the client
            # scales it while compositing. See preprocess.encode_mask.
            assert mask.size == (128, 128)

    def test_every_reply_carries_its_latency_and_verdict(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/segment") as ws:
            ws.send_bytes(_jpeg())
            reply = ws.receive_json()

        assert isinstance(reply["latency_ms"], (int, float))
        assert isinstance(reply["within_budget"], bool)
        assert reply["within_budget"] == (reply["latency_ms"] <= FRAME_BUDGET_MS)

    def test_a_corrupt_frame_is_reported_not_fatal(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/segment") as ws:
            ws.send_bytes(b"this is not a jpeg")
            reply = ws.receive_json()
            assert reply["type"] == "error"

            # The socket survives, so one bad frame does not end the stream.
            ws.send_bytes(_jpeg())
            assert ws.receive_json()["type"] == "mask"
