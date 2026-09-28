"""The inference engine, and a stand-in for when there is no checkpoint yet.

The service is written so the model is the replaceable part. `StubEngine`
exists so the queue, the drop policy, the metrics and the budget tests can all
be exercised before a trained checkpoint exists -- and so a container with no
model still starts and explains itself instead of crash-looping.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np

from service import metrics


class Engine(Protocol):
    """Anything that turns a batch of frames into a batch of masks."""

    name: str

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """(N, 3, H, W) float32 in [0, 1] -> (N, H, W) float32 in [0, 1]."""
        ...


class StubEngine:
    """A deterministic fake that costs a little time and returns a plausible mask.

    Not a mock in the testing sense: it runs in the real service when no
    checkpoint is present, so the endpoint, the metrics and the client all work
    end to end. Every response says which engine produced it, so a stub mask is
    never mistaken for a prediction.
    """

    name = "stub"

    def __init__(self, delay_ms: float = 8.0) -> None:
        self._delay_s = delay_ms / 1000.0

    def infer(self, batch: np.ndarray) -> np.ndarray:
        time.sleep(self._delay_s)
        # Fire is bright and red-shifted: a crude red-minus-blue response gives
        # a mask that moves with the image instead of a constant blob, which
        # makes a stubbed demo obviously alive but obviously not a model.
        red, blue = batch[:, 0], batch[:, 2]
        mask: np.ndarray = np.clip((red - blue) * 2.0, 0.0, 1.0).astype(np.float32)
        return mask


class OnnxEngine:
    """ONNX Runtime over the exported U-Net."""

    name = "onnx"

    def __init__(self, model_path: str | Path, threads: int = 2) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # Both thread pools are pinned. Left at their defaults, ORT sizes them
        # from the host's core count, which on a shared 2-vCPU container means
        # heavy oversubscription and worse latency than single-threaded.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, batch: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self._input_name: batch})
        logits: np.ndarray = outputs[0]
        # Squeeze the channel axis of a single-class segmentation head.
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        result: np.ndarray = logits.astype(np.float32)
        return result


def load_engine(model_path: str | Path, threads: int = 2) -> Engine:
    """The ONNX engine when a checkpoint exists, the stub when it does not.

    Falling back rather than raising is deliberate: a missing model should be
    visible on `/health` and `/metrics`, not a container that will not boot.
    """
    path = Path(model_path)
    if path.is_file():
        engine = OnnxEngine(path, threads=threads)
        metrics.model_loaded.set(1)
        return engine

    metrics.model_loaded.set(0)
    return StubEngine()
