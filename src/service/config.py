"""Service settings, and the latency budget the whole design answers to.

The budget is stated here rather than discovered from a benchmark, because a
target chosen after seeing the numbers is not a target. Everything else in the
service -- the queue bound, the drop policy, the batch size -- exists to keep
this promise or to report honestly that it could not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------
TARGET_FPS = 15
"""Frames per second the service aims to sustain.

15 is the low end of what reads as video to a person. Below about 10 the
motion visibly steps, which for a drone feed watching a fire line is the
difference between a tool and a slideshow.
"""

FRAME_BUDGET_MS = 1000.0 / TARGET_FPS
"""66.7 ms. The wall-clock budget for one frame, end to end.

End to end means decode + preprocess + inference + encode, not just the model.
Measuring only the model is how a service ships at 3x its stated latency.
"""

P95_BUDGET_MS = FRAME_BUDGET_MS
"""The budget applies at p95, not at the mean.

A mean hides the tail, and the tail is what a viewer actually notices: one
frame in twenty arriving late is visible stutter.
"""


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, all overridable by environment variable."""

    model_path: str = "models/unet_fire.onnx"
    """ONNX checkpoint. Absent, the service starts and reports itself unhealthy
    rather than refusing to boot: a container that will not start gives an
    operator nothing to look at."""

    input_size: int = 128
    """Square side the model expects. The source frames are 4K, so this is also
    the single biggest lever on preprocessing cost."""

    max_queue_depth: int = 8
    """Frames allowed to wait. Small on purpose.

    A deep queue does not prevent overload, it converts overload into latency:
    the frames still arrive, they just arrive late, and the viewer sees a feed
    running behind reality. Eight frames is about half a second at target rate,
    which is the most staleness worth keeping."""

    max_batch_size: int = 4
    """Upper bound on dynamic batching. The measured curve decides the real
    value; see scripts/benchmark.py."""

    batch_wait_ms: float = 5.0
    """How long the batcher waits for a batch to fill before running short.

    Waiting trades latency for throughput. At 66 ms of total budget, 5 ms is
    about as much as can be spent hoping for company."""

    onnx_threads: int = 0
    """Intra-op threads, 0 meaning "decide from the host" -- see `resolve_threads`."""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=os.getenv("WFS_MODEL_PATH", cls.model_path),
            input_size=int(os.getenv("WFS_INPUT_SIZE", cls.input_size)),
            max_queue_depth=int(os.getenv("WFS_MAX_QUEUE_DEPTH", cls.max_queue_depth)),
            max_batch_size=int(os.getenv("WFS_MAX_BATCH_SIZE", cls.max_batch_size)),
            batch_wait_ms=float(os.getenv("WFS_BATCH_WAIT_MS", cls.batch_wait_ms)),
            onnx_threads=int(os.getenv("WFS_ONNX_THREADS", cls.onnx_threads)),
        )


def resolve_threads(requested: int) -> int:
    """Intra-op threads to give ONNX Runtime.

    A fixed 2 was right for Hugging Face Spaces' free tier and wasteful
    everywhere else: on a 14-core laptop it left most of the machine idle.
    Letting ONNX Runtime decide is worse still -- it sizes from the host's core
    count, and measured here that was the slowest setting of all:

        1 thread   32.3 ms      4 threads  20.4 ms
        2 threads  17.8 ms      8 threads  14.0 ms
                               14 threads  48.6 ms   <- all cores, 3.5x worse

    The cap exists because past a point the threads spend longer synchronising
    over a 128x128 tensor than computing it. Half the cores, bounded at 8,
    stays on the good side of that on every machine tested.
    """
    if requested > 0:
        return requested
    cores = os.cpu_count() or 2
    return max(1, min(8, cores // 2))
