"""Prometheus metrics for the latency budget.

Histograms rather than averages, and the stages timed separately. A single
end-to-end mean cannot answer the question this service exists to ask: when a
frame misses its budget, which stage spent the time?

The bucket edges are chosen around the 66.7 ms budget so the percentile that
matters falls inside a bucket boundary rather than being interpolated across a
wide one.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Buckets in seconds, dense around the 66.7 ms budget.
_LATENCY_BUCKETS = (
    0.005,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
    0.0667,  # the budget itself
    0.080,
    0.100,
    0.150,
    0.200,
    0.300,
    0.500,
    1.0,
)

frame_latency = Histogram(
    "wfs_frame_latency_seconds",
    "End-to-end latency for one frame: decode, preprocess, inference, encode.",
    buckets=_LATENCY_BUCKETS,
)

stage_latency = Histogram(
    "wfs_stage_latency_seconds",
    "Latency of one pipeline stage.",
    labelnames=("stage",),
    buckets=_LATENCY_BUCKETS,
)
"""Stages are timed apart because the interesting answer is often that the
model was never the bottleneck. Decoding a 4K frame and resizing it to 128x128
can cost more than the inference it feeds."""

frames_received = Counter(
    "wfs_frames_received_total",
    "Frames accepted from clients.",
)

frames_dropped = Counter(
    "wfs_frames_dropped_total",
    "Frames refused because the queue was full.",
    labelnames=("reason",),
)
"""Drops are a design outcome, not an error. Under overload the service sheds
load instead of queueing it, so this counter rising while latency stays flat
is the system working as intended."""

frames_processed = Counter(
    "wfs_frames_processed_total",
    "Frames that produced a mask.",
)

budget_misses = Counter(
    "wfs_budget_misses_total",
    "Frames whose end-to-end latency exceeded the stated budget.",
)

queue_depth = Gauge(
    "wfs_queue_depth",
    "Frames currently waiting for inference.",
)

batch_size = Histogram(
    "wfs_batch_size",
    "Frames per inference call.",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16),
)
"""Batching only helps if batches actually form. A distribution stuck at 1
means the batch window is expiring empty and the added latency buys nothing."""

model_loaded = Gauge(
    "wfs_model_loaded",
    "1 when a model is loaded and able to serve, 0 otherwise.",
)
