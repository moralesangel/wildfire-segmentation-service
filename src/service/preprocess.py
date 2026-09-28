"""Turning an encoded frame into a model input, and a mask back into pixels.

This module is timed separately from inference on purpose. The source frames
in FLAME are 3840x2160, and decoding one JPEG of that size costs tens of
milliseconds on a CPU -- plausibly more than running a small U-Net over the
128x128 crop it produces. A service that reports only model latency would call
that fast and ship something that cannot hold frame rate.
"""

from __future__ import annotations

import io
import time
from typing import NamedTuple

import numpy as np
from PIL import Image
from PIL.Image import Resampling

from service import metrics


class Decoded(NamedTuple):
    """A frame ready for the model, with the size it arrived at."""

    tensor: np.ndarray
    """Shape (3, size, size), float32 in [0, 1], channels first."""

    source_size: tuple[int, int]
    """The frame's own (width, height), kept so the mask can be sent back at
    the resolution the client is showing."""


def decode(payload: bytes, size: int) -> Decoded:
    """Decode an encoded frame and resize it to the model's input.

    `Image.draft` is what makes 4K affordable: it lets the JPEG decoder skip
    straight to a smaller DCT scale while reading, so the full-resolution
    bitmap is never materialised. Without it the decode dominates the budget.
    """
    started = time.perf_counter()

    with Image.open(io.BytesIO(payload)) as opened:
        source_size = opened.size
        # Ask the decoder for the smallest scale that still covers the target.
        # A no-op for formats that do not support it, and the single largest
        # saving in the pipeline: 74.7 ms -> 35.7 ms on a 4K JPEG.
        opened.draft("RGB", (size, size))
        resized = opened.convert("RGB").resize((size, size), Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0

    tensor: np.ndarray = np.transpose(arr, (2, 0, 1))
    metrics.stage_latency.labels(stage="decode").observe(time.perf_counter() - started)
    return Decoded(tensor=tensor, source_size=source_size)


def encode_mask(mask: np.ndarray) -> bytes:
    """Encode a probability mask as a PNG, at the model's own resolution.

    Deliberately *not* upscaled to the source frame's size. Doing that was the
    first version, and the benchmark showed it costing 237 ms per frame against
    0.3 ms here -- nearly 800x, and 2.3 MB on the wire instead of 16 KB. It was
    three quarters of the entire frame budget spent producing pixels that carry
    no information the 128x128 mask did not already have.

    The client scales it instead, which its GPU does for free while compositing
    the overlay. Sent single-channel so the browser picks the colour.
    """
    started = time.perf_counter()

    arr = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    with Image.fromarray(arr, mode="L") as img:
        img.save(buffer, format="PNG", optimize=False)

    metrics.stage_latency.labels(stage="encode").observe(time.perf_counter() - started)
    return buffer.getvalue()
