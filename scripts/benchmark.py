"""Measure where the frame budget actually goes.

The question this answers is not "how fast is the model" but "which stage
spends the 66.7 ms". With 4K source frames the decode can cost more than the
inference it feeds, and a service tuned on model latency alone would optimise
the wrong half.

    python scripts/benchmark.py --frames 60
    python scripts/benchmark.py --source path/to/frame.jpg --sizes 128,256
"""

from __future__ import annotations

import argparse
import io
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from service import preprocess
from service.config import FRAME_BUDGET_MS, Settings
from service.engine import load_engine


def synthetic_frame(width: int, height: int) -> bytes:
    """A 4K-ish JPEG, for measuring decode cost without the dataset present."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[int(len(ordered) * 0.95) - 1] if len(ordered) >= 20 else max(ordered),
        "p99": ordered[int(len(ordered) * 0.99) - 1] if len(ordered) >= 100 else max(ordered),
        "mean": statistics.fmean(ordered),
    }


def run(payload: bytes, size: int, frames: int, model_path: str) -> None:
    settings = Settings(input_size=size, model_path=model_path)
    engine = load_engine(settings.model_path, threads=settings.onnx_threads)

    decode_ms: list[float] = []
    infer_ms: list[float] = []
    encode_ms: list[float] = []

    # One pass to warm caches and let ORT settle; timing the first call
    # measures initialisation, not steady state.
    warm = preprocess.decode(payload, size)
    engine.infer(warm.tensor[None])

    for _ in range(frames):
        started = time.perf_counter()
        decoded = preprocess.decode(payload, size)
        decode_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        mask = engine.infer(decoded.tensor[None])[0]
        infer_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        preprocess.encode_mask(mask)
        encode_ms.append((time.perf_counter() - started) * 1000)

    total = [d + i + e for d, i, e in zip(decode_ms, infer_ms, encode_ms, strict=True)]

    print(f"\n=== input {size}x{size} · engine {engine.name} · {frames} frames ===")
    print(f"{'stage':<12}{'p50':>9}{'p95':>9}{'mean':>9}   share of p50")
    for name, samples in (
        ("decode", decode_ms),
        ("inference", infer_ms),
        ("encode", encode_ms),
    ):
        stats = percentiles(samples)
        share = stats["p50"] / percentiles(total)["p50"] * 100
        print(
            f"{name:<12}{stats['p50']:>8.1f}{stats['p95']:>9.1f}"
            f"{stats['mean']:>9.1f}   {share:>5.1f}%"
        )

    stats = percentiles(total)
    verdict = "WITHIN" if stats["p95"] <= FRAME_BUDGET_MS else "OVER"
    print(f"{'TOTAL':<12}{stats['p50']:>8.1f}{stats['p95']:>9.1f}{stats['mean']:>9.1f}")
    print(f"\nbudget {FRAME_BUDGET_MS:.1f} ms at p95 -> {verdict} ({stats['p95']:.1f} ms)")
    print(f"sustainable rate at p95: {1000 / stats['p95']:.1f} FPS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--sizes", default="128", help="Comma-separated input sizes.")
    parser.add_argument("--source", help="A real frame; synthetic 4K if omitted.")
    parser.add_argument("--model", default="models/unet_fire.onnx")
    parser.add_argument("--source-width", type=int, default=3840)
    parser.add_argument("--source-height", type=int, default=2160)
    args = parser.parse_args()

    if args.source:
        payload = Path(args.source).read_bytes()
        with Image.open(io.BytesIO(payload)) as img:
            print(f"source: {args.source} ({img.size[0]}x{img.size[1]})")
    else:
        payload = synthetic_frame(args.source_width, args.source_height)
        print(f"source: synthetic {args.source_width}x{args.source_height}")

    for size in (int(s) for s in args.sizes.split(",")):
        run(payload, size, args.frames, args.model)


if __name__ == "__main__":
    main()
