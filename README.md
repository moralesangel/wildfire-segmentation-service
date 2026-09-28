# Wildfire Segmentation Service

[![CI](https://github.com/moralesangel/wildfire-segmentation-service/actions/workflows/ci.yml/badge.svg)](https://github.com/moralesangel/wildfire-segmentation-service/actions/workflows/ci.yml)
[![Budget](https://img.shields.io/badge/budget-66.7ms%20p95-38bdf8)](#the-budget)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A wildfire segmentation model served over a WebSocket, under a latency budget
that was **stated before it was measured**.

This is not a modelling project. The model is the least interesting part of it.
It is a serving project: what happens to latency under load, where the frame
budget actually goes, and what the service does when it cannot keep up.

## The budget

> **66.7 ms per frame at p95**, which is 15 FPS — the rate below which a drone
> feed stops reading as video and starts reading as a slideshow.

The budget is end to end: decode, preprocess, inference, encode. Measuring only
the model is how a service ships at three times its stated latency.

It is written in [`config.py`](src/service/config.py) and asserted in the tests,
so it cannot quietly become whatever the benchmark happened to produce.

## The result

**The budget holds.** With the trained U-Net, measured end to end on an idle
machine:

| Stage | p50 | p95 | share |
|---|---|---|---|
| decode | 44.5 ms | 48.1 ms | **80.9%** |
| inference | 9.9 ms | 11.1 ms | 18.0% |
| encode | 0.4 ms | 0.5 ms | 0.7% |
| **total** | **54.9 ms** | **58.6 ms** | |

58.6 ms against a 66.7 ms budget: **within, at 17.1 FPS sustainable**.

And the point of the exercise is in the third column. **Decoding one 4K frame
costs four and a half times the inference it feeds.** A service that reported
model latency would have called itself fast at 9.9 ms while spending 81% of its
budget somewhere it never looked.

The model: Dice **0.889**, IoU **0.800**, recall 0.895, precision 0.883 on a
held-out fifth of FLAME. 1.9M parameters. Best epoch was 40 of 40 — it hit the
budget still improving, so this is a floor rather than the architecture's
ceiling.

## What the measurement found

Two things, and the first one contradicted the design.

**The encode was three quarters of the budget.** The first version returned the
mask upscaled to the source frame's 3840×2160, because that is what a client
"wants". It cost **237 ms per frame against 0.3 ms** at the model's own
resolution — nearly 800×, and 2.3 MB on the wire instead of 16 KB — to produce
pixels carrying no information the 128×128 mask did not already have. The client
scales it while compositing, which its GPU does for free.

Measured with the stub engine, before and after:

| Stage | Before | After |
|---|---|---|
| decode | 35.7 ms | 35.2 ms |
| inference | 8.5 ms | 8.6 ms |
| **encode** | **89.8 ms** | **0.7 ms** |
| **total p95** | **137.4 ms** | **47.0 ms** |
| sustainable | 7.3 FPS | **21.3 FPS** |

**Then the decode dominates, not the model.** At 79% of the remaining time,
decoding one 4K JPEG costs four times the inference it feeds. `Image.draft()`
halves it — 74.7 ms → 35.7 ms — by letting the JPEG decoder skip to a smaller
DCT scale while reading, so the full-resolution bitmap is never built. Without
that one call the service does not meet its budget at all.

The lesson is the ordinary one in serving work: **the model was never the
bottleneck.** A service reporting only model latency would have called itself
fast at 8.6 ms and shipped something that could not hold frame rate.

Reproduce with `python scripts/benchmark.py --frames 40 --sizes 128,256`.

## Backpressure: it drops frames on purpose

When the queue is full the service **refuses the frame and says so**, rather
than accepting it and falling behind.

An unbounded queue does not make a service faster. It converts overload into
staleness: every frame still gets processed, each one further behind reality,
until the operator is watching a fire line as it looked ten seconds ago. For a
live feed that is worse than a gap. The client is told immediately, while the
frame it sent is still current.

The bound counts **frames outstanding**, not frames sitting in the queue. That
distinction was a real bug: the batcher drains the queue into a batch before
running inference, so a new frame could arrive to find the queue empty while a
full batch was still in the engine — accepted, queued behind it, and the
effective limit silently became `max_queue_depth + max_batch_size`.
[`tests/test_batcher.py`](tests/test_batcher.py) pins the corrected behaviour.

## Architecture

```
browser (drone video)
   │  frames over WebSocket
   ▼
FastAPI  ──►  bounded queue  ──►  ONNX Runtime  ──►  mask
   │           (drops, never         (U-Net)
   │            lags)
   ▼
/metrics  →  p50 / p95 / p99, FPS, queue depth, drops, budget misses
```

Each stage is timed separately in Prometheus histograms, with a bucket edge
**at** 66.7 ms so the percentile the service is judged on is a real count and
not an interpolation across a wide bucket.

## Running it

The trained model ships with the repo (7.8 MB), so a clone serves real
predictions rather than the stub.

```bash
uv venv && uv pip install -e ".[dev]"
uv run uvicorn service.app:app --port 8000
```

Or with Docker, which is the same image that deploys, so what runs locally is
what runs in production:

```bash
docker build -t wfs .
docker run --rm -p 7860:7860 wfs
```

Then open `http://127.0.0.1:8000` (or `:7860` under Docker), load a video or
point it at your camera, and press Start. `/health` and `/metrics` are there
too.

Two things worth doing once it is up:

- **Press Stress test.** Frames go out as fast as the browser can send them,
  the queue overflows, and *dropped* climbs while latency stays flat. That is
  the drop policy working — the service shedding load rather than falling
  behind.
- **Watch the split, not the total.** `/metrics` breaks latency down by stage,
  and on a 4K source the decode is most of it.

Note that Docker limits the container's CPU differently from a bare process, so
the numbers under Docker will be slower than the ones above. Both are honest;
they answer different questions.

**With no checkpoint the service still starts**, serving a stub engine that
returns a crude red-minus-blue response. That is deliberate: a container that
refuses to boot gives an operator nothing to look at. Every response carries
`"engine": "stub"` and `/health` reports `model_loaded: false`, so a placeholder
mask can never be mistaken for a prediction.

```bash
python scripts/benchmark.py --frames 40 --sizes 128,256   # where the budget goes
python -m pytest -q                                        # 13 tests
```

## Reproducing it

```bash
# 1. Cache the 4K frames at 256x256 (5.3 GB -> 282 MB, runs once)
python scripts/prepare_data.py --images Images.zip --masks Masks.zip

# 2. Train (40 epochs, ~40 s each on 14 CPU cores)
python scripts/train.py --epochs 40 --size 128 --base 16

# 3. Export, with parity against PyTorch checked and the file deleted if it fails
python scripts/export_onnx.py

# 4. Measure
python scripts/benchmark.py --frames 60 --sizes 128
```

The export reported `max |pytorch - onnx| = 8.08e-06` against a 1e-5 tolerance,
and `max |batched - single| = 0.0` — batching does not change per-item results,
so a frame's mask does not depend on who it was batched with.

## Status

Trained, exported, measured, and within budget. 28 tests, CI green.

Outstanding:

- [ ] Deploy to Hugging Face Spaces. Its free tier is 2 vCPU against the 14
      cores measured here, so **the budget will probably not hold there**. The
      deployed numbers will be published next to these either way.
- [ ] Multi-seed runs. Dice 0.889 is one run on one split, which makes it an
      observation rather than a measurement.
- [ ] Raise the epoch cap. Best epoch was 40 of 40, so training stopped on
      budget rather than on convergence.

## Data

[FLAME](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
(DOI 10.21227/qad6-r683): 2,003 aerial frames at 3840×2160 with pixel-level fire
masks. **Academic and non-commercial use only**, and it is not redistributed
here. Note that `Training.zip` and `Test.zip` are the *classification* subset —
segmentation needs item 9, "Fire segmentation frames" (4.98 GB), whose frames
are the ones the masks correspond to.

> A. Shamsoshoara, F. Afghah, A. Razi, L. Zheng, P. Fulé, E. Blasch, "The FLAME
> dataset: Aerial Imagery Pile burn detection using drones (UAVs)", IEEE
> Dataport, 2020.

## Limitations

- **Hugging Face Spaces' free tier is CPU with 2 vCPU.** The thread pools are
  pinned to match; left at their defaults ONNX Runtime sizes them from the
  host's core count and oversubscribes a shared container.
- **The Space sleeps.** The first request after idle pays a cold start of
  roughly 30 seconds. That is a property of the free tier, not of the service.
- **The measurements above are from one machine** (Intel Ultra 5 225H, no GPU)
  against a synthetic 4K frame. Absolute numbers will differ elsewhere; the
  shape — decode dominating inference — should not.
- **No GPU path.** The engine interface would take one, but nothing here has
  been tested against it.

## License

MIT for the code. The dataset carries its own terms; see above.
