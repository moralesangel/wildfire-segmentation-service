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

## What the measurement found

Two things, and the first one contradicted the design.

**The encode was three quarters of the budget.** The first version returned the
mask upscaled to the source frame's 3840×2160, because that is what a client
"wants". It cost **237 ms per frame against 0.3 ms** at the model's own
resolution — nearly 800×, and 2.3 MB on the wire instead of 16 KB — to produce
pixels carrying no information the 128×128 mask did not already have. The client
scales it while compositing, which its GPU does for free.

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

```bash
uv venv && uv pip install -e ".[dev]"
uv run uvicorn service.app:app --reload
```

Then `http://127.0.0.1:8000` for the demo, `/health`, `/metrics`.

**With no checkpoint the service still starts**, serving a stub engine that
returns a crude red-minus-blue response. That is deliberate: a container that
refuses to boot gives an operator nothing to look at. Every response carries
`"engine": "stub"` and `/health` reports `model_loaded: false`, so a placeholder
mask can never be mistaken for a prediction.

```bash
python scripts/benchmark.py --frames 40 --sizes 128,256   # where the budget goes
python -m pytest -q                                        # 13 tests
```

## Status

The service is complete and measured. **The model is not trained yet** — the
numbers above come from the stub engine, whose 8.6 ms is a stand-in rather than
a prediction of the real U-Net's cost.

Outstanding:

- [ ] Train the U-Net on FLAME (item 9 + masks), reusing the pipeline from
      [brain-tumor-segmentation](https://github.com/moralesangel/brain-tumor-segmentation)
- [ ] Export to ONNX with numerical parity verified against PyTorch
- [ ] Re-measure with the real model and publish the honest FPS
- [ ] Deploy to Hugging Face Spaces (its free tier is CPU; the budget may not
      hold there, and this README will say so either way)

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
