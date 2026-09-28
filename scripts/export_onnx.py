"""Export the trained U-Net to ONNX, and prove the export did not change it.

An export that silently alters the model is worse than one that fails: the
service would serve subtly different predictions and every metric would still
look healthy. So this does not just write the file, it runs both graphs over
the same inputs and refuses to keep the result if they disagree.

The sigmoid is folded into the exported graph. The service then has no
post-processing step to get wrong, and the mask it receives is already a
probability.

    python scripts/export_onnx.py --checkpoint models/unet_fire.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from train import UNet

# Tighter than float32 round-off but loose enough to survive ONNX Runtime
# choosing different kernels. The emergency-vehicle-detection port came out at
# 7.7e-9 against Keras; anything near that is a faithful export.
PARITY_TOLERANCE = 1e-5


class ServableUNet(nn.Module):
    """The trained net with its sigmoid attached, so the graph outputs a
    probability rather than a logit the caller has to remember to squash."""

    def __init__(self, net: UNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def check_parity(model: nn.Module, onnx_path: Path, size: int, trials: int = 8) -> float:
    """Largest absolute disagreement between PyTorch and ONNX Runtime.

    Random inputs rather than real frames on purpose: a real frame exercises
    the distribution the model was trained on, where both graphs are most
    likely to agree. Noise pushes activations through ranges a photo never
    would, which is where an export bug shows.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)
    worst = 0.0

    model.eval()
    for _ in range(trials):
        sample = rng.random((1, 3, size, size), dtype=np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(sample)).numpy()
        actual = session.run(None, {input_name: sample})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))

    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/unet_fire.pt"))
    parser.add_argument("--out", type=Path, default=Path("models/unet_fire.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    size = int(checkpoint.get("size", 128))
    base = int(checkpoint.get("base", 16))

    net = UNet(base=base)
    net.load_state_dict(checkpoint["state_dict"])
    model = ServableUNet(net).eval()
    print(f"loaded {args.checkpoint} (base={base}, size={size})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, size, size),
        str(args.out),
        input_names=["input"],
        output_names=["mask"],
        # The batch axis is dynamic because the service batches whatever
        # arrives in its window; a fixed axis would force a graph per size.
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    worst = check_parity(model, args.out, size)
    print(f"max |pytorch - onnx| over 8 random inputs: {worst:.3e}")

    if worst > PARITY_TOLERANCE:
        args.out.unlink(missing_ok=True)
        raise SystemExit(
            f"parity check failed: {worst:.3e} > {PARITY_TOLERANCE:.0e}. "
            "Export removed rather than shipped; a model that changed on export "
            "would serve different predictions with healthy-looking metrics."
        )

    print(f"parity OK (tolerance {PARITY_TOLERANCE:.0e})")

    # Batching must not change per-item results. If it does, a frame's mask
    # depends on who it happened to be batched with, which is not reproducible.
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    rng = np.random.default_rng(1)
    batch = rng.random((4, 3, size, size), dtype=np.float32)
    batched = session.run(None, {name: batch})[0]
    singles = np.concatenate([session.run(None, {name: batch[i : i + 1]})[0] for i in range(4)])
    drift = float(np.abs(batched - singles).max())
    print(f"max |batched - single| : {drift:.3e}")
    if drift > PARITY_TOLERANCE:
        raise SystemExit("batching changes per-item output; the service must not batch this graph")


if __name__ == "__main__":
    main()
