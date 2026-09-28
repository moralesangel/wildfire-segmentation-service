"""Train the U-Net on the prepared FLAME cache.

The architecture is the one validated in brain-tumor-segmentation: a 4-level
U-Net with BCE + soft Dice. What differs is the imbalance. Fire is 0.3-0.6% of
pixels against roughly 1.5% for tumours, so the background term dominates even
harder and pixel accuracy is worth nothing: a model predicting all-background
scores 99.6% and finds no fire at all. That baseline is printed explicitly.

    python scripts/train.py --data data/flame_256.npz --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

SEED = 42
SMOOTH = 1e-6


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level U-Net. `base` is deliberately small: this has to run on 2 vCPU
    inside the frame budget, and width costs more there than depth does."""

    def __init__(self, in_ch: int = 3, base: int = 16, levels: int = 4) -> None:
        super().__init__()
        self.levels = levels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = in_ch
        for i in range(levels):
            width = base * 2**i
            self.encoders.append(conv_block(channels, width))
            channels = width

        self.bottleneck = conv_block(channels, base * 2**levels)
        channels = base * 2**levels

        for i in reversed(range(levels)):
            width = base * 2**i
            self.ups.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
            self.decoders.append(conv_block(width * 2, width))
            channels = width

        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            x = up(x)
            x = decoder(torch.cat([x, skip], dim=1))

        # Logits. The sigmoid lives in the loss during training and is baked
        # into the graph at export, so the served model needs no post-step.
        return self.head(x)


# --------------------------------------------------------------------------
# Loss and metrics
# --------------------------------------------------------------------------
def dice_coefficient(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probs * target).sum()
    return (2 * intersection + SMOOTH) / (probs.sum() + target.sum() + SMOOTH)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE alone collapses to background at this imbalance; the Dice term is
    what gives the positive class any gradient worth having."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + (1 - dice_coefficient(torch.sigmoid(logits), target))


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, threshold: float = 0.5
) -> dict[str, float]:
    model.eval()
    intersection = predicted = actual = 0.0
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size])
        probs = torch.sigmoid(logits)
        hard = (probs > threshold).float()
        batch_y = y[start : start + batch_size]
        intersection += float((hard * batch_y).sum())
        predicted += float(hard.sum())
        actual += float(batch_y.sum())

    dice = (2 * intersection + SMOOTH) / (predicted + actual + SMOOTH)
    iou = (intersection + SMOOTH) / (predicted + actual - intersection + SMOOTH)
    recall = (intersection + SMOOTH) / (actual + SMOOTH)
    precision = (intersection + SMOOTH) / (predicted + SMOOTH)
    return {"dice": dice, "iou": iou, "recall": recall, "precision": precision}


def augment(
    x: torch.Tensor, y: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flips only. The same geometric transform is applied to image and mask;
    anything that moved one without the other would train against wrong labels
    and raise no error at all."""
    if torch.rand(1, generator=generator).item() < 0.5:
        x, y = torch.flip(x, dims=[3]), torch.flip(y, dims=[3])
    if torch.rand(1, generator=generator).item() < 0.5:
        x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/flame_256.npz"))
    parser.add_argument("--out", type=Path, default=Path("models/unet_fire.pt"))
    parser.add_argument("--history", type=Path, default=Path("models/history.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    cache = np.load(args.data)
    x_all, y_all = cache["x"], cache["y"]
    print(f"loaded {len(x_all)} frames at {x_all.shape[1]}x{x_all.shape[2]}")

    x = torch.from_numpy(x_all).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(y_all).unsqueeze(1).float()
    if x.shape[-1] != args.size:
        x = F.interpolate(x, size=(args.size, args.size), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(args.size, args.size), mode="nearest")
        print(f"resized to {args.size}x{args.size}")

    # Held out by index, deterministically, so the split survives a rerun.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x))
    cut = int(len(x) * 0.8)
    train_idx, val_idx = order[:cut], order[cut:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(f"train {len(x_train)} | val {len(x_val)}")

    fire_fraction = float(y.mean())
    print(f"\nfire pixels: {fire_fraction * 100:.3f}%")
    print(f"all-background baseline: accuracy {100 * (1 - fire_fraction):.2f}%, Dice 0.000")
    print("-> pixel accuracy is meaningless here; Dice and recall are the metrics\n")

    model = UNet(base=args.base)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=4
    )
    generator = torch.Generator().manual_seed(SEED)

    history: list[dict[str, float]] = []
    best_dice = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        permutation = torch.randperm(len(x_train), generator=generator)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, len(x_train), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            xb, yb = augment(x_train[batch], y_train[batch], generator)

            optimiser.zero_grad()
            loss = bce_dice_loss(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss)
            batches += 1

        metrics = evaluate(model, x_val, y_val, args.batch_size)
        scheduler.step(metrics["dice"])
        elapsed = time.perf_counter() - started

        history.append({"epoch": epoch, "loss": epoch_loss / batches, **metrics})
        print(
            f"epoch {epoch:3d}  loss {epoch_loss / batches:.4f}  "
            f"val dice {metrics['dice']:.4f}  iou {metrics['iou']:.4f}  "
            f"recall {metrics['recall']:.4f}  prec {metrics['precision']:.4f}  "
            f"{elapsed:.0f}s",
            flush=True,
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {"state_dict": model.state_dict(), "base": args.base, "size": args.size},
                args.out,
            )

    print(f"\nbest val Dice {best_dice:.4f} -> {args.out}")
    args.history.write_text(
        json.dumps(
            {
                "history": history,
                "best_dice": best_dice,
                "params": params,
                "size": args.size,
                "base": args.base,
                "fire_fraction": fire_fraction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
