"""Turn the FLAME zips into a training-ready cache.

The source frames are 3840x2160 and the archive is 5.3 GB. Decoding those on
every epoch would make the data loader, not the model, the thing being
trained -- so this runs once and writes a single compressed .npz the training
script memory-maps.

    python scripts/prepare_data.py --images ~/Downloads/Images.zip \\
                                   --masks ~/Downloads/Masks.zip

Two things about the masks that are easy to get wrong:

- They are **0/1 valued**, not 0/255. A `> 127` threshold silently yields an
  empty mask for every frame, and training proceeds against all-background
  labels without erroring.
- Fire is 0.3-0.6% of pixels. That imbalance is severe enough that pixel
  accuracy is meaningless and BCE alone collapses to predicting background.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import Resampling


def _ids(archive: zipfile.ZipFile, suffix: str) -> dict[int, str]:
    """Frame id -> entry name, for entries matching `image_<n><suffix>`."""
    found: dict[int, str] = {}
    for name in archive.namelist():
        stem = Path(name).name
        if not stem.startswith("image_") or not stem.endswith(suffix):
            continue
        try:
            found[int(stem[len("image_") : -len(suffix)])] = name
        except ValueError:
            continue
    return found


def prepare(images_zip: Path, masks_zip: Path, out: Path, size: int, limit: int | None) -> None:
    with zipfile.ZipFile(images_zip) as zi, zipfile.ZipFile(masks_zip) as zm:
        image_names = _ids(zi, ".jpg")
        mask_names = _ids(zm, ".png")

        paired = sorted(set(image_names) & set(mask_names))
        if not paired:
            raise SystemExit(
                "no image/mask pairs found. Images.zip must be item 9 "
                "('Fire segmentation frames'), not the classification Training.zip."
            )
        print(f"{len(image_names)} images, {len(mask_names)} masks, {len(paired)} paired")
        if limit:
            paired = paired[:limit]

        x = np.zeros((len(paired), size, size, 3), dtype=np.uint8)
        y = np.zeros((len(paired), size, size), dtype=np.uint8)

        for i, frame_id in enumerate(paired):
            with zi.open(image_names[frame_id]) as handle:
                payload = handle.read()
            with Image.open(io.BytesIO(payload)) as img:
                # Same draft() trick the service uses, for the same reason:
                # without it this pass decodes 2,003 full 4K bitmaps.
                img.draft("RGB", (size, size))
                x[i] = np.asarray(
                    img.convert("RGB").resize((size, size), Resampling.BILINEAR), dtype=np.uint8
                )

            with (
                zm.open(mask_names[frame_id]) as handle,
                Image.open(io.BytesIO(handle.read())) as mask,
            ):
                # NEAREST, not BILINEAR: interpolating a label map invents
                # in-between values that are neither fire nor background.
                resized = mask.convert("L").resize((size, size), Resampling.NEAREST)
                # > 0 rather than > 127, because the masks are 0/1.
                y[i] = (np.asarray(resized) > 0).astype(np.uint8)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paired)}")

    fire_fraction = float(y.mean())
    empty = int((y.sum(axis=(1, 2)) == 0).sum())
    print(f"\nfire pixels: {fire_fraction * 100:.3f}%")
    print(f"frames with no fire after resize: {empty} of {len(y)}")
    if fire_fraction == 0:
        raise SystemExit("every mask is empty; check the mask threshold")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x, y=y, ids=np.array(paired))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/flame_256.npz"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Fewer frames, for a quick pass.")
    args = parser.parse_args()

    prepare(args.images, args.masks, args.out, args.size, args.limit)


if __name__ == "__main__":
    main()
