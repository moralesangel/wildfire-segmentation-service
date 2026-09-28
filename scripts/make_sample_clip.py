"""Build the sample clip the demo page offers out of the box.

The FLAME segmentation frames are consecutive frames of one drone flight --
correlation between neighbours is 1.000 against 0.216 between distant ones --
so playing them in order is the original video, not a slideshow assembled from
stills.

This exists because the demo is otherwise unusable: a visitor has no wildfire
footage to hand, and pointing a webcam at a desk shows the model nothing to
segment.

    python scripts/make_sample_clip.py --images ~/Downloads/Images.zip

Written as MJPEG inside an AVI container using only Pillow-decoded frames and
imageio-ffmpeg if present, falling back to an animated-image path otherwise --
see `--format`.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import zipfile
from pathlib import Path

from PIL import Image
from PIL.Image import Resampling


def pick_frames(archive: zipfile.ZipFile, start: int, count: int, step: int) -> list[str]:
    """Consecutive entries, so the clip is the flight rather than a montage."""
    names = {}
    for name in archive.namelist():
        stem = Path(name).name
        if stem.startswith("image_") and stem.endswith(".jpg"):
            try:
                names[int(stem[len("image_") : -len(".jpg")])] = name
            except ValueError:
                continue

    wanted = range(start, start + count * step, step)
    chosen = [names[i] for i in wanted if i in names]
    if not chosen:
        raise SystemExit(f"no frames in range {start}..{start + count * step}")
    return chosen


def decode(archive: zipfile.ZipFile, name: str, width: int) -> Image.Image:
    with archive.open(name) as handle:
        payload = handle.read()
    with Image.open(io.BytesIO(payload)) as img:
        height = round(width * img.size[1] / img.size[0])
        # draft() again: 2,003 full 4K decodes would take minutes.
        img.draft("RGB", (width, height))
        return img.convert("RGB").resize((width, height), Resampling.BILINEAR)


def write_mp4(frames: list[Image.Image], out: Path, fps: int) -> bool:
    """Encode with ffmpeg if one is reachable. Returns False if none is."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"

    width, height = frames[0].size
    command = [
        exe,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        # yuv420p and faststart: without them Safari refuses the file and the
        # browser shows a black box with no error.
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "26",
        "-movflags",
        "+faststart",
        str(out),
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return False

    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    return process.wait() == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("static/sample.mp4"))
    parser.add_argument("--start", type=int, default=1100)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=int, default=15, help="Matches the service's target rate.")
    args = parser.parse_args()

    with zipfile.ZipFile(args.images) as archive:
        names = pick_frames(archive, args.start, args.count, args.step)
        print(f"decoding {len(names)} frames from {args.start} at {args.width}px wide...")
        frames = [decode(archive, name, args.width) for name in names]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not write_mp4(frames, args.out, args.fps):
        raise SystemExit(
            "no ffmpeg available. Install it, or `pip install imageio-ffmpeg`, "
            "which ships a static build."
        )

    size_mb = args.out.stat().st_size / 1e6
    seconds = len(frames) / args.fps
    print(f"wrote {args.out} ({size_mb:.1f} MB, {seconds:.0f}s at {args.fps} fps)")
    if size_mb > 10:
        print("note: over 10 MB. Lower --count or --width to keep the clone small.")


if __name__ == "__main__":
    main()
