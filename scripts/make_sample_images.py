"""Generate a tiny labeled image set (colored shapes) so the full pipeline can
be run end-to-end locally with no dataset download.

Usage:  python scripts/make_sample_images.py --out data/images --n 24
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 80),
    "blue": (40, 90, 220),
    "yellow": (230, 210, 40),
    "purple": (150, 40, 200),
}
SHAPES = ["circle", "square", "triangle"]


def draw(path: Path, color: str, shape: str, size: int = 256, seed: int = 0) -> None:
    rng = random.Random(seed)
    bg = tuple(rng.randint(20, 60) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    c = COLORS[color]
    m = size // 5
    box = [m, m, size - m, size - m]
    if shape == "circle":
        d.ellipse(box, fill=c)
    elif shape == "square":
        d.rectangle(box, fill=c)
    else:
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=c)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make synthetic sample images.")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    combos = [(c, s) for c in COLORS for s in SHAPES]
    for i in range(args.n):
        color, shape = combos[i % len(combos)]
        draw(out / f"{i:03d}_{color}_{shape}.png", color, shape, seed=i)
    print(f"Wrote {args.n} sample images to {out}")


if __name__ == "__main__":
    main()
