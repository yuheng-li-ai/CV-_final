#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a contact sheet from extracted video frames.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--width", type=int, default=180)
    parser.add_argument("--cols", type=int, default=4)
    return parser


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indexes = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[index] for index in indexes]


def resized(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> int:
    args = build_parser().parse_args()
    image_dir = Path(args.image_dir)
    paths = select_evenly(sorted(image_dir.glob("*.png")), args.count)
    if not paths:
        raise SystemExit(f"No PNG frames found under {image_dir}")

    thumbs = [(path, resized(path, args.width)) for path in paths]
    label_height = 24
    cell_height = max(image.height for _, image in thumbs) + label_height
    rows = (len(thumbs) + args.cols - 1) // args.cols
    canvas = Image.new("RGB", (args.width * args.cols, cell_height * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(thumbs):
        col = index % args.cols
        row = index // args.cols
        x = col * args.width
        y = row * cell_height
        draw.text((x + 6, y + 5), path.name, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_height))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
