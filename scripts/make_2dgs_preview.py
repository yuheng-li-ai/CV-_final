#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a compact GT/render preview grid for 2DGS.")
    parser.add_argument("--test_dir", required=True, help="Path like model/test/ours_5000.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    return parser


def resized(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> int:
    args = build_parser().parse_args()
    test_dir = Path(args.test_dir)
    render_paths = sorted((test_dir / "renders").glob("*.png"))[: args.count]
    if not render_paths:
        raise SystemExit(f"No render images found under {test_dir / 'renders'}")

    pairs = []
    for render_path in render_paths:
        gt_path = test_dir / "gt" / render_path.name
        if gt_path.exists():
            pairs.append((resized(gt_path, args.width), resized(render_path, args.width), render_path.name))

    label_height = 26
    row_height = max(max(gt.height, render.height) for gt, render, _ in pairs) + label_height
    canvas = Image.new("RGB", (args.width * 2, row_height * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (gt, render, name) in enumerate(pairs):
        y = index * row_height
        draw.text((8, y + 6), f"{name} GT", fill=(0, 0, 0))
        draw.text((args.width + 8, y + 6), f"{name} render", fill=(0, 0, 0))
        canvas.paste(gt, (0, y + label_height))
        canvas.paste(render, (args.width, y + label_height))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
