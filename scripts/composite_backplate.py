#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _fit_background(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image_ratio = image.size[0] / image.size[1]
    target_ratio = size[0] / size[1]
    if image_ratio > target_ratio:
        new_width = int(image.size[1] * target_ratio)
        left = (image.size[0] - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.size[1]))
    elif image_ratio < target_ratio:
        new_height = int(image.size[0] / target_ratio)
        top = (image.size[1] - new_height) // 2
        image = image.crop((0, top, image.size[0], top + new_height))
    return image.resize(size, Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Composite transparent Blender foreground over a 2DGS backplate.")
    parser.add_argument("--foreground", required=True)
    parser.add_argument("--backplate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    foreground_path = Path(args.foreground)
    backplate_path = Path(args.backplate)
    output_path = Path(args.output)
    foreground = Image.open(foreground_path).convert("RGBA")
    background = _fit_background(backplate_path, foreground.size).convert("RGBA")
    background.alpha_composite(foreground)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(output_path, quality=95)
    print(f"COMPOSITED: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
