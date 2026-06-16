#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ORDERED_INPUTS = [
    ("01_front.png", "a front.jpg"),
    ("02_front_right.png", "a front right.jpg"),
    ("03_left_front.png", "a left front.jpg"),
    ("04_right.png", "a right.jpg"),
    ("05_left_back.png", "a left back.jpg"),
    ("06_back.png", "a back.jpg"),
    ("07_top.png", "a top.jpg"),
]

SUBSETS = {
    "frames_sparse": ["01_front.png", "02_front_right.png", "03_left_front.png"],
    "frames_medium": [
        "01_front.png",
        "02_front_right.png",
        "03_left_front.png",
        "04_right.png",
        "05_left_back.png",
    ],
    "frames_dense": [
        "01_front.png",
        "02_front_right.png",
        "03_left_front.png",
        "04_right.png",
        "05_left_back.png",
        "06_back.png",
        "07_top.png",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage curated 7-view Object A images.")
    parser.add_argument("--src", default=".")
    parser.add_argument("--root", default="data/scene/object_a")
    parser.add_argument("--contact_sheet", default="reports/figures/object_a_reselect_7_contact.jpg")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_images(src_root: Path, target_dir: Path) -> list[Path]:
    reset_dir(target_dir)
    normalized: list[Path] = []
    for output_name, input_name in ORDERED_INPUTS:
        source = src_root / input_name
        if not source.exists():
            raise SystemExit(f"Missing input image: {source}")
        output = target_dir / output_name
        with Image.open(source) as image:
            image.convert("RGB").save(output)
        normalized.append(output)
    return normalized


def stage_subsets(root: Path, normalized_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for subset_name, names in SUBSETS.items():
        subset_dir = root / subset_name
        reset_dir(subset_dir)
        for index, name in enumerate(names, start=1):
            shutil.copy2(normalized_dir / name, subset_dir / f"frame_{index:06d}.png")
        counts[subset_name] = len(names)
    return counts


def resize_to_width(path: Path, width: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        height = round(rgb.height * width / rgb.width)
        return rgb.resize((width, height), Image.Resampling.LANCZOS)


def make_contact_sheet(paths: list[Path], output: Path, width: int = 220) -> None:
    thumbs = [resize_to_width(path, width) for path in paths]
    label_h = 28
    padding = 10
    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    cell_h = max(thumb.height for thumb in thumbs) + label_h
    canvas = Image.new("RGB", (columns * width + (columns + 1) * padding, rows * cell_h + (rows + 1) * padding), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (thumb, path) in enumerate(zip(thumbs, paths)):
        col = idx % columns
        row = idx // columns
        x = padding + col * (width + padding)
        y = padding + row * (cell_h + padding)
        draw.text((x, y), path.stem, fill=(0, 0, 0))
        canvas.paste(thumb, (x, y + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    normalized_dir = root / "reselect_7"
    normalized = normalize_images(Path(args.src), normalized_dir)
    counts = stage_subsets(root, normalized_dir)
    make_contact_sheet(normalized, Path(args.contact_sheet))
    print(f"normalized_dir={normalized_dir}")
    print(f"contact_sheet={args.contact_sheet}")
    for key, value in counts.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
