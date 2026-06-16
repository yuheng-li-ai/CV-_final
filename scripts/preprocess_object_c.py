#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cvhw3_scene.object_c import ObjectCPreprocessConfig, preprocess_object_c


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Object C auto-mask inputs for Magic123.")
    parser.add_argument("--input_image", default="data/scene/object_c/raw/object_c.jpg")
    parser.add_argument("--output_image", default="data/scene/object_c/masked/object_c_auto.png")
    parser.add_argument("--output_mask", default="data/scene/object_c/masks/object_c_auto.png")
    parser.add_argument("--output_metadata", default="data/scene/object_c/masks/object_c_auto.json")
    parser.add_argument("--threshold", type=int, default=245)
    parser.add_argument("--padding", type=int, default=64)
    parser.add_argument("--keep_largest_component", action="store_true")
    parser.add_argument("--close_kernel", type=int, default=0)
    parser.add_argument("--open_kernel", type=int, default=0)
    parser.add_argument("--feather_radius", type=int, default=0)
    parser.add_argument("--no_centered_crop", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = preprocess_object_c(
        ObjectCPreprocessConfig(
            input_image=Path(args.input_image),
            output_image=Path(args.output_image),
            output_mask=Path(args.output_mask),
            output_metadata=Path(args.output_metadata),
            threshold=args.threshold,
            padding=args.padding,
            keep_largest_component=args.keep_largest_component,
            close_kernel=args.close_kernel,
            open_kernel=args.open_kernel,
            feather_radius=args.feather_radius,
            centered_crop=not args.no_centered_crop,
            dry_run=args.dry_run,
        )
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
