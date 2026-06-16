#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cvhw3_scene.inputs import InputStagingPlan, stage_local_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage local Task 1 raw inputs into data/scene.")
    parser.add_argument("--object_a_video", default="任务A.mp4")
    parser.add_argument("--object_c_image", default="任务C.jpg")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    staged = stage_local_inputs(
        InputStagingPlan(
            object_a_video=Path(args.object_a_video),
            object_c_image=Path(args.object_c_image),
            data_root=Path(args.data_root),
            dry_run=args.dry_run,
        )
    )
    print(staged.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
