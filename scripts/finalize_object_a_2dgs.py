#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvhw3_scene.object_a import finalize_object_a_2dgs, parse_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize Phase3 Object A 2DGS metrics, previews, and missing-eval checks."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec formatted as run_id=path/to/model, e.g. full=runs/scene/object_a_2dgs_full.",
    )
    parser.add_argument("--out", default="reports/tables/object_a_2dgs_metrics.csv")
    parser.add_argument("--preview_dir", default="reports/figures")
    parser.add_argument("--expected_iteration", type=int, default=15000)
    parser.add_argument("--preview_count", type=int, default=4)
    parser.add_argument("--preview_width", type=int, default=320)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless all rows are complete.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = finalize_object_a_2dgs(
        models=[parse_model_spec(value) for value in args.model],
        out_csv=Path(args.out),
        preview_dir=Path(args.preview_dir),
        expected_iteration=args.expected_iteration,
        preview_count=args.preview_count,
        preview_width=args.preview_width,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
