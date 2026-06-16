#!/usr/bin/env python
from __future__ import annotations

import argparse

from cvhw3_scene.checklist import format_checklist_status, load_checklist_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Task 1 checklist completion state.")
    parser.add_argument("--checklist", default="docs/checklist.yaml")
    parser.add_argument("--required_section", default="required_keys")
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    status = load_checklist_status(args.checklist, required_section=args.required_section)
    print(format_checklist_status(status))
    if status.required_false and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
