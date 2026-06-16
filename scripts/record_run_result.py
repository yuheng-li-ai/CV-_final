#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cvhw3_scene.draft import append_draft_entry
from cvhw3_scene.results import collect_run_result, result_to_draft_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append an outputs/{run_id} result to docs/draft.md.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--draft", default="docs/draft.md")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--elapsed_time", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--cause_analysis", required=True)
    parser.add_argument("--next_step", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_result = collect_run_result(Path(args.outputs_root) / args.run_id)
    entry = result_to_draft_entry(
        run_result,
        phase=args.phase,
        goal=args.goal,
        elapsed_time=args.elapsed_time,
        result_status=args.result,
        cause_analysis=args.cause_analysis,
        next_step=args.next_step,
        date=args.date,
    )
    text = append_draft_entry(Path(args.draft), entry, dry_run=args.dry_run)
    print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
