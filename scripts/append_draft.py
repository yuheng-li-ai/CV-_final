#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from cvhw3_scene.draft import DraftEntry, append_draft_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append a Task 1 experiment entry to docs/draft.md.")
    parser.add_argument("--draft", default="docs/draft.md")
    parser.add_argument("--date", default=None)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--elapsed_time", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--metrics", default="none")
    parser.add_argument("--figure_paths", default="none")
    parser.add_argument("--video_paths", default="none")
    parser.add_argument("--cause_analysis", required=True)
    parser.add_argument("--next_step", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = append_draft_entry(
        Path(args.draft),
        DraftEntry(
            date=date,
            phase=args.phase,
            run_id=args.run_id,
            goal=args.goal,
            command=args.command,
            config=args.config,
            hardware=args.hardware,
            elapsed_time=args.elapsed_time,
            result=args.result,
            metrics=args.metrics,
            figure_paths=args.figure_paths,
            video_paths=args.video_paths,
            cause_analysis=args.cause_analysis,
            next_step=args.next_step,
        ),
        dry_run=args.dry_run,
    )
    print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
