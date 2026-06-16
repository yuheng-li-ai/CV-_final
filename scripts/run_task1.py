#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 1 common run wrapper.")
    parser.add_argument("--cli_command", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = [
        sys.executable,
        "-m",
        "cvhw3_scene",
        args.cli_command,
        "--config",
        args.config,
        "--run_id",
        args.run_id,
        "--device",
        args.device,
    ]
    if args.dry_run:
        command.append("--dry_run")
    if args.resume:
        command.append("--resume")
    print("TASK1_COMMAND:", " ".join(command))
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
