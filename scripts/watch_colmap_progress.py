#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


STAGE_WEIGHTS = {
    "waiting": 0.0,
    "feature": 0.2,
    "matching": 0.4,
    "mapping": 0.7,
    "done": 1.0,
    "failed": 1.0,
}


FINAL_ELAPSED_RE = re.compile(r"Elapsed time:\s+[0-9.]+\s+\[minutes\]\s*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a compact progress bar for COLMAP logs.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--images", type=int, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


def parse_progress(text: str, images: int) -> dict[str, object]:
    if "Traceback" in text or re.search(r"\bERROR\b|failed", text, re.IGNORECASE):
        stage = "failed"
    elif FINAL_ELAPSED_RE.search(text.strip()):
        stage = "done"
    elif "Registering image" in text or "Bundle adjustment" in text or "Mapper" in text:
        stage = "mapping"
    elif "Matching block" in text or "Matching image" in text or "Exhaustive feature matching" in text:
        stage = "matching"
    elif "Feature extraction" in text or "Processed file" in text:
        stage = "feature"
    else:
        stage = "waiting"

    processed = 0
    processed_matches = re.findall(r"Processed file \[(\d+)/(\d+)\]", text)
    if processed_matches:
        processed = int(processed_matches[-1][0])

    registered_numbers = [int(value) for value in re.findall(r"Registering image #\d+ \((\d+)\)", text)]
    registered = max(registered_numbers) if registered_numbers else 0

    if stage == "feature" and images:
        percent = min(0.2, 0.2 * processed / images)
    elif stage == "matching":
        percent = 0.4
    elif stage == "mapping" and images:
        percent = min(0.95, 0.4 + 0.5 * registered / images)
    else:
        percent = STAGE_WEIGHTS[stage]

    return {
        "stage": stage,
        "percent": percent,
        "processed": processed,
        "registered": registered,
        "done": stage in {"done", "failed"},
    }


def render_bar(progress: dict[str, object], images: int, width: int) -> str:
    percent = float(progress["percent"])
    filled = min(width, max(0, int(round(percent * width))))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"COLMAP [{bar}] {percent * 100:5.1f}% "
        f"stage={progress['stage']} "
        f"features={progress['processed']}/{images} "
        f"registered~={progress['registered']}/{images}"
    )


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    args = build_parser().parse_args()
    log_path = Path(args.log)
    while True:
        progress = parse_progress(read_text(log_path), args.images)
        print("\r" + render_bar(progress, args.images, args.width), end="", flush=True)
        if args.once or progress["done"]:
            print()
            return 1 if progress["stage"] == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
