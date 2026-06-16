#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


FRAME_RE = re.compile(r"\bFra:(\d+)\b")
TOTAL_RE = re.compile(r"FUSION_RENDER: animation frames=1\.\.(\d+)")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _progress_line(current: int, total: int, status: str) -> str:
    width = 30
    ratio = 0.0 if total <= 0 else min(max(current / total, 0.0), 1.0)
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    percent = ratio * 100.0
    return f"BLENDER {percent:6.2f}%|{bar}| {current}/{total} frame {status}"


def _extract_progress(text: str) -> tuple[int, int, str]:
    totals = [int(match.group(1)) for match in TOTAL_RE.finditer(text)]
    frames = [int(match.group(1)) for match in FRAME_RE.finditer(text)]
    total = totals[-1] if totals else 0
    current = max(frames) if frames else 0
    if "Error:" in text or "Traceback" in text:
        status = "failed"
    elif total and current >= total:
        status = "done"
    elif current:
        status = "running"
    elif "FUSION_RENDER: still frame 1/1" in text:
        total = 1
        current = 1 if "Saved:" in text or "Time:" in text else 0
        status = "done" if current else "running"
    else:
        status = "waiting"
    return current, total, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Blender fusion render progress from a nohup log.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    path = Path(args.log)
    while True:
        text = _read_text(path)
        current, total, status = _extract_progress(text)
        if total:
            print(_progress_line(current, total, status), flush=True)
        else:
            tail = text.strip().splitlines()[-1:] or ["log not created yet"]
            print(f"BLENDER waiting | {tail[0]}", flush=True)
        if args.once or status in {"done", "failed"}:
            return 0 if status != "failed" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
