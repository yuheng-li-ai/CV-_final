#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvhw3_scene.probe import probe_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Task 1 local inputs and external tools.")
    parser.add_argument(
        "--media",
        nargs="*",
        default=["data/scene/object_a/raw/object_a.mp4", "data/scene/object_c/raw/object_c.jpg"],
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["ffmpeg", "ffprobe", "scripts/colmap_clean_env.sh", "blender", "nvidia-smi"],
    )
    parser.add_argument("--out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = probe_report(args.media, args.tools)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
