#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["run_id", "fps", "image_dir", "frame_count", "first_frame", "last_frame"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect frame extraction counts into a CSV table.")
    parser.add_argument("--variant", action="append", required=True, help="run_id:fps:path/to/images")
    parser.add_argument("--out", required=True)
    return parser


def parse_variant(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--variant must be formatted as run_id:fps:path/to/images")
    return parts[0], parts[1], Path(parts[2])


def main() -> int:
    args = build_parser().parse_args()
    rows = []
    for value in args.variant:
        run_id, fps, image_dir = parse_variant(value)
        frames = sorted(image_dir.glob("*.png"))
        rows.append(
            {
                "run_id": run_id,
                "fps": fps,
                "image_dir": str(image_dir),
                "frame_count": len(frames),
                "first_frame": str(frames[0]) if frames else "",
                "last_frame": str(frames[-1]) if frames else "",
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
