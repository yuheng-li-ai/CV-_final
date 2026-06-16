#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


FIELDS = [
    "run_id",
    "model_path",
    "cameras",
    "images",
    "registered_images",
    "points",
    "observations",
    "mean_track_length",
    "mean_observations_per_image",
    "mean_reprojection_error_px",
]

PATTERNS = {
    "cameras": r"^Cameras:\s+(\d+)",
    "images": r"^Images:\s+(\d+)",
    "registered_images": r"^Registered images:\s+(\d+)",
    "points": r"^Points:\s+(\d+)",
    "observations": r"^Observations:\s+(\d+)",
    "mean_track_length": r"^Mean track length:\s+([0-9.]+)",
    "mean_observations_per_image": r"^Mean observations per image:\s+([0-9.]+)",
    "mean_reprojection_error_px": r"^Mean reprojection error:\s+([0-9.]+)px",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect COLMAP model_analyzer stats into CSV.")
    parser.add_argument("--model", action="append", required=True, help="run_id=path/to/sparse/0")
    parser.add_argument("--colmap", default="scripts/colmap_clean_env.sh")
    parser.add_argument("--out", required=True)
    return parser


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--model must be formatted as run_id=path/to/sparse/0")
    run_id, path = value.split("=", 1)
    return run_id, Path(path)


def parse_stats(text: str) -> dict[str, str]:
    stats: dict[str, str] = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.MULTILINE)
        stats[key] = match.group(1) if match else ""
    return stats


def analyze(colmap: str, model_path: Path) -> str:
    completed = subprocess.run(
        [colmap, "model_analyzer", "--path", str(model_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def main() -> int:
    args = build_parser().parse_args()
    rows = []
    for item in args.model:
        run_id, model_path = parse_model(item)
        stats = parse_stats(analyze(args.colmap, model_path))
        rows.append({"run_id": run_id, "model_path": str(model_path), **stats})

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
