#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_id",
    "model_path",
    "method",
    "iteration",
    "psnr",
    "ssim",
    "lpips",
    "render_count",
    "gt_count",
    "results_json",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect 2DGS metrics into a CSV table.")
    parser.add_argument("--model", action="append", required=True, help="run_id=path/to/model")
    parser.add_argument("--out", required=True)
    return parser


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--model must be formatted as run_id=path/to/model")
    run_id, path = value.split("=", 1)
    return run_id, Path(path)


def method_iteration(method: str) -> str:
    if "_" not in method:
        return ""
    return method.rsplit("_", 1)[1]


def collect_row(run_id: str, model_path: Path, method: str, metrics: dict[str, float]) -> dict[str, object]:
    test_dir = model_path / "test" / method
    render_count = len(list((test_dir / "renders").glob("*.png"))) if (test_dir / "renders").exists() else 0
    gt_count = len(list((test_dir / "gt").glob("*.png"))) if (test_dir / "gt").exists() else 0
    return {
        "run_id": run_id,
        "model_path": str(model_path),
        "method": method,
        "iteration": method_iteration(method),
        "psnr": metrics.get("PSNR", ""),
        "ssim": metrics.get("SSIM", ""),
        "lpips": metrics.get("LPIPS", ""),
        "render_count": render_count,
        "gt_count": gt_count,
        "results_json": str(model_path / "results.json"),
    }


def main() -> int:
    args = build_parser().parse_args()
    rows: list[dict[str, object]] = []
    for item in args.model:
        run_id, model_path = parse_model(item)
        results_path = model_path / "results.json"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for method, metrics in data.items():
            rows.append(collect_row(run_id, model_path, method, metrics))

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
