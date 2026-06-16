#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cvhw3_scene.gpu import format_gpu_report, query_gpu_memory, select_device, select_freest_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the GPU with the most free VRAM.")
    parser.add_argument("--device", default=None, help="Explicit device override, for example cuda:0.")
    parser.add_argument("--out", default=None, help="Optional path to write gpu.txt style report.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = None
    if args.device:
        device = select_device(args.device)
    else:
        rows = query_gpu_memory()
        device = select_freest_gpu(rows)
    report = format_gpu_report(device, rows)
    print(device)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
