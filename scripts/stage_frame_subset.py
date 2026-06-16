#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy an evenly spaced frame subset for smoke tests.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start", type=int, default=0, help="0-based start index before stride.")
    parser.add_argument("--stride", type=int, default=0, help="Use consecutive strided frames when >0.")
    parser.add_argument("--resume", action="store_true")
    return parser


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indexes = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[index] for index in indexes]


def main() -> int:
    args = build_parser().parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists() and any(dst.iterdir()) and not args.resume:
        raise SystemExit(f"Destination is non-empty: {dst}. Use --resume to refresh.")
    dst.mkdir(parents=True, exist_ok=True)

    frames = sorted(src.glob("*.png"))
    if args.stride > 0:
        selected = frames[args.start :: args.stride][: args.count]
    else:
        selected = select_evenly(frames, args.count)
    if not selected:
        raise SystemExit(f"No PNG frames found under {src}")
    for index, source in enumerate(selected, start=1):
        shutil.copy2(source, dst / f"frame_{index:06d}.png")
    print(f"copied={len(selected)} dst={dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
