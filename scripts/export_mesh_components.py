#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import trimesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export largest connected mesh components.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(input_path, process=False, force="mesh")
    components = sorted(mesh.split(only_watertight=False), key=lambda part: len(part.faces), reverse=True)
    rows = []
    for index, part in enumerate(components[: args.count]):
        out_path = out_dir / f"component_{index:02d}.ply"
        part.export(out_path)
        rows.append(
            {
                "component": index,
                "path": out_path,
                "vertices": len(part.vertices),
                "faces": len(part.faces),
                "bounds_min": " ".join(f"{value:.6f}" for value in part.bounds[0]),
                "bounds_max": " ".join(f"{value:.6f}" for value in part.bounds[1]),
                "extent": " ".join(f"{value:.6f}" for value in part.extents),
            }
        )

    with (out_dir / "components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(out_dir / "components.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
