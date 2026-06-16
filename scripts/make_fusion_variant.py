#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _asset_by_name(config: dict, name: str) -> dict:
    for asset in config["inputs"]["assets"]:
        if asset["name"] == name:
            return asset
    raise ValueError(f"Missing asset: {name}")


def _parse_vec(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z vector, got: {value}")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Phase6 fusion config variant.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--a_height", type=float)
    parser.add_argument("--b_height", type=float)
    parser.add_argument("--c_height", type=float)
    parser.add_argument("--a_location")
    parser.add_argument("--b_location")
    parser.add_argument("--c_location")
    parser.add_argument("--a_rotation")
    parser.add_argument("--b_rotation")
    parser.add_argument("--c_rotation")
    parser.add_argument("--shadow", choices=["true", "false"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["experiment"]["name"] = args.name
    config["fusion"]["variant"] = args.variant
    config["inputs"]["fused_scene"] = f"{args.run_dir}/scene.json"
    config["paths"]["output_dir"] = args.run_dir
    config["paths"]["video_path"] = f"{args.run_dir}/final_video.mp4"
    if args.shadow is not None:
        config["fusion"]["enable_shadow"] = args.shadow == "true"

    edits = {
        "object_a": (args.a_height, args.a_location, args.a_rotation),
        "object_b": (args.b_height, args.b_location, args.b_rotation),
        "object_c": (args.c_height, args.c_location, args.c_rotation),
    }
    for name, (height, location, rotation) in edits.items():
        asset = _asset_by_name(config, name)
        if height is not None:
            asset["target_height"] = height
        if location is not None:
            asset["location"] = _parse_vec(location)
        if rotation is not None:
            asset["rotation_deg"] = _parse_vec(rotation)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
