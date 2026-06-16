#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preserve the old final A/B foreground and replace only Object C."
    )
    parser.add_argument("--old_scene", required=True, help="Scene JSON containing the old Object C transform.")
    parser.add_argument("--new_scene", required=True, help="Scene JSON containing the tuned new Object C transform.")
    parser.add_argument("--old_final_foreground_dir", required=True)
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--background_depth_dir", required=True)
    parser.add_argument("--camera_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--contact_sheet", required=True)
    parser.add_argument("--blender", default="external/blender/blender-4.5.4-linux-x64/blender")
    parser.add_argument("--blender_script", default="scripts/blender_fusion_scene.py")
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--alpha_threshold", type=int, default=8)
    parser.add_argument("--depth_bias", type=float, default=0.08)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _only_asset_scene(source: Path, target: Path, asset_name: str) -> None:
    scene = json.loads(source.read_text(encoding="utf-8"))
    scene["assets"] = [asset for asset in scene["assets"] if asset["name"] == asset_name]
    if len(scene["assets"]) != 1:
        raise ValueError(f"Expected exactly one {asset_name} asset in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(scene, indent=2), encoding="utf-8")


def _run(command: list[str], dry_run: bool) -> None:
    print("COMMAND:", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _progress(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="frame")


def _combine_foregrounds(args: argparse.Namespace, output_dir: Path) -> Path:
    old_final_dir = Path(args.old_final_foreground_dir)
    old_c_dir = output_dir / "old_c_mask"
    new_c_dir = output_dir / "new_c"
    combined_dir = output_dir / "combined_foreground"
    combined_dir.mkdir(parents=True, exist_ok=True)

    dilate_size = max(1, int(args.dilate))
    if dilate_size % 2 == 0:
        dilate_size += 1

    for frame in _progress(range(1, args.frames + 1), args.frames, "C_SWAP"):
        old_foreground = Image.open(old_final_dir / f"frame_{frame:04d}.png").convert("RGBA")
        old_c = Image.open(old_c_dir / f"frame_{frame:04d}.png").convert("RGBA")
        new_c = Image.open(new_c_dir / f"frame_{frame:04d}.png").convert("RGBA")

        mask = old_c.getchannel("A").filter(ImageFilter.MaxFilter(dilate_size))
        old_array = np.asarray(old_foreground).copy()
        mask_array = np.asarray(mask)
        remove_mask = mask_array > int(args.alpha_threshold)
        old_array[remove_mask, :] = 0

        combined = Image.fromarray(old_array, mode="RGBA")
        combined.alpha_composite(new_c)
        combined.save(combined_dir / f"frame_{frame:04d}.png")
    return combined_dir


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_c_scene = output_dir / "scene_old_c_only.json"
    new_c_scene = output_dir / "scene_new_c_only.json"
    _only_asset_scene(Path(args.old_scene), old_c_scene, "object_c")
    _only_asset_scene(Path(args.new_scene), new_c_scene, "object_c")

    common_blender = [
        args.blender,
        "-b",
        "--python",
        args.blender_script,
        "--",
        "--frames",
        str(args.frames),
        "--resolution",
        args.resolution,
        "--samples",
        str(args.samples),
        "--transparent",
        "--environment",
        "backplate_only",
        "--camera-path",
        "2dgs_json",
        "--camera-json",
        args.camera_json,
        "--camera-indices",
        f"0:{args.frames}",
    ]

    _run(
        common_blender
        + [
            "--scene",
            str(old_c_scene),
            "--output",
            str(output_dir / "old_c_mask" / "frame_"),
        ],
        args.dry_run,
    )
    _run(
        common_blender
        + [
            "--scene",
            str(new_c_scene),
            "--output",
            str(output_dir / "new_c" / "frame_"),
        ],
        args.dry_run,
    )

    if args.dry_run:
        print("DRY_RUN: would combine foregrounds and composite video", flush=True)
        return 0

    combined_dir = _combine_foregrounds(args, output_dir)
    composite_command = [
        sys.executable,
        "scripts/composite_foreground_sequence.py",
        "--foreground_dir",
        str(combined_dir),
        "--foreground_glob",
        "*.png",
        "--background_dir",
        args.background_dir,
        "--background_glob",
        "*.png",
        "--output_dir",
        str(output_dir / "composite_frames"),
        "--output_video",
        args.output_video,
        "--contact_sheet",
        args.contact_sheet,
        "--fps",
        str(args.fps),
        "--background_indices",
        f"0:{args.frames}",
        "--background_depth_dir",
        args.background_depth_dir,
        "--background_depth_glob",
        "depth_*.tiff",
        "--approx_scene",
        args.new_scene,
        "--camera_json",
        args.camera_json,
        "--camera_indices",
        f"0:{args.frames}",
        "--depth_bias",
        str(args.depth_bias),
    ]
    _run(composite_command, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
