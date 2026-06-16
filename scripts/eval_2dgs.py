#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render test views and compute 2DGS metrics.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--render_script", required=True)
    parser.add_argument("--metrics_script", required=True)
    parser.add_argument("-s", "--source_scene", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", default="-1")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=None)
    parser.add_argument("--depth_trunc", default=None)
    parser.add_argument("--sdf_trunc", default=None)
    parser.add_argument("--num_cluster", default=None)
    parser.add_argument("--mesh_res", default=None)
    parser.add_argument("--unbounded", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_cuda_check", action="store_true")
    return parser


def run(command: list[str], dry_run: bool) -> None:
    print("COMMAND:", " ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = build_parser().parse_args()
    if not args.dry_run and not args.no_cuda_check and not cuda_available():
        print("ERROR: CUDA is not available in this process; 2DGS render/metrics require CUDA.")
        return 3
    model_path = Path(args.model_path)
    render_command = [
        args.python,
        args.render_script,
        "-s",
        args.source_scene,
        "-m",
        args.model_path,
        "--iteration",
        str(args.iteration),
    ]
    if args.skip_train:
        render_command.append("--skip_train")
    if args.skip_test:
        render_command.append("--skip_test")
    if args.skip_mesh:
        render_command.append("--skip_mesh")
    if args.quiet:
        render_command.append("--quiet")
    if args.render_path:
        render_command.append("--render_path")
    for value, flag in (
        (args.voxel_size, "--voxel_size"),
        (args.depth_trunc, "--depth_trunc"),
        (args.sdf_trunc, "--sdf_trunc"),
        (args.num_cluster, "--num_cluster"),
        (args.mesh_res, "--mesh_res"),
    ):
        if value is not None:
            render_command.extend([flag, str(value)])
    if args.unbounded:
        render_command.append("--unbounded")

    metrics_command = [
        args.python,
        args.metrics_script,
        "-m",
        args.model_path,
    ]

    run(render_command, args.dry_run)
    if not args.skip_test:
        run(metrics_command, args.dry_run)

    results_path = model_path / "results.json"
    if results_path.exists():
        print("RESULTS_JSON:", results_path)
        print(json.dumps(json.loads(results_path.read_text(encoding="utf-8")), indent=2))
    elif not args.dry_run and not args.skip_test:
        print(f"WARNING: metrics completed but {results_path} was not found.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
