#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run threestudio from the project harness.")
    parser.add_argument("--repo", default="external/threestudio")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "validate", "test", "export"], default="train")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gradio_progress", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    return parser


def _repo_relative_or_absolute(repo: Path, path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    repo_candidate = repo / candidate
    if repo_candidate.exists():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


def build_command(args: argparse.Namespace, project_root: Path) -> tuple[list[str], Path]:
    repo = (project_root / args.repo).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _repo_relative_or_absolute(repo, args.config)
    exp_root_dir = output_dir.parent
    name = output_dir.name

    command = [
        sys.executable,
        "launch.py",
        "--config",
        config,
        f"--{args.mode}",
        "--gpu",
        str(args.gpu),
    ]
    if args.gradio_progress:
        command.append("--gradio")

    overrides = [
        f"exp_root_dir={exp_root_dir}",
        f"name={name}",
        f"tag={args.tag}",
        "use_timestamp=false",
    ]
    if args.prompt:
        overrides.append(f"system.prompt_processor.prompt={args.prompt}")
    if args.image_path:
        overrides.append(f"data.image_path={(project_root / args.image_path).resolve()}")
        overrides.append(f"system.guidance_3d.cond_image_path={(project_root / args.image_path).resolve()}")
    if args.max_steps is not None:
        overrides.extend(
            [
                f"trainer.max_steps={args.max_steps}",
                f"checkpoint.every_n_train_steps={args.max_steps}",
                f"system.guidance.trainer_max_steps={args.max_steps}",
            ]
        )
    if args.width is not None:
        overrides.append(f"data.width={args.width}")
    if args.height is not None:
        overrides.append(f"data.height={args.height}")
    if args.batch_size is not None:
        overrides.append(f"data.batch_size={args.batch_size}")
    if args.val_check_interval is not None:
        overrides.append(f"trainer.val_check_interval={args.val_check_interval}")
    if args.resume:
        overrides.append(f"resume={(project_root / args.resume).resolve()}")
    overrides.extend(args.extra)
    return command + overrides, repo


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd().resolve()
    command, repo = build_command(args, project_root)
    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("TCNN_CUDA_ARCHITECTURES", "86")
    project_src = str(project_root / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        project_src if not current_pythonpath else project_src + os.pathsep + current_pythonpath
    )
    print("THREESTUDIO_COMMAND:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=repo, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
