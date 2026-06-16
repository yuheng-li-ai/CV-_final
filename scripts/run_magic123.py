#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official Magic123 from the project harness.")
    parser.add_argument("--repo", default="external/Magic123")
    parser.add_argument("--mode", choices=["preprocess", "coarse", "fine", "both"], default="coarse")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt", default="A high-resolution DSLR image of an object")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", default="object_c")
    parser.add_argument("--coarse_iters", type=int, default=5000)
    parser.add_argument("--fine_iters", type=int, default=5000)
    parser.add_argument("--sd_version", default="1.5")
    parser.add_argument("--hf_key", default=None)
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--lambda_guidance_coarse", default="1.0,40")
    parser.add_argument("--lambda_guidance_fine", default="1e-3,0.01")
    parser.add_argument("--guidance_scale_coarse", default="100,5")
    parser.add_argument("--guidance_scale_fine", default="100,5")
    parser.add_argument("--dataset_size_train", type=int, default=None)
    parser.add_argument("--dataset_size_valid", type=int, default=None)
    parser.add_argument("--dataset_size_test", type=int, default=None)
    parser.add_argument("--extra", action="append", default=[])
    return parser


def _pair(value: str) -> list[str]:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two values, got: {value}")
    return parts


def _workspace(output_dir: Path, tag: str, stage: str) -> Path:
    return output_dir / tag / stage


def _coarse_ckpt(output_dir: Path, tag: str) -> Path:
    workspace = _workspace(output_dir, tag, "coarse")
    return workspace / "checkpoints" / f"{workspace.name}.pth"


def _preprocess_command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "preprocess_image.py", "--path", str(Path(args.image_path).resolve())]


def _coarse_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    lambda_guidance = _pair(args.lambda_guidance_coarse)
    guidance_scale = _pair(args.guidance_scale_coarse)
    command = [
        sys.executable,
        "main.py",
        "-O",
        "--text",
        args.prompt,
        "--sd_version",
        args.sd_version,
        "--image",
        str(Path(args.image_path).resolve()),
        "--workspace",
        str(_workspace(output_dir, args.tag, "coarse").resolve()),
        "--optim",
        "adam",
        "--iters",
        str(args.coarse_iters),
        "--guidance",
        "SD",
        "zero123",
        "--lambda_guidance",
        *lambda_guidance,
        "--guidance_scale",
        *guidance_scale,
        "--latent_iter_ratio",
        "0",
        "--normal_iter_ratio",
        "0.2",
        "--t_range",
        "0.2",
        "0.6",
        "--bg_radius",
        "-1",
        "--save_mesh",
    ]
    if args.hf_key:
        command.extend(["--hf_key", args.hf_key])
    _append_dataset_sizes(command, args)
    command.extend(args.extra)
    return command


def _fine_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    lambda_guidance = _pair(args.lambda_guidance_fine)
    guidance_scale = _pair(args.guidance_scale_fine)
    init_ckpt = Path(args.init_ckpt).resolve() if args.init_ckpt else _coarse_ckpt(output_dir, args.tag)
    command = [
        sys.executable,
        "main.py",
        "-O",
        "--text",
        args.prompt,
        "--sd_version",
        args.sd_version,
        "--image",
        str(Path(args.image_path).resolve()),
        "--workspace",
        str(_workspace(output_dir, args.tag, "fine").resolve()),
        "--dmtet",
        "--init_ckpt",
        str(init_ckpt),
        "--iters",
        str(args.fine_iters),
        "--optim",
        "adam",
        "--known_view_interval",
        "4",
        "--latent_iter_ratio",
        "0",
        "--guidance",
        "SD",
        "zero123",
        "--lambda_guidance",
        *lambda_guidance,
        "--guidance_scale",
        *guidance_scale,
        "--rm_edge",
        "--bg_radius",
        "-1",
        "--save_mesh",
    ]
    if args.hf_key:
        command.extend(["--hf_key", args.hf_key])
    _append_dataset_sizes(command, args)
    command.extend(args.extra)
    return command


def _append_dataset_sizes(command: list[str], args: argparse.Namespace) -> None:
    values = {
        "--dataset_size_train": args.dataset_size_train,
        "--dataset_size_valid": args.dataset_size_valid,
        "--dataset_size_test": args.dataset_size_test,
    }
    for flag, value in values.items():
        if value is not None:
            command.extend([flag, str(value)])


def _run(command: list[str], repo: Path, env: dict[str, str]) -> int:
    print("MAGIC123_COMMAND:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=repo, env=env).returncode


def _prepend_ld_library_path(env: dict[str, str], paths: list[Path]) -> None:
    existing = env.get("LD_LIBRARY_PATH", "")
    values = [str(path) for path in paths if path.exists()]
    if existing:
        values.append(existing)
    if values:
        env["LD_LIBRARY_PATH"] = ":".join(values)


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd().resolve()
    repo = (project_root / args.repo).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env["CC"] = "/usr/bin/gcc-11"
    env["CXX"] = "/usr/bin/g++-11"
    env["CUDAHOSTCXX"] = "/usr/bin/g++-11"
    env["CMAKE_CUDA_HOST_COMPILER"] = "/usr/bin/g++-11"
    env.setdefault("MAX_JOBS", "2")
    env.setdefault("TORCH_EXTENSIONS_DIR", str(project_root / ".cache" / "torch_extensions_magic123"))
    pymeshlab_lib = Path(sysconfig.get_paths()["purelib"]) / "pymeshlab" / "lib"
    _prepend_ld_library_path(env, [pymeshlab_lib])
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if args.gpu == "auto":
        pass
    elif existing_visible_devices and args.gpu in {"0", "cuda:0"}:
        pass
    elif args.gpu.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = args.gpu.split(":", 1)[1]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.mode == "preprocess":
        return _run(_preprocess_command(args), repo, env)
    if args.mode == "coarse":
        return _run(_coarse_command(args, output_dir), repo, env)
    if args.mode == "fine":
        return _run(_fine_command(args, output_dir), repo, env)

    coarse_return = _run(_coarse_command(args, output_dir), repo, env)
    if coarse_return != 0:
        return coarse_return
    return _run(_fine_command(args, output_dir), repo, env)


if __name__ == "__main__":
    raise SystemExit(main())
