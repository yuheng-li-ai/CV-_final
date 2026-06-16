from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import os
from pathlib import Path
import subprocess

from cvhw3_scene.colmap import ColmapRunner
from cvhw3_scene.config import DryRunResult, SceneConfig
from cvhw3_scene.data import FrameExtractor
from cvhw3_scene.fusion import FusionRenderer
from cvhw3_scene.gaussian import GaussianEvaluator, GaussianTrainer
from cvhw3_scene.generation import ImageTo3DAssetGenerator, TextTo3DAssetGenerator
from cvhw3_scene.gpu import select_device
from cvhw3_scene.recording import ExperimentRecorder
from cvhw3_scene.run_manager import RunManager
from cvhw3_scene.tracking import build_tracking_plan


def _add_config_command(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", required=True, help="Path to a scene YAML config.")
    parser.add_argument("--run_id", default=None, help="Run ID under outputs/ for formal execution.")
    parser.add_argument("--device", default="auto", help="Device override such as cuda:0, or auto.")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Print planned work.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing run directory.")
    parser.set_defaults(handler=CONFIG_HANDLERS[name])


def _run_config_command(args: argparse.Namespace, factory: Callable[[SceneConfig], DryRunResult]) -> int:
    config = SceneConfig.from_yaml(args.config)
    result = factory(config)
    result = _apply_resume_to_command(config, result, resume=args.resume)
    if not args.dry_run:
        if not args.run_id:
            raise SystemExit("--run_id is required for non-dry formal execution.")
        if result.validation_issues:
            print(result.format())
            print("REFUSING_TO_RUN: fix validation issues or use --dry_run for inspection only.")
            return 2
        device = select_device(None if args.device == "auto" else args.device)
        manager = RunManager(
            config_path=Path(args.config),
            run_id=args.run_id,
            command=result.command,
            device=device,
            resume=args.resume,
        )
        run_dir = manager.prepare()
        print(f"RUN_DIR: {run_dir}")
        print(f"COMMAND: {' '.join(result.command)}")
        env = os.environ.copy()
        env.update(build_tracking_plan(config.data).command_env())
        if config.command in {"train-2dgs", "eval-2dgs"}:
            env.setdefault("DISABLE_2DGS_NETWORK_GUI", "1")
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        cache_dir = run_dir / "cache"
        matplotlib_dir = run_dir / "matplotlib"
        cache_dir.mkdir(exist_ok=True)
        matplotlib_dir.mkdir(exist_ok=True)
        env.setdefault("XDG_CACHE_HOME", str(cache_dir))
        env.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
        _prepare_command_inputs(config, resume=args.resume)
        with (run_dir / "log.txt").open("a", encoding="utf-8") as log:
            completed = subprocess.run(result.command, stdout=log, stderr=subprocess.STDOUT, env=env)
        return completed.returncode
    print(result.format())
    return 0


def _apply_resume_to_command(config: SceneConfig, result: DryRunResult, resume: bool) -> DryRunResult:
    if not resume or config.command != "generate-text3d":
        return result
    output_dir = config.path_value("paths.output_dir")
    tag = str(config.get("runtime.tag", "main"))
    checkpoint = output_dir / tag / "ckpts" / "last.ckpt"
    if not checkpoint.exists():
        return result
    return replace(result, command=[*result.command, "--resume", str(checkpoint)])


def _prepare_command_inputs(config: SceneConfig, resume: bool) -> None:
    if config.command == "prepare-object":
        image_dir = config.path_value("paths.image_dir")
        if image_dir.exists() and any(image_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"Frame directory already contains files: {image_dir}. Use --resume to append/overwrite."
            )
        image_dir.mkdir(parents=True, exist_ok=True)
    if config.command == "run-colmap":
        workspace_dir = config.path_value("paths.workspace_dir")
        if workspace_dir.exists() and any(workspace_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"COLMAP workspace already contains files: {workspace_dir}. Use --resume to reuse it."
            )
        workspace_dir.mkdir(parents=True, exist_ok=True)


def _record_experiment(args: argparse.Namespace) -> int:
    recorder = ExperimentRecorder()
    entry = recorder.record(
        name=args.name,
        stage=args.stage,
        status=args.status,
        command=args.command,
        notes=args.notes,
        dry_run=args.dry_run,
    )
    print(entry.strip())
    return 0


CONFIG_HANDLERS: dict[str, Callable[[SceneConfig], DryRunResult]] = {
    "prepare-object": lambda config: FrameExtractor(config).dry_run(),
    "run-colmap": lambda config: ColmapRunner(config).dry_run(),
    "train-2dgs": lambda config: GaussianTrainer(config).dry_run(),
    "eval-2dgs": lambda config: GaussianEvaluator(config).dry_run(),
    "generate-text3d": lambda config: TextTo3DAssetGenerator(config).dry_run(),
    "generate-image3d": lambda config: ImageTo3DAssetGenerator(config).dry_run(),
    "fuse-scene": lambda config: FusionRenderer(config).dry_run_fusion(),
    "render-video": lambda config: FusionRenderer(config).dry_run_render(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvhw3_scene",
        description="Task 1 scene reconstruction, generation, fusion, and rendering harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("prepare-object", "Extract object frames from raw video."),
        ("run-colmap", "Run COLMAP reconstruction for extracted frames."),
        ("train-2dgs", "Train a 2D Gaussian Splatting model."),
        ("eval-2dgs", "Render and evaluate a trained 2DGS model."),
        ("generate-text3d", "Generate a 3D asset from a text prompt."),
        ("generate-image3d", "Generate a 3D asset from a single image."),
        ("fuse-scene", "Fuse reconstructed and generated 3D assets."),
        ("render-video", "Render the final fused scene video."),
    ]:
        _add_config_command(subparsers, name, help_text)

    record = subparsers.add_parser(
        "record-experiment",
        help="Append an experiment entry to draft and registry logs.",
    )
    record.add_argument("--name", required=True)
    record.add_argument("--stage", default="scene")
    record.add_argument("--status", default="planned")
    record.add_argument("--command", required=True)
    record.add_argument("--notes", default="")
    record.add_argument("--dry-run", action="store_true")
    record.set_defaults(handler=_record_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    if args.command in CONFIG_HANDLERS:
        return _run_config_command(args, handler)
    return handler(args)
