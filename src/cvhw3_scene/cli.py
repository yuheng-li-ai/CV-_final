from __future__ import annotations

import argparse
from collections.abc import Callable

from cvhw3_scene.colmap import ColmapRunner
from cvhw3_scene.config import DryRunResult, SceneConfig
from cvhw3_scene.data import FrameExtractor
from cvhw3_scene.fusion import FusionRenderer
from cvhw3_scene.gaussian import GaussianTrainer
from cvhw3_scene.generation import ImageTo3DAssetGenerator, TextTo3DAssetGenerator
from cvhw3_scene.recording import ExperimentRecorder


def _add_config_command(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", required=True, help="Path to a scene YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without external execution.")
    parser.set_defaults(handler=CONFIG_HANDLERS[name])


def _run_config_command(args: argparse.Namespace, factory: Callable[[SceneConfig], DryRunResult]) -> int:
    config = SceneConfig.from_yaml(args.config)
    if not args.dry_run:
        raise SystemExit("Only --dry-run is implemented for this scaffold milestone.")
    print(factory(config).format())
    return 0


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
