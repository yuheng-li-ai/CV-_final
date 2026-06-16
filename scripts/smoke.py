#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvhw3_scene.config import SceneConfig
from cvhw3_scene.fusion import FusionRenderer
from cvhw3_scene.gaussian import GaussianEvaluator, GaussianTrainer
from cvhw3_scene.generation import TextTo3DAssetGenerator
from cvhw3_scene.gpu import select_device
from cvhw3_scene.run_manager import RunManager


def _get_nested(data: object, key: str) -> object:
    current = data
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def required_data_keys(config: SceneConfig) -> list[str]:
    command = config.command
    if command == "prepare-object":
        return ["inputs.video_path"]
    if command == "run-colmap":
        return ["paths.image_dir"]
    if command == "train-2dgs":
        return ["paths.source_scene", "tools.train_script"]
    if command == "eval-2dgs":
        return ["paths.source_scene", "paths.output_dir", "tools.render_script", "tools.metrics_script"]
    if command == "generate-text3d":
        return ["tools.launch_script"]
    if command == "generate-image3d":
        return ["inputs.image_path", "tools.launch_script"]
    if command == "fuse-scene":
        assets = config.get("inputs.assets", [])
        keys = ["inputs.background_scene"]
        for index, asset in enumerate(assets):
            keys.append(f"inputs.assets.{index}.path")
            if isinstance(asset, dict) and asset.get("texture"):
                keys.append(f"inputs.assets.{index}.texture")
        return keys
    if command == "render-video":
        return ["inputs.fused_scene", "render.camera_path"]
    return []


def validate_data_paths(config: SceneConfig) -> list[str]:
    missing: list[str] = []
    for key in required_data_keys(config):
        if key.startswith("inputs.assets."):
            value = _get_nested(config.data, key)
            if value is None or value == "" or not config.resolve_path(str(value)).exists():
                missing.append(key)
            continue
        if config.validate_required_paths([key]):
            missing.append(key)
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layered Task 1 smoke tests.")
    parser.add_argument("--stage", required=True, choices=["env", "data", "train", "render", "export"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_stage(stage: str, config: SceneConfig) -> dict[str, object]:
    if stage == "env":
        return {"stage": stage, "imports": ["yaml", "cvhw3_scene"], "status": "passed"}
    if stage == "data":
        missing = validate_data_paths(config)
        return {
            "stage": stage,
            "config": str(config.path),
            "required_paths": required_data_keys(config),
            "missing_paths": missing,
            "status": "passed" if not missing else "missing_data",
        }
    if stage == "train":
        if config.command == "generate-text3d":
            result = TextTo3DAssetGenerator(config).dry_run()
        else:
            result = GaussianTrainer(config).dry_run()
        missing = [issue.key for issue in result.validation_issues]
        return {
            "stage": stage,
            "command": result.command,
            "missing_paths": missing,
            "status": "passed" if not missing else "missing_data",
        }
    if stage == "render":
        if config.command in {"train-2dgs", "eval-2dgs"}:
            result = GaussianEvaluator(config).dry_run()
            missing = [issue.key for issue in result.validation_issues]
            return {
                "stage": stage,
                "command": result.command,
                "missing_paths": missing,
                "status": "passed" if not missing else "missing_data",
            }
        result = FusionRenderer(config).dry_run_render()
        missing = [issue.key for issue in result.validation_issues]
        return {
            "stage": stage,
            "command": result.command,
            "missing_paths": missing,
            "status": "passed" if not missing else "missing_data",
        }
    if stage == "export":
        result = FusionRenderer(config).dry_run_fusion()
        missing = [issue.key for issue in result.validation_issues]
        return {
            "stage": stage,
            "command": result.command,
            "missing_paths": missing,
            "status": "passed" if not missing else "missing_data",
        }
    raise ValueError(f"Unsupported smoke stage: {stage}")


def main() -> int:
    args = build_parser().parse_args()
    config = SceneConfig.from_yaml(args.config)
    metrics = run_stage(args.stage, config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics.get("status") != "passed":
        return 2
    if args.dry_run:
        return 0

    device = select_device(None if args.device == "auto" else args.device)
    manager = RunManager(
        config_path=Path(args.config),
        run_id=args.run_id,
        command=["python", "scripts/smoke.py", "--stage", args.stage],
        device=device,
        resume=args.resume,
    )
    run_dir = manager.prepare()
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
