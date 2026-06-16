#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Phase5 official Magic123 prerequisites.")
    parser.add_argument("--config", default="configs/scene/image3d_smoke.yaml")
    return parser


def _nested(config: dict[str, object], key: str, default: object = None) -> object:
    current: object = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return loaded


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    image_path = Path(str(_nested(config, "inputs.image_path", "")))
    repo_dir = Path(str(_nested(config, "tools.repo_dir", "external/Magic123")))
    wrapper = Path(str(_nested(config, "tools.wrapper_script", "scripts/run_magic123.py")))
    entrypoint = Path(str(_nested(config, "tools.entrypoint", "external/Magic123/main.py")))
    zero123_ckpt = repo_dir / "pretrained" / "zero123" / "105000.ckpt"
    midas_ckpt = repo_dir / "pretrained" / "midas" / "dpt_beit_large_512.pt"

    required_checks = {
        "config": config_path.exists(),
        "input_image": image_path.exists(),
        "wrapper_script": wrapper.exists(),
        "magic123_repo": repo_dir.exists(),
        "magic123_main": entrypoint.exists(),
        "magic123_install_script": (repo_dir / "install.sh").exists(),
        "magic123_run_scripts": (repo_dir / "scripts" / "magic123").exists(),
        "magic123_ldm_package": (repo_dir / "ldm").exists(),
        "zero123_105000_ckpt": zero123_ckpt.exists(),
        "midas_dpt_beit_large_512": midas_ckpt.exists(),
        "torch_cuda": torch.cuda.is_available(),
        "torch": _module_available("torch"),
        "diffusers": _module_available("diffusers"),
        "cv2": _module_available("cv2"),
        "trimesh": _module_available("trimesh"),
        "raymarching": _module_available("raymarching"),
        "shencoder": _module_available("shencoder"),
        "freqencoder": _module_available("freqencoder"),
        "gridencoder": _module_available("gridencoder"),
        "pymeshlab": _module_available("pymeshlab"),
        "easydict": _module_available("easydict"),
        "torch_ema": _module_available("torch_ema"),
        "tensorboardX": _module_available("tensorboardX"),
        "dearpygui": _module_available("dearpygui"),
        "termcolor": _module_available("termcolor"),
    }
    optional_checks = {
        "cubvh_base_mesh_optional": _module_available("cubvh"),
        "carvekit": _module_available("carvekit"),
    }
    payload = {"required": required_checks, "optional": optional_checks}
    print(json.dumps(payload, indent=2, sort_keys=True))
    missing = [name for name, ok in required_checks.items() if not ok]
    if missing:
        print("MISSING:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
