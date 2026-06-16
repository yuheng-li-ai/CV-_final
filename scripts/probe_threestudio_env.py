#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REQUIRED_IMPORTS = [
    "torch",
    "pytorch_lightning",
    "omegaconf",
    "jaxtyping",
    "typeguard",
    "diffusers",
    "transformers",
    "accelerate",
    "tensorboard",
    "nerfacc",
    "tinycudann",
    "nvdiffrast",
    "xatlas",
    "trimesh",
    "torchmetrics",
    "wandb",
    "igl",
    "mcubes",
    "envlight",
    "controlnet_aux",
    "einops",
    "kornia",
    "sentencepiece",
    "clip",
    "taming",
]

OPTIONAL_IMPORTS = [
    "pysdf",
    "xformers",
    "bitsandbytes",
]


def classify_import(name: str, required: bool) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "name": name,
            "required": required,
            "ok": True,
            "version": getattr(module, "__version__", ""),
        }
    except Exception as exc:
        return {
            "name": name,
            "required": required,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def nvcc_version(cuda_home: str) -> str:
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.exists():
        return "missing"
    completed = subprocess.run(
        [str(nvcc), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip().splitlines()[-1] if completed.stdout else "unknown"


def torch_status() -> dict[str, Any]:
    try:
        import torch

        return {
            "version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}


def igl_compat_status() -> dict[str, bool]:
    try:
        import sitecustomize  # noqa: F401
        import igl

        return {
            "fast_winding_number_for_meshes": hasattr(
                igl, "fast_winding_number_for_meshes"
            ),
            "point_mesh_squared_distance": hasattr(igl, "point_mesh_squared_distance"),
            "read_obj": hasattr(igl, "read_obj"),
        }
    except Exception:
        return {
            "fast_winding_number_for_meshes": False,
            "point_mesh_squared_distance": False,
            "read_obj": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe threestudio Phase4 environment.")
    parser.add_argument("--repo", default="external/threestudio")
    parser.add_argument("--cuda_home", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    imports = [classify_import(name, True) for name in REQUIRED_IMPORTS]
    imports.extend(classify_import(name, False) for name in OPTIONAL_IMPORTS)
    result = {
        "repo": str(repo),
        "repo_exists": repo.exists(),
        "launch_exists": (repo / "launch.py").exists(),
        "cuda_home": args.cuda_home,
        "nvcc": nvcc_version(args.cuda_home),
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "TCNN_CUDA_ARCHITECTURES": os.environ.get("TCNN_CUDA_ARCHITECTURES", ""),
        },
        "torch": torch_status(),
        "igl_compat": igl_compat_status(),
        "imports": imports,
    }
    missing_required = [
        item["name"] for item in imports if item["required"] and not item["ok"]
    ]
    igl_ok = all(result["igl_compat"].values())
    result["status"] = (
        "passed" if result["repo_exists"] and not missing_required and igl_ok else "missing_deps"
    )
    result["missing_required"] = missing_required
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
