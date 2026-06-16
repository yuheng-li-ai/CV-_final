from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


FIELDS = [
    "run_id",
    "model_path",
    "status",
    "method",
    "iteration",
    "psnr",
    "ssim",
    "lpips",
    "checkpoint_exists",
    "checkpoint_size",
    "render_count",
    "gt_count",
    "results_json",
    "preview_path",
    "suggested_eval_command",
]


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--model must be formatted as run_id=path/to/model")
    run_id, path = value.split("=", 1)
    if not run_id:
        raise ValueError("--model run_id cannot be empty")
    return run_id, Path(path)


def _read_results(model_path: Path) -> dict[str, Any]:
    results_path = model_path / "results.json"
    if not results_path.exists():
        return {}
    loaded = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"results.json must contain an object: {results_path}")
    return loaded


def _method_for_iteration(results: dict[str, Any], expected_iteration: int) -> str:
    preferred = f"ours_{expected_iteration}"
    if preferred in results:
        return preferred
    if results:
        return sorted(results)[-1]
    return preferred


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def _suggested_eval_command(run_id: str) -> str:
    config = Path(f"configs/scene/object_a_2dgs_{run_id}.yaml")
    if not config.exists():
        config = Path("configs/scene/object_a_2dgs_full.yaml")
    device = "cuda:2" if run_id == "half" else "cuda:1"
    return (
        "bash scripts/run_2dgs_eval.sh "
        f"--config {config} --run_id object_a_2dgs_{run_id}_eval --device {device}"
    )


def _resized(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_preview(test_dir: Path, out_path: Path, count: int = 4, width: int = 320) -> bool:
    render_paths = sorted((test_dir / "renders").glob("*.png"))[:count]
    pairs = []
    for render_path in render_paths:
        gt_path = test_dir / "gt" / render_path.name
        if gt_path.exists():
            pairs.append((_resized(gt_path, width), _resized(render_path, width), render_path.name))
    if not pairs:
        return False

    label_height = 26
    row_height = max(max(gt.height, render.height) for gt, render, _ in pairs) + label_height
    canvas = Image.new("RGB", (width * 2, row_height * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (gt, render, name) in enumerate(pairs):
        y = index * row_height
        draw.text((8, y + 6), f"{name} GT", fill=(0, 0, 0))
        draw.text((width + 8, y + 6), f"{name} render", fill=(0, 0, 0))
        canvas.paste(gt, (0, y + label_height))
        canvas.paste(render, (width, y + label_height))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def collect_object_a_2dgs_row(
    run_id: str,
    model_path: Path,
    preview_dir: Path,
    expected_iteration: int,
    preview_count: int = 4,
    preview_width: int = 320,
) -> dict[str, str]:
    checkpoint = model_path / "point_cloud" / f"iteration_{expected_iteration}" / "point_cloud.ply"
    checkpoint_exists = checkpoint.exists()
    results = _read_results(model_path)
    method = _method_for_iteration(results, expected_iteration)
    metrics = results.get(method, {}) if results else {}
    test_dir = model_path / "test" / method
    render_count = _count_pngs(test_dir / "renders")
    gt_count = _count_pngs(test_dir / "gt")

    preview_path = preview_dir / f"object_a_2dgs_{run_id}_preview.jpg"
    preview_text = ""
    if render_count and gt_count and make_preview(
        test_dir, preview_path, count=preview_count, width=preview_width
    ):
        preview_text = str(preview_path)

    if checkpoint_exists and metrics and render_count and gt_count:
        status = "complete"
        suggested_eval = ""
    elif checkpoint_exists:
        status = "needs_eval"
        suggested_eval = _suggested_eval_command(run_id)
    else:
        status = "missing_checkpoint"
        suggested_eval = ""

    return {
        "run_id": run_id,
        "model_path": str(model_path),
        "status": status,
        "method": method if metrics else "",
        "iteration": str(expected_iteration),
        "psnr": str(metrics.get("PSNR", "")),
        "ssim": str(metrics.get("SSIM", "")),
        "lpips": str(metrics.get("LPIPS", "")),
        "checkpoint_exists": str(checkpoint_exists).lower(),
        "checkpoint_size": str(checkpoint.stat().st_size if checkpoint_exists else ""),
        "render_count": str(render_count),
        "gt_count": str(gt_count),
        "results_json": str(model_path / "results.json") if results else "",
        "preview_path": preview_text,
        "suggested_eval_command": suggested_eval,
    }


def finalize_object_a_2dgs(
    models: list[tuple[str, Path]],
    out_csv: Path,
    preview_dir: Path,
    expected_iteration: int = 15000,
    preview_count: int = 4,
    preview_width: int = 320,
) -> dict[str, Any]:
    rows = [
        collect_object_a_2dgs_row(
            run_id,
            model_path,
            preview_dir,
            expected_iteration,
            preview_count=preview_count,
            preview_width=preview_width,
        )
        for run_id, model_path in models
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "complete": all(row["status"] == "complete" for row in rows),
        "out_csv": str(out_csv),
        "rows": rows,
    }
