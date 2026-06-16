from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from cvhw3_scene.object_a import finalize_object_a_2dgs


def write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


def make_model(root: Path, label: str, iteration: int = 15000) -> Path:
    model = root / label
    checkpoint = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("ply\n", encoding="utf-8")
    results = {
        f"ours_{iteration}": {
            "PSNR": 24.5,
            "SSIM": 0.8,
            "LPIPS": 0.2,
        }
    }
    (model / "results.json").write_text(json.dumps(results), encoding="utf-8")
    write_png(model / "test" / f"ours_{iteration}" / "gt" / "00000.png", (255, 0, 0))
    write_png(model / "test" / f"ours_{iteration}" / "renders" / "00000.png", (0, 255, 0))
    return model


def test_finalize_object_a_collects_metrics_and_preview(tmp_path: Path):
    full = make_model(tmp_path, "object_a_2dgs_full")
    half = make_model(tmp_path, "object_a_2dgs_half")
    out_csv = tmp_path / "reports" / "object_a.csv"
    preview_dir = tmp_path / "figures"

    summary = finalize_object_a_2dgs(
        models=[("full", full), ("half", half)],
        out_csv=out_csv,
        preview_dir=preview_dir,
        expected_iteration=15000,
    )

    assert summary["complete"] is True
    assert (preview_dir / "object_a_2dgs_full_preview.jpg").exists()
    assert (preview_dir / "object_a_2dgs_half_preview.jpg").exists()
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert [row["run_id"] for row in rows] == ["full", "half"]
    assert rows[0]["status"] == "complete"
    assert rows[0]["psnr"] == "24.5"
    assert rows[0]["checkpoint_exists"] == "true"
    assert rows[0]["render_count"] == "1"


def test_finalize_object_a_reports_missing_eval_without_failing_default(tmp_path: Path):
    model = tmp_path / "object_a_2dgs_full"
    checkpoint = model / "point_cloud" / "iteration_15000" / "point_cloud.ply"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("ply\n", encoding="utf-8")

    summary = finalize_object_a_2dgs(
        models=[("full", model)],
        out_csv=tmp_path / "object_a.csv",
        preview_dir=tmp_path / "figures",
        expected_iteration=15000,
    )

    assert summary["complete"] is False
    assert summary["rows"][0]["status"] == "needs_eval"
    assert "scripts/run_2dgs_eval.sh" in summary["rows"][0]["suggested_eval_command"]
