import csv
import json

from scripts.collect_2dgs_metrics import main


def test_collect_2dgs_metrics_writes_csv(tmp_path, monkeypatch):
    model = tmp_path / "model"
    render_dir = model / "test" / "ours_50" / "renders"
    gt_dir = model / "test" / "ours_50" / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    (render_dir / "00000.png").write_bytes(b"png")
    (gt_dir / "00000.png").write_bytes(b"png")
    (model / "results.json").write_text(
        json.dumps({"ours_50": {"PSNR": 12.5, "SSIM": 0.7, "LPIPS": 0.2}}),
        encoding="utf-8",
    )
    out = tmp_path / "metrics.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_2dgs_metrics.py",
            "--model",
            f"smoke={model}",
            "--out",
            str(out),
        ],
    )

    assert main() == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["run_id"] == "smoke"
    assert rows[0]["iteration"] == "50"
    assert rows[0]["render_count"] == "1"
