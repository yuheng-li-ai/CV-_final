import json

import yaml

from cvhw3_scene.results import collect_run_result, result_to_draft_entry


def test_collect_run_result_reads_run_manager_artifacts(tmp_path):
    run_dir = tmp_path / "outputs" / "bg_5k"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "videos").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"experiment": {"name": "background_2dgs_low"}}),
        encoding="utf-8",
    )
    (run_dir / "cmd.txt").write_text("python train.py --iterations 5000\n", encoding="utf-8")
    (run_dir / "gpu.txt").write_text("selected_device: cuda:0\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"psnr": 21.5, "ssim": 0.71, "lpips": 0.23}),
        encoding="utf-8",
    )
    (run_dir / "figures" / "render.png").write_text("fake image", encoding="utf-8")
    (run_dir / "videos" / "turntable.mp4").write_text("fake video", encoding="utf-8")
    (run_dir / "checkpoints" / "model.pt").write_text("fake checkpoint", encoding="utf-8")

    result = collect_run_result(run_dir)

    assert result.run_id == "bg_5k"
    assert result.command == "python train.py --iterations 5000"
    assert result.hardware == "selected_device: cuda:0"
    assert result.metrics["psnr"] == 21.5
    assert result.figure_paths == ["figures/render.png"]
    assert result.video_paths == ["videos/turntable.mp4"]
    assert result.checkpoint_paths == ["checkpoints/model.pt"]


def test_result_to_draft_entry_contains_run_evidence(tmp_path):
    run_dir = tmp_path / "outputs" / "object_b"
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "videos").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.yaml").write_text("experiment:\n  name: object_b\n", encoding="utf-8")
    (run_dir / "cmd.txt").write_text("python launch.py\n", encoding="utf-8")
    (run_dir / "gpu.txt").write_text("selected_device: cuda:1\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"geometry_score": 4}\n', encoding="utf-8")

    entry = result_to_draft_entry(
        collect_run_result(run_dir),
        phase="Phase4",
        goal="Record Object B SDS output.",
        elapsed_time="2h10m",
        result="completed",
        cause_analysis="Detailed prompt gave stable geometry.",
        next_step="Use mesh in fusion.",
        date="2026-05-30",
    )

    assert entry.phase == "Phase4"
    assert entry.run_id == "object_b"
    assert entry.metrics == '{"geometry_score": 4}'
    assert entry.hardware == "selected_device: cuda:1"
