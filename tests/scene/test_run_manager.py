from pathlib import Path

import pytest
import yaml

from cvhw3_scene.run_manager import RunManager


def test_run_manager_creates_non_overwriting_run_directory(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  name: smoke\n", encoding="utf-8")

    manager = RunManager(
        config_path=config,
        run_id="smoke_run",
        command=["python", "-m", "cvhw3_scene", "smoke-env"],
        device="cuda:2",
        outputs_root=tmp_path / "outputs",
    )
    run_dir = manager.prepare()

    assert run_dir == tmp_path / "outputs" / "smoke_run"
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "videos").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) == {
        "experiment": {"name": "smoke"}
    }
    assert (run_dir / "cmd.txt").read_text(encoding="utf-8").strip() == (
        "python -m cvhw3_scene smoke-env"
    )
    assert "cuda:2" in (run_dir / "gpu.txt").read_text(encoding="utf-8")
    assert (run_dir / "metrics.json").read_text(encoding="utf-8").strip() == "{}"

    with pytest.raises(FileExistsError):
        manager.prepare()


def test_run_manager_resume_preserves_existing_metrics(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  name: resume\n", encoding="utf-8")

    first = RunManager(
        config_path=config,
        run_id="resume_run",
        command=["python", "train.py"],
        device="cuda:0",
        outputs_root=tmp_path / "outputs",
    )
    run_dir = first.prepare()
    (run_dir / "metrics.json").write_text('{"psnr": 20.0}\n', encoding="utf-8")

    resumed = RunManager(
        config_path=config,
        run_id="resume_run",
        command=["python", "train.py", "--resume"],
        device="cuda:0",
        outputs_root=tmp_path / "outputs",
        resume=True,
    )
    assert resumed.prepare() == run_dir
    assert (run_dir / "metrics.json").read_text(encoding="utf-8") == '{"psnr": 20.0}\n'


def test_run_manager_allows_shell_redirect_only_directory(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("experiment:\n  name: redirect\n", encoding="utf-8")
    run_dir = tmp_path / "outputs" / "redirect_run"
    run_dir.mkdir(parents=True)
    (run_dir / "nohup.log").write_text("shell opened redirect first\n", encoding="utf-8")

    manager = RunManager(
        config_path=config,
        run_id="redirect_run",
        command=["python", "train.py"],
        device="cuda:0",
        outputs_root=tmp_path / "outputs",
    )

    assert manager.prepare() == run_dir
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "nohup.log").read_text(encoding="utf-8") == "shell opened redirect first\n"
