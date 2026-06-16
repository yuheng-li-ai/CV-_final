from pathlib import Path

from cvhw3_scene.cli import main as cli_main
from scripts import smoke


def test_formal_cli_refuses_to_run_when_validation_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "missing_train.yaml"
    config.write_text(
        """
experiment:
  name: missing_train
  command: train-2dgs
tools:
  python: python
  train_script: missing/train.py
paths:
  source_scene: missing/scene
  output_dir: runs/missing_train
training:
  iterations: 10
""",
        encoding="utf-8",
    )

    exit_code = cli_main(
        [
            "train-2dgs",
            "--config",
            str(config),
            "--run_id",
            "must_not_start",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "outputs" / "must_not_start").exists()


def test_smoke_main_returns_nonzero_for_missing_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "missing_train.yaml"
    config.write_text(
        """
experiment:
  name: missing_train
  command: train-2dgs
tools:
  python: python
  train_script: missing/train.py
paths:
  source_scene: missing/scene
  output_dir: runs/missing_train
training:
  iterations: 10
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke.py",
            "--stage",
            "train",
            "--config",
            str(config),
            "--run_id",
            "smoke_missing",
            "--device",
            "cpu",
        ],
    )

    assert smoke.main() == 2
    assert not (tmp_path / "outputs" / "smoke_missing").exists()
