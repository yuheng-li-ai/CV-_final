from pathlib import Path

from cvhw3_scene.recording import ExperimentRecorder


def test_experiment_recorder_dry_run_does_not_write_files(tmp_path):
    draft = tmp_path / "draft.md"
    registry = tmp_path / "experiment_registry.md"

    recorder = ExperimentRecorder(draft_path=draft, registry_path=registry)
    result = recorder.record(
        name="dry_run_scene",
        stage="scene",
        status="planned",
        command="PYTHONPATH=src python -m cvhw3_scene train-2dgs --dry-run",
        notes="No external training launched.",
        dry_run=True,
    )

    assert "dry_run_scene" in result
    assert not draft.exists()
    assert not registry.exists()


def test_experiment_recorder_appends_to_draft_and_registry(tmp_path):
    draft = tmp_path / "draft.md"
    registry = tmp_path / "docs" / "experiment_registry.md"

    recorder = ExperimentRecorder(draft_path=draft, registry_path=registry)
    recorder.record(
        name="scene_baseline",
        stage="scene",
        status="started",
        command="PYTHONPATH=src python -m cvhw3_scene train-2dgs",
        notes="Harness smoke test.",
        dry_run=False,
    )

    assert "scene_baseline" in draft.read_text(encoding="utf-8")
    assert "scene_baseline" in registry.read_text(encoding="utf-8")
