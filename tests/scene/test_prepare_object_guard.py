from pathlib import Path

import pytest

from cvhw3_scene.cli import _prepare_command_inputs
from cvhw3_scene.config import SceneConfig


def test_prepare_object_creates_frame_directory(tmp_path):
    config_path = tmp_path / "frames.yaml"
    config_path.write_text(
        """
experiment:
  name: frames
  command: prepare-object
inputs:
  video_path: input.mp4
paths:
  image_dir: frames
""",
        encoding="utf-8",
    )
    config = SceneConfig.from_yaml(config_path, project_root=tmp_path)

    _prepare_command_inputs(config, resume=False)

    assert (tmp_path / "frames").is_dir()


def test_prepare_object_refuses_nonempty_frame_directory_without_resume(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame_000001.png").write_bytes(b"png")
    config_path = tmp_path / "frames.yaml"
    config_path.write_text(
        """
experiment:
  name: frames
  command: prepare-object
inputs:
  video_path: input.mp4
paths:
  image_dir: frames
""",
        encoding="utf-8",
    )
    config = SceneConfig.from_yaml(config_path, project_root=tmp_path)

    with pytest.raises(FileExistsError):
        _prepare_command_inputs(config, resume=False)

    _prepare_command_inputs(config, resume=True)


def test_run_colmap_creates_workspace_directory(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    config_path = tmp_path / "colmap.yaml"
    config_path.write_text(
        """
experiment:
  name: colmap
  command: run-colmap
paths:
  image_dir: images
  workspace_dir: workspace
""",
        encoding="utf-8",
    )
    config = SceneConfig.from_yaml(config_path, project_root=tmp_path)

    _prepare_command_inputs(config, resume=False)

    assert (tmp_path / "workspace").is_dir()
