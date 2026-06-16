from pathlib import Path

from cvhw3_scene.config import SceneConfig


def test_scene_config_loads_yaml_and_resolves_paths():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap.yaml")

    assert config.name == "object_a_colmap"
    assert config.command == "prepare-object"
    assert config.project_root == Path.cwd()
    assert config.resolve_path("runs/scene/object_a_colmap") == Path.cwd() / "runs/scene/object_a_colmap"


def test_scene_config_reports_missing_required_paths():
    missing_path_config = Path("configs/scene/object_a_colmap.yaml")
    config = SceneConfig.from_yaml(missing_path_config)

    missing = config.validate_required_paths(["paths.image_dir"])

    assert len(missing) == 1
    assert missing[0].key == "paths.image_dir"
    assert "data/scene/object_a/frames" in str(missing[0].path)
