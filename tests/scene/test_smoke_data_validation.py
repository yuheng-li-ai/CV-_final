from copy import deepcopy

from cvhw3_scene.config import SceneConfig
from scripts.smoke import required_data_keys, run_stage


def config_with_missing_paths(config: SceneConfig) -> SceneConfig:
    data = deepcopy(config.data)
    data["paths"]["source_scene"] = "data/__missing/background"
    data["tools"]["train_script"] = "external/__missing_2dgs/train.py"
    return SceneConfig(path=config.path, data=data, project_root=config.project_root)


def test_smoke_data_required_keys_match_command_family():
    assert required_data_keys(
        SceneConfig.from_yaml("configs/scene/object_a_frames_medium.yaml")
    ) == ["inputs.video_path"]
    assert required_data_keys(
        SceneConfig.from_yaml("configs/scene/image3d_raw.yaml")
    ) == ["inputs.image_path", "tools.launch_script"]
    assert required_data_keys(
        SceneConfig.from_yaml("configs/scene/background_2dgs_low.yaml")
    ) == ["paths.source_scene", "tools.train_script"]
    assert required_data_keys(
        SceneConfig.from_yaml("configs/scene/fusion_main.yaml")
    ) == [
        "inputs.background_scene",
        "inputs.assets.0.path",
        "inputs.assets.1.path",
        "inputs.assets.1.texture",
        "inputs.assets.2.path",
        "inputs.assets.2.texture",
    ]


def test_smoke_train_reports_missing_validation_issues():
    metrics = run_stage(
        "train",
        config_with_missing_paths(SceneConfig.from_yaml("configs/scene/background_2dgs_low.yaml")),
    )

    assert metrics["status"] == "missing_data"
    assert "tools.train_script" in metrics["missing_paths"]
