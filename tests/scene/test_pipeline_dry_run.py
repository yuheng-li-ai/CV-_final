from copy import deepcopy

from cvhw3_scene.colmap import ColmapRunner
from cvhw3_scene.config import SceneConfig
from cvhw3_scene.data import FrameExtractor
from cvhw3_scene.gaussian import GaussianEvaluator, GaussianTrainer
from cvhw3_scene.generation import ImageTo3DAssetGenerator, TextTo3DAssetGenerator


def config_with_missing_paths(config: SceneConfig, updates: dict[str, str]) -> SceneConfig:
    data = deepcopy(config.data)
    for key, value in updates.items():
        current = data
        parts = key.split(".")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value
    return SceneConfig(path=config.path, data=data, project_root=config.project_root)


def test_prepare_object_dry_run_reports_command_and_missing_input():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap.yaml")
    result = FrameExtractor(config).dry_run()

    assert result.command[0] == "ffmpeg"
    assert all(issue.key != "inputs.video_path" for issue in result.validation_issues)


def test_colmap_dry_run_builds_automatic_reconstructor_command():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap.yaml")
    result = ColmapRunner(config).dry_run()

    command = " ".join(result.command)
    assert "automatic_reconstructor" in command
    assert "--workspace_path" in result.command
    assert any(issue.key == "paths.image_dir" for issue in result.validation_issues)


def test_colmap_smoke_dry_run_can_disable_gpu():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap_smoke.yaml")
    result = ColmapRunner(config).dry_run()

    assert "--use_gpu" in result.command
    assert "0" in result.command


def test_colmap_r2_configs_limit_threads():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap_dense.yaml")
    result = ColmapRunner(config).dry_run()

    assert "--num_threads" in result.command
    assert "90" in result.command


def test_gaussian_trainer_dry_run_builds_external_train_command():
    config = config_with_missing_paths(
        SceneConfig.from_yaml("configs/scene/background_2dgs.yaml"),
        {
            "paths.source_scene": "data/__missing/background",
            "tools.train_script": "external/__missing_2dgs/train.py",
        },
    )
    result = GaussianTrainer(config).dry_run()

    command = " ".join(result.command)
    assert "external/__missing_2dgs/train.py" in command
    assert "-s" in result.command
    assert any(issue.key == "paths.source_scene" for issue in result.validation_issues)
    assert any(issue.key == "tools.train_script" for issue in result.validation_issues)


def test_gaussian_trainer_includes_optional_network_port():
    config = SceneConfig.from_yaml("configs/scene/background_2dgs_high.yaml")
    result = GaussianTrainer(config).dry_run()

    assert "--ip" in result.command
    assert "127.0.0.1" in result.command
    assert "--port" in result.command
    assert "6011" in result.command


def test_gaussian_trainer_maps_named_resolution():
    config = SceneConfig.from_yaml("configs/scene/object_a_2dgs_half.yaml")
    result = GaussianTrainer(config).dry_run()

    assert "-r" in result.command
    assert "2" in result.command


def test_gaussian_evaluator_dry_run_builds_render_and_metrics_command():
    config = SceneConfig.from_yaml("configs/scene/background_2dgs_low.yaml")
    result = GaussianEvaluator(config).dry_run()

    command = " ".join(result.command)
    assert "scripts/eval_2dgs.py" in command
    assert "external/2d-gaussian-splatting/render.py" in command
    assert "external/2d-gaussian-splatting/metrics.py" in command
    assert "--skip_train" in result.command
    assert "--skip_mesh" in result.command
    assert "--iteration" in result.command


def test_text_and_image_generators_build_expected_external_commands():
    text_result = TextTo3DAssetGenerator(
        SceneConfig.from_yaml("configs/scene/text3d_baseline.yaml")
    ).dry_run()
    image_result = ImageTo3DAssetGenerator(
        SceneConfig.from_yaml("configs/scene/image3d_baseline.yaml")
    ).dry_run()

    assert "scripts/run_threestudio.py" in " ".join(text_result.command)
    assert "--repo" in text_result.command
    assert "external/threestudio" in text_result.command
    assert "--prompt" in text_result.command
    assert "A clean ceramic mug" in text_result.command
    assert all(issue.key != "tools.repo_dir" for issue in text_result.validation_issues)
    assert "scripts/run_magic123.py" in " ".join(image_result.command)
    assert "--image_path" in image_result.command
    assert "external/Magic123" in image_result.command
    assert "--coarse_iters" in image_result.command
    assert "--fine_iters" in image_result.command
    assert all(issue.key != "inputs.image_path" for issue in image_result.validation_issues)
    assert all(issue.key != "tools.wrapper_script" for issue in image_result.validation_issues)


def test_image3d_smoke_limits_magic123_dataset_sizes():
    result = ImageTo3DAssetGenerator(SceneConfig.from_yaml("configs/scene/image3d_smoke.yaml")).dry_run()

    assert "--dataset_size_train" in result.command
    assert "--dataset_size_valid" in result.command
    assert "--dataset_size_test" in result.command
    assert result.command[result.command.index("--dataset_size_train") + 1] == "4"
    assert result.command[result.command.index("--dataset_size_valid") + 1] == "1"
    assert result.command[result.command.index("--dataset_size_test") + 1] == "4"
