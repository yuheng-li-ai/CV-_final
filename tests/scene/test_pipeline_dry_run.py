from cvhw3_scene.colmap import ColmapRunner
from cvhw3_scene.config import SceneConfig
from cvhw3_scene.data import FrameExtractor
from cvhw3_scene.gaussian import GaussianTrainer
from cvhw3_scene.generation import ImageTo3DAssetGenerator, TextTo3DAssetGenerator


def test_prepare_object_dry_run_reports_command_and_missing_input():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap.yaml")
    result = FrameExtractor(config).dry_run()

    assert result.command[0] == "ffmpeg"
    assert any(issue.key == "inputs.video_path" for issue in result.validation_issues)


def test_colmap_dry_run_builds_automatic_reconstructor_command():
    config = SceneConfig.from_yaml("configs/scene/object_a_colmap.yaml")
    result = ColmapRunner(config).dry_run()

    command = " ".join(result.command)
    assert "colmap automatic_reconstructor" in command
    assert "--workspace_path" in result.command
    assert any(issue.key == "paths.image_dir" for issue in result.validation_issues)


def test_gaussian_trainer_dry_run_builds_external_train_command():
    config = SceneConfig.from_yaml("configs/scene/background_2dgs.yaml")
    result = GaussianTrainer(config).dry_run()

    command = " ".join(result.command)
    assert "external/2d-gaussian-splatting/train.py" in command
    assert "-s" in result.command
    assert any(issue.key == "paths.source_scene" for issue in result.validation_issues)


def test_text_and_image_generators_build_expected_external_commands():
    text_result = TextTo3DAssetGenerator(
        SceneConfig.from_yaml("configs/scene/text3d_baseline.yaml")
    ).dry_run()
    image_result = ImageTo3DAssetGenerator(
        SceneConfig.from_yaml("configs/scene/image3d_baseline.yaml")
    ).dry_run()

    assert "external/threestudio/launch.py" in " ".join(text_result.command)
    assert "system.prompt_processor.prompt=A clean ceramic mug" in text_result.command
    assert "external/Magic123/run.py" in " ".join(image_result.command)
    assert any(issue.key == "inputs.image_path" for issue in image_result.validation_issues)
