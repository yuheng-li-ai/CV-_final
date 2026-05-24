from cvhw3_scene.config import DryRunResult, SceneConfig


class TextTo3DAssetGenerator:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        python = str(self.config.get("tools.python", "python"))
        script = str(self.config.get("tools.launch_script", "external/threestudio/launch.py"))
        prompt = str(self.config.require("prompt.text"))
        return [
            python,
            script,
            "--config",
            str(self.config.get("prompt.config", "configs/external/threestudio/text3d.yaml")),
            "--train",
            "--gpu",
            str(self.config.get("runtime.gpu", 0)),
            f"system.prompt_processor.prompt={prompt}",
            f"trial_dir={self.config.require('paths.output_dir')}",
        ]

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=[],
            description=f"Dry run: text-to-3D generation for {self.config.name}",
        )


class ImageTo3DAssetGenerator:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        python = str(self.config.get("tools.python", "python"))
        script = str(self.config.get("tools.launch_script", "external/Magic123/run.py"))
        return [
            python,
            script,
            "--image",
            str(self.config.require("inputs.image_path")),
            "--prompt",
            str(self.config.get("prompt.text", "")),
            "--outdir",
            str(self.config.require("paths.output_dir")),
            "--gpu",
            str(self.config.get("runtime.gpu", 0)),
        ]

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(["inputs.image_path"]),
            description=f"Dry run: image-to-3D generation for {self.config.name}",
        )
