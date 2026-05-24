from cvhw3_scene.config import DryRunResult, SceneConfig


class GaussianTrainer:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        python = str(self.config.get("tools.python", "python"))
        script = str(self.config.get("tools.train_script", "external/2d-gaussian-splatting/train.py"))
        command = [
            python,
            script,
            "-s",
            str(self.config.require("paths.source_scene")),
            "-m",
            str(self.config.require("paths.output_dir")),
            "--iterations",
            str(self.config.get("training.iterations", 30000)),
        ]
        if self.config.get("training.eval", True):
            command.append("--eval")
        return command

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(["paths.source_scene"]),
            description=f"Dry run: 2DGS training for {self.config.name}",
        )
