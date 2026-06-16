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
        resolution = self.config.get("training.resolution")
        if resolution is not None:
            resolution_value = {"full": "1", "half": "2"}.get(str(resolution), str(resolution))
            command.extend(["-r", resolution_value])
        if self.config.get("training.ip") is not None:
            command.extend(["--ip", str(self.config.get("training.ip"))])
        if self.config.get("training.port") is not None:
            command.extend(["--port", str(self.config.get("training.port"))])
        return command

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(
                ["paths.source_scene", "tools.train_script"]
            ),
            description=f"Dry run: 2DGS training for {self.config.name}",
        )


class GaussianEvaluator:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        python = str(self.config.get("tools.python", "python"))
        command = [
            python,
            "scripts/eval_2dgs.py",
            "--python",
            python,
            "--render_script",
            str(self.config.get("tools.render_script", "external/2d-gaussian-splatting/render.py")),
            "--metrics_script",
            str(self.config.get("tools.metrics_script", "external/2d-gaussian-splatting/metrics.py")),
            "-s",
            str(self.config.require("paths.source_scene")),
            "-m",
            str(self.config.require("paths.output_dir")),
            "--iteration",
            str(self.config.get("evaluation.iteration", self.config.get("training.iterations", -1))),
        ]
        if self.config.get("evaluation.skip_train", True):
            command.append("--skip_train")
        if self.config.get("evaluation.skip_test", False):
            command.append("--skip_test")
        if self.config.get("evaluation.skip_mesh", True):
            command.append("--skip_mesh")
        if self.config.get("evaluation.quiet", True):
            command.append("--quiet")
        if self.config.get("evaluation.render_path", False):
            command.append("--render_path")
        for key, flag in (
            ("evaluation.voxel_size", "--voxel_size"),
            ("evaluation.depth_trunc", "--depth_trunc"),
            ("evaluation.sdf_trunc", "--sdf_trunc"),
            ("evaluation.num_cluster", "--num_cluster"),
            ("evaluation.mesh_res", "--mesh_res"),
        ):
            value = self.config.get(key)
            if value is not None:
                command.extend([flag, str(value)])
        if self.config.get("evaluation.unbounded", False):
            command.append("--unbounded")
        return command

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(
                [
                    "paths.source_scene",
                    "paths.output_dir",
                    "tools.render_script",
                    "tools.metrics_script",
                ]
            ),
            description=f"Dry run: 2DGS render/eval for {self.config.name}",
        )
