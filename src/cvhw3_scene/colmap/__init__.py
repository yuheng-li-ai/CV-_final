from cvhw3_scene.config import DryRunResult, SceneConfig


class ColmapRunner:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        colmap = str(self.config.get("tools.colmap", "colmap"))
        command = [
            colmap,
            "automatic_reconstructor",
            "--workspace_path",
            str(self.config.require("paths.workspace_dir")),
            "--image_path",
            str(self.config.require("paths.image_dir")),
            "--camera_model",
            str(self.config.get("colmap.camera_model", "SIMPLE_RADIAL")),
            "--single_camera",
            str(int(bool(self.config.get("colmap.single_camera", True)))),
        ]
        if self.config.get("colmap.use_gpu") is not None:
            use_gpu = int(bool(self.config.get("colmap.use_gpu")))
            command.extend(["--use_gpu", str(use_gpu)])
        if self.config.get("colmap.gpu_index") is not None:
            gpu_index = str(self.config.get("colmap.gpu_index"))
            command.extend(["--gpu_index", gpu_index])
        if self.config.get("colmap.num_threads") is not None:
            command.extend(["--num_threads", str(self.config.get("colmap.num_threads"))])
        return command

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(["paths.image_dir"]),
            description=f"Dry run: COLMAP reconstruction for {self.config.name}",
        )
