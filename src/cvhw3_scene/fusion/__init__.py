from cvhw3_scene.config import DryRunResult, SceneConfig, ValidationIssue


class FusionRenderer:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_fusion_command(self) -> list[str]:
        assets = ",".join(str(item) for item in self.config.get("inputs.asset_paths", []))
        return [
            "python",
            "-m",
            "cvhw3_scene.fusion",
            "--background",
            str(self.config.require("inputs.background_scene")),
            "--assets",
            assets,
            "--out",
            str(self.config.require("paths.output_dir")),
        ]

    def build_render_command(self) -> list[str]:
        return [
            "python",
            "-m",
            "cvhw3_scene.rendering",
            "--scene",
            str(self.config.require("inputs.fused_scene")),
            "--camera-path",
            str(self.config.require("render.camera_path")),
            "--out",
            str(self.config.require("paths.video_path")),
        ]

    def dry_run_fusion(self) -> DryRunResult:
        issues = self.config.validate_required_paths(["inputs.background_scene"])
        for index, asset in enumerate(self.config.get("inputs.asset_paths", [])):
            path = self.config.resolve_path(str(asset))
            if not path.exists():
                issues.append(
                    ValidationIssue(
                        key=f"inputs.asset_paths.{index}",
                        path=path,
                        message="path does not exist yet",
                    )
                )
        return DryRunResult(
            command=self.build_fusion_command(),
            validation_issues=issues,
            description=f"Dry run: scene fusion for {self.config.name}",
        )

    def dry_run_render(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_render_command(),
            validation_issues=self.config.validate_required_paths(
                ["inputs.fused_scene", "render.camera_path"]
            ),
            description=f"Dry run: video rendering for {self.config.name}",
        )
