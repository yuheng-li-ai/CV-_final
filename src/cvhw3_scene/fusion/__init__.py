import shutil

from cvhw3_scene.config import DryRunResult, SceneConfig, ValidationIssue


class FusionRenderer:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_fusion_command(self) -> list[str]:
        return [
            "python",
            "scripts/fuse_scene.py",
            "--config",
            str(self.config.path.relative_to(self.config.project_root)),
            "--run_id",
            str(self.config.name),
            "--device",
            "cpu",
        ]

    def build_render_command(self) -> list[str]:
        command = [
            str(self.config.get("tools.blender", "blender")),
            "-b",
            "--python",
            str(self.config.get("tools.blender_script", "scripts/blender_fusion_scene.py")),
            "--",
            "--scene",
            str(self.config.require("inputs.fused_scene")),
            "--output",
            str(self.config.require("paths.video_path")),
            "--frames",
            str(int(float(self.config.get("render.fps", 30)) * float(self.config.get("render.seconds", 1)))),
            "--resolution",
            str(self.config.get("render.resolution", "1280x720")),
        ]
        command.extend(["--samples", str(int(self.config.get("render.samples", 64)))])
        command.extend(["--camera-path", str(self.config.get("render.camera_path", "static"))])
        command.extend(["--environment", str(self.config.get("render.environment", "shadow_floor"))])
        background = self.config.get("inputs.background_backplate")
        if background:
            command.extend(["--background-image", str(background)])
        return command

    def dry_run_fusion(self) -> DryRunResult:
        issues = self.config.validate_required_paths(["inputs.background_scene"])
        for index, asset in enumerate(self.config.get("inputs.assets", [])):
            path = self.config.resolve_path(str(asset.get("path", "")))
            if not path.exists():
                issues.append(
                    ValidationIssue(
                        key=f"inputs.assets.{index}.path",
                        path=path,
                        message="path does not exist yet",
                    )
                )
            texture = asset.get("texture")
            if texture:
                texture_path = self.config.resolve_path(str(texture))
                if not texture_path.exists():
                    issues.append(
                        ValidationIssue(
                            key=f"inputs.assets.{index}.texture",
                            path=texture_path,
                            message="path does not exist yet",
                        )
                    )
        return DryRunResult(
            command=self.build_fusion_command(),
            validation_issues=issues,
            description=f"Dry run: scene fusion for {self.config.name}",
        )

    def dry_run_render(self) -> DryRunResult:
        issues = self.config.validate_required_paths(["inputs.fused_scene"])
        blender = str(self.config.get("tools.blender", "blender"))
        blender_path = shutil.which(blender)
        if blender_path is None:
            issues.append(
                ValidationIssue(
                    key="tools.blender",
                    path=self.config.project_root / blender,
                    message="Blender executable is not on PATH",
                )
            )
        script_path = self.config.resolve_path(str(self.config.get("tools.blender_script", "scripts/blender_fusion_scene.py")))
        if not script_path.exists():
            issues.append(
                ValidationIssue(
                    key="tools.blender_script",
                    path=script_path,
                    message="path does not exist yet",
                )
            )
        return DryRunResult(
            command=self.build_render_command(),
            validation_issues=issues,
            description=f"Dry run: video rendering for {self.config.name}",
        )
