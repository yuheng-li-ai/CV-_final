from cvhw3_scene.config import DryRunResult, SceneConfig


class TextTo3DAssetGenerator:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        python = str(self.config.get("tools.python", "python"))
        script = str(self.config.get("tools.wrapper_script", "scripts/run_threestudio.py"))
        prompt = str(self.config.require("prompt.text"))
        command = [
            python,
            script,
            "--repo",
            str(self.config.get("tools.repo_dir", "external/threestudio")),
            "--config",
            str(self.config.get("prompt.config", "configs/dreamfusion-sd.yaml")),
            "--mode",
            str(self.config.get("runtime.mode", "train")),
            "--gpu",
            str(self.config.get("runtime.gpu", 0)),
            "--prompt",
            prompt,
            "--output_dir",
            str(self.config.require("paths.output_dir")),
            "--tag",
            str(self.config.get("runtime.tag", "main")),
        ]
        optional_flags = {
            "--max_steps": "runtime.max_steps",
            "--width": "runtime.width",
            "--height": "runtime.height",
            "--batch_size": "runtime.batch_size",
            "--val_check_interval": "runtime.val_check_interval",
        }
        for flag, key in optional_flags.items():
            value = self.config.get(key)
            if value is not None:
                command.extend([flag, str(value)])
        if self.config.get("runtime.gradio_progress", True):
            command.append("--gradio_progress")
        for extra in self.config.get("runtime.extra_args", []):
            command.extend(["--extra", str(extra)])
        return command

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(
                ["tools.wrapper_script", "tools.repo_dir"]
            ),
            description=f"Dry run: text-to-3D generation for {self.config.name}",
        )


class ImageTo3DAssetGenerator:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config

    def build_command(self) -> list[str]:
        python = str(self.config.get("tools.python", "python"))
        script = str(self.config.get("tools.wrapper_script", "scripts/run_magic123.py"))
        command = [
            python,
            script,
            "--repo",
            str(self.config.get("tools.repo_dir", "external/Magic123")),
            "--mode",
            str(self.config.get("runtime.mode", "coarse")),
            "--image_path",
            str(self.config.require("inputs.image_path")),
            "--prompt",
            str(self.config.get("prompt.text", "")),
            "--output_dir",
            str(self.config.require("paths.output_dir")),
            "--gpu",
            str(self.config.get("runtime.gpu", 0)),
            "--tag",
            str(self.config.get("runtime.tag", "object_c")),
        ]
        optional_flags = {
            "--coarse_iters": "runtime.coarse_iters",
            "--fine_iters": "runtime.fine_iters",
            "--sd_version": "runtime.sd_version",
            "--hf_key": "runtime.hf_key",
            "--init_ckpt": "runtime.init_ckpt",
            "--lambda_guidance_coarse": "runtime.lambda_guidance_coarse",
            "--lambda_guidance_fine": "runtime.lambda_guidance_fine",
            "--guidance_scale_coarse": "runtime.guidance_scale_coarse",
            "--guidance_scale_fine": "runtime.guidance_scale_fine",
            "--dataset_size_train": "runtime.dataset_size_train",
            "--dataset_size_valid": "runtime.dataset_size_valid",
            "--dataset_size_test": "runtime.dataset_size_test",
        }
        for flag, key in optional_flags.items():
            value = self.config.get(key)
            if value is not None:
                command.extend([flag, str(value)])
        for extra in self.config.get("runtime.extra_args", []):
            command.extend(["--extra", str(extra)])
        return command

    def dry_run(self) -> DryRunResult:
        return DryRunResult(
            command=self.build_command(),
            validation_issues=self.config.validate_required_paths(
                ["inputs.image_path", "tools.wrapper_script", "tools.repo_dir", "tools.entrypoint"]
            ),
            description=f"Dry run: image-to-3D generation for {self.config.name}",
        )
