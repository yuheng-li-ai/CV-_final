from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys

from cvhw3_scene.gpu import format_gpu_report


@dataclass(frozen=True)
class RunManager:
    config_path: Path
    run_id: str
    command: list[str]
    device: str
    outputs_root: Path = Path("outputs")
    resume: bool = False

    @property
    def run_dir(self) -> Path:
        return self.outputs_root / self.run_id

    def prepare(self) -> Path:
        run_dir = self.run_dir
        if run_dir.exists() and not self.resume:
            if not self._is_redirect_only_run_dir(run_dir):
                raise FileExistsError(
                    f"Run directory already exists: {run_dir}. Use --resume to reuse it."
                )
        run_dir.mkdir(parents=True, exist_ok=True)

        for child in ["figures", "videos", "checkpoints"]:
            (run_dir / child).mkdir(exist_ok=True)

        self._copy_config_once(run_dir)
        self._write_text_once(run_dir / "cmd.txt", shlex.join(self.command) + "\n")
        self._write_text_once(run_dir / "git_hash.txt", self._git_hash() + "\n")
        self._write_text_once(run_dir / "env.txt", self._environment_report())
        self._write_text_once(run_dir / "gpu.txt", format_gpu_report(self.device))
        self._write_text_once(run_dir / "metrics.json", "{}\n")
        self._write_text_once(run_dir / "log.txt", "")
        return run_dir

    def _is_redirect_only_run_dir(self, run_dir: Path) -> bool:
        files = [path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file()]
        dirs = [path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_dir()]
        return files == [Path("nohup.log")] and not dirs

    def _copy_config_once(self, run_dir: Path) -> None:
        target = run_dir / "config.yaml"
        if target.exists() and self.resume:
            return
        shutil.copyfile(self.config_path, target)

    def _write_text_once(self, path: Path, text: str) -> None:
        if path.exists() and self.resume:
            return
        path.write_text(text, encoding="utf-8")

    def _git_hash(self) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
        return completed.stdout.strip() or "unknown"

    def _environment_report(self) -> str:
        return "\n".join(
            [
                f"python: {sys.version.split()[0]}",
                f"executable: {sys.executable}",
                f"platform: {platform.platform()}",
            ]
        ) + "\n"
