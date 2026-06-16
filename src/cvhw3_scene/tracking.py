from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_BACKENDS = {"none", "wandb", "swanlab"}


@dataclass(frozen=True)
class TrackingPlan:
    enabled: bool
    backend: str = "none"
    project: str = ""
    run_name: str = ""
    mode: str = "offline"

    def command_env(self) -> dict[str, str]:
        if not self.enabled or self.backend == "none":
            return {}
        if self.backend == "wandb":
            env = {"WANDB_PROJECT": self.project, "WANDB_MODE": self.mode}
            if self.run_name:
                env["WANDB_NAME"] = self.run_name
            return env
        if self.backend == "swanlab":
            env = {"SWANLAB_PROJECT": self.project, "SWANLAB_MODE": self.mode}
            if self.run_name:
                env["SWANLAB_RUN_NAME"] = self.run_name
            return env
        raise ValueError(f"Unsupported tracking backend: {self.backend}")


def build_tracking_plan(config_data: dict[str, Any]) -> TrackingPlan:
    raw = config_data.get("logging", {}) if isinstance(config_data, dict) else {}
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return TrackingPlan(enabled=False, backend="none")

    backend = str(raw.get("backend", "none")).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported tracking backend: {backend}")
    return TrackingPlan(
        enabled=backend != "none",
        backend=backend,
        project=str(raw.get("project", "cvhw3-task1")),
        run_name=str(raw.get("run_name", "")),
        mode=str(raw.get("mode", "offline")),
    )
