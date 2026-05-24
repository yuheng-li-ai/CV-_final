from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class ValidationIssue:
    key: str
    path: Path
    message: str


@dataclass(frozen=True)
class DryRunResult:
    command: list[str]
    validation_issues: list[ValidationIssue]
    description: str

    def format(self) -> str:
        lines = [self.description, "COMMAND:", shlex.join(self.command)]
        if self.validation_issues:
            lines.append("VALIDATION:")
            for issue in self.validation_issues:
                lines.append(f"- MISSING {issue.key}: {issue.path} ({issue.message})")
        else:
            lines.append("VALIDATION: ok")
        return "\n".join(lines)


@dataclass(frozen=True)
class SceneConfig:
    path: Path
    data: dict[str, Any]
    project_root: Path

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> "SceneConfig":
        config_path = Path(path)
        root = Path(project_root) if project_root is not None else Path.cwd()
        full_path = config_path if config_path.is_absolute() else root / config_path
        with full_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Scene config must be a YAML mapping: {full_path}")
        return cls(path=full_path, data=loaded, project_root=root.resolve())

    @property
    def name(self) -> str:
        return str(self.get("experiment.name", self.path.stem))

    @property
    def command(self) -> str:
        return str(self.get("experiment.command", ""))

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None or value == "":
            raise KeyError(f"Missing required config key: {key}")
        return value

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    def path_value(self, key: str) -> Path:
        return self.resolve_path(str(self.require(key)))

    def validate_required_paths(self, keys: Iterable[str]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for key in keys:
            value = self.get(key)
            if value is None or value == "":
                issues.append(
                    ValidationIssue(
                        key=key,
                        path=self.project_root,
                        message="config key is not set",
                    )
                )
                continue
            resolved = self.resolve_path(str(value))
            if not resolved.exists():
                issues.append(
                    ValidationIssue(
                        key=key,
                        path=resolved,
                        message="path does not exist yet",
                    )
                )
        return issues
