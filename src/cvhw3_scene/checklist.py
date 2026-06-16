from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ChecklistStatus:
    path: Path
    required_false: list[str]
    all_false: list[str]

    @property
    def is_complete(self) -> bool:
        return not self.required_false


def find_false_items(data: Any, prefix: str = "") -> list[str]:
    if isinstance(data, dict):
        items: list[str] = []
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(find_false_items(value, child_prefix))
        return items
    if data is False:
        return [prefix]
    return []


def load_checklist_status(
    path: str | Path = "docs/checklist.yaml",
    required_section: str = "required_keys",
) -> ChecklistStatus:
    checklist_path = Path(path)
    data = yaml.safe_load(checklist_path.read_text(encoding="utf-8")) or {}
    required = data.get(required_section, {})
    return ChecklistStatus(
        path=checklist_path,
        required_false=find_false_items(required, required_section),
        all_false=find_false_items(data),
    )


def format_checklist_status(status: ChecklistStatus) -> str:
    lines = [f"Checklist: {status.path}"]
    lines.append(f"Required complete: {str(status.is_complete).lower()}")
    if status.required_false:
        lines.append("Required false items:")
        lines.extend(f"- {item}" for item in status.required_false)
    if status.all_false:
        lines.append("All false items:")
        lines.extend(f"- {item}" for item in status.all_false)
    return "\n".join(lines)
