from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ExperimentRecorder:
    draft_path: Path = Path("draft.md")
    registry_path: Path = Path("docs/experiment_registry.md")

    def record(
        self,
        name: str,
        stage: str,
        status: str,
        command: str,
        notes: str,
        dry_run: bool = False,
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = (
            f"\n## {name}\n\n"
            f"- Time: {timestamp}\n"
            f"- Stage: {stage}\n"
            f"- Status: {status}\n"
            f"- Command: `{command}`\n"
            f"- Notes: {notes}\n"
        )
        if dry_run:
            return entry
        self.draft_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.draft_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        return entry
