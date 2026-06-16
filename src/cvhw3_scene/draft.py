from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DraftEntry:
    date: str
    phase: str
    run_id: str
    goal: str
    command: str
    config: str
    hardware: str
    elapsed_time: str
    result: str
    metrics: str
    figure_paths: str
    video_paths: str
    cause_analysis: str
    next_step: str


def build_draft_entry(entry: DraftEntry) -> str:
    return (
        f"\n## {entry.phase} - {entry.run_id}\n\n"
        f"- Date: {entry.date}\n"
        f"- Phase: {entry.phase}\n"
        f"- Run ID: {entry.run_id}\n"
        f"- Goal: {entry.goal}\n"
        f"- Command: `{entry.command}`\n"
        f"- Config: `{entry.config}`\n"
        f"- Hardware: {entry.hardware}\n"
        f"- Elapsed time: {entry.elapsed_time}\n"
        f"- Result: {entry.result}\n"
        f"- Metrics: {entry.metrics}\n"
        f"- Figure paths: {entry.figure_paths}\n"
        f"- Video paths: {entry.video_paths}\n"
        f"- Cause analysis: {entry.cause_analysis}\n"
        f"- Next step: {entry.next_step}\n"
    )


def append_draft_entry(path: Path, entry: DraftEntry, dry_run: bool = False) -> str:
    text = build_draft_entry(entry)
    if dry_run:
        return text
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
    return text
