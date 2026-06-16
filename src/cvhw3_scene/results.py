from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cvhw3_scene.draft import DraftEntry


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    command: str
    config_path: Path
    hardware: str
    metrics: dict[str, Any]
    figure_paths: list[str]
    video_paths: list[str]
    checkpoint_paths: list[str]


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8").strip()


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"metrics.json must contain an object: {path}")
    return loaded


def _relative_files(root: Path, child: str) -> list[str]:
    directory = root / child
    if not directory.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in directory.rglob("*") if path.is_file())


def collect_run_result(run_dir: str | Path) -> RunResult:
    root = Path(run_dir)
    return RunResult(
        run_id=root.name,
        run_dir=root,
        command=_read_text(root / "cmd.txt"),
        config_path=root / "config.yaml",
        hardware=_read_text(root / "gpu.txt"),
        metrics=_read_metrics(root / "metrics.json"),
        figure_paths=_relative_files(root, "figures"),
        video_paths=_relative_files(root, "videos"),
        checkpoint_paths=_relative_files(root, "checkpoints"),
    )


def result_to_draft_entry(
    run_result: RunResult,
    phase: str,
    goal: str,
    elapsed_time: str,
    cause_analysis: str,
    next_step: str,
    result_status: str | None = None,
    date: str | None = None,
    result: str | None = None,
) -> DraftEntry:
    status_text = result or result_status or ""
    metrics_text = json.dumps(run_result.metrics, ensure_ascii=False, sort_keys=True)
    return DraftEntry(
        date=date or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        phase=phase,
        run_id=run_result.run_id,
        goal=goal,
        command=run_result.command,
        config=str(run_result.config_path),
        hardware=run_result.hardware,
        elapsed_time=elapsed_time,
        result=status_text,
        metrics=metrics_text,
        figure_paths=", ".join(run_result.figure_paths) or "none",
        video_paths=", ".join(run_result.video_paths) or "none",
        cause_analysis=cause_analysis,
        next_step=next_step,
    )
