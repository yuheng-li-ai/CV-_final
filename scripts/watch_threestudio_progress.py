#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a compact progress bar for threestudio.")
    parser.add_argument("--progress", default=None, help="Path to threestudio gradio progress file.")
    parser.add_argument("--log", default=None, help="Fallback stdout/log file.")
    parser.add_argument("--config", default=None, help="Task1 YAML config used to infer paths.")
    parser.add_argument("--run_id", default=None, help="Task1 run_id used to infer outputs/<run_id>/log.txt.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


def parse_progress(text: str, max_steps: int) -> dict[str, object]:
    if "Traceback" in text or re.search(r"\b(ERROR|RuntimeError|CUDA out of memory)\b", text):
        return {"stage": "failed", "percent": 1.0, "step": 0, "done": True}
    if "Exporting mesh assets" in text:
        return {"stage": "exporting", "percent": 0.98, "step": max_steps, "done": False}
    if "Rendering video" in text:
        return {"stage": "testing", "percent": 0.95, "step": max_steps, "done": False}
    if "Generation progress:" in text:
        matches = re.findall(r"Generation progress:\s*([0-9.]+)%", text)
        percent = min(1.0, max(0.0, float(matches[-1]) / 100.0)) if matches else 0.0
        step = int(round(percent * max_steps))
        return {"stage": "training", "percent": percent, "step": step, "done": percent >= 1.0}
    step_matches = re.findall(r"\|\s*(\d+)/(\d+)", text)
    if step_matches:
        step, total = (int(value) for value in step_matches[-1])
        percent = min(1.0, max(0.0, step / max(total, 1)))
        return {"stage": "training", "percent": percent, "step": step, "done": percent >= 1.0}
    if "Trainer.fit" in text and "stopped:" in text:
        return {"stage": "done", "percent": 1.0, "step": max_steps, "done": True}
    epoch_matches = re.findall(r"Epoch\s+\d+:\s*:\s*(\d+)it", text)
    if epoch_matches:
        step = min(max_steps, int(epoch_matches[-1]))
        percent = min(1.0, max(0.0, step / max(max_steps, 1)))
        return {"stage": "training", "percent": percent, "step": step, "done": False}
    test_matches = re.findall(r"Testing DataLoader\s+\d+:\s*(\d+)%", text)
    if test_matches:
        percent = 0.90 + min(100, int(test_matches[-1])) / 1000.0
        return {"stage": "testing", "percent": min(0.99, percent), "step": max_steps, "done": False}
    return {"stage": "waiting", "percent": 0.0, "step": 0, "done": False}


def render_bar(progress: dict[str, object], width: int) -> str:
    percent = float(progress["percent"])
    filled = min(width, max(0, int(round(percent * width))))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"THREESTUDIO [{bar}] {percent * 100:5.1f}% "
        f"stage={progress['stage']} step~={progress['step']}"
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_tqdm(progress: dict[str, object], width: int, max_steps: int, start_time: float) -> str:
    percent = float(progress["percent"])
    step = int(progress["step"])
    filled = min(width, max(0, int(round(percent * width))))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - start_time
    rate = step / elapsed if elapsed > 0 and step > 0 else 0.0
    if rate > 0 and percent < 1.0:
        eta = (max_steps - step) / rate
    else:
        eta = 0.0
    return (
        f"THREESTUDIO {percent * 100:6.2f}%|{bar}| "
        f"{step}/{max_steps} [{format_duration(elapsed)}<{format_duration(eta)}, "
        f"{rate:5.2f}step/s] stage={progress['stage']}"
    )


def read_optional(path: str | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    if not candidate.exists():
        return ""
    text = candidate.read_text(encoding="utf-8", errors="replace")
    marker = "THREESTUDIO_COMMAND:"
    if marker in text:
        text = marker + text.rsplit(marker, 1)[-1]
    return text[-200_000:]


def _load_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config: {path}")
    return loaded


def _nested(config: dict[str, object], dotted: str, default: object = None) -> object:
    current: object = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def infer_paths(args: argparse.Namespace) -> tuple[str | None, str | None, int]:
    config = _load_config(args.config)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(_nested(config, "runtime.max_steps", 0) or 0)
    if max_steps <= 0:
        raise ValueError("--max_steps is required when --config does not define runtime.max_steps")

    progress = args.progress
    if progress is None:
        output_dir = _nested(config, "paths.output_dir")
        tag = str(_nested(config, "runtime.tag", "main"))
        if output_dir:
            progress = str(Path(str(output_dir)) / tag / "progress")

    log = args.log
    if log is None and args.run_id:
        run_log = Path("outputs") / args.run_id / "log.txt"
        nohup_log = Path("outputs") / args.run_id / "nohup.log"
        log = str(run_log if run_log.exists() else nohup_log)

    return progress, log, max_steps


def main() -> int:
    args = build_parser().parse_args()
    progress_path, log_path, max_steps = infer_paths(args)
    start_time = time.monotonic()
    while True:
        text = read_optional(progress_path) or read_optional(log_path)
        progress = parse_progress(text, max_steps)
        print(
            "\r" + render_tqdm(progress, args.width, max_steps, start_time),
            end="",
            flush=True,
        )
        if args.once or progress["done"]:
            print()
            return 1 if progress["stage"] == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
