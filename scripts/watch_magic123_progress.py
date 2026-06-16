#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a compact progress bar for official Magic123.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


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


def infer_log_and_steps(args: argparse.Namespace) -> tuple[str | None, int, int]:
    config = _load_config(args.config)
    max_steps = args.max_steps
    coarse = int(_nested(config, "runtime.coarse_iters", 0) or 0)
    fine = int(_nested(config, "runtime.fine_iters", 0) or 0)
    if max_steps is None:
        mode = str(_nested(config, "runtime.mode", "coarse"))
        max_steps = coarse + fine if mode == "both" else max(coarse, fine)
    if not max_steps:
        max_steps = 1

    log = args.log
    if log is None and args.run_id:
        run_log = Path("outputs") / args.run_id / "log.txt"
        nohup_log = Path("outputs") / args.run_id / "nohup.log"
        log = str(run_log if run_log.exists() else nohup_log)
    return log, max_steps, coarse


def read_log(path: str | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    if not candidate.exists():
        return ""
    text = candidate.read_text(encoding="utf-8", errors="replace")
    marker = "MAGIC123_COMMAND:"
    if marker in text:
        text = marker + text.rsplit(marker, 1)[-1]
    return text[-200_000:]


def _is_fine_stage(text: str) -> bool:
    fine_markers = [
        "/fine",
        "Trainer: fine",
        "runname': 'fine'",
        '"runname": "fine"',
        "/fine Epoch",
    ]
    return any(marker in text for marker in fine_markers)


def parse_progress(text: str, max_steps: int, coarse_steps: int = 0) -> dict[str, object]:
    if "Traceback" in text or re.search(r"\b(RuntimeError|CUDA out of memory|No such file)\b", text):
        return {"stage": "failed", "percent": 1.0, "step": 0, "done": True}
    in_fine_stage = _is_fine_stage(text)
    if "Saved mesh" in text or "save mesh" in text.lower() or "saving mesh" in text.lower():
        step = max_steps if in_fine_stage or coarse_steps <= 0 else coarse_steps
        return {"stage": "saving", "percent": 0.98, "step": step, "done": False}
    train_step_matches = re.findall(r"Train \[Step\]\s+(\d+)/(\d+)", text)
    if train_step_matches:
        step_text, total_text = train_step_matches[-1]
        stage_step = int(step_text)
        total = max(1, int(total_text))
        step = stage_step + coarse_steps if in_fine_stage and max_steps > total else stage_step
        denominator = max(max_steps, total)
        return {
            "stage": "training",
            "percent": min(0.99, step / denominator),
            "step": step,
            "done": False,
        }
    if "MAGIC123_COMMAND:" not in text:
        return {"stage": "waiting", "percent": 0.0, "step": 0, "done": False}
    tqdm_matches = re.findall(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", text)
    if tqdm_matches:
        percent_text, step_text, total_text = tqdm_matches[-1]
        step = int(step_text)
        total = max(1, int(total_text))
        percent = max(int(percent_text) / 100.0, step / total)
        return {"stage": "eval/render", "percent": min(0.99, percent), "step": step, "done": False}
    step_matches = re.findall(r"\biter(?:ation)?\s*[:=]\s*(\d+)\b", text, flags=re.IGNORECASE)
    if step_matches:
        step = min(max_steps, int(step_matches[-1]))
        return {
            "stage": "training",
            "percent": min(1.0, step / max(max_steps, 1)),
            "step": step,
            "done": step >= max_steps,
        }
    return {"stage": "starting", "percent": 0.0, "step": 0, "done": False}


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render(progress: dict[str, object], width: int, max_steps: int, start_time: float) -> str:
    percent = float(progress["percent"])
    step = int(progress["step"])
    filled = min(width, max(0, int(round(percent * width))))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - start_time
    rate = step / elapsed if step > 0 and elapsed > 0 else 0.0
    eta = (max_steps - step) / rate if 0.0 < percent < 1.0 and rate > 0 else 0.0
    return (
        f"MAGIC123 {percent * 100:6.2f}%|{bar}| "
        f"{step}/{max_steps} [{_duration(elapsed)}<{_duration(eta)}, "
        f"{rate:5.2f}step/s] stage={progress['stage']}"
    )


def main() -> int:
    args = build_parser().parse_args()
    log, max_steps, coarse_steps = infer_log_and_steps(args)
    start_time = time.monotonic()
    while True:
        progress = parse_progress(read_log(log), max_steps, coarse_steps)
        print("\r" + render(progress, args.width, max_steps, start_time), end="", flush=True)
        if args.once or progress["done"]:
            print()
            return 1 if progress["stage"] == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
