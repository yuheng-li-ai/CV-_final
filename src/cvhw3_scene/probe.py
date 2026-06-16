from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class MediaSummary:
    path: str
    codec: str
    width: int
    height: int
    duration_seconds: float
    frame_rate: float
    frames: int
    size_bytes: int


@dataclass(frozen=True)
class ToolSummary:
    name: str
    found: bool
    runnable: bool
    return_code: int
    path: str
    version: str


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _parse_frame_rate(value: object) -> float:
    text = str(value or "")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denom = _safe_float(denominator)
        return _safe_float(numerator) / denom if denom else 0.0
    return _safe_float(text)


def parse_ffprobe_json(text: str, path: str = "") -> MediaSummary:
    loaded = json.loads(text)
    streams = loaded.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        streams[0] if streams else {},
    )
    fmt = loaded.get("format", {})
    duration = _safe_float(video_stream.get("duration")) or _safe_float(fmt.get("duration"))
    return MediaSummary(
        path=path,
        codec=str(video_stream.get("codec_name", "")),
        width=_safe_int(video_stream.get("width")),
        height=_safe_int(video_stream.get("height")),
        duration_seconds=duration,
        frame_rate=_parse_frame_rate(video_stream.get("avg_frame_rate")),
        frames=_safe_int(video_stream.get("nb_frames")),
        size_bytes=_safe_int(fmt.get("size")),
    )


def _default_version_args(name: str) -> list[str]:
    base = Path(name).name
    if base in {"ffmpeg", "ffprobe"}:
        return ["-version"]
    if base.startswith("colmap") or "colmap" in name:
        return ["-h"]
    if base == "nvidia-smi":
        return []
    return ["--version"]


def probe_media(path: str | Path) -> MediaSummary:
    media_path = Path(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(media_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_ffprobe_json(completed.stdout, path=str(media_path))


def summarize_tool(name: str, version_args: list[str] | None = None) -> ToolSummary:
    explicit = Path(name)
    path = str(explicit) if explicit.exists() else shutil.which(name)
    if not path:
        return ToolSummary(name=name, found=False, runnable=False, return_code=127, path="", version="")
    args = version_args if version_args is not None else _default_version_args(name)
    command = ["bash", path, *args] if path.endswith(".sh") else [path, *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
        version = (completed.stdout or completed.stderr).splitlines()[0].strip()
        return_code = completed.returncode
    except (OSError, subprocess.TimeoutExpired):
        version = ""
        return_code = 124
    return ToolSummary(
        name=name,
        found=True,
        runnable=return_code == 0,
        return_code=return_code,
        path=path,
        version=version,
    )


def probe_report(media_paths: list[str | Path], tool_names: list[str]) -> dict[str, object]:
    return {
        "media": [asdict(probe_media(path)) for path in media_paths],
        "tools": [asdict(summarize_tool(name)) for name in tool_names],
    }
