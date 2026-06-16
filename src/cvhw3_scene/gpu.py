from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class GpuMemory:
    index: int
    used_mb: int
    total_mb: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def parse_nvidia_smi_csv(text: str) -> list[GpuMemory]:
    rows: list[GpuMemory] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected 3 CSV columns from nvidia-smi, got: {line}")
        rows.append(
            GpuMemory(
                index=int(parts[0]),
                used_mb=int(parts[1]),
                total_mb=int(parts[2]),
            )
        )
    return rows


def select_freest_gpu(rows: list[GpuMemory]) -> str:
    if not rows:
        raise ValueError("No GPU rows were reported by nvidia-smi")
    selected = sorted(rows, key=lambda row: (-row.free_mb, row.index))[0]
    return f"cuda:{selected.index}"


def query_gpu_memory() -> list[GpuMemory]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return parse_nvidia_smi_csv(completed.stdout)


def select_device(device_override: str | None = None) -> str:
    if device_override:
        return device_override
    return select_freest_gpu(query_gpu_memory())


def format_gpu_report(device: str, rows: list[GpuMemory] | None = None) -> str:
    lines = [f"selected_device: {device}"]
    if rows:
        lines.append("nvidia_smi_memory:")
        for row in rows:
            lines.append(
                f"- index={row.index} used_mb={row.used_mb} total_mb={row.total_mb} "
                f"free_mb={row.free_mb}"
            )
    return "\n".join(lines) + "\n"
