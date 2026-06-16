from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class InputStagingPlan:
    object_a_video: Path
    object_c_image: Path
    data_root: Path = Path("data")
    dry_run: bool = False


@dataclass(frozen=True)
class StagedInputs:
    object_a_source: Path
    object_a_target: Path
    object_c_source: Path
    object_c_target: Path
    dry_run: bool

    def format(self) -> str:
        mode = "dry-run" if self.dry_run else "copied"
        return "\n".join(
            [
                f"mode: {mode}",
                f"object_a_video: {self.object_a_source} -> {self.object_a_target}",
                f"object_c_image: {self.object_c_source} -> {self.object_c_target}",
            ]
        )


def stage_local_inputs(plan: InputStagingPlan) -> StagedInputs:
    object_a_target = plan.data_root / "scene/object_a/raw/object_a.mp4"
    object_c_target = plan.data_root / "scene/object_c/raw/object_c.jpg"
    staged = StagedInputs(
        object_a_source=plan.object_a_video,
        object_a_target=object_a_target,
        object_c_source=plan.object_c_image,
        object_c_target=object_c_target,
        dry_run=plan.dry_run,
    )
    if plan.dry_run:
        return staged

    for source, target in [
        (plan.object_a_video, object_a_target),
        (plan.object_c_image, object_c_target),
    ]:
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return staged
