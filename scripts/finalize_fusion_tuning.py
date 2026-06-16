#!/usr/bin/env python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TuningRun:
    name: str
    run_id: str
    config: Path
    foreground: Path
    composite: Path
    scale_score: int
    contact_score: int
    lighting_score: int
    viewpoint_stability_score: int
    artifact_count: int
    notes: str
    selection: str


RUNS = [
    TuningRun(
        name="v2_main",
        run_id="fusion_main_layout_v2",
        config=Path("configs/scene/fusion_main.yaml"),
        foreground=Path("runs/scene/fusion_main/garden_v2_foreground.png"),
        composite=Path("runs/scene/fusion_main/garden_v2_composite.jpg"),
        scale_score=2,
        contact_score=3,
        lighting_score=3,
        viewpoint_stability_score=2,
        artifact_count=4,
        notes="First real garden composite; objects are visible but too large and spread beyond the table footprint.",
        selection="rejected_too_large",
    ),
    TuningRun(
        name="scale_small",
        run_id="fusion_scale_small_layout_v2",
        config=Path("configs/scene/fusion_scale_ablation.yaml"),
        foreground=Path("runs/scene/fusion_scale_ablation/garden_scale_small_foreground.png"),
        composite=Path("runs/scene/fusion_scale_ablation/garden_scale_small_composite.jpg"),
        scale_score=3,
        contact_score=3,
        lighting_score=3,
        viewpoint_stability_score=3,
        artifact_count=3,
        notes="Global 0.78 scale reduces the mismatch, but A and C still sit near or outside the table edge and B occludes the vase.",
        selection="candidate_not_selected",
    ),
    TuningRun(
        name="table_insert_v3",
        run_id="fusion_table_insert_v3_layout",
        config=Path("configs/scene/fusion_table_insert_v3.yaml"),
        foreground=Path("runs/scene/fusion_table_insert_v3/garden_v3_foreground.png"),
        composite=Path("runs/scene/fusion_table_insert_v3/garden_v3_composite.jpg"),
        scale_score=4,
        contact_score=4,
        lighting_score=3,
        viewpoint_stability_score=4,
        artifact_count=2,
        notes="Global 0.62 scale with tighter x spacing keeps A/B/C on the garden table and is the current best still-frame insertion candidate.",
        selection="selected_for_next_video_smoke",
    ),
]


def _require_paths() -> None:
    missing = []
    for run in RUNS:
        for path in (run.config, run.foreground, run.composite):
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing fusion tuning artifacts: " + ", ".join(str(path) for path in missing))


def write_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "variant",
        "config",
        "foreground_frame",
        "composite_frame",
        "scale_score_1_to_5",
        "contact_score_1_to_5",
        "lighting_score_1_to_5",
        "viewpoint_stability_1_to_5",
        "artifact_count",
        "notes",
        "selection",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in RUNS:
            writer.writerow(
                {
                    "run_id": run.run_id,
                    "variant": run.name,
                    "config": run.config,
                    "foreground_frame": run.foreground,
                    "composite_frame": run.composite,
                    "scale_score_1_to_5": run.scale_score,
                    "contact_score_1_to_5": run.contact_score,
                    "lighting_score_1_to_5": run.lighting_score,
                    "viewpoint_stability_1_to_5": run.viewpoint_stability_score,
                    "artifact_count": run.artifact_count,
                    "notes": run.notes,
                    "selection": run.selection,
                }
            )


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 16, size[1] - 42), Image.Resampling.LANCZOS)
    x = (size[0] - image.size[0]) // 2
    y = 36 + (size[1] - 42 - image.size[1]) // 2
    canvas.paste(image, (x, y))
    return canvas


def write_contact_sheet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = (420, 280)
    sheet = Image.new("RGB", (cell[0] * len(RUNS), cell[1]), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, run in enumerate(RUNS):
        x = index * cell[0]
        image = _fit(run.composite, cell)
        image_draw = ImageDraw.Draw(image)
        image_draw.rectangle((0, 0, cell[0], 34), fill=(245, 245, 245))
        score = f"S{run.scale_score}/C{run.contact_score}/L{run.lighting_score}/V{run.viewpoint_stability_score}"
        image_draw.text((8, 5), run.name, fill=(20, 20, 20), font=font)
        image_draw.text((8, 18), f"{score} artifacts={run.artifact_count}", fill=(80, 80, 80), font=font)
        sheet.paste(image, (x, 0))
    sheet.save(path, quality=95)


def main() -> int:
    _require_paths()
    table = Path("reports/tables/fusion_garden_insertion_tuning.csv")
    figure = Path("reports/figures/fusion_garden_insertion_iterations.jpg")
    write_table(table)
    write_contact_sheet(figure)
    print(f"WROTE_TABLE: {table}")
    print(f"WROTE_FIGURE: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
