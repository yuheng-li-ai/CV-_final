#!/usr/bin/env python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Variant:
    name: str
    run_id: str
    scene: Path
    transform_table: Path
    layout: Path
    scale_score: int
    contact_score: int
    lighting_score: int
    viewpoint_stability_score: int
    artifact_count: int
    video_quality: str
    notes: str
    selection: str


VARIANTS = [
    Variant(
        name="main_v1_cluster10_yup",
        run_id="fusion_main_layout_v1",
        scene=Path("runs/scene/fusion_main/scene_v1.json"),
        transform_table=Path("runs/scene/fusion_main/transform_table_v1.csv"),
        layout=Path("runs/scene/fusion_main/layout_topdown_v1.jpg"),
        scale_score=2,
        contact_score=3,
        lighting_score=3,
        viewpoint_stability_score=2,
        artifact_count=4,
        video_quality="not_rendered_layout_only",
        notes="Object A used cluster10 mesh with y-up conversion; footprint was too large and sparse, so viewpoint stability would likely be weak.",
        selection="rejected_scale_orientation",
    ),
    Variant(
        name="main_v2_cluster1_zup",
        run_id="fusion_main_layout_v2",
        scene=Path("runs/scene/fusion_main/scene.json"),
        transform_table=Path("runs/scene/fusion_main/transform_table.csv"),
        layout=Path("runs/scene/fusion_main/layout_topdown.jpg"),
        scale_score=4,
        contact_score=4,
        lighting_score=3,
        viewpoint_stability_score=4,
        artifact_count=1,
        video_quality="not_rendered_layout_selected",
        notes="Object A switched to largest connected cluster, z-up conversion, and 0.60 target height; A/B/C footprints are balanced for the first Blender render.",
        selection="selected_for_blender_render",
    ),
    Variant(
        name="scale_small_v2",
        run_id="fusion_scale_small_layout_v2",
        scene=Path("runs/scene/fusion_scale_ablation/scene.json"),
        transform_table=Path("runs/scene/fusion_scale_ablation/transform_table.csv"),
        layout=Path("runs/scene/fusion_scale_ablation/layout_topdown.jpg"),
        scale_score=3,
        contact_score=4,
        lighting_score=3,
        viewpoint_stability_score=3,
        artifact_count=2,
        video_quality="not_rendered_layout_only",
        notes="Global 0.78 scale multiplier improves spacing but makes generated assets less legible against the background.",
        selection="candidate_not_selected",
    ),
    Variant(
        name="position_shift_right_v2",
        run_id="fusion_position_shift_layout_v2",
        scene=Path("runs/scene/fusion_position_ablation/scene.json"),
        transform_table=Path("runs/scene/fusion_position_ablation/transform_table.csv"),
        layout=Path("runs/scene/fusion_position_ablation/layout_topdown.jpg"),
        scale_score=4,
        contact_score=3,
        lighting_score=3,
        viewpoint_stability_score=3,
        artifact_count=2,
        video_quality="not_rendered_layout_only",
        notes="Position offset tests global placement sensitivity; right shift compresses B/C spacing and is less balanced for an orbit path.",
        selection="candidate_not_selected",
    ),
    Variant(
        name="shadow_off_v2",
        run_id="fusion_shadow_off_layout_v2",
        scene=Path("runs/scene/fusion_shadow_ablation/scene.json"),
        transform_table=Path("runs/scene/fusion_shadow_ablation/transform_table.csv"),
        layout=Path("runs/scene/fusion_shadow_ablation/layout_topdown.jpg"),
        scale_score=4,
        contact_score=3,
        lighting_score=2,
        viewpoint_stability_score=3,
        artifact_count=2,
        video_quality="not_rendered_layout_only",
        notes="Shadow-disabled control keeps geometry identical to main_v2; expected to look less grounded once Blender render is available.",
        selection="candidate_not_selected",
    ),
]


def _require_paths() -> None:
    missing = []
    for variant in VARIANTS:
        for path in (variant.scene, variant.transform_table, variant.layout):
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing fusion artifacts: " + ", ".join(str(path) for path in missing))


def write_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "variant",
        "scene",
        "transform_table",
        "layout_preview",
        "scale_score_1_to_5",
        "contact_score_1_to_5",
        "lighting_score_1_to_5",
        "viewpoint_stability_1_to_5",
        "artifact_count",
        "video_quality",
        "notes",
        "selection",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in VARIANTS:
            writer.writerow(
                {
                    "run_id": variant.run_id,
                    "variant": variant.name,
                    "scene": variant.scene,
                    "transform_table": variant.transform_table,
                    "layout_preview": variant.layout,
                    "scale_score_1_to_5": variant.scale_score,
                    "contact_score_1_to_5": variant.contact_score,
                    "lighting_score_1_to_5": variant.lighting_score,
                    "viewpoint_stability_1_to_5": variant.viewpoint_stability_score,
                    "artifact_count": variant.artifact_count,
                    "video_quality": variant.video_quality,
                    "notes": variant.notes,
                    "selection": variant.selection,
                }
            )


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 14, size[1] - 36), Image.Resampling.LANCZOS)
    x = (size[0] - image.size[0]) // 2
    y = 30 + (size[1] - 36 - image.size[1]) // 2
    canvas.paste(image, (x, y))
    return canvas


def _label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.size[0], 30), fill=(245, 245, 245))
    draw.text((8, 5), title, fill=(20, 20, 20), font=font)
    draw.text((8, 17), subtitle, fill=(80, 80, 80), font=font)
    return image


def write_contact_sheet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = (440, 310)
    sheet = Image.new("RGB", (cell[0] * 2, cell[1] * 3), "white")
    for index, variant in enumerate(VARIANTS):
        x = (index % 2) * cell[0]
        y = (index // 2) * cell[1]
        image = _fit_image(variant.layout, cell)
        subtitle = f"S{variant.scale_score}/C{variant.contact_score}/L{variant.lighting_score}/V{variant.viewpoint_stability_score}"
        image = _label(image, variant.name, subtitle)
        sheet.paste(image, (x, y))
    sheet.save(path, quality=95)


def main() -> int:
    _require_paths()
    table_path = Path("reports/tables/fusion_r6_transform_shadow_ablation.csv")
    figure_path = Path("reports/figures/fusion_r6_layout_contact_sheet.jpg")
    write_table(table_path)
    write_contact_sheet(figure_path)
    print(f"WROTE_TABLE: {table_path}")
    print(f"WROTE_FIGURE: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
