#!/usr/bin/env python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ReviewVariant:
    name: str
    foreground: Path
    composite: Path
    scale_score: int
    contact_score: int
    lighting_score: int
    viewpoint_stability_score: int
    artifact_count: int
    notes: str
    recommendation: str


VARIANTS = [
    ReviewVariant(
        name="default_shadow",
        foreground=Path("runs/scene/fusion_table_insert_v3/review_default_foreground.png"),
        composite=Path("runs/scene/fusion_table_insert_v3/review_default_composite.jpg"),
        scale_score=4,
        contact_score=4,
        lighting_score=3,
        viewpoint_stability_score=4,
        artifact_count=2,
        notes="Current table_insert_v3 composition; balanced placement, but generated assets remain brighter than the garden table.",
        recommendation="baseline_candidate",
    ),
    ReviewVariant(
        name="soft_shadow",
        foreground=Path("runs/scene/fusion_table_insert_v3/review_soft_foreground.png"),
        composite=Path("runs/scene/fusion_table_insert_v3/review_soft_composite.jpg"),
        scale_score=4,
        contact_score=4,
        lighting_score=4,
        viewpoint_stability_score=4,
        artifact_count=2,
        notes="Lower light energy reduces harshness and keeps contact shadows; best still-frame candidate before user review.",
        recommendation="recommended_for_user_review",
    ),
    ReviewVariant(
        name="side_light",
        foreground=Path("runs/scene/fusion_table_insert_v3/review_side_light_foreground.png"),
        composite=Path("runs/scene/fusion_table_insert_v3/review_side_light_composite.jpg"),
        scale_score=4,
        contact_score=4,
        lighting_score=3,
        viewpoint_stability_score=4,
        artifact_count=2,
        notes="Side light gives more shape contrast but is less consistent with the diffuse garden illumination.",
        recommendation="candidate_not_selected",
    ),
    ReviewVariant(
        name="no_shadow",
        foreground=Path("runs/scene/fusion_table_insert_v3/review_no_shadow_foreground.png"),
        composite=Path("runs/scene/fusion_table_insert_v3/review_no_shadow_composite.jpg"),
        scale_score=4,
        contact_score=2,
        lighting_score=2,
        viewpoint_stability_score=3,
        artifact_count=3,
        notes="No-shadow control looks less grounded on the table; useful as R6 evidence but not final candidate.",
        recommendation="rejected_shadow_off",
    ),
]


def _require_paths() -> None:
    missing = []
    for variant in VARIANTS:
        for path in (variant.foreground, variant.composite):
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing review artifacts: " + ", ".join(str(path) for path in missing))


def write_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "foreground_frame",
        "composite_frame",
        "scale_score_1_to_5",
        "contact_score_1_to_5",
        "lighting_score_1_to_5",
        "viewpoint_stability_1_to_5",
        "artifact_count",
        "notes",
        "recommendation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            writer.writerow(
                {
                    "variant": variant.name,
                    "foreground_frame": variant.foreground,
                    "composite_frame": variant.composite,
                    "scale_score_1_to_5": variant.scale_score,
                    "contact_score_1_to_5": variant.contact_score,
                    "lighting_score_1_to_5": variant.lighting_score,
                    "viewpoint_stability_1_to_5": variant.viewpoint_stability_score,
                    "artifact_count": variant.artifact_count,
                    "notes": variant.notes,
                    "recommendation": variant.recommendation,
                }
            )


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 16, size[1] - 46), Image.Resampling.LANCZOS)
    x = (size[0] - image.size[0]) // 2
    y = 42 + (size[1] - 46 - image.size[1]) // 2
    canvas.paste(image, (x, y))
    return canvas


def write_contact_sheet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = (360, 260)
    sheet = Image.new("RGB", (cell[0] * 2, cell[1] * 2), "white")
    font = ImageFont.load_default()
    for index, variant in enumerate(VARIANTS):
        x = (index % 2) * cell[0]
        y = (index // 2) * cell[1]
        image = _fit(variant.composite, cell)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, cell[0], 38), fill=(245, 245, 245))
        score = f"S{variant.scale_score}/C{variant.contact_score}/L{variant.lighting_score}/V{variant.viewpoint_stability_score}"
        draw.text((8, 5), variant.name, fill=(20, 20, 20), font=font)
        draw.text((8, 20), f"{score} artifacts={variant.artifact_count}", fill=(80, 80, 80), font=font)
        sheet.paste(image, (x, y))
    sheet.save(path, quality=95)


def main() -> int:
    _require_paths()
    table = Path("reports/tables/fusion_pre_final_review.csv")
    figure = Path("reports/figures/fusion_pre_final_review_sheet.jpg")
    write_table(table)
    write_contact_sheet(figure)
    print(f"WROTE_TABLE: {table}")
    print(f"WROTE_FIGURE: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
