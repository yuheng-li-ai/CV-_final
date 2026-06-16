#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ply_counts(path: Path) -> tuple[int | None, int | None]:
    vertices = None
    faces = None
    if not path.exists():
        return vertices, faces
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                faces = int(line.split()[-1])
            elif line == "end_header":
                break
    return vertices, faces


def _size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _contact_sheet(
    items: list[tuple[str, Path]],
    out_path: Path,
    *,
    columns: int,
    thumb_width: int,
) -> None:
    label_height = 34
    margin = 8
    thumbs: list[tuple[str, Image.Image]] = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        thumbs.append((label, _resize_to_width(image, thumb_width)))

    thumb_height = max(image.height for _, image in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(thumbs):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        draw.text((x + margin, y + 10), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def make_fusion_orientation_contact() -> Path:
    out_path = Path("reports/figures/fusion_v7_orientation_contact.jpg")
    _contact_sheet(
        [
            (
                "v6 baseline",
                Path("runs/scene/fusion_table_insert_v6_position/review_composite.jpg"),
            ),
            (
                "A flip X",
                Path("runs/scene/fusion_table_insert_v7_a_flip_x/review_composite.jpg"),
            ),
            (
                "A flip Y",
                Path("runs/scene/fusion_table_insert_v7_a_flip_y/review_composite.jpg"),
            ),
            (
                "A flip X + B flip X",
                Path("runs/scene/fusion_table_insert_v7_a_flip_x_b_flip_x/review_composite.jpg"),
            ),
            (
                "A flip X + B flip Y",
                Path("runs/scene/fusion_table_insert_v7_a_flip_x_b_flip_y/review_composite.jpg"),
            ),
            (
                "A flip Y + B flip X",
                Path("runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/review_composite.jpg"),
            ),
            (
                "A flip Y + B flip Y",
                Path("runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_y/review_composite.jpg"),
            ),
        ],
        out_path,
        columns=2,
        thumb_width=460,
    )
    return out_path


def make_object_a_render_pair_sheet() -> Path:
    out_path = Path("reports/figures/object_a_2dgs_render_gt_samples.jpg")
    items: list[tuple[str, Path]] = []
    for frame_id in ("00000", "00006", "00012", "00018"):
        items.append((f"GT {frame_id}", Path(f"runs/scene/object_a_2dgs_dense/test/ours_15000/gt/{frame_id}.png")))
        items.append((f"2DGS {frame_id}", Path(f"runs/scene/object_a_2dgs_dense/test/ours_15000/renders/{frame_id}.png")))
    _contact_sheet(items, out_path, columns=4, thumb_width=260)
    return out_path


def make_object_a_stage_table() -> Path:
    colmap_rows = _read_csv(Path("reports/tables/object_a_colmap_r2_stats.csv"))
    dense_colmap = next(row for row in colmap_rows if row["run_id"] == "object_a_dense")
    metric_rows = _read_csv(Path("reports/tables/object_a_2dgs_metrics.csv"))
    dense_2dgs = next(row for row in metric_rows if row["run_id"] == "dense_half")

    mesh_1024 = Path("runs/scene/object_a_2dgs_dense/train/ours_15000/fuse_post_meshres1024.ply")
    mesh_256 = Path("runs/scene/object_a_2dgs_dense/train/ours_15000/fuse_post.ply")
    component_02 = Path("runs/scene/object_a_2dgs_dense/train/ours_15000/components/component_02.ply")
    mesh_1024_v, mesh_1024_f = _ply_counts(mesh_1024)
    mesh_256_v, mesh_256_f = _ply_counts(mesh_256)
    comp_v, comp_f = _ply_counts(component_02)

    rows: list[dict[str, object]] = [
        {
            "stage": "COLMAP dense reconstruction",
            "artifact": "reports/tables/object_a_colmap_r2_stats.csv",
            "evidence": (
                f"{dense_colmap['registered_images']}/{dense_colmap['images']} images registered; "
                f"{dense_colmap['points']} sparse points; "
                f"mean reprojection error {float(dense_colmap['mean_reprojection_error_px']):.4f}px"
            ),
            "status": "pass",
            "paper_claim": "Camera registration and multiview coverage are sufficient; Object A is not failing at COLMAP.",
        },
        {
            "stage": "2DGS novel-view rendering",
            "artifact": dense_2dgs["preview_path"],
            "evidence": (
                f"PSNR {float(dense_2dgs['psnr']):.4f}; "
                f"SSIM {float(dense_2dgs['ssim']):.4f}; "
                f"LPIPS {float(dense_2dgs['lpips']):.4f}; "
                f"{dense_2dgs['render_count']} eval views"
            ),
            "status": "pass",
            "paper_claim": "The color and view synthesis quality are preserved through 2DGS image rendering.",
        },
        {
            "stage": "2DGS mesh export, mesh_res=1024",
            "artifact": str(mesh_1024),
            "evidence": (
                f"{mesh_1024_v} vertices; {mesh_1024_f} faces; "
                f"{_size_mb(mesh_1024):.1f} MB post mesh"
            ),
            "status": "fail_for_fusion",
            "paper_claim": "The first clear failure appears during TSDF/mesh extraction: table/background shells dominate the mesh.",
        },
        {
            "stage": "2DGS mesh export, mesh_res=256",
            "artifact": str(mesh_256),
            "evidence": (
                f"{mesh_256_v} vertices; {mesh_256_f} faces; "
                f"{_size_mb(mesh_256):.1f} MB post mesh"
            ),
            "status": "partial",
            "paper_claim": "Lower mesh resolution makes the asset manageable but still keeps non-object geometry.",
        },
        {
            "stage": "Connected-component filtering",
            "artifact": str(component_02),
            "evidence": f"selected component_02 with {comp_v} vertices and {comp_f} faces",
            "status": "partial",
            "paper_claim": "Component selection removes most table/background shell but cannot fully repair incomplete object geometry.",
        },
        {
            "stage": "Fusion orientation and placement",
            "artifact": "reports/figures/fusion_v7_orientation_contact.jpg",
            "evidence": "A/B rotation variants rendered as still composites before final video; no final video rendered yet",
            "status": "under_review",
            "paper_claim": "Remaining visible issues are coordinate/orientation and contact tuning, not failure of the 2DGS image renderer.",
        },
    ]
    out_path = Path("reports/tables/object_a_stage_failure_evidence.csv")
    _write_csv(out_path, rows)
    return out_path


def make_object_a_stage_sheet() -> Path:
    out_path = Path("reports/figures/object_a_stage_evidence.jpg")
    _contact_sheet(
        [
            ("GT vs 2DGS samples", Path("reports/figures/object_a_2dgs_render_gt_samples.jpg")),
            (
                "Raw mesh fusion has shell",
                Path("runs/scene/fusion_table_insert_v5/review_v5_composite.jpg"),
            ),
            (
                "Mesh components expose failure",
                Path("runs/scene/fusion_object_a_components/components_preview.png"),
            ),
            (
                "Component fusion before final fix",
                Path("runs/scene/fusion_table_insert_v6_position/review_composite.jpg"),
            ),
        ],
        out_path,
        columns=2,
        thumb_width=460,
    )
    return out_path


def make_object_c_summary() -> Path:
    rows_in = _read_csv(Path("reports/tables/object_c_magic123_r5_ablation.csv"))
    rows: list[dict[str, object]] = []
    for row in rows_in:
        rows.append(
            {
                "variant": row["variant"],
                "background_mode": row["background_mode"],
                "vertices": row["vertices"],
                "faces": row["faces"],
                "geometry_score": row["geometry_score"],
                "texture_score": row["texture_score"],
                "fusion_readiness_score": row["fusion_readiness_score"],
                "selection": row["selection"],
                "paper_claim": row["artifact_notes"],
            }
        )
    out_path = Path("reports/tables/object_c_limitations_summary.csv")
    _write_csv(out_path, rows)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report evidence tables and contact sheets.")
    parser.add_argument("--all", action="store_true", help="Generate every evidence artifact.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.all:
        raise SystemExit("Use --all to generate report evidence artifacts.")

    outputs = [
        make_fusion_orientation_contact(),
        make_object_a_render_pair_sheet(),
        make_object_a_stage_table(),
        make_object_a_stage_sheet(),
        make_object_c_summary(),
    ]
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
