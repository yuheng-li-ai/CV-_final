#!/usr/bin/env python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import trimesh
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Variant:
    name: str
    run_id: str
    input_image: Path
    mask_image: Path | None
    workspace: Path
    background_mode: str
    centered_crop: bool
    geometry_score: int
    texture_score: int
    fusion_readiness_score: int
    artifact_notes: str
    selection: str

    @property
    def mesh_dir(self) -> Path:
        return self.workspace / "fine" / "mesh"

    @property
    def results_dir(self) -> Path:
        return self.workspace / "fine" / "results"


VARIANTS = [
    Variant(
        name="raw",
        run_id="object_c_magic123_raw_full",
        input_image=Path("data/scene/object_c/raw/object_c.jpg"),
        mask_image=None,
        workspace=Path("runs/scene/image3d_raw/raw"),
        background_mode="raw_image",
        centered_crop=False,
        geometry_score=3,
        texture_score=2,
        fusion_readiness_score=3,
        artifact_notes="recognizable object but raw background introduces stronger texture contamination and less clean silhouette",
        selection="candidate_not_selected",
    ),
    Variant(
        name="auto_mask",
        run_id="object_c_magic123_auto_mask_full",
        input_image=Path("data/scene/object_c/masked/object_c_auto.png"),
        mask_image=Path("data/scene/object_c/masks/object_c_auto.png"),
        workspace=Path("runs/scene/image3d_auto_mask/auto_mask"),
        background_mode="automatic_background_removal",
        centered_crop=True,
        geometry_score=4,
        texture_score=4,
        fusion_readiness_score=4,
        artifact_notes="best balance: clean silhouette, stable side/back views, visible eye/detail retention, and complete decimated mesh with only minor edge noise",
        selection="selected_for_fusion",
    ),
    Variant(
        name="refined_mask",
        run_id="object_c_magic123_refined_mask_full",
        input_image=Path("data/scene/object_c/masked/object_c_refined.png"),
        mask_image=Path("data/scene/object_c/masks/object_c_refined.png"),
        workspace=Path("runs/scene/image3d_refined_mask/refined_mask"),
        background_mode="refined_mask",
        centered_crop=True,
        geometry_score=3,
        texture_score=3,
        fusion_readiness_score=3,
        artifact_notes="least background pollution in the input, but generated geometry is more inflated/distorted from side views and texture atlas shows stronger fragmentation",
        selection="candidate_not_selected",
    ),
]


def _load_mesh_shape(path: Path) -> tuple[int, int]:
    mesh = trimesh.load(path, process=False, force="mesh")
    return int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])


def _image_size(path: Path) -> str:
    image = Image.open(path)
    return f"{image.size[0]}x{image.size[1]}"


def _fit_image(path: Path | None, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    if path is None:
        draw = ImageDraw.Draw(canvas)
        draw.text((size[0] // 2 - 24, size[1] // 2 - 8), "N/A", fill=(80, 80, 80))
        return canvas

    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    image = background.convert("RGB")
    image.thumbnail((size[0] - 16, size[1] - 28), Image.Resampling.LANCZOS)
    x = (size[0] - image.size[0]) // 2
    y = (size[1] - image.size[1]) // 2 + 8
    canvas.paste(image, (x, y))
    return canvas


def _label_cell(image: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.size[0], 24), fill=(245, 245, 245))
    draw.text((8, 6), label, fill=(20, 20, 20), font=ImageFont.load_default())
    return image


def write_contact_sheet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = (240, 190)
    columns = [
        ("input", lambda v: v.input_image),
        ("mask", lambda v: v.mask_image),
        ("lambertian", lambda v: v.results_dir / "fine_ep0050_lambertian.jpg"),
        ("normal", lambda v: v.results_dir / "fine_ep0050_normal_image.jpg"),
        ("mesh_albedo", lambda v: v.mesh_dir / "albedo.png"),
    ]
    header_h = 34
    sheet = Image.new("RGB", (cell[0] * len(columns), header_h + cell[1] * len(VARIANTS)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (label, _) in enumerate(columns):
        draw.text((col * cell[0] + 8, 10), label, fill=(20, 20, 20), font=ImageFont.load_default())
    for row, variant in enumerate(VARIANTS):
        y = header_h + row * cell[1]
        for col, (_, resolver) in enumerate(columns):
            resolved = resolver(variant)
            if resolved is not None and not resolved.exists():
                resolved = None
            image = _fit_image(resolved, cell)
            if col == 0:
                image = _label_cell(image, variant.name)
            sheet.paste(image, (col * cell[0], y))
    sheet.save(path, quality=95)


def write_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "variant",
        "background_mode",
        "centered_crop",
        "input_image",
        "mask",
        "checkpoint",
        "mesh",
        "texture",
        "texture_resolution",
        "vertices",
        "faces",
        "video",
        "geometry_score",
        "texture_score",
        "fusion_readiness_score",
        "artifact_notes",
        "selection",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in VARIANTS:
            mesh_path = variant.mesh_dir / "mesh.obj"
            texture_path = variant.mesh_dir / "albedo.png"
            vertices, faces = _load_mesh_shape(mesh_path)
            writer.writerow(
                {
                    "run_id": variant.run_id,
                    "variant": variant.name,
                    "background_mode": variant.background_mode,
                    "centered_crop": str(variant.centered_crop).lower(),
                    "input_image": variant.input_image,
                    "mask": variant.mask_image or "none",
                    "checkpoint": variant.workspace / "fine" / "checkpoints" / "fine.pth",
                    "mesh": mesh_path,
                    "texture": texture_path,
                    "texture_resolution": _image_size(texture_path),
                    "vertices": vertices,
                    "faces": faces,
                    "video": variant.results_dir / "fine_ep0050_lambertian.mp4",
                    "geometry_score": variant.geometry_score,
                    "texture_score": variant.texture_score,
                    "fusion_readiness_score": variant.fusion_readiness_score,
                    "artifact_notes": variant.artifact_notes,
                    "selection": variant.selection,
                }
            )


def main() -> int:
    table_path = Path("reports/tables/object_c_magic123_r5_ablation.csv")
    figure_path = Path("reports/figures/object_c_magic123_r5_contact_sheet.jpg")
    write_table(table_path)
    write_contact_sheet(figure_path)
    print(f"WROTE_TABLE: {table_path}")
    print(f"WROTE_FIGURE: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
