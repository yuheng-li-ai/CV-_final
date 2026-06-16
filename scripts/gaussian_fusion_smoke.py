#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814


def _load_xyz(ply_path: Path) -> np.ndarray:
    vertex = PlyData.read(ply_path)["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def _bounds_from_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz = _load_xyz(path)
    return xyz.min(axis=0), xyz.max(axis=0)


def _matrix_from_rows(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def _rotation_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    diag = np.diag(matrix)
    index = int(np.argmax(diag))
    if index == 0:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif index == 1:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat / max(np.linalg.norm(quat), 1e-12)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _transform_xyz(xyz: np.ndarray, transform: dict) -> np.ndarray:
    source_to_fusion = _matrix_from_rows(transform["source_to_fusion_matrix"])
    rotation = _matrix_from_rows(transform["rotation_matrix"])
    center = np.asarray(transform["center"], dtype=np.float64)
    contact_shift = np.asarray(transform["contact_shift"], dtype=np.float64)
    location = np.asarray(transform["location"], dtype=np.float64)
    scale = float(transform["final_scale"])
    return ((xyz @ source_to_fusion.T - center) * scale) @ rotation.T + contact_shift + location


def _dc_to_rgb(vertex: np.ndarray) -> np.ndarray:
    rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
    return np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)


def _draw_projection(
    draw: ImageDraw.ImageDraw,
    xy: np.ndarray,
    colors: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(40, 40, 40), width=2)
    draw.text((x0 + 8, y0 + 8), label, fill=(0, 0, 0))
    if len(xy) == 0:
        return
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-8)
    scale = min((x1 - x0 - 28) / span[0], (y1 - y0 - 34) / span[1])
    px = x0 + 14 + (xy[:, 0] - xy[:, 0].min()) * scale
    py = y1 - 14 - (xy[:, 1] - xy[:, 1].min()) * scale
    order = np.argsort(py)
    for index in order:
        color = tuple(int(value * 255) for value in colors[index])
        x = int(np.clip(px[index], x0 + 2, x1 - 3))
        y = int(np.clip(py[index], y0 + 22, y1 - 3))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)


def _write_contact(path: Path, xyz: np.ndarray, colors: np.ndarray, title: str) -> None:
    if len(xyz) > 18000:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(xyz), size=18000, replace=False)
        xyz = xyz[keep]
        colors = colors[keep]
    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 14), title, fill=(0, 0, 0))
    _draw_projection(draw, xyz[:, [0, 2]], colors, (25, 55, 385, 735), "front: x/z")
    _draw_projection(draw, xyz[:, [1, 2]], colors, (420, 55, 780, 735), "side: y/z")
    _draw_projection(draw, xyz[:, [0, 1]], colors, (815, 55, 1175, 735), "top: x/y")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def _asset(scene: dict, name: str) -> dict:
    for asset in scene["assets"]:
        if asset["name"] == name:
            return asset
    raise ValueError(f"Missing asset in scene: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop and transform Object A 2DGS Gaussians for a Phase6 smoke.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--object_gaussian", required=True)
    parser.add_argument("--component_mesh", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--asset_name", default="object_a")
    parser.add_argument("--bbox_margin", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scene_path = Path(args.scene)
    object_gaussian = Path(args.object_gaussian)
    component_mesh = Path(args.component_mesh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    transform = _asset(scene, args.asset_name)["transform"]
    ply = PlyData.read(object_gaussian)
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    comp_min, comp_max = _bounds_from_ply(component_mesh)
    extent = np.maximum(comp_max - comp_min, 1e-8)
    crop_min = comp_min - extent * args.bbox_margin
    crop_max = comp_max + extent * args.bbox_margin
    mask = ((xyz >= crop_min) & (xyz <= crop_max)).all(axis=1)
    cropped = vertex[mask].copy()
    source_cropped = cropped.copy()
    cropped_xyz = xyz[mask]
    transformed_xyz = _transform_xyz(cropped_xyz, transform)
    cropped["x"] = transformed_xyz[:, 0].astype(np.float32)
    cropped["y"] = transformed_xyz[:, 1].astype(np.float32)
    cropped["z"] = transformed_xyz[:, 2].astype(np.float32)

    scale = float(transform["final_scale"])
    for key in ("scale_0", "scale_1"):
        if key in cropped.dtype.names:
            cropped[key] = (cropped[key].astype(np.float64) + math.log(scale)).astype(np.float32)

    rotation = _matrix_from_rows(transform["rotation_matrix"])
    scene_quat = _rotation_to_quat_wxyz(rotation)[None, :]
    rot_keys = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(key in cropped.dtype.names for key in rot_keys):
        original = np.stack([cropped[key] for key in rot_keys], axis=1).astype(np.float64)
        rotated = _quat_multiply_wxyz(np.repeat(scene_quat, len(cropped), axis=0), original)
        for index, key in enumerate(rot_keys):
            cropped[key] = rotated[:, index].astype(np.float32)

    source_ply = out_dir / "object_a_gaussian_crop_source.ply"
    out_ply = out_dir / "object_a_gaussian_crop_fusion.ply"
    PlyData([PlyElement.describe(source_cropped, "vertex")], text=False).write(source_ply)
    PlyData([PlyElement.describe(cropped, "vertex")], text=False).write(out_ply)

    colors = _dc_to_rgb(cropped)
    before_fig = out_dir / "object_a_gaussian_crop_source_views.jpg"
    after_fig = out_dir / "object_a_gaussian_crop_fusion_views.jpg"
    _write_contact(before_fig, cropped_xyz, colors, "Object A Gaussian crop before fusion transform")
    _write_contact(after_fig, transformed_xyz, colors, "Object A Gaussian crop after fusion transform")

    stats_path = out_dir / "gaussian_fusion_smoke_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_gaussians",
                "cropped_gaussians",
                "crop_ratio",
                "bbox_margin",
                "source_ply_mb",
                "source_crop_ply_mb",
                "output_ply_mb",
                "crop_min",
                "crop_max",
                "fusion_min",
                "fusion_max",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_gaussians": len(vertex),
                "cropped_gaussians": len(cropped),
                "crop_ratio": len(cropped) / max(len(vertex), 1),
                "bbox_margin": args.bbox_margin,
                "source_ply_mb": object_gaussian.stat().st_size / (1024 * 1024),
                "source_crop_ply_mb": source_ply.stat().st_size / (1024 * 1024),
                "output_ply_mb": out_ply.stat().st_size / (1024 * 1024),
                "crop_min": " ".join(f"{value:.6f}" for value in crop_min),
                "crop_max": " ".join(f"{value:.6f}" for value in crop_max),
                "fusion_min": " ".join(f"{value:.6f}" for value in transformed_xyz.min(axis=0)),
                "fusion_max": " ".join(f"{value:.6f}" for value in transformed_xyz.max(axis=0)),
                "notes": "Gaussian crop keeps the 2DGS color representation and bypasses TSDF mesh extraction; true garden 3D insertion still needs COLMAP-coordinate alignment.",
            }
        )

    print(source_ply)
    print(out_ply)
    print(before_fig)
    print(after_fig)
    print(stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
