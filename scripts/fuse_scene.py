#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import trimesh


def _nested(data: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _as_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _load_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False, force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"Asset has no vertices: {path}")
    return np.asarray(mesh.vertices, dtype=np.float64)


def _rotation_matrix_xyz(degrees: list[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(value)) for value in degrees]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mz @ my @ mx


def _source_to_fusion(vertices: np.ndarray, source_up: str) -> np.ndarray:
    if source_up == "z":
        return vertices.copy()
    if source_up == "y":
        return np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    raise ValueError(f"Unsupported source_up: {source_up}")


def _source_to_fusion_matrix(source_up: str) -> np.ndarray:
    if source_up == "z":
        return np.eye(3, dtype=np.float64)
    if source_up == "y":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    raise ValueError(f"Unsupported source_up: {source_up}")


def _robust_bounds(vertices: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(vertices, low, axis=0), np.percentile(vertices, high, axis=0)


def _transform_vertices(vertices: np.ndarray, asset: dict[str, Any], variant: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    source_up = str(asset.get("source_up", "y"))
    fusion_vertices = _source_to_fusion(vertices, source_up)
    low = float(asset.get("robust_percentile_low", 5))
    high = float(asset.get("robust_percentile_high", 95))
    robust_min, robust_max = _robust_bounds(fusion_vertices, low, high)
    robust_extent = robust_max - robust_min
    target_height = float(asset["target_height"])
    base_scale = target_height / max(float(robust_extent[2]), 1e-8)
    scale = base_scale * float(asset.get("scale", 1.0)) * float(variant.get("scale_multiplier", 1.0))

    center = (robust_min + robust_max) / 2.0
    aligned = (fusion_vertices - center) * scale
    rot = _rotation_matrix_xyz(list(asset.get("rotation_deg", [0, 0, 0])))
    aligned = aligned @ rot.T
    min_after = aligned.min(axis=0)
    contact_shift = np.array([0.0, 0.0, -float(min_after[2])], dtype=np.float64)
    aligned += contact_shift
    location = np.asarray(asset.get("location", [0, 0, 0]), dtype=np.float64)
    offset = np.asarray(variant.get("position_offset", [0, 0, 0]), dtype=np.float64)
    transformed = aligned + location + offset
    bounds_min, bounds_max = transformed.min(axis=0), transformed.max(axis=0)
    metadata = {
        "source_up": source_up,
        "robust_percentile": [low, high],
        "base_scale": base_scale,
        "final_scale": scale,
        "target_height": target_height,
        "center": center.tolist(),
        "source_to_fusion_matrix": _source_to_fusion_matrix(source_up).tolist(),
        "rotation_matrix": rot.tolist(),
        "contact_shift": contact_shift.tolist(),
        "location": (location + offset).tolist(),
        "rotation_deg": list(asset.get("rotation_deg", [0, 0, 0])),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "extent": (bounds_max - bounds_min).tolist(),
    }
    return transformed, metadata


def _write_transform_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "asset",
        "kind",
        "path",
        "target_height",
        "base_scale",
        "final_scale",
        "location_x",
        "location_y",
        "location_z",
        "rotation_deg",
        "extent_x",
        "extent_y",
        "extent_z",
        "contact_z",
        "shadow_enabled",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _draw_preview(path: Path, transformed: list[tuple[str, np.ndarray]], shadow_enabled: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin = 70
    colors = {
        "object_a": (44, 113, 181),
        "object_b": (215, 95, 2),
        "object_c": (35, 139, 69),
    }

    all_points = np.concatenate([points for _, points in transformed], axis=0)
    xy_min = all_points[:, :2].min(axis=0)
    xy_max = all_points[:, :2].max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    draw.rectangle((margin, margin, width - margin, height - margin), outline=(210, 210, 210), width=2)
    draw.text((margin, 20), f"Phase6 fusion top-down layout (shadow={'on' if shadow_enabled else 'off'})", fill=(20, 20, 20), font=font)
    for index in range(7):
        x = margin + index * (width - 2 * margin) / 6
        y = margin + index * (height - 2 * margin) / 6
        draw.line((x, margin, x, height - margin), fill=(230, 230, 230))
        draw.line((margin, y, width - margin, y), fill=(230, 230, 230))

    for name, points in transformed:
        if len(points) > 5000:
            rng = np.random.default_rng(0)
            points = points[rng.choice(len(points), size=5000, replace=False)]
        xy = points[:, :2]
        px = margin + (xy[:, 0] - xy_min[0]) * scale
        py = height - margin - (xy[:, 1] - xy_min[1]) * scale
        color = colors.get(name, (80, 80, 80))
        for x, y in zip(px.astype(int), py.astype(int), strict=False):
            image.putpixel((max(0, min(width - 1, x)), max(0, min(height - 1, y))), color)
        bmin = xy.min(axis=0)
        bmax = xy.max(axis=0)
        x0 = margin + (bmin[0] - xy_min[0]) * scale
        x1 = margin + (bmax[0] - xy_min[0]) * scale
        y0 = height - margin - (bmax[1] - xy_min[1]) * scale
        y1 = height - margin - (bmin[1] - xy_min[1]) * scale
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), name, fill=color, font=font)

    image.save(path, quality=95)


def _write_scene_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Phase6 fusion scene manifest and transform preview.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    config_path = _as_path(project_root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = _as_path(project_root, _nested(config, "paths.output_dir"))
    fused_scene = _as_path(project_root, _nested(config, "inputs.fused_scene"))
    variant = _nested(config, "fusion", {}) or {}
    shadow_enabled = bool(variant.get("enable_shadow", True))
    assets = _nested(config, "inputs.assets", [])
    if not assets:
        raise ValueError("inputs.assets must contain named asset definitions")

    transformed_for_preview: list[tuple[str, np.ndarray]] = []
    scene_assets: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for asset in assets:
        path = _as_path(project_root, asset["path"])
        vertices = _load_vertices(path)
        transformed, meta = _transform_vertices(vertices, asset, variant)
        name = str(asset["name"])
        transformed_for_preview.append((name, transformed))
        scene_asset = {
            "name": name,
            "kind": asset.get("kind", "mesh"),
            "path": str(path.relative_to(project_root)),
            "texture": asset.get("texture"),
            "material_color": asset.get("material_color"),
            "use_vertex_color": asset.get("use_vertex_color"),
            "emission_strength": asset.get("emission_strength"),
            "source_up": asset.get("source_up", "y"),
            "transform": meta,
        }
        scene_assets.append(scene_asset)
        loc = meta["location"]
        extent = meta["extent"]
        table_rows.append(
            {
                "asset": name,
                "kind": asset.get("kind", "mesh"),
                "path": str(path.relative_to(project_root)),
                "target_height": meta["target_height"],
                "base_scale": meta["base_scale"],
                "final_scale": meta["final_scale"],
                "location_x": loc[0],
                "location_y": loc[1],
                "location_z": loc[2],
                "rotation_deg": " ".join(str(value) for value in meta["rotation_deg"]),
                "extent_x": extent[0],
                "extent_y": extent[1],
                "extent_z": extent[2],
                "contact_z": meta["bounds_min"][2],
                "shadow_enabled": str(shadow_enabled).lower(),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_data = {
        "run_id": args.run_id,
        "config": str(config_path.relative_to(project_root)),
        "variant": variant.get("variant", "main"),
        "device": args.device,
        "background": {
            "point_cloud": _nested(config, "inputs.background_scene"),
            "backplate": _nested(config, "inputs.background_backplate"),
        },
        "shadow_enabled": shadow_enabled,
        "assets": scene_assets,
        "render": _nested(config, "render", {}),
        "outputs": {
            "output_dir": str(output_dir.relative_to(project_root)),
            "fused_scene": str(fused_scene.relative_to(project_root)),
            "transform_table": str((output_dir / "transform_table.csv").relative_to(project_root)),
            "layout_preview": str((output_dir / "layout_topdown.jpg").relative_to(project_root)),
        },
    }
    if not args.dry_run:
        _write_scene_json(fused_scene, scene_data)
        _write_transform_table(output_dir / "transform_table.csv", table_rows)
        _draw_preview(output_dir / "layout_topdown.jpg", transformed_for_preview, shadow_enabled)

    print(f"FUSED_SCENE: {fused_scene}")
    print(f"TRANSFORM_TABLE: {output_dir / 'transform_table.csv'}")
    print(f"LAYOUT_PREVIEW: {output_dir / 'layout_topdown.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
