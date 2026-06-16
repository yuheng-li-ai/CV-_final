#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def _first_image_size(directory: Path, pattern: str) -> tuple[int, int]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No background frames found in {directory} matching {pattern}")
    with Image.open(paths[0]) as image:
        return image.size


def _clear_matching_files(directory: Path, pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def _default_indices(frames: int) -> str:
    return f"0:{frames}"


def _parse_pixel_targets(value: str) -> dict[str, tuple[int, int]]:
    targets: dict[str, tuple[int, int]] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        name, coords = item.split(":", 1)
        x_text, y_text = coords.split(",", 1)
        targets[name.strip()] = (int(x_text), int(y_text))
    return targets


def _parse_asset_float_map(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    parsed: dict[str, float] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        name, raw_value = item.split(":", 1)
        parsed[name.strip()] = float(raw_value)
    return parsed


def _parse_asset_rotation_delta_map(value: str | None) -> dict[str, list[float]]:
    if not value:
        return {}
    parsed: dict[str, list[float]] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        name, raw_value = item.split(":", 1)
        values = [float(part.strip()) for part in raw_value.split(",")]
        if len(values) != 3:
            raise ValueError(f"Expected three rotation values for {name}, got {raw_value}")
        parsed[name.strip()] = values
    return parsed


def _unproject_depth_pixel(camera: dict, depth: np.ndarray, pixel: tuple[int, int]) -> list[float]:
    x, y = pixel
    if y < 0 or y >= depth.shape[0] or x < 0 or x >= depth.shape[1]:
        raise ValueError(f"Placement pixel {pixel} is outside depth image shape {depth.shape}")
    z = float(depth[y, x])
    if z <= 0.0:
        raise ValueError(f"Placement pixel {pixel} has invalid depth {z}")
    scale_x = float(camera["width"]) / float(depth.shape[1])
    scale_y = float(camera["height"]) / float(depth.shape[0])
    u = x * scale_x
    v = y * scale_y
    camera_point = np.array(
        [
            (u - float(camera["width"]) * 0.5) / float(camera["fx"]) * z,
            (v - float(camera["height"]) * 0.5) / float(camera["fy"]) * z,
            z,
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    position = np.asarray(camera["position"], dtype=np.float64)
    return (rotation @ camera_point + position).tolist()


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


def _load_asset_vertices(project_root: Path, asset: dict) -> np.ndarray:
    import trimesh

    path = Path(asset["path"])
    if not path.is_absolute():
        path = project_root / path
    mesh = trimesh.load(path, process=False, force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"Asset has no vertices: {path}")
    return np.asarray(mesh.vertices, dtype=np.float64)


def _update_transform_for_location_scale_rotation(
    project_root: Path,
    asset: dict,
    location: np.ndarray,
    scale_multiplier: float,
    rotation_delta_deg: list[float],
) -> None:
    transform = asset["transform"]
    vertices = _load_asset_vertices(project_root, asset)
    source_up = str(transform.get("source_up", asset.get("source_up", "z")))
    fusion_vertices = _source_to_fusion(vertices, source_up)
    center = np.asarray(transform["center"], dtype=np.float64)
    rotation_deg = [
        float(base) + float(delta)
        for base, delta in zip(transform["rotation_deg"], rotation_delta_deg, strict=True)
    ]
    final_scale = float(transform["final_scale"]) * scale_multiplier

    aligned = (fusion_vertices - center) * final_scale
    rotation = _rotation_matrix_xyz(rotation_deg)
    aligned = aligned @ rotation.T
    min_after = aligned.min(axis=0)
    contact_shift = np.array([0.0, 0.0, -float(min_after[2])], dtype=np.float64)
    transformed = aligned + contact_shift + location
    bounds_min = transformed.min(axis=0)
    bounds_max = transformed.max(axis=0)

    transform["final_scale"] = final_scale
    transform["rotation_deg"] = rotation_deg
    transform["rotation_matrix"] = rotation.tolist()
    transform["contact_shift"] = contact_shift.tolist()
    transform["location"] = location.tolist()
    transform["bounds_min"] = bounds_min.tolist()
    transform["bounds_max"] = bounds_max.tolist()
    transform["extent"] = (bounds_max - bounds_min).tolist()


def _apply_height_offset(location: np.ndarray, height_offset: float) -> np.ndarray:
    adjusted = location.copy()
    adjusted[2] += float(height_offset)
    return adjusted


def _write_image_table_scene(
    scene_path: Path,
    camera_json: Path,
    depth_path: Path,
    output_scene: Path,
    targets: dict[str, tuple[int, int]],
    placement_camera_index: int,
    scale_multipliers: dict[str, float],
    rotation_deltas: dict[str, list[float]],
    height_offsets: dict[str, float],
) -> Path:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    cameras = json.loads(camera_json.read_text(encoding="utf-8"))
    camera = cameras[placement_camera_index]
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd()

    for asset in scene["assets"]:
        name = asset["name"]
        if name not in targets:
            continue
        new_location = np.asarray(_unproject_depth_pixel(camera, depth, targets[name]), dtype=np.float64)
        new_location = _apply_height_offset(new_location, height_offsets.get(name, 0.0))
        _update_transform_for_location_scale_rotation(
            project_root,
            asset,
            new_location,
            scale_multipliers.get(name, 1.0),
            rotation_deltas.get(name, [0.0, 0.0, 0.0]),
        )

    scene["synchronized_placement"] = {
        "method": "2dgs_depth_unprojected_image_table",
        "camera_json": str(camera_json),
        "depth_path": str(depth_path),
        "placement_camera_index": placement_camera_index,
        "pixel_targets": {name: list(pixel) for name, pixel in targets.items()},
        "scale_multipliers": scale_multipliers,
        "rotation_deltas": rotation_deltas,
        "height_offsets": height_offsets,
    }
    output_scene.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return output_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render mesh foreground with 2DGS cameras and composite over 2DGS background frames."
    )
    parser.add_argument("--scene", default="runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json")
    parser.add_argument("--camera_json", default="runs/scene/background_2dgs_high/cameras.json")
    parser.add_argument(
        "--background_dir",
        default="runs/scene/background_2dgs_high/test/ours_30000/renders",
    )
    parser.add_argument("--background_glob", default="*.png")
    parser.add_argument(
        "--background_depth_dir",
        default="runs/scene/background_2dgs_high/test/ours_30000/vis",
    )
    parser.add_argument("--background_depth_glob", default="depth_*.tiff")
    parser.add_argument("--output_dir", default="runs/scene/final_video_sync")
    parser.add_argument("--output_video", default="runs/scene/final_video_sync/synchronized_composite.mp4")
    parser.add_argument("--contact_sheet", default="reports/figures/final_video_sync_8f_contact.jpg")
    parser.add_argument("--blender", default="external/blender/blender-4.5.4-linux-x64/blender")
    parser.add_argument("--blender_script", default="scripts/blender_fusion_scene.py")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--camera_indices")
    parser.add_argument("--background_indices")
    parser.add_argument("--resolution")
    parser.add_argument(
        "--placement",
        default="image_table",
        choices=["image_table", "scene"],
        help="image_table reprojects selected table pixels from 2DGS depth into world space.",
    )
    parser.add_argument(
        "--placement_pixels",
        default="object_a:610,480;object_b:815,500;object_c:990,480",
        help="Semicolon-separated image-space table points in the 2DGS background resolution.",
    )
    parser.add_argument(
        "--placement_depth",
        default="runs/scene/background_2dgs_high/test/ours_30000/vis/depth_00000.tiff",
    )
    parser.add_argument("--placement_camera_index", type=int, default=0)
    parser.add_argument(
        "--asset_scale_multipliers",
        default="object_a:1.55;object_b:1.20;object_c:1.7",
        help="Semicolon-separated per-asset scale multipliers applied after the selected fusion transform.",
    )
    parser.add_argument(
        "--asset_rotation_delta_deg",
        default="object_a:-30,90,0;object_b:130,10,0;object_c:90,-20,20",
        help="Semicolon-separated per-asset XYZ rotation deltas in degrees.",
    )
    parser.add_argument(
        "--asset_height_offsets",
        default="object_a:-0.65;object_b:-0.6;object_c:-0.6",
        help="Semicolon-separated per-asset world-space Z offsets applied after 2DGS depth placement.",
    )
    parser.add_argument("--disable_depth_occlusion", action="store_true")
    parser.add_argument(
        "--use_foreground_depth",
        action="store_true",
        help="Render Blender foreground Z-depth and use it for depth-aware compositing.",
    )
    parser.add_argument(
        "--approx_depth_occlusion",
        action="store_true",
        help="Use the legacy asset-center depth approximation for depth-aware compositing.",
    )
    parser.add_argument("--depth_bias", type=float, default=0.08)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    foreground_dir = output_dir / "foreground"
    foreground_depth_dir = output_dir / "foreground_depth"
    composite_dir = output_dir / "composite_frames"
    use_depth_occlusion = not args.disable_depth_occlusion and (
        args.use_foreground_depth or args.approx_depth_occlusion
    )
    foreground_dir.mkdir(parents=True, exist_ok=True)
    if args.use_foreground_depth:
        foreground_depth_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)
    _clear_matching_files(foreground_dir, "*.png")
    if args.use_foreground_depth:
        _clear_matching_files(foreground_depth_dir, "*.npy")
    _clear_matching_files(composite_dir, "*.jpg")

    resolution = args.resolution
    if not resolution:
        width, height = _first_image_size(Path(args.background_dir), args.background_glob)
        resolution = f"{width}x{height}"

    scene_path = Path(args.scene)
    if args.placement == "image_table":
        scene_path = _write_image_table_scene(
            scene_path,
            Path(args.camera_json),
            Path(args.placement_depth),
            output_dir / "scene_2dgs_table_aligned.json",
            _parse_pixel_targets(args.placement_pixels),
            args.placement_camera_index,
            _parse_asset_float_map(args.asset_scale_multipliers),
            _parse_asset_rotation_delta_map(args.asset_rotation_delta_deg),
            _parse_asset_float_map(args.asset_height_offsets),
        )

    camera_indices = args.camera_indices or _default_indices(args.frames)
    foreground_prefix = foreground_dir / "frame_"
    blender_command = [
        args.blender,
        "-b",
        "--python",
        args.blender_script,
        "--",
        "--scene",
        str(scene_path),
        "--output",
        str(foreground_prefix),
        "--frames",
        str(args.frames),
        "--resolution",
        resolution,
        "--samples",
        str(args.samples),
        "--transparent",
        "--environment",
        "backplate_only",
        "--camera-path",
        "2dgs_json",
        "--camera-json",
        args.camera_json,
        "--camera-indices",
        camera_indices,
    ]
    if args.use_foreground_depth and not args.disable_depth_occlusion:
        blender_command.extend(["--depth-output-dir", str(foreground_depth_dir)])
    print("BLENDER_COMMAND:", " ".join(blender_command), flush=True)
    subprocess.run(blender_command, check=True)

    composite_command = [
        sys.executable,
        "scripts/composite_foreground_sequence.py",
        "--foreground_dir",
        str(foreground_dir),
        "--foreground_glob",
        "*.png",
        "--background_dir",
        args.background_dir,
        "--background_glob",
        args.background_glob,
        "--output_dir",
        str(composite_dir),
        "--output_video",
        args.output_video,
        "--contact_sheet",
        args.contact_sheet,
        "--fps",
        str(args.fps),
    ]
    if args.background_indices:
        composite_command.extend(["--background_indices", args.background_indices])
    if use_depth_occlusion:
        composite_command.extend([
            "--background_depth_dir",
            args.background_depth_dir,
            "--background_depth_glob",
            args.background_depth_glob,
        ])
        if args.use_foreground_depth:
            composite_command.extend([
                "--foreground_depth_dir",
                str(foreground_depth_dir),
                "--foreground_depth_glob",
                "depth_*.npy",
            ])
        elif args.approx_depth_occlusion:
            composite_command.extend([
                "--approx_scene",
                str(scene_path),
                "--camera_json",
                args.camera_json,
                "--camera_indices",
                camera_indices,
            ])
        composite_command.extend(["--depth_bias", str(args.depth_bias)])
    print("COMPOSITE_COMMAND:", " ".join(composite_command), flush=True)
    subprocess.run(composite_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
