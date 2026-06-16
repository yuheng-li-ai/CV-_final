#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from sklearn.cluster import DBSCAN
import trimesh


SH_C0 = 0.28209479177387814


def _parse_bounds(value: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 6:
        raise ValueError("--crop_bounds must be xmin,xmax,ymin,ymax,zmin,zmax")
    return np.array([parts[0], parts[2], parts[4]], dtype=np.float64), np.array(
        [parts[1], parts[3], parts[5]], dtype=np.float64
    )


def _load_gaussian(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(path)["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float64)))
    rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float64)
    colors = np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)
    return xyz, opacity, colors


def _select_cluster(
    xyz: np.ndarray,
    opacity: np.ndarray,
    opacity_threshold: float,
    eps: float,
    min_samples: int,
    cluster_rank: int,
) -> np.ndarray:
    mask = opacity >= opacity_threshold
    points = xyz[mask]
    if len(points) == 0:
        raise ValueError("No Gaussian points survived the opacity threshold")
    q = np.quantile(points, [0.05, 0.95], axis=0)
    scale = np.maximum(q[1] - q[0], 1e-8)
    normalized = (points - np.median(points, axis=0)) / scale
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(normalized)
    unique, counts = np.unique(labels, return_counts=True)
    clusters = sorted(
        [(int(label), int(count)) for label, count in zip(unique, counts) if label != -1],
        key=lambda item: item[1],
        reverse=True,
    )
    if cluster_rank >= len(clusters):
        raise ValueError(f"cluster_rank={cluster_rank} but only {len(clusters)} clusters were found")
    label = clusters[cluster_rank][0]
    selected = np.zeros(len(xyz), dtype=bool)
    selected[np.flatnonzero(mask)[labels == label]] = True
    return selected


def _apply_crop(xyz: np.ndarray, mask: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if bounds is None:
        return mask
    lower, upper = bounds
    crop_mask = ((xyz >= lower) & (xyz <= upper)).all(axis=1)
    return mask & crop_mask


def _write_contact(path: Path, xyz: np.ndarray, colors: np.ndarray, title: str) -> None:
    if len(xyz) > 16000:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(xyz), 16000, replace=False)
        xyz = xyz[keep]
        colors = colors[keep]
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), title, fill=(0, 0, 0))
    _draw_projection(draw, xyz, colors, (25, 50, 385, 700), "x/z front", (0, 2))
    _draw_projection(draw, xyz, colors, (420, 50, 780, 700), "y/z side", (1, 2))
    _draw_projection(draw, xyz, colors, (815, 50, 1175, 700), "x/y top", (0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def _draw_projection(
    draw: ImageDraw.ImageDraw,
    xyz: np.ndarray,
    colors: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    axes: tuple[int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(35, 35, 35), width=2)
    draw.text((x0 + 6, y0 + 6), label, fill=(0, 0, 0))
    if len(xyz) == 0:
        return
    values = xyz[:, axes]
    q = np.quantile(values, [0.02, 0.98], axis=0)
    clipped = np.clip(values, q[0], q[1])
    span = np.maximum(q[1] - q[0], 1e-8)
    scale = min((x1 - x0 - 20) / span[0], (y1 - y0 - 28) / span[1])
    px = x0 + 10 + (clipped[:, 0] - q[0, 0]) * scale
    py = y1 - 10 - (clipped[:, 1] - q[0, 1]) * scale
    order = np.argsort(py)
    for index in order:
        color = tuple(int(value * 255) for value in colors[index])
        x = int(np.clip(px[index], x0 + 2, x1 - 3))
        y = int(np.clip(py[index], y0 + 22, y1 - 3))
        draw.rectangle((x, y, x + 1, y + 1), fill=color)


def _parse_color(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--override_color must be r,g,b with values in [0, 1]")
    return np.clip(np.asarray(parts, dtype=np.float64), 0.0, 1.0)


def _build_proxy_mesh(
    xyz: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    radius: float,
    override_color: np.ndarray | None,
) -> trimesh.Trimesh:
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(xyz), max_points, replace=False)
        xyz = xyz[keep]
        colors = colors[keep]
    base = trimesh.creation.icosphere(subdivisions=0, radius=radius)
    vertices = []
    faces = []
    vertex_colors = []
    base_vertices = np.asarray(base.vertices, dtype=np.float64)
    base_faces = np.asarray(base.faces, dtype=np.int64)
    if override_color is not None:
        colors = np.repeat(override_color[None, :], len(xyz), axis=0)
    for point, color in zip(xyz, colors):
        offset = len(vertices)
        vertices.extend(base_vertices + point)
        faces.extend(base_faces + offset)
        rgba = np.array([color[0], color[1], color[2], 1.0]) * 255.0
        vertex_colors.extend([rgba.astype(np.uint8)] * len(base_vertices))
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=np.asarray(vertex_colors, dtype=np.uint8),
        process=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a colored proxy mesh from a 2DGS Gaussian PLY.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--opacity_threshold", type=float, default=0.95)
    parser.add_argument("--dbscan_eps", type=float, default=0.055)
    parser.add_argument("--dbscan_min_samples", type=int, default=20)
    parser.add_argument("--cluster_rank", type=int, default=0)
    parser.add_argument("--crop_bounds", help="Optional xmin,xmax,ymin,ymax,zmin,zmax in source coordinates.")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--sphere_radius", type=float, default=0.035)
    parser.add_argument("--override_color", help="Optional fixed proxy color as r,g,b in [0, 1].")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz, opacity, colors = _load_gaussian(input_path)
    selected = _select_cluster(
        xyz,
        opacity,
        args.opacity_threshold,
        args.dbscan_eps,
        args.dbscan_min_samples,
        args.cluster_rank,
    )
    selected = _apply_crop(xyz, selected, _parse_bounds(args.crop_bounds))
    selected_xyz = xyz[selected]
    selected_colors = colors[selected]
    if len(selected_xyz) == 0:
        raise ValueError("No Gaussian points survived cluster selection and crop")

    contact = out_dir / "proxy_contact.jpg"
    _write_contact(
        contact,
        selected_xyz,
        selected_colors,
        f"Gaussian proxy: opacity>={args.opacity_threshold}, cluster_rank={args.cluster_rank}, n={len(selected_xyz)}",
    )
    override_color = _parse_color(args.override_color)
    mesh = _build_proxy_mesh(selected_xyz, selected_colors, args.max_points, args.sphere_radius, override_color)
    mesh_path = out_dir / "proxy_mesh.ply"
    mesh.export(mesh_path)

    stats_path = out_dir / "proxy_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_points",
                "selected_points",
                "mesh_vertices",
                "mesh_faces",
                "opacity_threshold",
                "dbscan_eps",
                "dbscan_min_samples",
                "cluster_rank",
                "crop_bounds",
                "source_bounds_min",
                "source_bounds_max",
                "source_extent",
                "proxy_mesh",
                "contact",
                "override_color",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_points": len(xyz),
                "selected_points": len(selected_xyz),
                "mesh_vertices": len(mesh.vertices),
                "mesh_faces": len(mesh.faces),
                "opacity_threshold": args.opacity_threshold,
                "dbscan_eps": args.dbscan_eps,
                "dbscan_min_samples": args.dbscan_min_samples,
                "cluster_rank": args.cluster_rank,
                "crop_bounds": args.crop_bounds or "",
                "source_bounds_min": " ".join(f"{value:.6f}" for value in selected_xyz.min(axis=0)),
                "source_bounds_max": " ".join(f"{value:.6f}" for value in selected_xyz.max(axis=0)),
                "source_extent": " ".join(f"{value:.6f}" for value in selected_xyz.ptp(axis=0)),
                "proxy_mesh": str(mesh_path),
                "contact": str(contact),
                "override_color": args.override_color or "",
            }
        )

    print(mesh_path)
    print(contact)
    print(stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
