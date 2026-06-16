#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def _fit_background(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image_ratio = image.width / image.height
    target_ratio = size[0] / size[1]
    if image_ratio > target_ratio:
        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    elif image_ratio < target_ratio:
        new_height = int(image.width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))
    return image.resize(size, Image.Resampling.LANCZOS)


def _sample_paths(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(paths) < count:
        raise ValueError(f"Need at least {count} background frames, got {len(paths)}")
    if count == 1:
        return [paths[0]]
    indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def _visible_foreground_mask(
    alpha: np.ndarray,
    foreground_depth: np.ndarray,
    background_depth: np.ndarray,
    depth_bias: float,
) -> np.ndarray:
    foreground_valid = np.isfinite(foreground_depth) & (foreground_depth > 0)
    background_valid = np.isfinite(background_depth) & (background_depth > 0)
    foreground_visible = (~background_valid) | (foreground_depth <= background_depth + depth_bias)
    return (alpha > 0) & foreground_valid & foreground_visible


def _should_skip_depth_occlusion(
    alpha: np.ndarray,
    foreground_depth: np.ndarray,
    min_valid_ratio: float = 0.25,
) -> bool:
    alpha_mask = alpha > 0
    alpha_pixels = int(alpha_mask.sum())
    if alpha_pixels == 0:
        return False
    foreground_valid = np.isfinite(foreground_depth) & (foreground_depth > 0)
    valid_pixels = int((foreground_valid & alpha_mask).sum())
    return (valid_pixels / alpha_pixels) < min_valid_ratio


def _load_depth(path: Path, size: tuple[int, int]) -> np.ndarray:
    if path.suffix == ".npy":
        depth_array = np.load(path).astype(np.float32)
        if depth_array.shape[::-1] == size:
            return depth_array
        image = Image.fromarray(depth_array)
        return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    depth = Image.open(path)
    if depth.size != size:
        depth = depth.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(depth, dtype=np.float32)


def _parse_indices(value: str | None, count: int) -> list[int]:
    if not value:
        return list(range(count))
    value = value.strip()
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text)
        indices = list(range(start, end))
    else:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(indices) != count:
        raise ValueError(f"Index count {len(indices)} does not match frame count {count}")
    return indices


def _asset_centers(scene_path: Path) -> list[np.ndarray]:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    centers = []
    for asset in scene["assets"]:
        transform = asset["transform"]
        bounds_min = np.asarray(transform["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(transform["bounds_max"], dtype=np.float64)
        centers.append((bounds_min + bounds_max) * 0.5)
    return centers


def _project_asset_centers(
    centers: list[np.ndarray],
    camera: dict,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    position = np.asarray(camera["position"], dtype=np.float64)
    projected = []
    depths = []
    scale_x = size[0] / float(camera["width"])
    scale_y = size[1] / float(camera["height"])
    for center in centers:
        camera_point = rotation.T @ (center - position)
        z = max(float(camera_point[2]), 1e-6)
        x = (float(camera["fx"]) * float(camera_point[0]) / z + float(camera["width"]) * 0.5) * scale_x
        y = (float(camera["fy"]) * float(camera_point[1]) / z + float(camera["height"]) * 0.5) * scale_y
        projected.append([x, y])
        depths.append(z)
    return np.asarray(projected, dtype=np.float32), np.asarray(depths, dtype=np.float32)


def _approximate_foreground_depth(
    alpha: np.ndarray,
    projected_centers: np.ndarray,
    center_depths: np.ndarray,
) -> np.ndarray:
    height, width = alpha.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    if len(projected_centers) == 0:
        return depth
    distances = []
    for center in projected_centers:
        distances.append((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    nearest = np.argmin(np.stack(distances, axis=0), axis=0)
    for index, value in enumerate(center_depths):
        depth[nearest == index] = value
    depth[alpha == 0] = 0.0
    return depth


def _write_contact_sheet(frames: list[tuple[str, Image.Image]], out_path: Path) -> None:
    thumb_width = 320
    label_height = 28
    columns = 4
    thumbs = []
    for label, frame in frames:
        height = round(frame.height * thumb_width / frame.width)
        thumbs.append((label, frame.resize((thumb_width, height), Image.Resampling.LANCZOS)))
    thumb_height = max(image.height for _, image in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(thumbs):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def _sample_labeled_frames(
    frames: list[tuple[str, Image.Image]],
    max_frames: int,
) -> list[tuple[str, Image.Image]]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = [round(index * (len(frames) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [frames[index] for index in indices]


def _sample_index_set(count: int, max_frames: int) -> set[int]:
    if max_frames <= 0 or count <= max_frames:
        return set(range(count))
    return {round(index * (count - 1) / (max_frames - 1)) for index in range(max_frames)}


def _progress(iterable, total: int):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="COMPOSITE", unit="frame")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Composite transparent Blender foreground frames over moving 2DGS background frames.")
    parser.add_argument("--foreground_dir", required=True)
    parser.add_argument("--foreground_glob", default="*.png")
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--background_glob", default="*.png")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--contact_sheet")
    parser.add_argument("--contact_sheet_max_frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--background_indices", help="Comma-separated 0-based background frame indices to use.")
    parser.add_argument("--foreground_depth_dir")
    parser.add_argument("--foreground_depth_glob", default="*.tif*")
    parser.add_argument("--background_depth_dir")
    parser.add_argument("--background_depth_glob", default="depth_*.tiff")
    parser.add_argument("--approx_scene")
    parser.add_argument("--camera_json")
    parser.add_argument("--camera_indices")
    parser.add_argument("--depth_bias", type=float, default=0.08)
    return parser


def _parse_indices(value: str, count: int | None = None) -> list[int]:
    value = value.strip()
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text)
        indices = list(range(start, end))
    else:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if count is not None and len(indices) != count:
        raise ValueError(f"Index count {len(indices)} does not match frame count {count}")
    return indices


def main() -> int:
    args = build_parser().parse_args()
    foreground_paths = sorted(Path(args.foreground_dir).glob(args.foreground_glob))
    if not foreground_paths:
        raise SystemExit(f"No foreground frames found in {args.foreground_dir}")
    all_background_paths = sorted(Path(args.background_dir).glob(args.background_glob))
    if not all_background_paths:
        raise SystemExit(f"No background frames found in {args.background_dir}")
    if args.background_indices:
        indices = _parse_indices(args.background_indices)
        background_paths = [all_background_paths[index] for index in indices]
        if len(background_paths) != len(foreground_paths):
            raise SystemExit("background_indices count must match foreground frame count")
    else:
        background_paths = _sample_paths(all_background_paths, len(foreground_paths))

    foreground_depth_paths: list[Path] = []
    background_depth_paths: list[Path] = []
    use_depth_occlusion = bool(args.background_depth_dir and (args.foreground_depth_dir or args.approx_scene))
    centers: list[np.ndarray] = []
    cameras: list[dict] = []
    camera_indices: list[int] = []
    if use_depth_occlusion:
        if args.foreground_depth_dir:
            foreground_depth_paths = sorted(Path(args.foreground_depth_dir).glob(args.foreground_depth_glob))
            if len(foreground_depth_paths) != len(foreground_paths):
                raise SystemExit(
                    f"Depth foreground count {len(foreground_depth_paths)} does not match foreground count {len(foreground_paths)}"
                )
        elif args.approx_scene:
            if not args.camera_json:
                raise SystemExit("--camera_json is required with --approx_scene")
            centers = _asset_centers(Path(args.approx_scene))
            cameras = json.loads(Path(args.camera_json).read_text(encoding="utf-8"))
            camera_indices = _parse_indices(args.camera_indices, len(foreground_paths))
        all_background_depth_paths = sorted(Path(args.background_depth_dir).glob(args.background_depth_glob))
        if not all_background_depth_paths:
            raise SystemExit(f"No background depth frames found in {args.background_depth_dir}")
        if args.background_indices:
            background_depth_paths = [all_background_depth_paths[index] for index in indices]
        else:
            background_depth_paths = _sample_paths(all_background_depth_paths, len(foreground_paths))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_frames: list[tuple[str, Image.Image]] = []
    contact_indices = _sample_index_set(len(foreground_paths), args.contact_sheet_max_frames)

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    pairs = zip(foreground_paths, background_paths, strict=True)
    with imageio.get_writer(output_video, fps=args.fps, codec="libx264", macro_block_size=1) as writer:
        for index, (foreground_path, background_path) in enumerate(_progress(pairs, len(foreground_paths)), 1):
            foreground = Image.open(foreground_path).convert("RGBA")
            if use_depth_occlusion:
                foreground_array = np.asarray(foreground).copy()
                if foreground_depth_paths:
                    foreground_depth = _load_depth(foreground_depth_paths[index - 1], foreground.size)
                else:
                    projected_centers, center_depths = _project_asset_centers(
                        centers,
                        cameras[camera_indices[index - 1]],
                        foreground.size,
                    )
                    foreground_depth = _approximate_foreground_depth(
                        foreground_array[:, :, 3],
                        projected_centers,
                        center_depths,
                    )
                background_depth = _load_depth(background_depth_paths[index - 1], foreground.size)
                if _should_skip_depth_occlusion(foreground_array[:, :, 3], foreground_depth):
                    print(
                        f"WARNING: skipping depth occlusion for frame {index:04d}; foreground depth is invalid",
                        flush=True,
                    )
                else:
                    mask = _visible_foreground_mask(
                        foreground_array[:, :, 3],
                        foreground_depth,
                        background_depth,
                        args.depth_bias,
                    )
                    foreground_array[:, :, 3] = np.where(mask, foreground_array[:, :, 3], 0).astype(np.uint8)
                    foreground = Image.fromarray(foreground_array, mode="RGBA")
            background = _fit_background(background_path, foreground.size).convert("RGBA")
            background.alpha_composite(foreground)
            composite = background.convert("RGB")
            composite.save(output_dir / f"frame_{index:04d}.jpg", quality=95)
            writer.append_data(np.asarray(composite))
            if (index - 1) in contact_indices:
                labeled_frames.append((f"frame {index} bg {background_path.stem}", composite.copy()))
    if args.contact_sheet:
        _write_contact_sheet(_sample_labeled_frames(labeled_frames, args.contact_sheet_max_frames), Path(args.contact_sheet))
    print(output_video)
    if args.contact_sheet:
        print(args.contact_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
