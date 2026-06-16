from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image


@dataclass(frozen=True)
class ObjectCPreprocessConfig:
    input_image: Path
    output_image: Path
    output_mask: Path
    output_metadata: Path
    threshold: int = 245
    padding: int = 32
    centered_crop: bool = True
    keep_largest_component: bool = False
    close_kernel: int = 0
    open_kernel: int = 0
    feather_radius: int = 0
    dry_run: bool = False


@dataclass(frozen=True)
class ObjectCPreprocessResult:
    input_image: Path
    output_image: Path
    output_mask: Path
    output_metadata: Path
    bbox: list[int]
    foreground_pixels: int
    image_size: list[int]
    dry_run: bool

    def to_json(self) -> str:
        payload = asdict(self)
        for key in ["input_image", "output_image", "output_mask", "output_metadata"]:
            payload[key] = str(payload[key])
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _border_background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    border: list[tuple[int, int, int]] = []
    for x in range(width):
        border.append(pixels[x, 0])
        border.append(pixels[x, height - 1])
    for y in range(height):
        border.append(pixels[0, y])
        border.append(pixels[width - 1, y])
    return (
        int(median(pixel[0] for pixel in border)),
        int(median(pixel[1] for pixel in border)),
        int(median(pixel[2] for pixel in border)),
    )


def _foreground_mask(image: Image.Image, threshold: int) -> Image.Image:
    if "A" in image.getbands():
        return image.getchannel("A")
    rgb = image.convert("RGB")
    background = _border_background_color(rgb)
    mask = Image.new("L", rgb.size, 0)
    pixels = rgb.load()
    mask_pixels = mask.load()
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            distance = math.sqrt(
                (r - background[0]) ** 2 + (g - background[1]) ** 2 + (b - background[2]) ** 2
            )
            mask_pixels[x, y] = 255 if distance > threshold else 0
    return mask


def _padded_bbox(mask: Image.Image, padding: int) -> tuple[list[int], int]:
    bbox = mask.getbbox()
    if bbox is None:
        width, height = mask.size
        return [0, 0, width, height], 0
    left, top, right, bottom = bbox
    width, height = mask.size
    padded = [
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    ]
    histogram = mask.histogram()
    foreground = sum(count for value, count in enumerate(histogram) if value > 0)
    return padded, foreground


def _apply_alpha(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def _refine_mask(
    mask: Image.Image,
    keep_largest_component: bool,
    close_kernel: int,
    open_kernel: int,
    feather_radius: int,
) -> Image.Image:
    if not any([keep_largest_component, close_kernel > 0, open_kernel > 0, feather_radius > 0]):
        return mask

    import cv2
    import numpy as np

    array = np.array(mask)
    binary = np.where(array > 0, 255, 0).astype("uint8")

    if keep_largest_component:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = np.where(labels == largest, 255, 0).astype("uint8")

    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), dtype="uint8")
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if open_kernel > 0:
        kernel = np.ones((open_kernel, open_kernel), dtype="uint8")
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if feather_radius > 0:
        radius = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        binary = cv2.GaussianBlur(binary, (radius, radius), 0)

    return Image.fromarray(binary, mode="L")


def preprocess_object_c(config: ObjectCPreprocessConfig) -> ObjectCPreprocessResult:
    image = Image.open(config.input_image)
    mask = _foreground_mask(image, config.threshold)
    mask = _refine_mask(
        mask,
        keep_largest_component=config.keep_largest_component,
        close_kernel=config.close_kernel,
        open_kernel=config.open_kernel,
        feather_radius=config.feather_radius,
    )
    bbox, foreground_pixels = _padded_bbox(mask, config.padding)

    output_image = _apply_alpha(image, mask)
    output_mask = mask
    if config.centered_crop:
        crop_box = tuple(bbox)
        output_image = output_image.crop(crop_box)
        output_mask = output_mask.crop(crop_box)

    result = ObjectCPreprocessResult(
        input_image=config.input_image,
        output_image=config.output_image,
        output_mask=config.output_mask,
        output_metadata=config.output_metadata,
        bbox=bbox,
        foreground_pixels=foreground_pixels,
        image_size=[image.size[0], image.size[1]],
        dry_run=config.dry_run,
    )

    if config.dry_run:
        return result

    config.output_image.parent.mkdir(parents=True, exist_ok=True)
    config.output_mask.parent.mkdir(parents=True, exist_ok=True)
    config.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(config.output_image)
    output_mask.save(config.output_mask)
    config.output_metadata.write_text(result.to_json() + "\n", encoding="utf-8")
    return result
