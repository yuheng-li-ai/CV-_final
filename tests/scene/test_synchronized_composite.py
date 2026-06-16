from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_camera_indices_range_and_list():
    blender_scene = _load_script("scripts/blender_fusion_scene.py")

    assert blender_scene._parse_camera_indices("2:5", 3) == [2, 3, 4]
    assert blender_scene._parse_camera_indices("0,3,7", 3) == [0, 3, 7]


def test_parse_pixel_targets():
    composite = _load_script("scripts/render_synchronized_composite.py")

    assert composite._parse_pixel_targets("object_a:10,20;object_b:30,40") == {
        "object_a": (10, 20),
        "object_b": (30, 40),
    }


def test_parse_asset_float_map_defaults_missing_assets():
    composite = _load_script("scripts/render_synchronized_composite.py")

    assert composite._parse_asset_float_map("object_a:1.5;object_c:2.0") == {
        "object_a": 1.5,
        "object_c": 2.0,
    }


def test_parse_asset_rotation_delta_map():
    composite = _load_script("scripts/render_synchronized_composite.py")

    assert composite._parse_asset_rotation_delta_map("object_a:90,0,0;object_b:0,-90,15") == {
        "object_a": [90.0, 0.0, 0.0],
        "object_b": [0.0, -90.0, 15.0],
    }


def test_clear_matching_files_removes_only_matching_files(tmp_path):
    composite = _load_script("scripts/render_synchronized_composite.py")
    keep = tmp_path / "keep.txt"
    remove = tmp_path / "frame_0001.png"
    keep.write_text("keep", encoding="utf-8")
    remove.write_text("remove", encoding="utf-8")

    composite._clear_matching_files(tmp_path, "*.png")

    assert keep.exists()
    assert not remove.exists()


def test_unproject_depth_pixel_identity_camera():
    composite = _load_script("scripts/render_synchronized_composite.py")
    camera = {
        "width": 4,
        "height": 4,
        "fx": 2.0,
        "fy": 2.0,
        "position": [1.0, 2.0, 3.0],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    depth = np.full((4, 4), 2.0, dtype=np.float32)

    assert composite._unproject_depth_pixel(camera, depth, (2, 2)) == [1.0, 2.0, 5.0]


def test_apply_height_offset_changes_only_z_without_mutating_input():
    composite = _load_script("scripts/render_synchronized_composite.py")
    location = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    adjusted = composite._apply_height_offset(location, 0.12)

    assert adjusted.tolist() == [1.0, 2.0, 3.12]
    assert location.tolist() == [1.0, 2.0, 3.0]


def test_depth_occlusion_keeps_nearer_foreground_and_hides_farther_foreground():
    compositor = _load_script("scripts/composite_foreground_sequence.py")
    alpha = np.array([[0, 255, 255]], dtype=np.uint8)
    foreground_depth = np.array([[1.0, 2.0, 5.0]], dtype=np.float32)
    background_depth = np.array([[1.0, 3.0, 4.0]], dtype=np.float32)

    mask = compositor._visible_foreground_mask(alpha, foreground_depth, background_depth, depth_bias=0.05)

    assert mask.tolist() == [[False, True, False]]


def test_depth_occlusion_skips_invalid_foreground_depth():
    compositor = _load_script("scripts/composite_foreground_sequence.py")
    alpha = np.array([[0, 255, 255, 255]], dtype=np.uint8)
    invalid_depth = np.zeros((1, 4), dtype=np.float32)
    valid_depth = np.array([[0.0, 2.0, 2.1, 2.2]], dtype=np.float32)

    assert compositor._should_skip_depth_occlusion(alpha, invalid_depth)
    assert not compositor._should_skip_depth_occlusion(alpha, valid_depth)
