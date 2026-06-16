#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render a Phase6 fusion scene in Blender.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument(
        "--camera-path",
        default="static",
        choices=["static", "phase7_walkthrough", "phase7_table_orbit", "2dgs_json"],
    )
    parser.add_argument("--camera-json", help="2DGS cameras.json used to synchronize foreground rendering.")
    parser.add_argument(
        "--camera-indices",
        help="Comma-separated camera indices, or start:end range, matching the 2DGS background frames.",
    )
    parser.add_argument("--camera-location", default="0,-4,1.8")
    parser.add_argument("--camera-rotation", default="1.18,0,0")
    parser.add_argument("--light-location", default="0,-3,4")
    parser.add_argument("--light-energy", type=float, default=450.0)
    parser.add_argument("--background-image")
    parser.add_argument("--background-sequence-dir")
    parser.add_argument("--background-sequence-glob", default="*.png")
    parser.add_argument("--depth-output-dir", help="Optional directory for camera-space Z-depth frames.")
    parser.add_argument(
        "--environment",
        default="shadow_floor",
        choices=["shadow_floor", "proxy_garden_table", "backplate_only", "proxy_garden_ring"],
    )
    return parser.parse_args(argv)


def _import_bpy():
    import bpy  # type: ignore
    return bpy


def _import_mathutils():
    import mathutils  # type: ignore
    return mathutils


def _find_project_root(scene_path: Path) -> Path:
    for parent in [scene_path.parent, *scene_path.parents]:
        if (parent / "scripts" / "blender_fusion_scene.py").exists():
            return parent
    return Path.cwd()


def _clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _matrix_from_rows(mathutils, rows: list[list[float]]):
    return mathutils.Matrix([[float(value) for value in row] for row in rows])


def _asset_matrix(mathutils, asset: dict):
    transform = asset["transform"]
    source_to_fusion = _matrix_from_rows(mathutils, transform["source_to_fusion_matrix"]).to_4x4()
    rotation = _matrix_from_rows(mathutils, transform["rotation_matrix"]).to_4x4()
    scale = mathutils.Matrix.Diagonal((float(transform["final_scale"]),) * 3 + (1.0,))
    center = mathutils.Matrix.Translation(tuple(-float(value) for value in transform["center"]))
    contact_shift = mathutils.Matrix.Translation(tuple(float(value) for value in transform["contact_shift"]))
    location = mathutils.Matrix.Translation(tuple(float(value) for value in transform["location"]))
    return location @ contact_shift @ rotation @ scale @ center @ source_to_fusion


def _import_asset(bpy, mathutils, project_root: Path, asset: dict):
    path = project_root / asset["path"]
    if path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix.lower() == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported asset type: {path}")
    imported = list(bpy.context.selected_objects)
    if not imported:
        raise RuntimeError(f"Blender did not import any object from {path}")
    matrix = _asset_matrix(mathutils, asset)
    for index, obj in enumerate(imported):
        obj.name = asset["name"] if index == 0 else f"{asset['name']}_{index}"
        obj.matrix_world = matrix @ obj.matrix_world
    return imported


def _setup_materials(bpy, project_root: Path, objects, asset: dict) -> None:
    texture = asset.get("texture")
    color = asset.get("material_color") or [0.8, 0.8, 0.8, 1.0]
    material = bpy.data.materials.new(f"{asset['name']}_material")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = tuple(color)
        bsdf.inputs["Roughness"].default_value = 0.55
        emission_strength = float(asset.get("emission_strength", 0.0) or 0.0)
        if emission_strength > 0:
            bsdf.inputs["Emission Color"].default_value = tuple(color)
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    if texture:
        texture_path = project_root / texture
        if texture_path.exists() and bsdf is not None:
            tex_node = material.node_tree.nodes.new("ShaderNodeTexImage")
            tex_node.image = bpy.data.images.load(str(texture_path))
            material.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    elif asset.get("kind") == "ply_mesh" and bool(asset.get("use_vertex_color", True)) and bsdf is not None:
        color_layer = _ensure_corner_vertex_color(objects, "Col")
        try:
            color_node = material.node_tree.nodes.new("ShaderNodeVertexColor")
            color_node.layer_name = color_layer
        except RuntimeError:
            color_node = material.node_tree.nodes.new("ShaderNodeAttribute")
            color_node.attribute_name = color_layer
        material.node_tree.links.new(color_node.outputs["Color"], bsdf.inputs["Base Color"])
    for obj in objects:
        if hasattr(obj, "data") and hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(material)


def _ensure_corner_vertex_color(objects, source_name: str) -> str:
    target_name = f"{source_name}_corner"
    for obj in objects:
        mesh = getattr(obj, "data", None)
        color_attributes = getattr(mesh, "color_attributes", None)
        if color_attributes is None:
            continue
        source = color_attributes.get(source_name)
        if source is None:
            continue
        if source.domain == "CORNER":
            return source_name
        target = color_attributes.get(target_name)
        if target is None:
            target = color_attributes.new(name=target_name, type=source.data_type, domain="CORNER")
        loops = mesh.loops
        for loop_index, loop in enumerate(loops):
            target.data[loop_index].color = source.data[loop.vertex_index].color
        color_attributes.active_color = target
        mesh.update()
        return target_name
    return source_name


def _parse_vector(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {value}")
    return parts[0], parts[1], parts[2]


def _principled_material(bpy, name: str, color: tuple[float, float, float, float], roughness: float = 0.65):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    material.diffuse_color = color
    return material


def _image_material(bpy, image_path: Path, name: str):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(str(image_path.resolve()))
    if bsdf is not None:
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Emission Strength"].default_value = 0.25
        material.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        material.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Emission Color"])
    return material


def _setup_shadow_floor(bpy, scene: dict, transparent: bool) -> None:
    bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0, 0, -0.01))
    floor = bpy.context.object
    floor.name = "contact_floor"
    if bool(scene.get("shadow_enabled", True)):
        if transparent:
            floor.is_shadow_catcher = True
        floor.data.materials.append(_principled_material(bpy, "matte_floor", (0.64, 0.66, 0.62, 1.0)))
    else:
        floor.hide_render = True


def _setup_proxy_garden_table(bpy, background_image: str | None) -> None:
    wood = _principled_material(bpy, "proxy_warm_wood", (0.42, 0.30, 0.20, 1.0), 0.72)
    dark_wood = _principled_material(bpy, "proxy_dark_wood", (0.25, 0.18, 0.12, 1.0), 0.78)
    ground_mat = _principled_material(bpy, "proxy_garden_ground", (0.28, 0.42, 0.24, 1.0), 0.85)

    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=1.28, depth=0.08, location=(0, 0, -0.04))
    tabletop = bpy.context.object
    tabletop.name = "proxy_round_tabletop"
    tabletop.data.materials.append(wood)

    for x, y in [(-0.72, -0.72), (0.72, -0.72), (-0.72, 0.72), (0.72, 0.72)]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, -0.62))
        leg = bpy.context.object
        leg.name = "proxy_table_leg"
        leg.dimensions = (0.08, 0.08, 1.16)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        leg.data.materials.append(dark_wood)

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0, 0, -1.22))
    ground = bpy.context.object
    ground.name = "proxy_garden_ground"
    ground.data.materials.append(ground_mat)

    if background_image:
        image_path = Path(background_image)
        if image_path.exists():
            mat = _image_material(bpy, image_path, "proxy_garden_reference_material")
            panels = [
                ((0, 3.05, 1.05), (math.radians(90), 0, 0)),
                ((0, -3.05, 1.05), (math.radians(90), 0, math.radians(180))),
                ((3.05, 0, 1.05), (math.radians(90), 0, math.radians(-90))),
                ((-3.05, 0, 1.05), (math.radians(90), 0, math.radians(90))),
            ]
            for location, rotation in panels:
                bpy.ops.mesh.primitive_plane_add(size=3.4, location=location, rotation=rotation)
                panel = bpy.context.object
                panel.name = "proxy_garden_reference_panel"
                panel.scale = (1.55, 1.0, 1.0)
                panel.data.materials.append(mat)


def _sample_background_sequence(directory: str | None, pattern: str, count: int) -> list[Path]:
    if not directory:
        return []
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        return []
    if len(paths) == 1:
        return paths * count
    indices = [round(index * (len(paths) - 1) / max(1, count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def _parse_camera_indices(value: str | None, frames: int) -> list[int]:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if not value:
        return list(range(frames))
    value = value.strip()
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text)
        indices = list(range(start, end))
    else:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(indices) != frames:
        raise ValueError(f"camera-indices count {len(indices)} does not match frames {frames}")
    return indices


def _load_2dgs_cameras(path: str | None) -> list[dict]:
    if not path:
        raise ValueError("--camera-json is required when --camera-path=2dgs_json")
    camera_path = Path(path)
    if not camera_path.exists():
        raise FileNotFoundError(f"2DGS camera JSON does not exist: {camera_path}")
    return json.loads(camera_path.read_text(encoding="utf-8"))


def _setup_proxy_garden_ring(
    bpy,
    background_image: str | None,
    background_sequence_dir: str | None,
    background_sequence_glob: str,
) -> None:
    wood = _principled_material(bpy, "proxy_warm_wood", (0.42, 0.30, 0.20, 1.0), 0.72)
    ground_mat = _principled_material(bpy, "proxy_garden_ground", (0.24, 0.37, 0.22, 1.0), 0.85)

    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=1.30, depth=0.08, location=(0, 0, -0.04))
    tabletop = bpy.context.object
    tabletop.name = "proxy_round_tabletop_directional"
    tabletop.data.materials.append(wood)

    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0, 0, -1.22))
    ground = bpy.context.object
    ground.name = "proxy_garden_ground_directional"
    ground.data.materials.append(ground_mat)

    panel_count = 8
    image_paths = _sample_background_sequence(background_sequence_dir, background_sequence_glob, panel_count)
    if not image_paths and background_image:
        image_path = Path(background_image)
        if image_path.exists():
            image_paths = [image_path] * panel_count
    if not image_paths:
        return

    radius = 4.2
    panel_width = 3.7
    panel_height = 2.1
    for index, image_path in enumerate(image_paths):
        angle = math.radians(-90 + index * 360 / panel_count)
        location = (radius * math.cos(angle), radius * math.sin(angle), 1.05)
        mat = _image_material(bpy, image_path, f"proxy_garden_ring_material_{index:02d}")
        bpy.ops.mesh.primitive_plane_add(
            size=1.0,
            location=location,
            rotation=(math.radians(90), 0, angle + math.radians(90)),
        )
        panel = bpy.context.object
        panel.name = f"proxy_garden_directional_panel_{index:02d}"
        panel.dimensions = (panel_width, panel_height, 1.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        panel.data.materials.append(mat)


def _setup_camera_light_floor(
    bpy,
    scene: dict,
    output: Path,
    resolution: str,
    samples: int,
    transparent: bool,
    camera_location: tuple[float, float, float],
    camera_rotation: tuple[float, float, float],
    light_location: tuple[float, float, float],
    light_energy: float,
    background_image: str | None,
    background_sequence_dir: str | None,
    background_sequence_glob: str,
    environment: str,
) -> None:
    width, height = [int(part) for part in resolution.lower().split("x", 1)]
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = str(output)
    if transparent or output.suffix.lower() == ".png":
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
    elif output.suffix.lower() in {".mp4", ".mkv", ".mov"}:
        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.ffmpeg.format = "MPEG4"
        bpy.context.scene.render.ffmpeg.codec = "H264"
        bpy.context.scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    bpy.context.scene.render.film_transparent = transparent

    if environment == "proxy_garden_table":
        _setup_proxy_garden_table(bpy, background_image)
        bpy.context.scene.world.color = (0.74, 0.79, 0.82)
    elif environment == "proxy_garden_ring":
        _setup_proxy_garden_ring(bpy, background_image, background_sequence_dir, background_sequence_glob)
        bpy.context.scene.world.color = (0.74, 0.79, 0.82)
    elif environment == "backplate_only":
        bpy.context.scene.world.color = (0.74, 0.79, 0.82)
    else:
        _setup_shadow_floor(bpy, scene, transparent)

    bpy.ops.object.light_add(type="AREA", location=light_location)
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = light_energy
    light.data.size = 4.0

    bpy.ops.object.camera_add(location=camera_location, rotation=camera_rotation)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    if background_image and not transparent and environment in {"shadow_floor", "backplate_only"}:
        _attach_camera_backplate(bpy, Path(background_image), camera, width / height)
    return camera


def _setup_depth_output(bpy, depth_output_dir: str | None) -> None:
    if not depth_output_dir:
        return
    path = Path(depth_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.use_alpha = False
    tree.links.new(render_layers.outputs["Depth"], viewer.inputs["Image"])


def _frame_output_path(output: Path, frame: int) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}_{frame:04d}{output.suffix}")
    return output.parent / f"{output.name}{frame:04d}.png"


def _save_viewer_depth(bpy, output_dir: str, frame: int) -> None:
    import numpy as np

    image = bpy.data.images.get("Viewer Node")
    if image is None:
        raise RuntimeError("Depth viewer node did not produce an image")
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))
    depth = np.flipud(pixels[:, :, 0])
    out_path = Path(output_dir) / f"depth_{frame:04d}.npy"
    np.save(out_path, depth)


def _attach_camera_backplate(bpy, image_path: Path, camera, aspect: float) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Background image does not exist: {image_path}")
    distance = 8.0
    height = 2.0 * distance * 0.50
    width = height * aspect
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    plane = bpy.context.object
    plane.name = "camera_backplate"
    plane.parent = camera
    plane.location = (0.0, 0.0, -distance)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.scale = (width, height, 1.0)
    mat = bpy.data.materials.new("camera_backplate_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Roughness"].default_value = 1.0
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(str(image_path.resolve()))
    if bsdf is not None:
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Emission Strength"].default_value = 0.35
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Emission Color"])
    plane.data.materials.append(mat)


def _look_at(mathutils, camera, target: tuple[float, float, float]) -> None:
    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _asset_center(scene: dict, name: str) -> tuple[float, float, float]:
    for asset in scene["assets"]:
        if asset["name"] == name:
            transform = asset["transform"]
            bmin = transform["bounds_min"]
            bmax = transform["bounds_max"]
            return (
                (float(bmin[0]) + float(bmax[0])) * 0.5,
                (float(bmin[1]) + float(bmax[1])) * 0.5,
                (float(bmin[2]) + float(bmax[2])) * 0.5,
            )
    return (0.0, 0.0, 0.25)


def _scene_center(scene: dict) -> tuple[float, float, float]:
    mins = []
    maxs = []
    for asset in scene["assets"]:
        transform = asset["transform"]
        mins.append([float(value) for value in transform["bounds_min"]])
        maxs.append([float(value) for value in transform["bounds_max"]])
    if not mins:
        return (0.0, 0.0, 0.25)
    bmin = [min(values[index] for values in mins) for index in range(3)]
    bmax = [max(values[index] for values in maxs) for index in range(3)]
    return ((bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5, (bmin[2] + bmax[2]) * 0.45)


def _setup_phase7_camera_path(bpy, mathutils, scene: dict, camera, frames: int) -> None:
    center = _scene_center(scene)
    object_a = _asset_center(scene, "object_a")
    object_b = _asset_center(scene, "object_b")
    object_c = _asset_center(scene, "object_c")
    keyframes = [
        (1, (0.0, -4.6, 1.75), (center[0], center[1], 0.30)),
        (max(2, int(frames * 0.18)), (0.0, -3.4, 1.35), (center[0], center[1], 0.30)),
        (max(3, int(frames * 0.34)), (-0.78, -2.95, 1.15), (object_a[0], object_a[1], object_a[2])),
        (max(4, int(frames * 0.50)), (0.00, -2.75, 1.12), (object_b[0], object_b[1], object_b[2])),
        (max(5, int(frames * 0.66)), (0.78, -2.95, 1.15), (object_c[0], object_c[1], object_c[2])),
        (max(6, int(frames * 0.82)), (1.05, -3.30, 1.35), (center[0], center[1], 0.32)),
        (frames, (0.0, -4.35, 1.70), (center[0], center[1], 0.30)),
    ]
    camera.data.lens = 34
    for frame, location, target in keyframes:
        bpy.context.scene.frame_set(frame)
        camera.location = location
        _look_at(mathutils, camera, target)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    for fcurve in camera.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"


def _setup_phase7_table_orbit(bpy, mathutils, scene: dict, camera, frames: int) -> None:
    center = _scene_center(scene)
    radius = 3.05
    height = 1.35
    camera.data.lens = 36
    if frames == 1:
        angles = [math.radians(-90)]
    else:
        angles = [math.radians(-90 + index * 360 / frames) for index in range(frames)]
    for index, angle in enumerate(angles, 1):
        bpy.context.scene.frame_set(index)
        camera.location = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            height,
        )
        _look_at(mathutils, camera, (center[0], center[1], 0.30))
        camera.keyframe_insert(data_path="location", frame=index)
        camera.keyframe_insert(data_path="rotation_euler", frame=index)
    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def _setup_2dgs_camera_path(
    bpy,
    mathutils,
    camera,
    frames: int,
    camera_json: str | None,
    camera_indices: str | None,
) -> None:
    cameras = _load_2dgs_cameras(camera_json)
    indices = _parse_camera_indices(camera_indices, frames)
    first = cameras[indices[0]]
    camera.data.type = "PERSP"
    camera.data.angle_x = 2.0 * math.atan(float(first["width"]) / (2.0 * float(first["fx"])))
    camera.data.sensor_fit = "AUTO"

    cv_to_blender = mathutils.Matrix.Diagonal((1.0, -1.0, -1.0)).to_3x3()
    for frame, camera_index in enumerate(indices, 1):
        if camera_index < 0 or camera_index >= len(cameras):
            raise ValueError(f"Camera index out of range: {camera_index}")
        entry = cameras[camera_index]
        bpy.context.scene.frame_set(frame)
        rotation_cv = mathutils.Matrix(entry["rotation"])
        rotation_blender = rotation_cv @ cv_to_blender
        camera.location = tuple(float(value) for value in entry["position"])
        camera.rotation_euler = rotation_blender.to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "CONSTANT"


def main() -> int:
    args = _parse_args()
    bpy = _import_bpy()
    mathutils = _import_mathutils()
    scene_path = Path(args.scene).resolve()
    project_root = _find_project_root(scene_path)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    _clear_scene(bpy)
    for asset in scene["assets"]:
        objects = _import_asset(bpy, mathutils, project_root, asset)
        _setup_materials(bpy, project_root, objects, asset)
        print(f"FUSION_ASSET: {asset['name']} objects={len(objects)}", flush=True)
    camera = _setup_camera_light_floor(
        bpy,
        scene,
        output,
        args.resolution,
        args.samples,
        args.transparent,
        _parse_vector(args.camera_location),
        _parse_vector(args.camera_rotation),
        _parse_vector(args.light_location),
        args.light_energy,
        args.background_image,
        args.background_sequence_dir,
        args.background_sequence_glob,
        args.environment,
    )
    _setup_depth_output(bpy, args.depth_output_dir)

    if args.frames <= 1:
        if args.camera_path == "2dgs_json":
            _setup_2dgs_camera_path(
                bpy,
                mathutils,
                camera,
                1,
                args.camera_json,
                args.camera_indices,
            )
        print("FUSION_RENDER: still frame 1/1", flush=True)
        bpy.ops.render.render(write_still=True)
        if args.depth_output_dir:
            _save_viewer_depth(bpy, args.depth_output_dir, 1)
    else:
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = args.frames
        if args.camera_path == "phase7_walkthrough":
            _setup_phase7_camera_path(bpy, mathutils, scene, camera, args.frames)
        elif args.camera_path == "phase7_table_orbit":
            _setup_phase7_table_orbit(bpy, mathutils, scene, camera, args.frames)
        elif args.camera_path == "2dgs_json":
            _setup_2dgs_camera_path(
                bpy,
                mathutils,
                camera,
                args.frames,
                args.camera_json,
                args.camera_indices,
            )
        print(f"FUSION_RENDER: animation frames=1..{args.frames}", flush=True)
        if args.depth_output_dir:
            original_output = output
            for frame in range(1, args.frames + 1):
                bpy.context.scene.frame_set(frame)
                bpy.context.scene.render.filepath = str(_frame_output_path(original_output, frame))
                bpy.ops.render.render(write_still=True)
                _save_viewer_depth(bpy, args.depth_output_dir, frame)
        else:
            bpy.ops.render.render(animation=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
