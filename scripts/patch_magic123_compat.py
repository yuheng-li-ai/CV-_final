#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


CPP17_FILES = [
    "gridencoder/backend.py",
    "gridencoder/setup.py",
    "shencoder/backend.py",
    "shencoder/setup.py",
    "freqencoder/backend.py",
    "freqencoder/setup.py",
    "raymarching/backend.py",
    "raymarching/setup.py",
    "render/renderutils/ops.py",
]

SD_BEFORE = """        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_path, torch_dtype=self.precision_t, local_files_only=False)
"""

SD_AFTER = """        offline = (
            os.environ.get('HF_HUB_OFFLINE') == '1'
            or os.environ.get('TRANSFORMERS_OFFLINE') == '1'
            or os.environ.get('DIFFUSERS_OFFLINE') == '1'
        )

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_path, torch_dtype=self.precision_t, local_files_only=offline)
"""

MESH_SWALLOW_BEFORE = """            if opt.save_mesh:
                try:
                    trainer.save_mesh()
                except:
                    pass
"""

MESH_SWALLOW_AFTER = """            if opt.save_mesh:
                try:
                    trainer.save_mesh()
                except Exception:
                    logger.exception('mesh export failed')
                    raise
"""

RAW_MESH_BEFORE = """        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(os.path.join(path, f'mesh.ply'))
"""

RAW_MESH_AFTER = """        import trimesh
        raw_mesh_path = os.path.join(path, 'mesh_raw.ply')
        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(raw_mesh_path)
        logger.info(f'[INFO] writing raw mesh to {raw_mesh_path}')
"""

TEXTURED_EXPORT_BEFORE = """        _export(v, f)
"""

TEXTURED_EXPORT_AFTER = """        try:
            _export(v, f)
        except Exception:
            logger.exception('[WARN] textured obj export failed; raw mesh was saved first.')
"""

MESHUTILS_IMPORT_BEFORE = """        from meshutils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
"""

MESHUTILS_IMPORT_AFTER = """        from meshutils import decimate_mesh, clean_mesh
"""


def _replace_once(path: Path, before: str, after: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if after in text:
        return False
    if before not in text:
        raise RuntimeError(f"Expected pattern not found in {path}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")
    return True


def patch_cpp17(repo: Path) -> list[Path]:
    changed: list[Path] = []
    for relative in CPP17_FILES:
        path = repo / relative
        text = path.read_text(encoding="utf-8")
        updated = text.replace("-std=c++14", "-std=c++17")
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)
    return changed


def patch_offline_sd(repo: Path) -> bool:
    return _replace_once(repo / "guidance" / "sd_utils.py", SD_BEFORE, SD_AFTER)


def patch_mesh_export(repo: Path) -> list[Path]:
    changed: list[Path] = []
    main_path = repo / "main.py"
    renderer_path = repo / "nerf" / "renderer.py"

    main_text = main_path.read_text(encoding="utf-8")
    if MESH_SWALLOW_AFTER not in main_text:
        if MESH_SWALLOW_BEFORE not in main_text:
            raise RuntimeError(f"Expected mesh exception pattern not found in {main_path}")
        main_path.write_text(main_text.replace(MESH_SWALLOW_BEFORE, MESH_SWALLOW_AFTER), encoding="utf-8")
        changed.append(main_path)

    renderer_text = renderer_path.read_text(encoding="utf-8")
    renderer_updated = renderer_text
    if MESHUTILS_IMPORT_AFTER not in renderer_updated:
        if MESHUTILS_IMPORT_BEFORE not in renderer_updated:
            raise RuntimeError(f"Expected meshutils import pattern not found in {renderer_path}")
        renderer_updated = renderer_updated.replace(MESHUTILS_IMPORT_BEFORE, MESHUTILS_IMPORT_AFTER, 1)
    if RAW_MESH_AFTER not in renderer_updated:
        if RAW_MESH_BEFORE not in renderer_updated:
            raise RuntimeError(f"Expected raw mesh pattern not found in {renderer_path}")
        renderer_updated = renderer_updated.replace(RAW_MESH_BEFORE, RAW_MESH_AFTER, 1)
    if TEXTURED_EXPORT_AFTER not in renderer_updated:
        if TEXTURED_EXPORT_BEFORE not in renderer_updated:
            raise RuntimeError(f"Expected textured export pattern not found in {renderer_path}")
        renderer_updated = renderer_updated.replace(TEXTURED_EXPORT_BEFORE, TEXTURED_EXPORT_AFTER, 1)
    if renderer_updated != renderer_text:
        renderer_path.write_text(renderer_updated, encoding="utf-8")
        changed.append(renderer_path)

    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Task 1 Magic123 compatibility patches.")
    parser.add_argument("repo", nargs="?", default="external/Magic123")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Magic123 repo not found: {repo}")
    changed = patch_cpp17(repo)
    sd_changed = patch_offline_sd(repo)
    mesh_changed = patch_mesh_export(repo)
    for path in changed:
        print(f"PATCHED_CPP17: {path}")
    if sd_changed:
        print(f"PATCHED_OFFLINE_SD: {repo / 'guidance' / 'sd_utils.py'}")
    for path in mesh_changed:
        print(f"PATCHED_MESH_EXPORT: {path}")
    if not changed and not sd_changed and not mesh_changed:
        print("MAGIC123_COMPAT_ALREADY_PATCHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
