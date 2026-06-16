from __future__ import annotations


def _patch_igl_legacy_names() -> None:
    try:
        import igl
    except Exception:
        return
    if not hasattr(igl, "fast_winding_number_for_meshes") and hasattr(
        igl, "fast_winding_number"
    ):
        igl.fast_winding_number_for_meshes = igl.fast_winding_number
    if not hasattr(igl, "read_obj") and hasattr(igl, "readOBJ"):
        igl.read_obj = igl.readOBJ


_patch_igl_legacy_names()
