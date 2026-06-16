# Phase6 C-Swap Correction Materials

## Current Final-Video Job

- Run ID: `c_swap_preserve_old_ab_900f_1080p`
- Goal: generate a 15-second 1080p/60fps video that preserves the old final A/B appearance and replaces only Object C with the tuned mouse C.
- Running command:

```bash
python scripts/run_c_only_swap_video.py \
  --old_scene runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json \
  --new_scene runs/scene/fusion_old_ab_mouse_c/a_color_compare_origA_6f/scene_2dgs_table_aligned.json \
  --old_final_foreground_dir runs/scene/final_video_900f_1080p/foreground \
  --background_dir runs/scene/background_2dgs_high/traj/ours_30000/renders \
  --background_depth_dir runs/scene/background_2dgs_high/traj/ours_30000/vis \
  --camera_json runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json \
  --output_dir runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p \
  --output_video runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p/final_scene_fusion_c_swap_15s_60fps_1080p.mp4 \
  --contact_sheet reports/figures/c_swap_preserve_old_ab_900f_1080p_contact.jpg \
  --frames 900 \
  --fps 60 \
  --samples 16 \
  --resolution 1920x1080
```

Expected outputs:

- `runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p/final_scene_fusion_c_swap_15s_60fps_1080p.mp4`
- `reports/figures/c_swap_preserve_old_ab_900f_1080p_contact.jpg`
- `runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p/composite_frames/frame_0001.jpg ... frame_0900.jpg`

## Why Object A Looked Wrong

Object A is the old final Object A, not the newly trained mouse A. The old final foreground already contains the best available A/B appearance, including the red/blue texture visible on Object A. When the replacement-C pipeline rerendered A through the current Blender PLY path, A became gray/white. The measured crop comparison confirms this regression:

- `bg00000`: old final A mean RGB `[149.27, 137.80, 125.87]`; rerendered A mean RGB `[85.96, 80.75, 74.61]`.
- `bg00156`: old final A mean RGB `[198.48, 189.63, 159.86]`; rerendered A mean RGB `[83.98, 77.52, 68.66]`.

Evidence:

- `reports/figures/object_a_old_final_vs_new_mouse_c_color_compare.jpg`
- Old final foreground sample: `runs/scene/final_video_900f_1080p/foreground/frame_0470.png`
- Rerendered gray A sample: `runs/scene/fusion_old_ab_mouse_c/a_color_compare_origA_vertexcolor_6f/foreground/frame_0004.png`

The broader Object A failure chain should still be described in the report. A was reconstructed from real multi-view images with COLMAP and 2DGS, and image-space 2DGS rendering preserved recognizable appearance. The problem is that the captured object occupied only part of each frame while the table/background occupied a large area. During COLMAP and 2DGS optimization, those static background regions were also modeled as confident scene content. When the 2DGS result was later converted to a mesh, TSDF/mesh export therefore contained table/background shells in addition to the real object. Connected-component filtering only partially removed these shells, and the remaining component had incomplete geometry and unstable color transfer. The observed symptoms are: some viewpoints preserve the umbrella-side texture reasonably well, the opposite side becomes blurred or incomplete, and Blender rerendering can turn the old A gray/white. This is why the final correction avoids rerendering A/B and instead preserves the old final A/B foreground directly.

## Old C vs New Mouse C

Old C:

- Scene: `runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json`
- Mesh: `runs/scene/image3d_auto_mask/auto_mask/fine/mesh/mesh.obj`
- Texture: `runs/scene/image3d_auto_mask/auto_mask/fine/mesh/albedo.png`
- Location: `[0.9876, 1.6392, 0.1995]`
- Scale: `0.544534`
- Rotation: `[90.0, -20.0, 4.0]`
- Extent: `[0.5127, 0.3770, 0.4995]`

New tuned mouse C:

- Scene: `runs/scene/fusion_old_ab_mouse_c/a_color_compare_origA_6f/scene_2dgs_table_aligned.json`
- Mesh: `runs/scene/image3d_mouse_final_clean/auto_mask/fine/mesh/mesh.obj`
- Texture: `runs/scene/image3d_mouse_final_clean/auto_mask/fine/mesh/albedo.png`
- Location: `[1.0747, 1.4571, 0.2869]`
- Scale: `0.549409`
- Rotation: `[70.0, -20.0, 4.0]`
- Extent: `[0.4861, 0.3370, 0.5569]`

The tuned mouse C is lower-profile in x/y footprint but taller in z extent. The visual strategy is to keep it as the only newly rendered object, so it can be positioned and scaled without perturbing the already accepted old A/B foreground.

Evidence:

- `reports/figures/object_c_old_vs_mouse_reconstruction_comparison.jpg`
- `reports/figures/c_swap_preserve_old_ab_6f_contact.jpg`
- `runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_6f/preview.mp4`

Interpretation:

- The old C validation images preserve a toy-like silhouette but the fused result reads as a yellow blob on the table. This is consistent with the earlier conclusion that the original single-image C reconstruction was not reliable enough for final insertion.
- The mouse replacement is a simpler object with a lower-profile shape and cleaner single-image mask. Its final validation depth/normal are not perfect, but the fused preview has a clearer boundary and a more interpretable object identity.
- The C comparison should be presented as a limitation of single-image generation: even when Magic123 returns plausible validation views, the output may still fail after scale/orientation/contact constraints are imposed in the real scene.

## Final Method To Describe

The final correction is an image-space C-only swap:

1. Use the old final foreground frames from `runs/scene/final_video_900f_1080p/foreground` to preserve A/B exactly.
2. Render old C only with the old final C scene to obtain an alpha mask.
3. Render new mouse C only with the tuned mouse-C scene.
4. Remove old C from the old foreground using a dilated old-C alpha mask.
5. Alpha-composite new mouse C into the foreground.
6. Composite the corrected foreground over the 2DGS garden RGB/depth background using the same approximate depth-occlusion path as the old final.

This is not a physically unified 3D insertion. It is a controlled image-space correction that preserves the best verified A/B result and replaces only the failed/obsolete C asset.
