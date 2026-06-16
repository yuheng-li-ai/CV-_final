# CV HW3 Task 1: 2DGS And AIGC Scene Fusion

This repository tracks Task 1 of the CV final assignment: real multi-view reconstruction, text-to-3D generation, image-to-3D generation, 2DGS background reconstruction, fused scene rendering, and report evidence collection.

Current status: engineering harness is ready; formal long experiments are not assumed complete until their logs, metrics, screenshots, videos, and output paths are recorded in `docs/draft.md`.

## GitHub

Final public repository URL: https://github.com/yuheng-li-ai/CV-_final

## Model Weights

Large checkpoints, pretrained models, external repositories, and downloaded datasets are intentionally excluded from Git. They can be regenerated with the scripts and configs in this repository.

## Final Videos

Final and reference videos are collected under `videos/`:

- `videos/final_scene_fusion_15s_60fps_1080p.mp4`
- `videos/final_scene_fusion_15s_60fps_1080p_no_ghost.mp4`
- `videos/final_scene_fusion_c_swap_15s_60fps_1080p.mp4`
- `videos/task_a_input.mp4`
- `videos/object_a_mouse_input.mp4`

## Requirements

Use the scene environment for Task 1:

```bash
conda env create -f envs/scene.yml
conda activate cvhw3-scene
python -m pip install -e .
```

External tools expected for the full pipeline:

- COLMAP for Object A camera pose estimation.
- 2D Gaussian Splatting implementation under `external/2d-gaussian-splatting`.
- threestudio under `external/threestudio`.
- Magic123 under `external/Magic123`.
- FFmpeg for frame and video handling.
- Blender for the primary textured-mesh fusion/render route.

Record environment details in `environment.md` and every run directory under `outputs/{run_id}`.

## Data Preparation

Stage the provided local Task 1 inputs:

```bash
PYTHONPATH=src python scripts/stage_local_inputs.py --object_a_video "任务A.mp4" --object_c_image "任务C.jpg"
```

Expected canonical paths:

```text
data/scene/object_a/raw/object_a.mp4
data/scene/object_c/raw/object_c.jpg
```

Create the reproducible auto-mask Object C branch:

```bash
PYTHONPATH=src python scripts/preprocess_object_c.py --input_image data/scene/object_c/raw/object_c.jpg --output_image data/scene/object_c/masked/object_c_auto.png --output_mask data/scene/object_c/masks/object_c_auto.png --output_metadata data/scene/object_c/masks/object_c_auto.json --threshold 45 --padding 96
```

This produces the `image3d_auto_mask.yaml` input. The `image3d_refined_mask.yaml` branch still requires a manually refined or stronger model-generated mask.

Prepare the open-source background scene, preferably Mip-NeRF360 `garden` or `counter`, under:

```text
data/scene/background/garden/colmap
```

## Smoke Gates

Run smoke gates before every long experiment family:

```bash
bash scripts/smoke_env.sh --config configs/scene/background_2dgs_low.yaml --run_id smoke_env_bg --device cpu --dry_run
bash scripts/smoke_data.sh --config configs/scene/background_2dgs_low.yaml --run_id smoke_data_bg --device cpu --dry_run
bash scripts/smoke_train.sh --config configs/scene/background_2dgs_low.yaml --run_id smoke_train_bg --device cuda:0 --dry_run
bash scripts/smoke_render.sh --config configs/scene/fusion_main.yaml --run_id smoke_render_fusion --device cpu --dry_run
bash scripts/smoke_export.sh --config configs/scene/fusion_main.yaml --run_id smoke_export_fusion --device cpu --dry_run
```

If any smoke command prints `status: "missing_data"` or exits nonzero, do not launch the formal `nohup` command. Fix the missing paths first. Formal `run_*.sh` commands also refuse to start when validation fails.

## Train

Formal long runs are launched manually with `nohup`. Create the output directory before redirecting logs.

Background 2DGS low iteration example:

Smoke training check:

```bash
bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_smoke.yaml --run_id smoke_bg_2dgs_50iter --device cuda:1
```

```bash
mkdir -p outputs/bg_garden_5k
nohup bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_low.yaml --run_id bg_garden_5k --device cuda:0 > outputs/bg_garden_5k/nohup.log 2>&1 &
tail -f outputs/bg_garden_5k/nohup.log
```

After training, render held-out views and compute PSNR/SSIM/LPIPS:

```bash
mkdir -p outputs/bg_garden_5k_eval
nohup bash scripts/run_2dgs_eval.sh --config configs/scene/background_2dgs_low.yaml --run_id bg_garden_5k_eval --device cuda:0 > outputs/bg_garden_5k_eval/nohup.log 2>&1 &
tail -f outputs/bg_garden_5k_eval/nohup.log
```

Object A frame extraction and COLMAP:

```bash
mkdir -p outputs/object_a_frames_medium
nohup bash scripts/run_object_a_frames.sh --config configs/scene/object_a_frames_medium.yaml --run_id object_a_frames_medium --device cuda:0 > outputs/object_a_frames_medium/nohup.log 2>&1 &

mkdir -p outputs/object_a_colmap_medium
nohup bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_medium.yaml --run_id object_a_colmap_medium --device cpu > outputs/object_a_colmap_medium/nohup.log 2>&1 &
tail -f outputs/object_a_colmap_medium/nohup.log
python scripts/watch_colmap_progress.py --log outputs/object_a_colmap_medium/log.txt --images 87

scripts/colmap_clean_env.sh model_analyzer --path runs/scene/object_a_colmap_medium/colmap/sparse/0
```

Object A 2DGS:

```bash
mkdir -p outputs/object_a_2dgs_half
nohup bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_half.yaml --run_id object_a_2dgs_half --device cuda:0 > outputs/object_a_2dgs_half/nohup.log 2>&1 &
```

Object B SDS:

```bash
mkdir -p outputs/object_b_sds_detailed
nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed.yaml --run_id object_b_sds_detailed --device cuda:0 > outputs/object_b_sds_detailed/nohup.log 2>&1 &
```

Object C Magic123:

```bash
mkdir -p outputs/object_c_magic123_refined
nohup bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_refined_mask.yaml --run_id object_c_magic123_refined --device cuda:0 > outputs/object_c_magic123_refined/nohup.log 2>&1 &
```

## Test And Evaluation

Check harness tests:

```bash
python -m pytest tests/scene
```

Check remaining required checklist items:

```bash
PYTHONPATH=src python scripts/check_checklist.py --allow_incomplete
```

After a formal run completes and the user has reviewed the logs, append evidence to `docs/draft.md`:

```bash
PYTHONPATH=src python scripts/record_run_result.py --run_id bg_garden_5k --phase Phase2 --goal "Background 2DGS low iteration ablation" --elapsed_time "TODO" --result "completed" --cause_analysis "TODO" --next_step "Run mid iteration ablation"
```

## Render

Fusion:

```bash
mkdir -p outputs/fusion_main
nohup bash scripts/run_fusion.sh --config configs/scene/fusion_main.yaml --run_id fusion_main --device cuda:0 > outputs/fusion_main/nohup.log 2>&1 &
```

Final video:

```bash
mkdir -p outputs/final_video_main
nohup bash scripts/run_final_video.sh --config configs/scene/final_video.yaml --run_id final_video_main --device cuda:0 > outputs/final_video_main/nohup.log 2>&1 &
```

## Reproduce Task 1

1. Stage local Object A and Object C inputs.
2. Prepare Mip-NeRF360 background data.
3. Run smoke gates.
4. Run M1 background low/mid/high 2DGS ablation.
5. Run M2 Object A frame sampling and resolution ablations.
6. Run M3 Object B prompt ablation.
7. Run M4 Object C background-removal ablation.
8. Run M5 fusion scale/position/shadow ablation.
9. Run M6 final multi-view walkthrough video.
10. Append every run result to `docs/draft.md`.
11. Export WandB/SwanLab curves, report figures, final video, and weight links.
12. Update `docs/checklist.yaml` until required submission items have no `false` values.

See `docs/runbook_task1.md` for detailed nohup, tail, resume, expected output, and debug commands for every formal run family.
