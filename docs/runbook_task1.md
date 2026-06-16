# Task 1 Formal Runbook

All formal experiments are launched manually by the user. The agent only runs dry-runs and smoke tests.

Before any formal `nohup` command, create the run directory because shell redirection opens `outputs/<run_id>/nohup.log` before the script can run:

```bash
mkdir -p outputs/<run_id>
```

Every script supports:

```text
--config --run_id --device --dry_run --resume
```

## Smoke Gate

Run these before each long experiment family. Use `--dry_run` first, then a real smoke run with a smoke run ID if the dry-run is clean.

```bash
bash scripts/smoke_env.sh --config configs/scene/background_2dgs.yaml --run_id smoke_env_bg --device cpu --dry_run
bash scripts/smoke_data.sh --config configs/scene/background_2dgs.yaml --run_id smoke_data_bg --device cpu --dry_run
bash scripts/smoke_train.sh --config configs/scene/background_2dgs_low.yaml --run_id smoke_train_bg --device cuda:0 --dry_run
bash scripts/smoke_render.sh --config configs/scene/fusion_baseline.yaml --run_id smoke_render_fusion --device cpu --dry_run
bash scripts/smoke_export.sh --config configs/scene/fusion_baseline.yaml --run_id smoke_export_fusion --device cpu --dry_run
```

Success:

- command exits 0;
- config parses;
- planned command is printed;
- missing data paths are understood before formal launch.

If a smoke command prints `status: "missing_data"` or exits nonzero, stop and fix the listed paths. The formal `run_*.sh` wrappers refuse to start when validation issues remain.

Failure recovery:

- fix missing config paths;
- verify external repo paths under `external/`;
- rerun the same smoke command;
- do not launch formal `nohup` until the smoke gate passes.

## M1 Background 2DGS

Run script:

```text
scripts/run_background_2dgs.sh
```

Config YAML:

```text
configs/scene/background_2dgs_smoke.yaml, `configs/scene/background_2dgs_low.yaml`, `configs/scene/background_2dgs_mid.yaml`, or `configs/scene/background_2dgs_high.yaml`
```

Smoke command already verified by the agent:

```bash
bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_smoke.yaml --run_id smoke_bg_2dgs_50iter --device cuda:1
```

Nohup command:

```bash
mkdir -p outputs/bg_garden_5k
nohup bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_low.yaml --run_id bg_garden_5k --device cuda:0 > outputs/bg_garden_5k/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/bg_garden_5k/nohup.log
tail -f outputs/bg_garden_5k/log.txt
```

Resume:

```bash
nohup bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_low.yaml --run_id bg_garden_5k --device cuda:0 --resume > outputs/bg_garden_5k/nohup.log 2>&1 &
```

Expected outputs:

- `outputs/bg_garden_5k/config.yaml`
- `outputs/bg_garden_5k/gpu.txt`
- `outputs/bg_garden_5k/log.txt`
- external 2DGS checkpoint from the config output path

Render and metrics command after training:

```bash
mkdir -p outputs/bg_garden_5k_eval
nohup bash scripts/run_2dgs_eval.sh --config configs/scene/background_2dgs_low.yaml --run_id bg_garden_5k_eval --device cuda:0 > outputs/bg_garden_5k_eval/nohup.log 2>&1 &
```

Tail eval log:

```bash
tail -f outputs/bg_garden_5k_eval/nohup.log
tail -f outputs/bg_garden_5k_eval/log.txt
```

Expected eval outputs:

- `runs/scene/background_2dgs_low/test/ours_5000/renders`
- `runs/scene/background_2dgs_low/test/ours_5000/gt`
- `runs/scene/background_2dgs_low/results.json`
- `runs/scene/background_2dgs_low/per_view.json`
- PSNR/SSIM/LPIPS metrics in `results.json`

Failure/debug checklist:

- Mip-NeRF360 scene path exists;
- COLMAP sparse model is readable;
- external 2DGS training script exists;
- external 2DGS render and metrics scripts exist;
- selected CUDA device has enough VRAM;
- reduce resolution or iterations for smoke failures.

## M2 Object A Frame Extraction

Run script:

```text
scripts/run_object_a_frames.sh
```

Config YAML:

```text
configs/scene/object_a_frames_dense.yaml, configs/scene/object_a_frames_medium.yaml, or configs/scene/object_a_frames_sparse.yaml
```

Nohup command:

```bash
mkdir -p outputs/object_a_frames_medium
nohup bash scripts/run_object_a_frames.sh --config configs/scene/object_a_frames_medium.yaml --run_id object_a_frames_medium --device cuda:0 > outputs/object_a_frames_medium/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/object_a_frames_medium/nohup.log
tail -f outputs/object_a_frames_medium/log.txt
```

Resume:

```bash
nohup bash scripts/run_object_a_frames.sh --config configs/scene/object_a_frames_medium.yaml --run_id object_a_frames_medium --device cuda:0 --resume > outputs/object_a_frames_medium/nohup.log 2>&1 &
```

Expected outputs:

- extracted frame directory;
- frame count;
- representative extracted images.

Failure/debug checklist:

- source video path exists;
- FFmpeg can decode the phone video;
- dense/medium/sparse FPS settings produce enough overlap;
- remove blurry or duplicate frames before COLMAP if registration fails.

## M2 Object A COLMAP

Run frame extraction first for the matching dense/medium/sparse variant.

Run script:

```text
scripts/run_object_a_colmap.sh
```

Config YAML:

```text
configs/scene/object_a_colmap_dense.yaml, configs/scene/object_a_colmap_medium.yaml, or configs/scene/object_a_colmap_sparse.yaml
```

Nohup command:

```bash
mkdir -p outputs/object_a_colmap_medium
nohup bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_medium.yaml --run_id object_a_colmap_medium --device cpu > outputs/object_a_colmap_medium/nohup.log 2>&1 &
```

Progress and tail log:

```bash
python scripts/watch_colmap_progress.py --log outputs/object_a_colmap_medium/log.txt --images 87
tail -f outputs/object_a_colmap_medium/nohup.log
tail -f outputs/object_a_colmap_medium/log.txt
```

Resume:

```bash
nohup bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_medium.yaml --run_id object_a_colmap_medium --device cpu --resume > outputs/object_a_colmap_medium/nohup.log 2>&1 &
```

Expected outputs:

- extracted image folder;
- COLMAP database;
- sparse reconstruction;
- registered image count;
- sparse point count.

Success check:

```bash
scripts/colmap_clean_env.sh model_analyzer --path runs/scene/object_a_colmap_medium/colmap/sparse/0
```

Collect sparse/medium/dense stats after the formal runs finish:

```bash
python scripts/collect_colmap_stats.py \
  --model object_a_sparse=runs/scene/object_a_colmap_sparse/colmap/sparse/0 \
  --model object_a_medium=runs/scene/object_a_colmap_medium/colmap/sparse/0 \
  --model object_a_dense=runs/scene/object_a_colmap_dense/colmap/sparse/0 \
  --out reports/tables/object_a_colmap_r2_stats.csv
```

Failure/debug checklist:

- video exists and is readable by FFmpeg;
- use `scripts/colmap_clean_env.sh` for COLMAP in this conda environment;
- frames are sharp and sufficiently overlapping;
- camera model is appropriate;
- try lower FPS for duplicate frames or higher FPS for registration gaps.
- if `No good initial image pair found` appears, use more temporally adjacent frames or filter out extreme tilt/close-up frames.

## M2 Object A 2DGS

Run script:

```text
scripts/run_object_a_2dgs.sh
```

Config YAML:

```text
`configs/scene/object_a_2dgs_full.yaml` or `configs/scene/object_a_2dgs_half.yaml`
```

Nohup command:

```bash
mkdir -p outputs/object_a_2dgs_medium_half
nohup bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_half.yaml --run_id object_a_2dgs_medium_half --device cuda:0 > outputs/object_a_2dgs_medium_half/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/object_a_2dgs_medium_half/nohup.log
tail -f outputs/object_a_2dgs_medium_half/log.txt
```

Resume:

```bash
nohup bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_half.yaml --run_id object_a_2dgs_medium_half --device cuda:0 --resume > outputs/object_a_2dgs_medium_half/nohup.log 2>&1 &
```

Expected outputs:

- Object A 2DGS checkpoint;
- novel-view renders;
- PSNR/SSIM/LPIPS if a held-out split is available;
- checkpoint size and peak VRAM.

Failure/debug checklist:

- COLMAP output has enough registered images;
- image paths match 2DGS loader expectations;
- reduce resolution for CUDA OOM;
- inspect for floating or sparse geometry.

## M3 Object B SDS

Run script:

```text
scripts/run_object_b_sds.sh
```

Config YAML:

```text
`configs/scene/text3d_simple.yaml`, `configs/scene/text3d_detailed.yaml`, or `configs/scene/text3d_style_constrained.yaml`
```

Nohup command:

```bash
mkdir -p outputs/object_b_sds_detailed
nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed.yaml --run_id object_b_sds_detailed --device cuda:0 > outputs/object_b_sds_detailed/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/object_b_sds_detailed/nohup.log
tail -f outputs/object_b_sds_detailed/log.txt
```

Resume:

```bash
nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed.yaml --run_id object_b_sds_detailed --device cuda:0 --resume > outputs/object_b_sds_detailed/nohup.log 2>&1 &
```

Expected outputs:

- exported mesh;
- texture image;
- preview renders;
- vertices/faces count;
- geometry, texture, and fusion-readiness scores.

Failure/debug checklist:

- threestudio repo path and config exist;
- prompt does not invite multiple objects or ambiguous views;
- record Janus artifacts instead of hiding them;
- select the best fusion-ready prompt, not just the prettiest preview.

## M4 Object C Magic123

Phase5/Object C uses the official Magic123 repository, not threestudio. The expected repo
path is `external/Magic123`, with `main.py`, `pretrained/zero123/105000.ckpt`,
and `pretrained/midas/dpt_beit_large_512.pt` present before any formal run.

Before the auto-mask variant, generate the reproducible threshold mask:

```bash
PYTHONPATH=src python scripts/preprocess_object_c.py --input_image data/scene/object_c/raw/object_c.jpg --output_image data/scene/object_c/masked/object_c_auto.png --output_mask data/scene/object_c/masks/object_c_auto.png --output_metadata data/scene/object_c/masks/object_c_auto.json --threshold 45 --padding 96
```

Environment/material probe:

```bash
PYTHONPATH=src python scripts/probe_magic123_env.py --config configs/scene/image3d_smoke.yaml
```

Repository and model preparation commands:

```bash
bash scripts/setup_magic123_repo.sh external/Magic123
bash scripts/download_magic123_models.sh external/Magic123
```

Run script:

```text
scripts/run_object_c_magic123.sh
```

Config YAML:

```text
`configs/scene/image3d_raw.yaml`, `configs/scene/image3d_auto_mask.yaml`, or `configs/scene/image3d_refined_mask.yaml`
```

Nohup command:

```bash
mkdir -p outputs/object_c_magic123_masked
nohup bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_refined_mask.yaml --run_id object_c_magic123_masked --device cuda:0 > outputs/object_c_magic123_masked/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/object_c_magic123_masked/nohup.log
tail -f outputs/object_c_magic123_masked/log.txt
python scripts/watch_magic123_progress.py --config configs/scene/image3d_refined_mask.yaml --run_id object_c_magic123_masked --interval 5
```

Resume:

```bash
nohup bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_refined_mask.yaml --run_id object_c_magic123_masked --device cuda:0 --resume > outputs/object_c_magic123_masked/nohup.log 2>&1 &
```

Expected outputs:

- input mask variant;
- exported mesh;
- texture image;
- preview renders;
- background pollution and boundary noise notes.

Failure/debug checklist:

- `external/Magic123/main.py` exists and imports in the intended conda environment;
- `pretrained/zero123/105000.ckpt` and `pretrained/midas/dpt_beit_large_512.pt` exist;
- input image has a single centered object;
- mask does not cut off object parts;
- auto-mask may include bottom shadow and edge noise; record this as part of R5;
- use centered crop if framing fails;
- compare raw, auto-mask, and refined-mask variants.

## M5 Fusion

Run script:

```text
scripts/run_fusion.sh
```

Config YAML:

```text
`configs/scene/fusion_main.yaml`, `configs/scene/fusion_scale_ablation.yaml`, `configs/scene/fusion_position_ablation.yaml`, or `configs/scene/fusion_shadow_ablation.yaml`
```

Nohup command:

```bash
mkdir -p outputs/fusion_main
nohup bash scripts/run_fusion.sh --config configs/scene/fusion_main.yaml --run_id fusion_main --device cuda:0 > outputs/fusion_main/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/fusion_main/nohup.log
tail -f outputs/fusion_main/log.txt
```

Resume:

```bash
nohup bash scripts/run_fusion.sh --config configs/scene/fusion_main.yaml --run_id fusion_main --device cuda:0 --resume > outputs/fusion_main/nohup.log 2>&1 &
```

Expected outputs:

- fused scene descriptor;
- A/B/C transform table;
- still frames;
- scale/contact/lighting/viewpoint stability scores.

Failure/debug checklist:

- all asset paths exist;
- coordinate axes are documented;
- object contact points touch plausible surfaces;
- shadow settings do not hide object identity.

## M6 Final Video

Run script:

```text
scripts/run_final_video.sh
```

Config YAML:

```text
configs/scene/final_video.yaml
```

Nohup command:

```bash
mkdir -p outputs/final_video_main
nohup bash scripts/run_final_video.sh --config configs/scene/final_video.yaml --run_id final_video_main --device cuda:0 > outputs/final_video_main/nohup.log 2>&1 &
```

Tail log:

```bash
tail -f outputs/final_video_main/nohup.log
tail -f outputs/final_video_main/log.txt
python scripts/watch_blender_progress.py --log outputs/final_video_main/nohup.log --interval 10
```

Resume:

```bash
nohup bash scripts/run_final_video.sh --config configs/scene/final_video.yaml --run_id final_video_main --device cuda:0 --resume > outputs/final_video_main/nohup.log 2>&1 &
```

Expected outputs:

- final MP4;
- key frames;
- video metadata;
- final camera path config.

Failure/debug checklist:

- fused scene path exists;
- camera path frames the background and all three objects;
- first segment shows background, middle approaches/orbits A/B/C, final segment is wide;
- no severe object clipping or text occlusion in report frames.
