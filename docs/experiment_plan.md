# Task 1 Experiment Plan

This plan implements the high value plan B path for "2DGS and AIGC multi-source asset generation and real-scene fusion".

Current Phase0 limitation: the HW3 PDF was not found in the repository or parent CV directory. This plan must be audited again after the PDF path is supplied.

## Operating Contract

The agent owns engineering only:

- write and check scripts, configs, harness code, docs, and lightweight smoke tests;
- run only tiny subset, low resolution, 10 to 50 iteration smoke tests;
- never launch formal long training, reconstruction, generation, or rendering jobs;
- update `docs/draft.md` only from real user-provided logs, metrics, screenshots, or output paths.

The user owns formal experiments:

- run all long jobs manually with `nohup`;
- provide logs, metrics, screenshots, videos, and output paths for analysis;
- decide when to spend extra GPU-hours beyond the main closure path.

Target budget: 70 to 140 GPU-hours.

Priority order:

1. Main pipeline closure.
2. Required ablations R1 to R6.
3. Optional Gaussian/point-cloud stitching or extra 3D baseline only after closure.

## Repository Layout Target

```text
src/configs/
src/data/
src/recon/
src/generation/
src/fusion/
src/render/
src/eval/
scripts/
docs/
outputs/
```

The existing `src/cvhw3_scene` package remains available as the Python package namespace unless a later phase migrates modules.

## Phase0 Deliverables

- `docs/requirements_checklist.md`
- `docs/checklist.yaml`
- `docs/experiment_plan.md`
- target directory placeholders

## Phase1 Harness

Required implementation:

- `RunManager` with non-overwriting `outputs/{run_id}` creation.
- `scripts/select_gpu.py` using `nvidia-smi`, with `--device` override.
- smoke commands: `smoke_env`, `smoke_data`, `smoke_train`, `smoke_render`, `smoke_export`.
- `scripts/append_draft.py` appending to `docs/draft.md`.
- WandB/SwanLab logging settings that are opt-in and safe for smoke tests.
- `scripts/run_*.sh` wrappers supporting `--config`, `--run_id`, `--device`, `--dry_run`, `--resume`.

Every run directory must contain:

```text
outputs/{run_id}/
  config.yaml
  cmd.txt
  git_hash.txt
  env.txt
  gpu.txt
  metrics.json
  log.txt
  figures/
  videos/
  checkpoints/
```

Default behavior: fail if `outputs/{run_id}` already exists. `--resume` is required to reuse a run directory.

## Formal Experiment Matrix

| Matrix ID | Purpose | Required variants | Primary outputs |
|---|---|---|---|
| M1 | Background 2DGS | counter/garden/bicycle, then iteration low/mid/high | checkpoints, metrics, renders |
| M2 | Object A real reconstruction | phone video, frame extraction, COLMAP, 2DGS | COLMAP model, 2DGS checkpoint, renders |
| M3 | Object B text to 3D | simple/detailed/style-constrained prompt | mesh, texture, preview renders |
| M4 | Object C image to 3D | raw/auto background removal/refined mask, optional centered crop | masks, mesh, texture, preview renders |
| M5 | Fusion | scale/position/shadow variants | transform table, fused scene, still frames |
| M6 | Final video | background intro, approach, orbit A/B/C, wide shot | MP4 video, key frames |

## Required Metrics

2DGS runs:

- PSNR, SSIM, LPIPS;
- elapsed time;
- peak VRAM;
- input view count;
- registered images;
- sparse points;
- checkpoint size;
- representative render image paths.

Object B and C generation:

- generation time;
- vertices and faces;
- texture resolution;
- geometry score 1 to 5;
- texture score 1 to 5;
- fusion readiness score 1 to 5;
- artifact notes, including Janus artifacts for text-to-3D and background pollution for image-to-3D.

Fusion and final video:

- scale score 1 to 5;
- contact score 1 to 5;
- lighting score 1 to 5;
- viewpoint stability score 1 to 5;
- artifact count;
- video quality notes.

## Formal Run Template

Every formal run must provide these fields before the user launches it:

- run script;
- config YAML;
- nohup command;
- tail log command;
- resume command;
- expected outputs;
- failure and debug checklist.

Command pattern:

```bash
nohup bash scripts/run_xxx.sh --config configs/xxx.yaml --run_id xxx --device cuda:0 > outputs/xxx/nohup.log 2>&1 &
```

Because shell redirection creates `outputs/xxx/nohup.log` before the script starts, the user must create the output directory first or the script must provide a separate `scripts/prepare_run_dir.py` helper in Phase1.

## Phase2 Background 2DGS

Dataset choice:

- Preferred: Mip-NeRF360 `garden` if storage and image count are manageable.
- Backup: `counter` for faster iteration.
- Avoid optional scene switching after smoke tests unless the first dataset fails.

Variants:

- low: 5k iterations.
- mid: 15k iterations.
- high/final: 30k iterations after selecting the best settings.

Manual commands after Phase1 script creation:

```bash
nohup bash scripts/run_background_2dgs.sh --config configs/background_2dgs_low.yaml --run_id bg_garden_5k --device cuda:0 > outputs/bg_garden_5k/nohup.log 2>&1 &
tail -f outputs/bg_garden_5k/nohup.log
bash scripts/run_background_2dgs.sh --config configs/background_2dgs_low.yaml --run_id bg_garden_5k --device cuda:0 --resume
```

Success:

- training completes without NaN;
- final checkpoint and render images exist;
- metrics include PSNR/SSIM/LPIPS;
- `docs/draft.md` entry is appended after the user provides results.

Debug checklist:

- dataset path and COLMAP sparse model exist;
- CUDA device is visible;
- image resolution is not too high for VRAM;
- external 2DGS commit hash is recorded;
- resume uses the exact same config unless explicitly documented.

## Phase3 Object A

Inputs:

- user phone video or multi-view image folder;
- object should be static, textured, and visible from many angles;
- avoid reflective, transparent, or textureless objects.

Variants:

- frame sampling: dense, medium, sparse;
- resolution: full and half.

Success:

- COLMAP registers enough views for stable reconstruction;
- sparse point count is recorded;
- 2DGS runs and renders from novel views;
- failure modes are documented even if a variant is worse.

## Phase4 Object B

Tool: threestudio plus SDS.

Prompt variants:

- simple;
- detailed;
- style-constrained.

Success:

- mesh and texture export exist;
- Janus or multi-face artifacts are noted;
- best prompt is selected for fusion rather than only standalone appearance.

## Phase5 Object C

Tool: background removal plus Magic123.

Variants:

- raw image;
- auto background removal;
- refined mask;
- optional centered crop if Magic123 framing fails.

Success:

- mesh and texture export exist;
- background pollution and boundary noise are recorded;
- best variant is selected for fusion.

## Phase6 Fusion

Primary route: Blender textured mesh rendering with a common coordinate frame.

Record:

- scale, rotation, translation for A/B/C;
- coordinate conversion;
- contact points and shadows;
- manual score rubric values.

Variants:

- scale;
- position;
- shadow/lighting.

Optional only after main closure:

- point-cloud or Gaussian stitching. Record success or failure, but do not block the main Blender route.

## Phase7 Final Video

Camera path:

1. show background scene;
2. approach the three inserted objects;
3. orbit object A;
4. orbit object B;
5. orbit object C;
6. finish with a wide shot containing the background and all assets.

Required outputs:

- final MP4;
- key frames;
- command and config;
- video metadata.

## Phase8 Evaluation Materials

Create:

- method comparison table;
- ablation table R1 to R6;
- input and output gallery;
- WandB/SwanLab curves;
- failure mode panel;
- asset transform table.

## Phase9 README And Cleanup

Final checks:

- `docs/checklist.yaml` contains no `false` values for required submission items;
- README has setup, data, train, test, render, reproduce, and artifact links;
- weights/assets link is present;
- report references the same run IDs as `docs/draft.md`;
- generated large files are ignored by Git.
