# Task 1 Preflight Notes

Last updated: 2026-05-31 UTC.

## Local Inputs

The provided local inputs were staged into canonical data paths:

```text
data/scene/object_a/raw/object_a.mp4
data/scene/object_c/raw/object_c.jpg
```

Probe summary from `outputs/probe_task1_inputs/probe.json`:

| Input | Codec | Size | Duration / Frames |
|---|---|---|---|
| Object A video | HEVC | 720 x 1280 | 49.37 s, 987 frames, ~19.99 fps |
| Object C image | MJPEG/JPEG | 1760 x 2432 | single image |

Object C is a yellow toy-like object on a mostly white background. It is suitable for the raw-image Magic123 branch, but the refined-mask branch still needs a clean foreground mask before the formal run.

The reproducible auto-mask branch has been generated at:

```text
data/scene/object_c/masked/object_c_auto.png
data/scene/object_c/masks/object_c_auto.png
data/scene/object_c/masks/object_c_auto.json
```

The mask keeps the full object but includes some bottom shadow and edge noise. Treat that as the `auto_mask` ablation condition, not as the final refined mask.

## Tool Status

| Tool | Status | Notes |
|---|---|---|
| FFmpeg | runnable | From the active `cvhw3-scene` environment. |
| FFprobe | runnable | Used for media metadata. |
| COLMAP | runnable through `scripts/colmap_clean_env.sh` | Bare `/usr/bin/colmap` conflicts with conda libraries; use the wrapper. |
| Blender | missing | Required before Phase6/Phase7 Blender fusion and rendering. |
| nvidia-smi | not runnable in current shell | Current probe cannot communicate with the NVIDIA driver; use explicit `--device cuda:N` and verify GPU visibility before formal training. |

## Immediate Risks

- Mip-NeRF360 background data is not present at `data/scene/background/garden/colmap`.
- External repos are not verified yet: `external/2d-gaussian-splatting`, `external/threestudio`, `external/Magic123`.
- Formal wrappers now refuse to start if required external scripts or input paths are missing.
- Blender must be installed or made available on PATH before final fusion/video.
- Formal GPU jobs must be launched by the user; agent only runs dry-runs and tiny smoke checks.

## Safe Checks

```bash
PYTHONPATH=src python scripts/probe_task1_inputs.py --out outputs/probe_task1_inputs/probe.json
PYTHONPATH=src python scripts/preprocess_object_c.py --input_image data/scene/object_c/raw/object_c.jpg --output_image data/scene/object_c/masked/object_c_auto.png --output_mask data/scene/object_c/masks/object_c_auto.png --output_metadata data/scene/object_c/masks/object_c_auto.json --threshold 45 --padding 96
PYTHONPATH=src bash scripts/smoke_data.sh --config configs/scene/object_a_frames_medium.yaml --run_id smoke_data_object_a --device cpu --dry_run
PYTHONPATH=src bash scripts/smoke_data.sh --config configs/scene/image3d_raw.yaml --run_id smoke_data_object_c --device cpu --dry_run
bash scripts/colmap_clean_env.sh -h
```
