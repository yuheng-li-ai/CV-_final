# Environment Inventory

Student: yuhengli  
Student ID: 23307130334  
Repository: https://github.com/yuheng-li-ai/CV-_final

This file is the human-readable environment checklist for CV HW3. Update it whenever dependencies, external repository commits, datasets, or hardware assignments change.

## Observed Local System

Current observations from `/home/yuhengli` on 2026-05-23:

| Item | Value |
|---|---|
| Python in base shell | 3.12.7 |
| PyTorch in base shell | 2.6.0+cu126 |
| torchvision in base shell | 0.21.0+cu126 |
| GPU | 3 x NVIDIA RTX A6000 |
| GPU memory | 49140 MiB each |
| NVIDIA driver | 550.54.14 |
| CUDA shown by driver | 12.4 |
| COLMAP | `/usr/bin/colmap` |
| FFmpeg | `/usr/bin/ffmpeg` |
| Blender | not found in current PATH |
| SwanLab | not installed in base shell |
| LeRobot | not installed in base shell |

## Required Conda Environments

Use two environments to avoid dependency conflicts:

```bash
conda env create -f envs/scene.yml
conda env create -f envs/act.yml
```

Update existing environments with:

```bash
conda env update -f envs/scene.yml --prune
conda env update -f envs/act.yml --prune
```

## Scene Environment Checklist

Environment file:

```text
envs/scene.yml
```

Purpose:

- COLMAP orchestration.
- 2D Gaussian Splatting experiments.
- threestudio text-to-3D experiments.
- Magic123 image-to-3D experiments.
- mesh export and scene fusion.
- final rendering pipeline.

External tools to record after installation:

| Tool | Path | Version / Commit | Notes |
|---|---|---|---|
| COLMAP | `/usr/bin/colmap` | TODO | System install detected |
| FFmpeg | `/usr/bin/ffmpeg` | TODO | System install detected |
| Blender | TODO | TODO | Install if needed |
| 2DGS repo | `external/2d-gaussian-splatting` | TODO | Record commit |
| threestudio repo | `external/threestudio` | TODO | Record commit |
| Magic123 repo | `external/Magic123` | TODO | Record commit |

## ACT Environment Checklist

Environment file:

```text
envs/act.yml
```

Purpose:

- LeRobot installation.
- CALVIN dataset loading.
- ACT training.
- zero-shot evaluation on environment D.
- SwanLab metric tracking.

External tools and datasets to record:

| Item | Path / Version | Notes |
|---|---|---|
| LeRobot | TODO | pip package or git commit |
| CALVIN data root | TODO | Must include environments A/B/C/D |
| evaluation script | TODO | Use official or wrapped script |
| SwanLab project | TODO | Do not store token |

## Reproducibility Settings

Global defaults:

| Setting | Value |
|---|---|
| random seeds | 0, 1, 2 for main ACT experiments |
| tracking | SwanLab |
| scene output root | `runs/scene/` |
| ACT output root | `runs/act/` |
| checkpoint root | `checkpoints/` |
| report figure root | `reports/figures/` |
| data root | `data/` |

Do not store secrets in this file. SwanLab tokens, cloud-drive credentials, and GitHub credentials must stay outside the repository.

## Required Version Commands

Run and paste outputs here before final experiments:

```bash
uname -a
conda --version
conda env list
nvidia-smi
colmap -h | head
ffmpeg -version | head
blender --version
```

Scene environment:

```bash
conda activate cvhw3-scene
python -V
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import cv2, numpy, scipy, matplotlib, pandas, skimage; print('scene core imports ok')"
python -c "import swanlab; print(swanlab.__version__)"
```

ACT environment:

```bash
conda activate cvhw3-act
python -V
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import lerobot; print('lerobot import ok')"
python -c "import swanlab; print(swanlab.__version__)"
```

## Known Risks

- Blender is not currently available in PATH and may need installation.
- GitHub HTTPS failed through the current proxy; use SSH with the local proxy if needed.
- 2DGS, threestudio, and Magic123 may require specific CUDA/PyTorch combinations. Keep them in `cvhw3-scene`.
- LeRobot/CALVIN may require package versions that conflict with 3D generation tools. Keep them in `cvhw3-act`.
- Large datasets and checkpoints must stay outside Git.
