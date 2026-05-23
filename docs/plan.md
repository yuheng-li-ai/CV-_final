# CV HW3 Implementation Plan

Student: yuhengli  
Student ID: 23307130334  
GitHub repository: https://github.com/yuheng-li-ai/CV-_final  
Deadline: 2026-06-23 23:59 Beijing time

## 1. Objectives

This repository will implement the two large assignments from `HW3_Computer Vision` as two separated, object-oriented, reproducible codebases.

- Task 1: 2DGS and AIGC multi-source 3D asset generation, reconstruction, fusion, and rendering.
- Task 2: LeRobot ACT policy training and zero-shot cross-environment generalization on CALVIN.

The final submission must include:

- A public GitHub repository.
- A professional `README.md` with setup, data preparation, train, test, render, and report commands.
- Conda environment files.
- A detailed environment inventory.
- SwanLab visualizations for training and validation metrics.
- Best model weights uploaded to cloud storage.
- A two-column English academic report of at least 35 pages.
- Complete baseline experiments plus as many extension experiments as feasible.

The hard rule is that all PDF requirements must be satisfied before optional extensions are treated as complete.

## 2. Repository Structure

The implementation should be created under:

```text
/home/yuhengli/study/CV/assignment3
```

Required structure:

```text
assignment3/
  README.md
  draft.md
  environment.md
  pyproject.toml
  requirements-dev.txt
  envs/
    scene.yml
    act.yml
  configs/
    scene/
    act/
  docs/
    plan.md
    experiment_registry.md
  reports/
    final_report.tex
    figures/
    tables/
  scripts/
    scene/
    act/
    report/
  src/
    cvhw3_scene/
    cvhw3_act/
  tests/
    scene/
    act/
  data/
  external/
  runs/
  checkpoints/
  assets/
```

Git should track source code, configs, scripts, docs, small report figures, and LaTeX files. Git should ignore raw datasets, downloaded external repositories unless intentionally added as submodules, large generated videos, checkpoints, SwanLab local cache, and temporary outputs.

## 3. Git Workflow

The repository must remain recoverable at all times.

Before each implementation or experiment block:

```bash
git pull --ff-only
git status --short --branch
```

After each scaffold, baseline, experiment, analysis update, or report milestone:

```bash
git status --short
git diff
git add <changed-files>
git commit -m "<type>: <description>"
git push
```

Recommended commit sequence:

- `chore: scaffold cv hw3 repository`
- `docs: add environment inventory and implementation plan`
- `feat: add scene reconstruction harness`
- `feat: add act training harness`
- `exp: record 2dgs object baseline`
- `exp: record 2dgs background baseline`
- `exp: record text to 3d baseline`
- `exp: record image to 3d baseline`
- `exp: record scene fusion baseline`
- `exp: record act environment b baseline`
- `exp: record act abc joint baseline`
- `exp: record act environment d evaluation`
- `docs: update final report results`

The configured remote should be:

```bash
git remote add origin https://github.com/yuheng-li-ai/CV-_final
```

If HTTPS fails in this environment, switch the remote to SSH and use the local proxy workflow:

```bash
git remote set-url origin git@github.com:yuheng-li-ai/CV-_final.git
git config core.sshCommand "ssh -i /home/yuhengli/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o ProxyCommand='nc -X connect -x 127.0.0.1:10080 %h %p'"
```

Never commit:

- raw datasets,
- model checkpoints,
- generated long videos,
- API keys,
- SwanLab tokens,
- cloud-drive credentials,
- private notes with secrets.

## 4. Environment Plan

The project must maintain three environment artifacts:

- `envs/scene.yml`: Conda environment for Task 1.
- `envs/act.yml`: Conda environment for Task 2.
- `environment.md`: human-readable environment inventory and reproducibility checklist.

The environment inventory must be updated before major experiments and before final report writing. It must record:

- OS, kernel, CPU, RAM, disk.
- GPU model/count/VRAM.
- NVIDIA driver and CUDA versions.
- Python and Conda versions.
- PyTorch and torchvision versions.
- COLMAP path/version.
- FFmpeg path/version.
- Blender path/version or absence.
- 2DGS, threestudio, Magic123, and LeRobot repository commit hashes.
- CALVIN dataset path and split layout.
- SwanLab project settings without secrets.
- Dataset roots.
- GPU IDs used.
- random seeds.
- known compatibility risks.

Initial local facts already observed:

- Python: 3.12.7 in the current base shell.
- PyTorch: 2.6.0+cu126.
- torchvision: 0.21.0+cu126.
- GPU: 3 x NVIDIA RTX A6000, 48 GB each.
- COLMAP exists at `/usr/bin/colmap`.
- FFmpeg exists at `/usr/bin/ffmpeg`.
- Blender was not found in the current shell path.
- SwanLab, WandB, and LeRobot were not installed in the current base shell.

Because 2DGS, threestudio, Magic123, LeRobot, and CALVIN dependencies may conflict, the plan uses two Conda environments instead of one shared environment.

## 5. Architecture Rules

All code must be object-oriented and decoupled.

The two assignments must not share task-specific code. Shared utilities are allowed only for generic concerns such as config loading, metric serialization, seed setup, logging, and draft recording.

Scene package:

```text
src/cvhw3_scene/
  config.py
  cli.py
  data/
  colmap/
  gaussian/
  generation/
  fusion/
  rendering/
  metrics/
  recording/
```

Core classes:

- `SceneConfig`
- `DatasetPreparer`
- `FrameExtractor`
- `ColmapRunner`
- `GaussianTrainer`
- `TextTo3DAssetGenerator`
- `ImageTo3DAssetGenerator`
- `MeshExporter`
- `AssetNormalizer`
- `FusionRenderer`
- `SceneMetricEvaluator`
- `ExperimentRecorder`

ACT package:

```text
src/cvhw3_act/
  config.py
  cli.py
  data/
  training/
  evaluation/
  metrics/
  recording/
```

Core classes:

- `ActConfig`
- `CalvinDatasetBuilder`
- `EnvironmentMixer`
- `ActTrainer`
- `ActEvaluator`
- `SwanLabLogger`
- `MetricAggregator`
- `ExperimentRecorder`

Every experiment command should:

- load a YAML config,
- validate required paths,
- set seed,
- copy the config into the run directory,
- record the current Git commit,
- write metrics to JSON/CSV,
- append a summary to `draft.md`,
- log curves and artifacts to SwanLab when enabled.

## 6. Task 1 Required Baseline

### 6.1 Object A: Real Multi-view Reconstruction

Input:

- User-provided phone video or multi-view image set.

Pipeline:

1. Extract frames if video is provided.
2. Filter blurry frames if necessary.
3. Run COLMAP feature extraction, matching, sparse reconstruction, and image undistortion.
4. Train 2DGS on the prepared object sequence.
5. Render held-out views if available.
6. Export qualitative images and metrics.

Required analysis:

- COLMAP pose quality.
- reconstruction geometry quality.
- texture fidelity.
- training time.
- GPU memory.
- failure cases such as reflective surfaces, thin structures, or poor coverage.

### 6.2 Background Scene Reconstruction

Default dataset:

- Mip-NeRF 360 `garden`.

Fallbacks:

- `bicycle`
- `counter`

Pipeline:

1. Prepare dataset into 2DGS-compatible layout.
2. Train 2DGS baseline.
3. Render validation views.
4. Compute PSNR, SSIM, and LPIPS where possible.
5. Export representative renderings for the report.

### 6.3 Object B: Text-to-3D Generation

Tool:

- threestudio with SDS loss.

Baseline:

- One fixed prompt, one fixed seed, default model settings.

Required outputs:

- generated 3D asset,
- rendered turntable,
- exported mesh or compatible representation,
- runtime and GPU memory.

Required analysis:

- geometry plausibility,
- texture completeness,
- prompt adherence,
- SDS artifacts,
- comparison with Object A and Object C.

### 6.4 Object C: Single-image-to-3D Generation

Input:

- User-provided single photo of a real object.

Pipeline:

1. Remove background manually or with a segmentation model.
2. Prepare foreground image.
3. Run Magic123.
4. Export 3D model.
5. Render multi-view inspection images.

Required analysis:

- front-view fidelity,
- back-side hallucination quality,
- texture consistency,
- sensitivity to background removal.

### 6.5 Scene Fusion And Rendering

Required:

- Insert Objects A, B, and C into the reconstructed background scene.
- Use reasonable scale, orientation, and location.
- Render a multi-view walkthrough video.

Primary fusion route:

- Export generated objects as textured meshes.
- Normalize object scale and coordinate frames.
- Compose in Blender or a scripted renderer.
- Render final images and video.

Extension fusion route:

- Sample meshes into point clouds.
- Convert or approximate them as Gaussian-like primitives if feasible.
- Compare against mesh-based composition.

Required report focus:

- Explain how implicit/mesh assets from AIGC are unified with explicit Gaussian scene representations.
- Explain why the chosen fusion route is technically valid.

## 7. Task 1 Extension Experiments

The goal is to generate enough evidence for a rich 35+ page report.

2DGS reconstruction extensions:

- training iterations: short, medium, long;
- input resolution: low vs high;
- COLMAP frame sampling rate;
- COLMAP matcher choice where relevant;
- Gaussian density/pruning settings where supported;
- background scene choice if compute permits.

Object A extensions:

- compare video-frame sampling intervals;
- compare all frames vs filtered frames;
- compare object surface types if additional captures exist.

Object B extensions:

- prompt variants;
- random seed variants;
- SDS guidance scale;
- mesh resolution;
- texture export setting;
- object category difficulty.

Object C extensions:

- manual background removal vs automatic mask;
- Magic123 guidance settings;
- camera prior settings;
- mesh simplification vs full mesh.

Fusion extensions:

- mesh composition vs point/Gaussian conversion;
- object scale variants;
- lighting variants;
- camera path variants;
- close-up inspection renders;
- final video at different resolutions.

Metrics:

- PSNR;
- SSIM;
- LPIPS;
- COLMAP reprojection error;
- mesh vertex/face count;
- Gaussian count;
- render FPS;
- training time;
- peak GPU memory;
- qualitative failure tags.

## 8. Task 1 Mathematical Notes To Record

The report must include method-level mathematical explanation. The following should be recorded in `draft.md` during implementation:

- camera projection and reprojection error in COLMAP;
- 2D Gaussian parameterization;
- Gaussian projection to image plane;
- alpha compositing for differentiable splatting;
- photometric reconstruction objective;
- SDS objective and gradient intuition;
- Magic123 single-view prior and multi-view consistency;
- coordinate normalization for fusion;
- rigid transform and scale alignment for inserted objects.

## 9. Task 2 Required Baseline

Tool:

- LeRobot ACT.

Dataset:

- CALVIN.

Required training:

1. Train ACT using only environment B.
2. Train ACT using mixed environments A, B, and C.
3. Use the same ACT architecture and hyperparameters for the two required models.
4. Evaluate both models zero-shot on unseen environment D.

Required comparisons:

- Action L1 Loss curves.
- validation curves.
- zero-shot success rate or action error.
- convergence speed.
- robustness under visual distribution shift.

Required report focus:

- Explain ACT architecture.
- Explain action chunking.
- Analyze why action chunking may help or fail under cross-environment visual distribution shift.

## 10. Task 2 Extension Experiments

Main extensions:

- run at least 3 seeds for environment-B-only and A+B+C models;
- action chunk size ablation: small, default, large;
- dataset mixing ablation: natural ratio vs balanced environment sampling;
- visual augmentation ablation: none vs color jitter vs crop/resize;
- training schedule ablation: default learning rate vs lower learning rate vs longer training;
- robustness evaluation on environment D with brightness/color/noise perturbations.

Optional extensions if supported:

- camera view ablation;
- proprioception on/off ablation;
- transformer depth/width ablation;
- temporal horizon ablation;
- action normalization ablation.

Metrics:

- training Action L1 Loss;
- validation Action L1 Loss;
- success rate;
- action error;
- episode length;
- convergence steps;
- seed mean/std;
- failure mode counts.

## 11. Task 2 Mathematical Notes To Record

The final report should explain:

- ACT action chunking formulation;
- L1 action loss;
- CVAE-style latent variable objective if used by the chosen LeRobot ACT implementation;
- temporal ensembling or chunk aggregation if used;
- how multi-environment training changes empirical risk;
- visual distribution shift from A/B/C to D;
- why larger or smaller chunks may affect robustness.

## 12. Experiment Logging Protocol

Every completed experiment must append to `draft.md`.

Template:

```markdown
## EXP-ID: short_name

- Date:
- Git commit:
- Task:
- Config:
- Command:
- Dataset:
- GPU:
- Seed:
- Runtime:
- Checkpoint:
- SwanLab run:
- Output artifacts:

### Metrics

| Metric | Value |
|---|---:|

### Observations

### Failure Cases

### Report Notes

### Mathematical / Method Notes
```

Also maintain `docs/experiment_registry.md`:

```markdown
| ID | Task | Purpose | Config | Commit | Status | Key Result |
|---|---|---|---|---|---|---|
```

No experiment is considered complete until:

- metrics are saved,
- SwanLab curves are logged or exported,
- `draft.md` is updated,
- artifacts are named and indexed,
- changes are committed and pushed.

## 13. SwanLab Tracking

Use SwanLab, not WandB, as the primary visualization platform.

Each training run should log:

- training loss;
- validation loss;
- task-specific metrics;
- learning rate;
- batch size;
- seed;
- runtime;
- GPU memory if easy to capture;
- final summary metrics.

Task 1 should additionally log:

- rendered images;
- metric curves;
- reconstruction quality snapshots;
- final video preview or selected frames.

Task 2 should additionally log:

- Action L1 Loss;
- success rate or action error;
- environment split;
- chunk size;
- seed;
- robustness condition.

All SwanLab figures used in the report should also be exported locally under `reports/figures/` or `assets/`.

## 14. README Requirements

The root `README.md` must include:

- assignment overview;
- repository structure;
- environment setup commands;
- link to `environment.md`;
- data preparation instructions;
- Object A and Object C capture instructions;
- Mip-NeRF 360 preparation;
- CALVIN preparation;
- train commands;
- evaluation commands;
- rendering commands;
- report build command;
- model weight cloud-drive link;
- GitHub repository link;
- note about ignored large files.

Required command categories:

- `prepare-scene-data`;
- `run-colmap`;
- `train-2dgs-object`;
- `train-2dgs-background`;
- `generate-text3d`;
- `generate-image3d`;
- `fuse-scene`;
- `render-video`;
- `train-act-b`;
- `train-act-abc`;
- `eval-act-d`;
- `summarize-results`;
- `build-report`.

## 15. Final Report Plan

The final report must be written in English with a two-column academic template.

Target:

- at least 35 pages;
- dense with real experiments, tables, figures, and analysis;
- not padded with filler text.

Required sections:

1. Abstract.
2. Introduction.
3. Related Work.
4. Datasets and Assets.
5. Method: COLMAP and 2DGS.
6. Method: Text-to-3D with SDS.
7. Method: Single-image-to-3D with Magic123.
8. Method: Scene Fusion and Representation Unification.
9. Task 1 Experimental Setup.
10. Task 1 Results and Ablations.
11. Method: LeRobot ACT.
12. Method: Action Chunking and Cross-environment Generalization.
13. Task 2 Experimental Setup.
14. Task 2 Results and Ablations.
15. Discussion.
16. Limitations.
17. Conclusion.
18. Appendix.

Mandatory report artifacts:

- GitHub repository link;
- cloud-drive weight link;
- SwanLab charts;
- hyperparameter tables;
- metric tables;
- qualitative reconstruction figures;
- fused scene frames;
- video frame sequence;
- ACT convergence curves;
- zero-shot comparison table;
- extension experiment analysis;
- mathematical derivations.

## 16. Verification Plan

Before each push:

```bash
git status --short --branch
git diff
ruff check .
pytest
```

Also run:

- import checks for both packages;
- config validation;
- secret scan;
- large-file staged-file scan;
- LaTeX compile check when report files change.

Smoke tests:

- scene CLI help;
- ACT CLI help;
- config schema load;
- mocked scene dry run;
- mocked ACT training loop;
- experiment recorder append test;
- metrics JSON writer test.

Acceptance criteria:

- all required PDF items are mapped to repo artifacts;
- both task baselines run end to end;
- every completed experiment is logged in `draft.md`;
- SwanLab charts exist;
- README has copy-pasteable commands;
- final report compiles;
- GitHub repo is public;
- best weights are uploaded and linked.

## 17. Pending Inputs

Still needed from the user:

- Object A phone video or multi-view photos.
- Object C single object photo.
- cloud-drive choice and final weight link.
- GitHub push authentication confirmation if HTTPS proxy remains unavailable.

Defaults:

- tracking: SwanLab;
- background scene: Mip-NeRF 360 `garden`;
- fallback scenes: `bicycle`, `counter`;
- primary fusion route: textured mesh composition;
- experiment scale: A6000 strengthened version;
- baseline completion before optional expansions.
