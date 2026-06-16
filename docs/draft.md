# Task 1 Draft Log

This file is append-only for Task 1 experiment evidence. Every completed formal experiment must be appended here, including failures and worse variants.

## Required Entry Fields

- Date:
- Phase:
- Run ID:
- Goal:
- Command:
- Config:
- Hardware:
- Elapsed time:
- Result:
- Metrics:
- Figure paths:
- Video paths:
- Cause analysis:
- Next step:


## Phase2 - background_2dgs_low

- Date: 2026-06-01 01:04:14 UTC
- Phase: Phase2
- Run ID: background_2dgs_low
- Goal: Mip-NeRF360 garden background 2DGS low-iteration ablation training
- Command: `nohup python external/2d-gaussian-splatting/train.py -s data/scene/background/garden/colmap -m runs/scene/background_2dgs_low --iterations 5000 --eval > logs/background_2dgs_low_cuda1.log 2>&1 &`
- Config: `configs/scene/background_2dgs_low.yaml`
- Hardware: cuda:1 inferred from log filename; exact gpu.txt unavailable because this run was launched outside RunManager
- Elapsed time: 2026-05-31 12:11:14 to 12:18:44 log timestamps; train progress 5m52s
- Result: success: training completed and saved iteration_5000 checkpoint; render/eval not run yet
- Metrics: iterations=5000; cameras=185; initial_points=138766; final_points=2155064; final_loss=0.06974; checkpoint=runs/scene/background_2dgs_low/point_cloud/iteration_5000/point_cloud.ply; checkpoint_size=525837125 bytes; PSNR/SSIM/LPIPS=pending
- Figure paths: none yet; render.py output pending
- Video paths: none
- Cause analysis: The valid evidence is logs/background_2dgs_low_cuda1.log plus runs/scene/background_2dgs_low. outputs/bg_garden_5k/log.txt is an older failed attempt and must not be used as the experiment result.
- Next step: Run render.py with --skip_train --skip_mesh for test renders, then metrics.py to produce results.json/per_view.json; after metrics exist, update checklist and comparison table.

## Phase2 - bg_garden_5k_eval

- Date: 2026-06-01 01:23:59 UTC
- Phase: Phase2
- Run ID: bg_garden_5k_eval
- Goal: Render held-out Mip-NeRF360 garden views and compute PSNR/SSIM/LPIPS for background_2dgs_low
- Command: `nohup bash scripts/run_2dgs_eval.sh --config configs/scene/background_2dgs_low.yaml --run_id bg_garden_5k_eval --device cuda:1 > outputs/bg_garden_5k_eval/nohup.log 2>&1 &`
- Config: `configs/scene/background_2dgs_low.yaml`
- Hardware: requested device cuda:1; RunManager sets CUDA_VISIBLE_DEVICES=1 so 2DGS internal cuda:0 maps to physical GPU1
- Elapsed time: render plus metrics completed; exact wall time not parsed from nohup, metrics progress about 1m09s after LPIPS weight download
- Result: success: held-out renders and metrics generated for iteration 5000
- Metrics: ours_5000: SSIM=0.7677450776100159, PSNR=25.190664291381836, LPIPS=0.25556084513664246; rendered files=72 under runs/scene/background_2dgs_low/test/ours_5000; results_json=runs/scene/background_2dgs_low/results.json
- Figure paths: runs/scene/background_2dgs_low/test/ours_5000/renders and runs/scene/background_2dgs_low/test/ours_5000/gt
- Video paths: none
- Cause analysis: The earlier FileExistsError was a harness/run-directory precreation conflict from shell redirection; it was fixed in RunManager. The eval run then completed and downloaded LPIPS VGG weights into outputs/bg_garden_5k_eval/cache.
- Next step: Record mid/high background ablations or select low as provisional baseline only after comparing quality; do not mark R1 complete until low/mid/high trend is available.

## Phase8 - background_2dgs_low_report_materials

- Date: 2026-06-01 01:27:51 UTC
- Phase: Phase8
- Run ID: background_2dgs_low_report_materials
- Goal: Collect report-ready metrics table and GT/render preview for background_2dgs_low
- Command: `python scripts/collect_2dgs_metrics.py --model background_2dgs_low=runs/scene/background_2dgs_low --out reports/tables/background_2dgs_metrics.csv; python scripts/make_2dgs_preview.py --test_dir runs/scene/background_2dgs_low/test/ours_5000 --out reports/figures/background_2dgs_low_preview.jpg --count 4 --width 320`
- Config: `configs/scene/background_2dgs_low.yaml`
- Hardware: CPU/PIL report-material generation; no GPU required
- Elapsed time: under 1 minute
- Result: success: CSV table and preview image generated
- Metrics: CSV row: background_2dgs_low ours_5000 PSNR=25.190664291381836 SSIM=0.7677450776100159 LPIPS=0.25556084513664246 render_count=24 gt_count=24
- Figure paths: reports/figures/background_2dgs_low_preview.jpg
- Video paths: none
- Cause analysis: Generated after bg_garden_5k_eval produced results.json and held-out render/gt images; preview confirms render/GT pairing is visually valid.
- Next step: Run mid/high background training ablations, then regenerate the same CSV with all variants and choose the best background.

## Phase2 - smoke_bg_2dgs_50iter

- Date: 2026-06-01 01:31:37 UTC
- Phase: Phase2
- Run ID: smoke_bg_2dgs_50iter
- Goal: Smoke-test background 2DGS training startup on Mip-NeRF360 garden before mid/high formal ablations
- Command: `bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_smoke.yaml --run_id smoke_bg_2dgs_50iter --device cuda:1`
- Config: `configs/scene/background_2dgs_smoke.yaml`
- Hardware: cuda:1; RunManager selected_device=cuda:1
- Elapsed time: 2026-06-01 01:30:04 to 01:31:08 log timestamps; training loop 50 iterations in about 3 seconds after data loading
- Result: success: 50/50 iterations completed and iteration_50 checkpoint saved
- Metrics: cameras=185; initial_points=138766; final_loss=0.21488; checkpoint=runs/scene/background_2dgs_smoke_50iter/point_cloud/iteration_50/point_cloud.ply; checkpoint_size=33860412 bytes; PSNR/SSIM/LPIPS not computed for smoke
- Figure paths: none
- Video paths: none
- Cause analysis: This confirms external 2DGS train.py, dataset path, CUDA device mapping, RunManager logging, and checkpoint writing work before launching longer mid/high background ablations.
- Next step: User should launch formal mid/high background runs with run_background_2dgs.sh; agent will evaluate and collect metrics after logs/checkpoints are available.

## Phase3 - object_a_frame_sampling

- Date: 2026-06-01 01:35:18 UTC
- Phase: Phase3
- Run ID: object_a_frame_sampling
- Goal: Extract sparse/medium/dense frame sets from Object A phone video for COLMAP frame-sampling ablation
- Command: `bash scripts/run_object_a_frames.sh --config configs/scene/object_a_frames_sparse.yaml --run_id object_a_frames_sparse --device cpu; bash scripts/run_object_a_frames.sh --config configs/scene/object_a_frames_medium.yaml --run_id object_a_frames_medium --device cpu; bash scripts/run_object_a_frames.sh --config configs/scene/object_a_frames_dense.yaml --run_id object_a_frames_dense --device cpu`
- Config: `configs/scene/object_a_frames_sparse.yaml; configs/scene/object_a_frames_medium.yaml; configs/scene/object_a_frames_dense.yaml`
- Hardware: CPU ffmpeg preprocessing
- Elapsed time: each extraction completed in about 1.3 seconds per ffmpeg log
- Result: success: sparse/medium/dense frame directories generated
- Metrics: sparse fps=1 frame_count=49; medium fps=2 frame_count=99; dense fps=3 frame_count=148; source video duration=49.37s, 720x1280, 19.99fps; table=reports/tables/object_a_frame_counts.csv
- Figure paths: reports/figures/object_a_frames_medium_contact.jpg
- Video paths: data/scene/object_a/raw/object_a.mp4
- Cause analysis: Frame extraction is complete and lightweight. Medium set gives a practical COLMAP starting point; some late frames are highly tilted/close-up and may need filtering if COLMAP registration drops.
- Next step: Run COLMAP smoke on a tiny subset, then launch formal COLMAP medium/dense/sparse reconstructions and compare registered images/sparse points.

## Phase3 - smoke_object_a_colmap_sampling

- Date: 2026-06-01 01:41:37 UTC
- Phase: Phase3
- Run ID: smoke_object_a_colmap_sampling
- Goal: Smoke-test Object A COLMAP reconstruction on tiny frame subsets before formal sparse/medium/dense runs
- Command: `python scripts/stage_frame_subset.py --src data/scene/object_a/frames_medium --dst data/scene/object_a/frames_colmap_smoke --count 12; bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_smoke.yaml --run_id smoke_object_a_colmap_12f_cpu_v3 --device cpu; python scripts/stage_frame_subset.py --src data/scene/object_a/frames_medium --dst data/scene/object_a/frames_colmap_smoke_head --count 20 --start 0 --stride 1; bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_smoke_head.yaml --run_id smoke_object_a_colmap_20f_head_cpu --device cpu`
- Config: `configs/scene/object_a_colmap_smoke.yaml; configs/scene/object_a_colmap_smoke_head.yaml`
- Hardware: CPU COLMAP via scripts/colmap_clean_env.sh; system COLMAP 3.6 without CUDA
- Elapsed time: uniform 12-frame smoke about 1.29 min; consecutive 20-frame smoke about 0.99 min
- Result: mixed: uniform 12-frame smoke failed to find a good initial pair; consecutive 20-frame smoke produced a valid sparse model
- Metrics: uniform subset: no sparse/0 model; consecutive head subset: cameras=1, images=12, registered_images=12, sparse_points=671, observations=1962, mean_track_length=2.923994, mean_observations_per_image=163.5, mean_reprojection_error=1.014050px
- Figure paths: reports/figures/object_a_frames_medium_contact.jpg
- Video paths: none
- Cause analysis: Object A video has useful overlap when sampled continuously, while widely spaced frames can be too dissimilar. Formal sparse/medium/dense ablation should expect registration rate to be the key criterion, not only frame count.
- Next step: Run formal COLMAP sparse/medium/dense reconstructions; start with medium. If registration drops, create a filtered frame set that removes highly tilted/close-up frames.

## Phase2 - bg_garden_30k_failed_port_conflict

- Date: 2026-06-01 01:46:57 UTC
- Phase: Phase2
- Run ID: bg_garden_30k_failed_port_conflict
- Goal: Background 2DGS high 30k training attempt on garden
- Command: `nohup bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_high.yaml --run_id bg_garden_30k --device cuda:2 --resume > outputs/bg_garden_30k/nohup.log 2>&1 &`
- Config: `configs/scene/background_2dgs_high.yaml`
- Hardware: requested cuda:2; run failed before CUDA training because network GUI port was occupied
- Elapsed time: failed immediately before data loading; no model directory created
- Result: failed: train.py raised OSError Errno 98 Address already in use from gaussian_renderer/network_gui.py listener.bind
- Metrics: no checkpoint; runs/scene/background_2dgs_high missing; outputs/bg_garden_30k/log.txt contains duplicated port bind failure
- Figure paths: none
- Video paths: none
- Cause analysis: 2DGS train.py defaults to GUI port 6009. bg_garden_15k was already running and occupied that port, so the concurrent bg_garden_30k launch exited before training.
- Next step: Retry high run with updated config training.port=6011 and a fresh run_id to avoid stale RunManager metadata, while bg_garden_15k continues on its existing port.

## Phase2 - bg_garden_15k

- Date: 2026-06-01 03:40:19 UTC
- Phase: Phase2
- Run ID: bg_garden_15k
- Goal: background 2DGS mid iteration training for R1 ablation
- Command: `nohup bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_mid.yaml --run_id bg_garden_15k --device cuda:1 > outputs/bg_garden_15k/nohup.log 2>&1 &`
- Config: `configs/scene/background_2dgs_mid.yaml`
- Hardware: cuda:1, 48GB GPU, 185 garden cameras
- Elapsed time: about 22 minutes from log progress plus data loading
- Result: success; Training complete; checkpoint saved at runs/scene/background_2dgs_mid/point_cloud/iteration_15000/point_cloud.ply (595815105 bytes)
- Metrics: train log includes ITER 7000 test PSNR 25.7292; full render metrics pending eval run
- Figure paths: pending mid eval render preview
- Video paths: none
- Cause analysis: 15k run completed without the previous 6009 port conflict; current config now has a distinct port for future reruns
- Next step: run eval-2dgs for bg_garden_15k_eval, collect PSNR/SSIM/LPIPS, generate preview, then compare with 5k and 30k

## Phase3 - object_a_colmap_medium

- Date: 2026-06-01 03:40:19 UTC
- Phase: Phase3
- Run ID: object_a_colmap_medium
- Goal: Object A medium frame COLMAP reconstruction
- Command: `nohup bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_medium.yaml --run_id object_a_colmap_medium --device cpu > outputs/object_a_colmap_medium/nohup.log 2>&1 &`
- Config: `configs/scene/object_a_colmap_medium.yaml`
- Hardware: CPU COLMAP 3.6 without CUDA; medium frame set with 99 extracted frames
- Elapsed time: 6.583 minutes reported by COLMAP log
- Result: success; sparse/0 is the main model with 87 registered images out of 99, 9837 sparse points; sparse/1 is a small secondary model with 12 registered images
- Metrics: sparse/0: cameras=1, images=87, points=9837, observations=41228, mean_track_length=4.191115, mean_observations_per_image=473.885057, mean_reprojection_error=1.126519px
- Figure paths: reports/figures/object_a_frames_medium_contact.jpg
- Video paths: none
- Cause analysis: medium frame sampling gives enough overlap for COLMAP; prior uniform 12-frame smoke failed, while head subset and full medium succeeded
- Next step: use sparse/0 as Object A COLMAP input for Object A 2DGS; sparse/dense formal ablation outputs are still not present and should be run later if time permits

## Phase2 - bg_garden_30k_port6011

- Date: 2026-06-01 03:40:19 UTC
- Phase: Phase2
- Run ID: bg_garden_30k_port6011
- Goal: background 2DGS high iteration final/R1 ablation candidate
- Command: `nohup bash scripts/run_background_2dgs.sh --config configs/scene/background_2dgs_high.yaml --run_id bg_garden_30k_port6011 --device cuda:2 > outputs/bg_garden_30k_port6011/nohup.log 2>&1 &`
- Config: `configs/scene/background_2dgs_high.yaml`
- Hardware: cuda:2, 48GB GPU, 185 garden cameras
- Elapsed time: about 44 minutes from log progress plus data loading
- Result: success; Training complete; checkpoint saved at runs/scene/background_2dgs_high/point_cloud/iteration_30000/point_cloud.ply (596359713 bytes)
- Metrics: train log ITER 30000 test PSNR 26.6557, train PSNR 29.9007; full render metrics pending eval run
- Figure paths: pending high eval render preview
- Video paths: none
- Cause analysis: rerun with distinct GUI/network port resolved the previous address-in-use failure
- Next step: run eval-2dgs for bg_garden_30k_eval, collect PSNR/SSIM/LPIPS, and choose background checkpoint for fusion

## Phase2 - bg_garden_15k_eval

- Date: 2026-06-01 03:46:29 UTC
- Phase: Phase2
- Run ID: bg_garden_15k_eval
- Goal: Evaluate background 2DGS 15k checkpoint with held-out renders and metrics
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/background_2dgs_mid.yaml --run_id bg_garden_15k_eval --device cuda:1 --resume`
- Config: `configs/scene/background_2dgs_mid.yaml`
- Hardware: cuda:1 outside sandbox for CUDA access; 24 held-out test views
- Elapsed time: about 2 minutes 15 seconds for metric loop after LPIPS weight download
- Result: success; renders and GT stored under runs/scene/background_2dgs_mid/test/ours_15000; metrics written to results.json
- Metrics: PSNR=26.206167221069336, SSIM=0.8208866119384766, LPIPS=0.18119029700756073, renders=24, gt=24
- Figure paths: reports/figures/background_2dgs_mid_preview.jpg
- Video paths: none
- Cause analysis: 15k improves substantially over 5k. Initial sandbox eval failed because torch saw zero CUDA devices; rerun outside sandbox succeeded.
- Next step: compare with 30k and use best checkpoint for background fusion

## Phase2 - bg_garden_30k_eval

- Date: 2026-06-01 03:46:29 UTC
- Phase: Phase2
- Run ID: bg_garden_30k_eval
- Goal: Evaluate background 2DGS 30k checkpoint with held-out renders and metrics
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/background_2dgs_high.yaml --run_id bg_garden_30k_eval --device cuda:2 --resume`
- Config: `configs/scene/background_2dgs_high.yaml`
- Hardware: cuda:2 outside sandbox for CUDA access; 24 held-out test views
- Elapsed time: about 2 minutes 15 seconds for metric loop after LPIPS weight download
- Result: success; renders and GT stored under runs/scene/background_2dgs_high/test/ours_30000; metrics written to results.json
- Metrics: PSNR=26.650115966796875, SSIM=0.8336549401283264, LPIPS=0.1637781411409378, renders=24, gt=24
- Figure paths: reports/figures/background_2dgs_high_preview.jpg
- Video paths: none
- Cause analysis: 30k is the best background among 5k/15k/30k on PSNR, SSIM, and LPIPS, so use high checkpoint unless visual inspection finds unacceptable artifacts.
- Next step: mark background_2dgs and R1 complete; proceed to Object A 2DGS using COLMAP sparse/0

## Phase2 - background_2dgs_ablation_materials

- Date: 2026-06-01 03:46:29 UTC
- Phase: Phase2
- Run ID: background_2dgs_ablation_materials
- Goal: Collect background 2DGS ablation table and preview figures
- Command: `python scripts/collect_2dgs_metrics.py --model background_2dgs_low=runs/scene/background_2dgs_low --model background_2dgs_mid=runs/scene/background_2dgs_mid --model background_2dgs_high=runs/scene/background_2dgs_high --out reports/tables/background_2dgs_metrics.csv; python scripts/make_2dgs_preview.py for mid/high`
- Config: `configs/scene/background_2dgs_low.yaml, configs/scene/background_2dgs_mid.yaml, configs/scene/background_2dgs_high.yaml`
- Hardware: CPU table/preview scripts; source renders from completed GPU evals
- Elapsed time: under 5 seconds for CSV and preview generation
- Result: success; background R1 table and preview figures generated
- Metrics: 5k: PSNR=25.1907 SSIM=0.7677 LPIPS=0.2556; 15k: PSNR=26.2062 SSIM=0.8209 LPIPS=0.1812; 30k: PSNR=26.6501 SSIM=0.8337 LPIPS=0.1638
- Figure paths: reports/figures/background_2dgs_low_preview.jpg, reports/figures/background_2dgs_mid_preview.jpg, reports/figures/background_2dgs_high_preview.jpg, reports/tables/background_2dgs_metrics.csv
- Video paths: none
- Cause analysis: Metrics improve monotonically from 5k to 30k; 30k is selected as current background checkpoint.
- Next step: use runs/scene/background_2dgs_high for fusion background and README/report tables

## Phase3 - object_a_colmap_medium_dense_stage

- Date: 2026-06-01 03:53:00 UTC
- Phase: Phase3
- Run ID: object_a_colmap_medium_dense_stage
- Goal: Prepare Object A COLMAP output for 2DGS by undistorting sparse/0 model
- Command: `scripts/colmap_clean_env.sh image_undistorter --image_path data/scene/object_a/frames_medium --input_path runs/scene/object_a_colmap_medium/colmap/sparse/0 --output_path runs/scene/object_a_colmap_medium/colmap/dense --output_type COLMAP`
- Config: `configs/scene/object_a_colmap_medium.yaml`
- Hardware: CPU COLMAP image_undistorter
- Elapsed time: 0.013 minutes reported by COLMAP
- Result: success; generated dense/images for 87 registered views and dense/sparse model; copied sparse files into dense/sparse/0 for 2DGS reader compatibility
- Metrics: dense model analyzer matches sparse/0: registered_images=87, points=9837, mean_reprojection_error=1.126519px
- Figure paths: reports/tables/object_a_colmap_stats.csv
- Video paths: none
- Cause analysis: 2DGS expects source_scene/images and source_scene/sparse/0 with PINHOLE/SIMPLE_PINHOLE camera model; original COLMAP workspace only had sparse output with SIMPLE_RADIAL model.
- Next step: run Object A 2DGS smoke and then formal full/half training

## Phase3 - object_a_2dgs_smoke_50iter

- Date: 2026-06-01 03:53:00 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_smoke_50iter
- Goal: Smoke test Object A 2DGS training on prepared COLMAP dense scene
- Command: `bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_smoke.yaml --run_id object_a_2dgs_smoke_50iter --device cuda:1`
- Config: `configs/scene/object_a_2dgs_smoke.yaml`
- Hardware: cuda:1 outside sandbox for CUDA access
- Elapsed time: about 5 seconds after camera loading
- Result: success; 50 iteration checkpoint saved at runs/scene/object_a_2dgs_smoke/point_cloud/iteration_50/point_cloud.ply
- Metrics: read 87 cameras; initial points=9837; final smoke loss=0.14800; checkpoint size=2401734 bytes
- Figure paths: none at train stage
- Video paths: none
- Cause analysis: Prepared dense scene is readable by 2DGS; named resolution half correctly maps to -r 2.
- Next step: run smoke eval/render, then provide formal Object A 2DGS full/half nohup commands

## Phase3 - object_a_2dgs_smoke_eval

- Date: 2026-06-01 03:53:00 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_smoke_eval
- Goal: Smoke test Object A 2DGS render and metrics path
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/object_a_2dgs_smoke.yaml --run_id object_a_2dgs_smoke_eval --device cuda:1`
- Config: `configs/scene/object_a_2dgs_smoke.yaml`
- Hardware: cuda:1 outside sandbox for CUDA access; 11 held-out test views
- Elapsed time: about 1 minute 5 seconds including LPIPS weight download
- Result: success; test renders, GT images, and results.json generated for 50 iteration smoke checkpoint
- Metrics: PSNR=16.177278518676758, SSIM=0.6774211525917053, LPIPS=0.5354579091072083, renders=11, gt=11
- Figure paths: reports/figures/object_a_2dgs_smoke_preview.jpg
- Video paths: none
- Cause analysis: Very low iteration render is expectedly blurry, but the pipeline is functional: train, checkpoint, render, and metrics all complete.
- Next step: start formal Object A 2DGS full/half runs; use smoke preview only as a sanity check, not as final quality evidence

## Phase3 - object_a_2dgs_full_interrupted_foreground

- Date: 2026-06-01 03:59:01 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_full_interrupted_foreground
- Goal: Object A full-resolution 2DGS formal run attempt
- Command: `bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_full.yaml --run_id object_a_2dgs_full --device cuda:1 --resume`
- Config: `configs/scene/object_a_2dgs_full.yaml`
- Hardware: cuda:1 foreground terminal run
- Elapsed time: interrupted after about 180/15000 iterations
- Result: failed/interrupted by KeyboardInterrupt; no iteration_15000 checkpoint was produced
- Metrics: read 87 cameras, initial points=9837, training reached about 180/15000 iterations before Ctrl-C
- Figure paths: none
- Video paths: none
- Cause analysis: The command was run in the foreground instead of nohup background mode. Ctrl-C interrupted the wrapper and child train.py before a formal checkpoint could be saved.
- Next step: Rerun the same run_id with nohup and --resume so RunManager reuses the existing outputs directory and logs continue to outputs/object_a_2dgs_full/nohup.log

## Phase3 - object_a_2dgs_half

- Date: 2026-06-01 04:08:03 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_half
- Goal: Object A half-resolution 2DGS formal training for R3 resolution ablation
- Command: `nohup bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_half.yaml --run_id object_a_2dgs_half --device cuda:2 > outputs/object_a_2dgs_half/nohup.log 2>&1 &`
- Config: `configs/scene/object_a_2dgs_half.yaml`
- Hardware: cuda:2, Object A COLMAP medium dense scene with 87 registered views
- Elapsed time: about 4 minutes 32 seconds from log progress plus camera loading
- Result: success; Training complete; checkpoint saved at runs/scene/object_a_2dgs_half/point_cloud/iteration_15000/point_cloud.ply
- Metrics: ITER 7000 test PSNR=22.8334; final checkpoint size=94846016 bytes; final point count around 388707 before save
- Figure paths: none
- Video paths: none
- Cause analysis: Half resolution trains quickly and produces a formal 15k checkpoint; this is a valid candidate for Object A and R3.
- Next step: run eval for object_a_2dgs_half and compare with full after full training completes

## Phase3 - object_a_2dgs_half_eval

- Date: 2026-06-01 04:08:03 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_half_eval
- Goal: Evaluate Object A half-resolution 2DGS checkpoint
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/object_a_2dgs_half.yaml --run_id object_a_2dgs_half_eval --device cuda:2`
- Config: `configs/scene/object_a_2dgs_half.yaml`
- Hardware: cuda:2 outside sandbox; 11 held-out test views
- Elapsed time: about 1 minute 2 seconds for metrics after LPIPS weight download
- Result: success; Object A half renders, GT, results.json, CSV row, and preview generated
- Metrics: PSNR=22.424612045288086, SSIM=0.7944319248199463, LPIPS=0.3245447874069214, renders=11, gt=11
- Figure paths: reports/figures/object_a_2dgs_half_preview.jpg, reports/tables/object_a_2dgs_metrics.csv
- Video paths: none
- Cause analysis: Half-resolution Object A is quantitatively usable; full-resolution comparison is still missing because full training was interrupted earlier.
- Next step: wait for object_a_2dgs_full checkpoint, then run full eval and finalize R3 full-vs-half comparison

## Phase3 - object_a_2dgs_full

- Date: 2026-06-01 04:12:58 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_full
- Goal: Object A full-resolution 2DGS formal training for R3 resolution ablation
- Command: `nohup bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_full.yaml --run_id object_a_2dgs_full --device cuda:1 --resume > outputs/object_a_2dgs_full/nohup.log 2>&1 &`
- Config: `configs/scene/object_a_2dgs_full.yaml`
- Hardware: cuda:1, Object A COLMAP medium dense scene with 87 registered views
- Elapsed time: about 9 minutes 58 seconds from log progress plus camera loading on the successful rerun
- Result: success; Training complete; checkpoint saved at runs/scene/object_a_2dgs_full/point_cloud/iteration_15000/point_cloud.ply
- Metrics: ITER 7000 test PSNR=22.4475; final checkpoint size=72491224 bytes; final point count around 297089 before save
- Figure paths: none
- Video paths: none
- Cause analysis: Foreground interrupted attempt was rerun via nohup/resume; full resolution finished and is available for R3 comparison.
- Next step: run full eval and compare full vs half metrics for Object A selection

## Phase3 - object_a_2dgs_full_eval

- Date: 2026-06-01 04:12:58 UTC
- Phase: Phase3
- Run ID: object_a_2dgs_full_eval
- Goal: Evaluate Object A full-resolution 2DGS checkpoint
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/object_a_2dgs_full.yaml --run_id object_a_2dgs_full_eval --device cuda:1`
- Config: `configs/scene/object_a_2dgs_full.yaml`
- Hardware: cuda:1 outside sandbox; 11 held-out test views
- Elapsed time: about 58 seconds for metrics after LPIPS weight download
- Result: success; Object A full renders, GT, results.json, CSV row, and preview generated
- Metrics: PSNR=21.813936233520508, SSIM=0.841981828212738, LPIPS=0.3548964560031891, renders=11, gt=11
- Figure paths: reports/figures/object_a_2dgs_full_preview.jpg, reports/tables/object_a_2dgs_metrics.csv
- Video paths: none
- Cause analysis: Full has higher SSIM than half but lower PSNR and worse LPIPS. Half remains the current quantitative pick unless visual inspection favors full.
- Next step: inspect previews, choose Object A final checkpoint, and proceed to remaining Phase3 R2 only if time permits

## Phase3 - object_a_colmap_sparse

- Date: 2026-06-01 04:25:26 UTC
- Phase: Phase3
- Run ID: object_a_colmap_sparse
- Goal: R2 sparse frame sampling COLMAP ablation for Object A
- Command: `bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_sparse.yaml --run_id object_a_colmap_sparse --device cpu`
- Config: `configs/scene/object_a_colmap_sparse.yaml`
- Hardware: CPU COLMAP 3.6.0; colmap.num_threads=90 on 128 logical CPUs (~70% target); use_gpu=false
- Elapsed time: 0.364 minutes from COLMAP log
- Result: success with limited sparse-view reconstruction; model exported at runs/scene/object_a_colmap_sparse/colmap/sparse/0
- Metrics: registered_images=35, sparse_points=4130, observations=13892, mean_track_length=3.363680, mean_observations_per_image=396.914286, mean_reprojection_error_px=0.972736; see reports/tables/object_a_colmap_r2_stats.csv
- Figure paths: none
- Video paths: none
- Cause analysis: Sparse sampling is fast but weak for Object A coverage. The log ends with 'No good initial image pair found' after registering a subset, so it is useful as an ablation lower bound rather than the final reconstruction source.
- Next step: Compare against medium and dense frame sampling, then keep the best-coverage COLMAP source for any follow-up Object A rerun.

## Phase3 - object_a_colmap_dense

- Date: 2026-06-01 04:25:35 UTC
- Phase: Phase3
- Run ID: object_a_colmap_dense
- Goal: R2 dense frame sampling COLMAP ablation for Object A
- Command: `bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_dense.yaml --run_id object_a_colmap_dense --device cpu`
- Config: `configs/scene/object_a_colmap_dense.yaml`
- Hardware: CPU COLMAP 3.6.0; colmap.num_threads=90 on 128 logical CPUs (~70% target); use_gpu=false
- Elapsed time: 3.440 minutes from COLMAP log
- Result: success; dense frame sampling produced the largest Object A reconstruction at runs/scene/object_a_colmap_dense/colmap/sparse/0
- Metrics: registered_images=148, sparse_points=24125, observations=114704, mean_track_length=4.754570, mean_observations_per_image=775.027027, mean_reprojection_error_px=1.129602; see reports/tables/object_a_colmap_r2_stats.csv
- Figure paths: none
- Video paths: none
- Cause analysis: Dense sampling costs more time than sparse/medium but gives much stronger coverage, more points, and longer tracks. Reprojection error is slightly higher than sparse but comparable to medium, so dense is the best R2 COLMAP source if later Object A rerun time is available.
- Next step: Mark R2 complete and keep medium-trained 2DGS as current finished Object A result unless time permits a dense-source final rerun.

## Phase3 - object_a_colmap_r2_ablation

- Date: 2026-06-01 04:25:47 UTC
- Phase: Phase3
- Run ID: object_a_colmap_r2_ablation
- Goal: R2 Object A frame sampling comparison: sparse vs medium vs dense
- Command: `python scripts/collect_colmap_stats.py --model object_a_sparse=runs/scene/object_a_colmap_sparse/colmap/sparse/0 --model object_a_medium=runs/scene/object_a_colmap_medium/colmap/sparse/0 --model object_a_dense=runs/scene/object_a_colmap_dense/colmap/sparse/0 --out reports/tables/object_a_colmap_r2_stats.csv`
- Config: `configs/scene/object_a_colmap_sparse.yaml, configs/scene/object_a_colmap_medium.yaml, configs/scene/object_a_colmap_dense.yaml`
- Hardware: CPU-only COLMAP comparison; each config uses colmap.num_threads=90 (~70% of 128 logical CPUs)
- Elapsed time: sparse 0.364 min; medium previously completed; dense 3.440 min
- Result: success; R2 comparison table generated and dense selected as best COLMAP coverage while medium remains the current already-trained 2DGS source
- Metrics: sparse: 35 images/4130 points/0.972736px; medium: 87 images/9837 points/1.126519px; dense: 148 images/24125 points/1.129602px; table=reports/tables/object_a_colmap_r2_stats.csv
- Figure paths: none
- Video paths: none
- Cause analysis: Increasing frame density substantially improves registration count, point count, observations, and track length. Dense is best for reconstruction completeness; sparse is too weak; medium is a practical compromise already used by the finished full/half 2DGS experiments.
- Next step: Use dense as the recommended source for optional final Object A rerun; otherwise proceed to later phases with the completed medium-source Object A 2DGS assets.

## Phase4 - object_b_sds_smoke

- Date: 2026-06-01 04:55:44 UTC
- Phase: Phase4
- Run ID: object_b_sds_smoke
- Goal: 1-step threestudio DreamFusion-SD smoke for Object B
- Command: `HF_HOME=/home/yuhengli/.cache/huggingface CUDA_HOME=/usr/local/cuda-12.4 TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 bash scripts/run_object_b_sds.sh --config configs/scene/text3d_smoke.yaml --run_id object_b_sds_smoke --device cuda:1`
- Config: `configs/scene/text3d_smoke.yaml`
- Hardware: cuda:1 selected by RunManager; RTX A6000; server CUDA 12.4; PyTorch CUDA 12.1; HF cache at /home/yuhengli/.cache/huggingface
- Elapsed time: about 7 seconds before import failure
- Result: failed before training; threestudio import stopped with ModuleNotFoundError: No module named 'igl'
- Metrics: no training step completed; no checkpoint or mesh generated
- Figure paths: none
- Video paths: none
- Cause analysis: The initial dependency probe incorrectly treated igl/libigl as optional, but threestudio imports igl unconditionally from threestudio.utils.ops during startup.
- Next step: Install official hard-import dependencies libigl/envlight/wandb and export helpers, update the environment probe, then rerun the same smoke with --resume or a fresh run_id.

## Phase4 - object_b_sds_smoke_resume_igl_api

- Date: 2026-06-01 04:58:20 UTC
- Phase: Phase4
- Run ID: object_b_sds_smoke_resume_igl_api
- Goal: Resume 1-step threestudio DreamFusion-SD smoke after installing official dependencies
- Command: `HF_HOME=/home/yuhengli/.cache/huggingface CUDA_HOME=/usr/local/cuda-12.4 TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 bash scripts/run_object_b_sds.sh --config configs/scene/text3d_smoke.yaml --run_id object_b_sds_smoke --device cuda:1 --resume`
- Config: `configs/scene/text3d_smoke.yaml`
- Hardware: cuda:1 selected by RunManager; RTX A6000; server CUDA 12.4; PyTorch CUDA 12.1; HF cache at /home/yuhengli/.cache/huggingface
- Elapsed time: about 5 seconds before import API failure
- Result: failed before training; libigl imported but lacks threestudio's expected fast_winding_number_for_meshes/read_obj legacy names
- Metrics: no training step completed; no checkpoint or mesh generated
- Figure paths: none
- Video paths: none
- Cause analysis: Installed libigl 2.6.2 exposes fast_winding_number and readOBJ, while this threestudio commit imports older snake_case names. A narrow compatibility shim is safer than changing external source or downgrading blindly.
- Next step: Add project-side igl compatibility shim through sitecustomize and rerun smoke.

## Phase4 - object_b_sds_smoke_hf_model_access

- Date: 2026-06-01 05:02:13 UTC
- Phase: Phase4
- Run ID: object_b_sds_smoke_hf_model_access
- Goal: Resume 1-step threestudio smoke after startup dependencies and igl shim
- Command: `HF_HOME=/home/yuhengli/.cache/huggingface CUDA_HOME=/usr/local/cuda-12.4 TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 bash scripts/run_object_b_sds.sh --config configs/scene/text3d_smoke.yaml --run_id object_b_sds_smoke --device cuda:1 --resume`
- Config: `configs/scene/text3d_smoke.yaml`
- Hardware: cuda:1 selected by RunManager; RTX A6000; server CUDA 12.4; PyTorch CUDA 12.1; HF cache at /home/yuhengli/.cache/huggingface
- Elapsed time: about 16 seconds before prompt embedding failure
- Result: failed after threestudio startup; prompt embedding process could not access stabilityai/stable-diffusion-2-1-base on HuggingFace
- Metrics: Lightning reached on_fit_start; no training step completed; no checkpoint or mesh generated
- Figure paths: none
- Video paths: none
- Cause analysis: The environment and CUDA import path are now functional, but the selected SD 2.1 base model identifier is unavailable without authentication or is blocked in the current network/cache state.
- Next step: Probe accessible Stable Diffusion model IDs and switch Phase4 configs to an accessible SD checkpoint, then rerun the same 1-step smoke.

## Phase4 - object_b_sds_smoke_manojb_export_fail

- Date: 2026-06-01 05:30:15 UTC
- Phase: Phase4
- Run ID: object_b_sds_smoke_manojb_export_fail
- Goal: 1-step threestudio DreamFusion-SD smoke with accessible Manojb SD2.1 mirror
- Command: `HF_HOME=/home/yuhengli/.cache/huggingface CUDA_HOME=/usr/local/cuda-12.4 TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 bash scripts/run_object_b_sds.sh --config configs/scene/text3d_smoke.yaml --run_id object_b_sds_smoke --device cuda:1 --resume`
- Config: `configs/scene/text3d_smoke.yaml`
- Hardware: cuda:1 RTX A6000; server CUDA 12.4; PyTorch CUDA 12.1; HF cache /home/yuhengli/.cache/huggingface; model Manojb/stable-diffusion-2-1-base
- Elapsed time: about 8 minutes including first model download
- Result: partial success then failure; Stable Diffusion loaded, 1 training step and 120 test-view renders completed, final automatic mesh export failed with ValueError: max() arg is an empty sequence
- Metrics: checkpoint directory created; tensorboard events created; test renders under runs/scene/text3d_smoke/smoke/save/it1-test; no mesh exported
- Figure paths: runs/scene/text3d_smoke/smoke/save/it1-test
- Video paths: none
- Cause analysis: The accessible Manojb SD2.1 mirror works. Failure happened only because --gradio triggers mesh export after a 1-step smoke; the density field has no isosurface yet, so outlier removal receives an empty component list.
- Next step: Disable gradio/export for the 1-step train smoke, use a fresh smoke output directory, then rerun to validate the train path cleanly.

## Phase4 - object_b_sds_smoke_trainonly_sandbox_fail

- Date: 2026-06-01 05:32:14 UTC
- Phase: Phase4
- Run ID: object_b_sds_smoke_trainonly_sandbox_fail
- Goal: Rerun 1-step train-only smoke after disabling automatic export
- Command: `HF_HOME=/home/yuhengli/.cache/huggingface CUDA_HOME=/usr/local/cuda-12.4 TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 bash scripts/run_object_b_sds.sh --config configs/scene/text3d_smoke.yaml --run_id object_b_sds_smoke_trainonly --device cuda:1`
- Config: `configs/scene/text3d_smoke.yaml`
- Hardware: sandboxed command environment; intended cuda:1 RTX A6000
- Elapsed time: about 6 seconds
- Result: failed before training in sandbox; tiny-cuda-nn raised Unknown compute capability after PyTorch warned Can't initialize NVML
- Metrics: no training step; no checkpoint; no threestudio trial directory
- Figure paths: none
- Video paths: none
- Cause analysis: This was an execution-environment failure, not a Phase4 code/model failure. The sandbox can expose nvidia-smi but PyTorch/tiny-cuda-nn could not query compute capability through NVML.
- Next step: Rerun the same train-only smoke outside sandbox with a fresh run_id.

## Phase4 - object_b_sds_smoke_trainonly_real

- Date: 2026-06-01 05:32:27 UTC
- Phase: Phase4
- Run ID: object_b_sds_smoke_trainonly_real
- Goal: Validate Phase4 threestudio DreamFusion-SD train path with accessible SD2.1 mirror
- Command: `HF_HOME=/home/yuhengli/.cache/huggingface CUDA_HOME=/usr/local/cuda-12.4 TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 bash scripts/run_object_b_sds.sh --config configs/scene/text3d_smoke.yaml --run_id object_b_sds_smoke_trainonly_real --device cuda:1`
- Config: `configs/scene/text3d_smoke.yaml`
- Hardware: cuda:1 RTX A6000; server CUDA 12.4; PyTorch CUDA 12.1; model cache Manojb/stable-diffusion-2-1-base at /home/yuhengli/.cache/huggingface
- Elapsed time: about 30 seconds with model already cached
- Result: success; Stable Diffusion loaded, trainer reached max_steps=1, validation ran, 120 test-view renders saved, checkpoints saved
- Metrics: ckpts: last.ckpt and epoch=0-step=1.ckpt, 49M each; 120 PNG test renders at 1536x512; model cache size 4.9G; trial size 114M
- Figure paths: runs/scene/text3d_smoke_trainonly/smoke_trainonly/save/it1-test
- Video paths: none
- Cause analysis: The Phase4 environment, CUDA extension stack, SD2.1 mirror, and threestudio DreamFusion-SD training path are now functional. Export was intentionally disabled for this 1-step smoke because 1 step is too short to produce a valid mesh isosurface.
- Next step: Proceed to formal Object B prompt ablation commands; use gradio/export only for sufficiently trained runs or run export after a trained checkpoint exists.

## Phase4 - object_b_sds_simple_hf_cache_fail

- Date: 2026-06-01 05:42:25 UTC
- Phase: Phase4
- Run ID: object_b_sds_simple_hf_cache_fail
- Goal: Formal Object B SDS simple prompt training on cuda:1
- Command: `nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_simple.yaml --run_id object_b_sds_simple --device cuda:1 > outputs/object_b_sds_simple/nohup.log 2>&1 &`
- Config: `configs/scene/text3d_simple.yaml`
- Hardware: cuda:1 requested; RTX A6000; RunManager set XDG_CACHE_HOME to outputs/object_b_sds_simple/cache; HF_HOME was not set in the original script
- Elapsed time: failed during startup before training steps
- Result: failed; StableDiffusionPipeline tried to reach huggingface.co for Manojb/stable-diffusion-2-1-base and network was unreachable
- Metrics: no training step completed; no checkpoint saved; trial directory created at runs/scene/text3d_simple/main
- Figure paths: none
- Video paths: none
- Cause analysis: The global HF cache from smoke was not visible because HF_HOME was unset and RunManager sets a per-run XDG_CACHE_HOME. The script then used an empty run-local cache and attempted network access.
- Next step: Patch scripts/run_object_b_sds.sh to default HF_HOME=/home/yuhengli/.cache/huggingface, then resume the same run_id on cuda:1.

## Phase4 - object_b_sds_detailed_hf_cache_fail

- Date: 2026-06-01 05:42:39 UTC
- Phase: Phase4
- Run ID: object_b_sds_detailed_hf_cache_fail
- Goal: Formal Object B SDS detailed prompt training on cuda:2
- Command: `nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed.yaml --run_id object_b_sds_detailed --device cuda:2 > outputs/object_b_sds_detailed/nohup.log 2>&1 &`
- Config: `configs/scene/text3d_detailed.yaml`
- Hardware: cuda:2 requested; RTX A6000; RunManager set XDG_CACHE_HOME to outputs/object_b_sds_detailed/cache; HF_HOME was not set in the original script
- Elapsed time: failed during startup before training steps
- Result: failed; prompt embedding subprocess could not find tokenizer/config.json in cache and network access to HuggingFace failed
- Metrics: no training step completed; no checkpoint saved; trial directory created at runs/scene/text3d_detailed/main
- Figure paths: none
- Video paths: none
- Cause analysis: The detailed prompt required text embedding generation. Because HF_HOME was unset, transformers searched the run-local cache instead of the global HF cache populated during smoke.
- Next step: Patch scripts/run_object_b_sds.sh to default HF_HOME=/home/yuhengli/.cache/huggingface, then resume the same run_id on cuda:2.

## Phase4 - object_b_sds_parallel_remote_id_retry_fail

- Date: 2026-06-01 05:49:37 UTC
- Phase: Phase4
- Run ID: object_b_sds_parallel_remote_id_retry_fail
- Goal: Retry simple and detailed Object B SDS runs in parallel on cuda:1/cuda:2
- Command: `nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_simple.yaml --run_id object_b_sds_simple --device cuda:1 --resume > outputs/object_b_sds_simple/nohup.log 2>&1 &; nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed.yaml --run_id object_b_sds_detailed --device cuda:2 --resume > outputs/object_b_sds_detailed/nohup.log 2>&1 &`
- Config: `configs/scene/text3d_simple.yaml, configs/scene/text3d_detailed.yaml`
- Hardware: cuda:1 and cuda:2 requested; both processes stopped before training; GPU1/GPU2 returned to idle
- Elapsed time: about 2 minutes before both exited
- Result: failed; both runs stopped during StableDiffusionPipeline.from_pretrained because the config still used remote model id Manojb/stable-diffusion-2-1-base and the server could not reach huggingface.co
- Metrics: no training step completed; no checkpoint; GPU memory never exceeded CUDA context level around 382MiB
- Figure paths: none
- Video paths: none
- Cause analysis: Setting HF_HOME was not sufficient because diffusers still calls HuggingFace model_info for a remote repo id. Use the local snapshot path directly and force offline mode.
- Next step: Update Phase4 configs to the local HF snapshot path and restart both run_ids with --resume.

## Phase5_ObjectC_Magic123_prep - object_c_magic123_prep

- Date: 2026-06-01 06:19:17 UTC
- Phase: Phase5_ObjectC_Magic123_prep
- Run ID: object_c_magic123_prep
- Goal: Prepare official Magic123 harness and Object C input variants without launching training
- Command: `PYTHONPATH=src python scripts/probe_magic123_env.py --config configs/scene/image3d_smoke.yaml; bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_smoke.yaml --run_id object_c_magic123_smoke --device cuda:1 --dry_run`
- Config: `configs/scene/image3d_smoke.yaml; configs/scene/image3d_raw.yaml; configs/scene/image3d_auto_mask.yaml; configs/scene/image3d_refined_mask.yaml`
- Hardware: no GPU training launched; current probe only
- Elapsed time: <1 min
- Result: corrected Phase5 to official guochengqian/Magic123 entrypoints; Object C raw/auto/refined inputs prepared; dry-run command generated; formal run blocked until external/Magic123 and official pretrained checkpoints are installed
- Metrics: tests: 15 passed; probe missing: magic123_repo, main.py, run scripts, ldm package, zero123 105000.ckpt, midas dpt_beit_large_512.pt
- Figure paths: data/scene/object_c/raw/object_c.jpg; data/scene/object_c/masked/object_c_auto.png; data/scene/object_c/masked/object_c_refined.png
- Video paths: none
- Cause analysis: Previous Phase5 route incorrectly used threestudio Magic123 config. Official project requirement is Magic123 single-image-to-3D, so the harness now points to external/Magic123/main.py and official coarse/fine command structure.
- Next step: Clone external/Magic123, install dependencies in a compatible environment without downloading CUDA, download Zero123 and MiDaS weights, then let the agent run a 1-iter smoke before formal raw/auto/refined runs.

## Phase4_ObjectB_SDS - object_b_sds_simple

- Date: 2026-06-01 06:46:26 UTC
- Phase: Phase4_ObjectB_SDS
- Run ID: object_b_sds_simple
- Goal: Formal Object B SDS simple prompt training and export
- Command: `nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_simple.yaml --run_id object_b_sds_simple --device cuda:1 --resume > outputs/object_b_sds_simple/nohup.log 2>&1 &`
- Config: `configs/scene/text3d_simple.yaml`
- Hardware: RTX A6000 cuda:1 requested; final artifacts under runs/scene/text3d_simple/main
- Elapsed time: about 45-50 min from log progress
- Result: completed 10000 training steps, checkpoint saved, test video rendered, mesh export succeeded
- Metrics: checkpoint: runs/scene/text3d_simple/main/ckpts/last.ckpt; preview: runs/scene/text3d_simple/main/save/it10000-0.png; video: runs/scene/text3d_simple/main/save/it10000-test.mp4; mesh: runs/scene/text3d_simple/main/save/it10000-export/model.obj; texture: runs/scene/text3d_simple/main/save/it10000-export/texture_kd.jpg
- Figure paths: runs/scene/text3d_simple/main/save/it10000-0.png; runs/scene/text3d_simple/main/save/it10000-export/texture_kd.jpg
- Video paths: runs/scene/text3d_simple/main/save/it10000-test.mp4
- Cause analysis: The simple prompt produced a valid isosurface and exportable textured OBJ, so it is a viable Object B candidate pending visual scoring.
- Next step: Inspect preview/video/mesh, assign geometry/texture/fusion-readiness scores, then compare against detailed and style-constrained prompts.

## Phase4_ObjectB_SDS - object_b_sds_detailed

- Date: 2026-06-01 06:46:26 UTC
- Phase: Phase4_ObjectB_SDS
- Run ID: object_b_sds_detailed
- Goal: Formal Object B SDS detailed prompt training and export
- Command: `nohup bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed.yaml --run_id object_b_sds_detailed --device cuda:2 --resume > outputs/object_b_sds_detailed/nohup.log 2>&1 &`
- Config: `configs/scene/text3d_detailed.yaml`
- Hardware: RTX A6000 cuda:2 requested; final artifacts under runs/scene/text3d_detailed/main
- Elapsed time: about 45-50 min from log progress
- Result: completed 10000 training steps and rendered test video, but mesh export failed with ValueError: max() arg is an empty sequence during isosurface outlier removal
- Metrics: checkpoint: runs/scene/text3d_detailed/main/ckpts/last.ckpt; preview: runs/scene/text3d_detailed/main/save/it10000-0.png; video: runs/scene/text3d_detailed/main/save/it10000-test.mp4; mesh export: missing
- Figure paths: runs/scene/text3d_detailed/main/save/it10000-0.png
- Video paths: runs/scene/text3d_detailed/main/save/it10000-test.mp4
- Cause analysis: The detailed prompt produced renderable output but the extracted isosurface was empty or fully removed as outlier components, so no OBJ/texture was generated. This is a prompt/export failure mode to record for R4.
- Next step: Inspect the preview/video for qualitative value; if visually promising, retry export from last.ckpt with relaxed isosurface/outlier settings, otherwise mark detailed as less fusion-ready than simple.

## Phase4_ObjectB_SDS - object_b_sds_detailed_export_retry

- Date: 2026-06-01 06:51:01 UTC
- Phase: Phase4_ObjectB_SDS
- Run ID: object_b_sds_detailed_export_retry
- Goal: Retry detailed Object B SDS mesh export from last.ckpt with outlier removal disabled
- Command: `bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed_export_retry.yaml --run_id object_b_sds_detailed_export_retry --device cuda:2 --resume`
- Config: `configs/scene/text3d_detailed_export_retry.yaml`
- Hardware: RTX A6000 cuda:2 requested; export-only run from existing detailed checkpoint
- Elapsed time: <1 min
- Result: failed; export progressed past outlier removal but coarse-to-fine isosurface produced an empty mesh and raised IndexError: amin() expected non-zero size
- Metrics: checkpoint: runs/scene/text3d_detailed/main/ckpts/last.ckpt; mesh export: still missing after first retry
- Figure paths: none
- Video paths: none
- Cause analysis: Disabling outlier removal alone is insufficient because the coarse isosurface under the default threshold remains empty.
- Next step: Retry export with system.geometry.isosurface_coarse_to_fine=false and system.geometry.isosurface_threshold=auto.

## Phase4_ObjectB_SDS - object_b_sds_detailed_export_retry_auto

- Date: 2026-06-01 06:51:11 UTC
- Phase: Phase4_ObjectB_SDS
- Run ID: object_b_sds_detailed_export_retry_auto
- Goal: Retry detailed Object B SDS mesh export from last.ckpt with auto isosurface threshold
- Command: `bash scripts/run_object_b_sds.sh --config configs/scene/text3d_detailed_export_retry_auto.yaml --run_id object_b_sds_detailed_export_retry_auto --device cuda:2 --resume`
- Config: `configs/scene/text3d_detailed_export_retry_auto.yaml`
- Hardware: RTX A6000 cuda:2 requested; export-only run from existing detailed checkpoint
- Elapsed time: <1 min
- Result: success; detailed prompt checkpoint exported textured OBJ/MTL/texture after disabling coarse-to-fine/outlier removal and using auto isosurface threshold
- Metrics: auto threshold: 0.4878173768520355; vertices: 3560; faces: 7060; obj: runs/scene/text3d_detailed/main/save/it10000-export/model.obj; texture: runs/scene/text3d_detailed/main/save/it10000-export/texture_kd.jpg
- Figure paths: runs/scene/text3d_detailed/main/save/it10000-0.png; runs/scene/text3d_detailed/main/save/it10000-export/texture_kd.jpg
- Video paths: runs/scene/text3d_detailed/main/save/it10000-test.mp4
- Cause analysis: The detailed model had a renderable density field but the default export threshold/coarse pass returned an empty coarse mesh. Auto threshold selected a lower valid level set and produced a usable mesh.
- Next step: Visually score simple/detailed/style-constrained prompts after style-constrained finishes; use the best mesh for Object B fusion.

## Phase5_magic123_env - object_c_magic123_smoke_small

- Date: 2026-06-01 08:31:00 UTC
- Phase: Phase5_magic123_env
- Run ID: object_c_magic123_smoke_small
- Goal: Verify reduced Magic123 smoke after dataset_size wiring
- Command: `bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_smoke.yaml --run_id object_c_magic123_smoke_small --device cuda:1`
- Config: `configs/scene/image3d_smoke.yaml`
- Hardware: sandbox, requested cuda:1, no CUDA access
- Elapsed time: failed during SD load
- Result: failed: Magic123 SD loader attempted Hugging Face API despite local SD1.5 cache; proxy/network unavailable in sandbox
- Metrics: none
- Figure paths: none
- Video paths: none
- Cause analysis: external/Magic123/guidance/sd_utils.py used local_files_only=False, so diffusers called huggingface.co/model_info before local cache use
- Next step: Patch SD loader to honor HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/DIFFUSERS_OFFLINE and retry outside sandbox for CUDA access

## Phase5_magic123_env - object_c_magic123_smoke_small_offline

- Date: 2026-06-01 08:31:12 UTC
- Phase: Phase5_magic123_env
- Run ID: object_c_magic123_smoke_small_offline
- Goal: Verify offline SD1.5 local loading after patch
- Command: `bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_smoke.yaml --run_id object_c_magic123_smoke_small_offline --device cuda:1`
- Config: `configs/scene/image3d_smoke.yaml`
- Hardware: sandbox, requested cuda:1, no CUDA access
- Elapsed time: failed at first CUDA train op
- Result: failed: SD1.5 loaded locally, but sandbox process had no visible CUDA device
- Metrics: dataset_size_train=4,dataset_size_valid=1,dataset_size_test=4 reached argparse; no training metrics
- Figure paths: none
- Video paths: none
- Cause analysis: This run validated the offline loader patch; remaining failure was execution boundary, not Magic123 config
- Next step: Run the same smoke with escalated CUDA access outside sandbox

## Phase5_magic123_env - object_c_magic123_smoke_small_cuda

- Date: 2026-06-01 08:31:29 UTC
- Phase: Phase5_magic123_env
- Run ID: object_c_magic123_smoke_small_cuda
- Goal: Verify Magic123 smoke with gcc11, C++17, offline SD1.5, device mapping, reduced dataset sizes
- Command: `bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_smoke.yaml --run_id object_c_magic123_smoke_small_cuda --device cuda:1`
- Config: `configs/scene/image3d_smoke.yaml`
- Hardware: cuda:1, peak GPU memory reported by Magic123 10.7GB, MAX_JOBS=2, gcc/g++ 11
- Elapsed time: 0.8617 minutes training time; total wall time about 1.5 minutes after model load
- Result: success: one coarse smoke epoch completed; eval 1/1 and test 4/4 progress bars shown; checkpoints/images/videos exported
- Metrics: dataset_size_train=4,dataset_size_valid=1,dataset_size_test=4; checkpoint coarse.pth 57MB; full checkpoint 204MB; SD1.5 cache 5.2GB; torch extension cache 8.6MB
- Figure paths: runs/scene/image3d_smoke_small/smoke/coarse/results/coarse_ep0001_lambertian.jpg; runs/scene/image3d_smoke_small/smoke/coarse/results/coarse_ep0001_normal.jpg
- Video paths: runs/scene/image3d_smoke_small/smoke/coarse/results/coarse_ep0001_lambertian.mp4; runs/scene/image3d_smoke_small/smoke/coarse/results/coarse_ep0001_normal.mp4
- Cause analysis: Compatibility fixes are sufficient for lightweight Phase5 smoke. Mesh directory was created but no mesh file appeared in this 1-iter smoke, likely too few iterations or empty marching-cubes surface; formal runs still need mesh export verification.
- Next step: Prepare Phase5 formal raw/auto/refined mask commands; keep sd_version 1.5 and run outside sandbox on a free GPU

## Phase4_ObjectB_SDS - object_b_sds_prompt_ablation_closeout

- Date: 2026-06-01 08:42:11 UTC
- Phase: Phase4_ObjectB_SDS
- Run ID: object_b_sds_prompt_ablation_closeout
- Goal: Close Object B SDS prompt ablation and select the mesh for fusion
- Command: `posthoc inspection of object_b_sds_simple, object_b_sds_detailed, object_b_sds_style_constrained outputs`
- Config: `configs/scene/text3d_simple.yaml; configs/scene/text3d_detailed.yaml; configs/scene/text3d_style_constrained.yaml`
- Hardware: RTX A6000: simple cuda:1, detailed cuda:2, style-constrained cuda:1
- Elapsed time: simple 49:41 train, detailed 50:03 train, style-constrained 47:39 train; export completed after training
- Result: success: all three prompt variants produced checkpoint, test video, exported OBJ/MTL/1024x1024 texture; style-constrained selected for Phase6 fusion
- Metrics: simple: 33999 vertices, 67786 faces, geometry/texture/fusion=3/3/3; detailed: 3560 vertices, 7060 faces, 2/2/2 and required auto-threshold export retry; style-constrained: 39314 vertices, 78628 faces, 4/3/4; table reports/tables/object_b_sds_prompt_ablation.csv
- Figure paths: runs/scene/text3d_simple/main/save/it10000-0.png; runs/scene/text3d_detailed/main/save/it10000-0.png; runs/scene/text3d_style_constrained/main/save/it10000-0.png; runs/scene/text3d_style_constrained/main/save/it10000-export/texture_kd.jpg
- Video paths: runs/scene/text3d_simple/main/save/it10000-test.mp4; runs/scene/text3d_detailed/main/save/it10000-test.mp4; runs/scene/text3d_style_constrained/main/save/it10000-test.mp4
- Cause analysis: The style-constrained prompt reduced Janus/ghosting and produced the cleanest silhouette and stable handle/body. Simple retained richer color but had more fuzzy surface noise and floating artifacts. Detailed was over-constrained and only exported after an auto-threshold retry, making it less fusion-ready.
- Next step: Use runs/scene/text3d_style_constrained/main/save/it10000-export/model.obj and texture_kd.jpg as Object B in Phase6 fusion; include the prompt ablation table in the report.

## Phase5_magic123_env - object_c_magic123_refined_mask_gpu2_smoke

- Date: 2026-06-01 08:48:57 UTC
- Phase: Phase5_magic123_env
- Run ID: object_c_magic123_refined_mask_gpu2_smoke
- Goal: Verify refined_mask Magic123 smoke on physical GPU2
- Command: `bash scripts/run_object_c_magic123.sh --config /tmp/image3d_refined_mask_smoke.yaml --run_id object_c_magic123_refined_mask_gpu2_smoke --device cuda:2`
- Config: `/tmp/image3d_refined_mask_smoke.yaml`
- Hardware: selected_device=cuda:2; fine log reports cuda fp16 and GPU=9.9GB; nvidia-smi after completion shows GPU2 3MiB used, free
- Elapsed time: coarse+fine smoke under lightweight run; fine training log 0.0505 min
- Result: success: coarse and fine 1-iter Magic123 smoke completed; produced checkpoints, result images, mp4 previews
- Metrics: coarse_iters=1, fine_iters=1, dataset_size_train=4, dataset_size_valid=1, dataset_size_test=4, sd_version=1.5
- Figure paths: runs/scene/image3d_refined_mask_smoke/refined_mask_smoke/fine/results/fine_ep0001_lambertian.jpg; runs/scene/image3d_refined_mask_smoke/refined_mask_smoke/fine/results/images/
- Video paths: runs/scene/image3d_refined_mask_smoke/refined_mask_smoke/fine/results/fine_ep0001_lambertian.mp4; runs/scene/image3d_refined_mask_smoke/refined_mask_smoke/fine/results/fine_ep0001_normal_image.mp4
- Cause analysis: Previous refined check was dry-run only; this run verified real CUDA execution, SD1.5/Zero123 loading, refined input compatibility, coarse-to-fine checkpoint handoff, and output export on mapped physical GPU2.
- Next step: User can run full refined_mask on cuda:2 if GPU2 remains free; keep auto_mask run on cuda:1 undisturbed.

## Phase5_magic123_full_launch - object_c_magic123_refined_mask_full

- Date: 2026-06-01 08:51:41 UTC
- Phase: Phase5_magic123_full_launch
- Run ID: object_c_magic123_refined_mask_full
- Goal: Launch full Magic123 refined_mask run on cuda:2
- Command: `nohup bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_refined_mask.yaml --run_id object_c_magic123_refined_mask_full --device cuda:2 > outputs/object_c_magic123_refined_mask_full/nohup.log 2>&1 &`
- Config: `configs/scene/image3d_refined_mask.yaml`
- Hardware: selected_device=cuda:2 recorded in outputs/object_c_magic123_refined_mask_full/gpu.txt
- Elapsed time: failed during startup/import before training progress
- Result: failed: first attempt was interrupted during Magic123 import at matplotlib font_manager; second attempt failed immediately because RunManager refused to overwrite existing outputs/object_c_magic123_refined_mask_full without --resume
- Metrics: progress=0/10000; no Magic123 training checkpoint produced
- Figure paths: none
- Video paths: none
- Cause analysis: The run directory was created by the first failed launch. Reusing the same run_id without --resume correctly triggered FileExistsError. Startup also exposed a matplotlib font cache/import fragility, so scripts/run_object_c_magic123.sh now sets MPLCONFIGDIR to a project-local cache and precreates it.
- Next step: Relaunch with --resume on cuda:2 after the script fix; monitor with watch_magic123_progress.

## Phase6_prep_object_A_mesh - object_a_2dgs_mesh_export_dry

- Date: 2026-06-01 08:57:45 UTC
- Phase: Phase6_prep_object_A_mesh
- Run ID: object_a_2dgs_mesh_export_dry
- Goal: Verify Object A 2DGS mesh export readiness without using occupied GPUs
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/object_a_2dgs_mesh_export.yaml --run_id object_a_2dgs_mesh_export_dry --device cuda:0 --dry_run`
- Config: `configs/scene/object_a_2dgs_mesh_export.yaml`
- Hardware: CPU/static validation only; nvidia-smi showed all GPUs occupied during check
- Elapsed time: under 1 minute
- Result: success: harness mesh export command validates; Object A 15000-iter point_cloud.ply is readable by Open3D
- Metrics: point_cloud_exists=true, points=297089, has_normals=true, has_colors=false; pytest tests/scene/test_required_configs.py tests/scene/test_checklist.py passed 3/3
- Figure paths: none
- Video paths: none
- Cause analysis: 2DGS render.py exports bounded TSDF mesh when --skip_mesh is absent, but actual export requires CUDA rendering. With all GPUs busy, only command/path/dependency/readability validation was safe.
- Next step: When a GPU is free, run object_a_2dgs_mesh_export_full with this config and verify train/ours_15000/fuse.ply and fuse_post.ply.

## Phase5_magic123_raw_scheduler - object_c_magic123_raw_full_scheduler

- Date: 2026-06-01 09:11:37 UTC
- Phase: Phase5_magic123_raw_scheduler
- Run ID: object_c_magic123_raw_full_scheduler
- Goal: Authorized scheduler to launch raw Magic123 full run on GPU1 after auto_mask finishes
- Command: `setsid bash scripts/wait_then_run_magic123.sh --wait_checkpoint runs/scene/image3d_auto_mask/auto_mask/fine/checkpoints/fine.pth --wait_pattern image3d_auto_mask/auto_mask --config configs/scene/image3d_raw.yaml --run_id object_c_magic123_raw_full --device cuda:1 --interval 900 --min_free_mb 35000 --post_launch_min_step 20 --post_launch_timeout 3600 --post_launch_interval 30 > outputs/object_c_raw_after_auto_scheduler/nohup.log 2>&1 < /dev/null &`
- Config: `configs/scene/image3d_raw.yaml`
- Hardware: scheduler PID 4181140; target cuda:1; checks every 900s; requires >=35000MiB free before launch
- Elapsed time: scheduler armed at 2026-06-01 09:11 UTC; launch pending
- Result: pending: scheduler is running and did not launch raw immediately; first check reported checkpoint=0 process_running=1
- Metrics: post_launch_min_step=20, post_launch_timeout=3600s, post_launch_interval=30s
- Figure paths: none
- Video paths: none
- Cause analysis: User explicitly authorized this single long-training automation. setsid was used because a prior nohup-only scheduler did not remain attached; the active scheduler has PPID=1 and logs status to outputs/object_c_raw_after_auto_scheduler/nohup.log.
- Next step: Scheduler will launch object_c_magic123_raw_full after auto_mask fine checkpoint exists, auto_mask process exits, and GPU1 memory is free; then it will wait until raw reaches at least step 20.

## Phase5_magic123_training_status - object_c_magic123_raw_scheduler_return_check

- Date: 2026-06-01 10:45:52 UTC
- Phase: Phase5_magic123_training_status
- Run ID: object_c_magic123_raw_scheduler_return_check
- Goal: Check Phase5 Magic123 runs after 1h absence and verify scheduled raw launch
- Command: `tail outputs/object_c_raw_after_auto_scheduler/nohup.log; inspect auto/refined/raw logs, checkpoints, result videos, mesh directories`
- Config: `configs/scene/image3d_auto_mask.yaml; configs/scene/image3d_refined_mask.yaml; configs/scene/image3d_raw.yaml`
- Hardware: auto_mask cuda:1, refined_mask cuda:2, raw cuda:1; GPUs 1 and 2 free at return check
- Elapsed time: auto fine training 16.7240 min, refined fine training 17.8718 min, raw fine training 16.8629 min; raw launched at 09:41:12 UTC and reached step 100 by 09:42:12 UTC
- Result: success with caveat: three Magic123 runs produced fine_ep0050 checkpoints and final result images/videos; mesh directories were created but empty
- Metrics: checkpoints: fine.pth and fine_ep0050.pth exist for raw/auto/refined; result mp4/jpg exist for lambertian/depth/mask/normal/normal_image; mesh files count=0 for all variants
- Figure paths: runs/scene/image3d_auto_mask/auto_mask/fine/results; runs/scene/image3d_refined_mask/refined_mask/fine/results; runs/scene/image3d_raw/raw/fine/results
- Video paths: runs/scene/image3d_auto_mask/auto_mask/fine/results/fine_ep0050_lambertian.mp4; runs/scene/image3d_refined_mask/refined_mask/fine/results/fine_ep0050_lambertian.mp4; runs/scene/image3d_raw/raw/fine/results/fine_ep0050_lambertian.mp4
- Cause analysis: Scheduler worked as intended. Magic123 reports Saving mesh but produced empty mesh directories, consistent across raw/auto/refined; likely export/marching-cubes produced no surface or silently skipped mesh file writing. This requires a lightweight mesh export/debug step before Phase6 fusion.
- Next step: Inspect final result videos/images, then run a focused Magic123 mesh export/debug pass or use checkpoints for alternative export; do not mark R5 complete until mesh/export status is resolved.

## Phase5_magic123_mesh_export - object_c_magic123_mesh_export_fix

- Date: 2026-06-01 11:06:28 UTC
- Phase: Phase5_magic123_mesh_export
- Run ID: object_c_magic123_mesh_export_fix
- Goal: Fix Magic123 mesh export and export Object C meshes for raw/auto/refined variants.
- Command: `Patched Magic123 PyMeshLab runtime path, pinned numpy==1.26.4/rich==13.9.4, then ran main.py --test --save_mesh for image3d_refined_mask, image3d_auto_mask, image3d_raw.`
- Config: `configs/scene/image3d_refined_mask.yaml; configs/scene/image3d_auto_mask.yaml; configs/scene/image3d_raw.yaml`
- Hardware: GPU cuda:1 for export; CPU PyMeshLab clean/remesh/decimate; PyMeshLab bundled Qt via LD_LIBRARY_PATH; NumPy 1.26.4.
- Elapsed time: debug+export about 15 min; final per-variant export about 35-55 sec.
- Result: SUCCESS: PyMeshLab import/segfault fixed; clean_mesh/remesh/decimate succeeded; textured mesh.obj, mesh.mtl, albedo.png, and mesh_raw.ply exported for raw, auto_mask, refined_mask.
- Metrics: auto OBJ vertices=25874 faces=50000; refined OBJ vertices=27090 faces=50000; raw OBJ vertices=18695 faces=34856; pip check residual: open3d-cpu platform metadata warning only.
- Figure paths: runs/scene/image3d_auto_mask/auto_mask/fine/mesh/albedo.png; runs/scene/image3d_refined_mask/refined_mask/fine/mesh/albedo.png; runs/scene/image3d_raw/raw/fine/mesh/albedo.png
- Video paths: none
- Cause analysis: Original Magic123 swallowed mesh export exceptions. First failure was Qt library precedence loading /home/yuhengli/study/enter/lib instead of pymeshlab/lib. After fixing LD_LIBRARY_PATH, pymeshlab still segfaulted because pymeshlab 2022.2.post4 is incompatible with NumPy 2.2.6 ndarray conversion. Pinning NumPy 1.26.4 fixed PyMeshLab Mesh/filter operations.
- Next step: Use refined/auto/raw mesh visual quality to choose Object C asset for Phase6 Blender fusion; keep refined_mask as likely default unless visual inspection shows artifacts.

## Phase5_R5_object_c_background_removal - object_c_magic123_r5_ablation

- Date: 2026-06-01 11:10:56 UTC
- Phase: Phase5_R5_object_c_background_removal
- Run ID: object_c_magic123_r5_ablation
- Goal: Close Object C R5 raw/auto-mask/refined-mask background-removal ablation with report-ready table and visual comparison.
- Command: `python scripts/finalize_object_c_r5.py`
- Config: `configs/scene/image3d_raw.yaml; configs/scene/image3d_auto_mask.yaml; configs/scene/image3d_refined_mask.yaml`
- Hardware: No training; CPU/PIL/trimesh table and contact-sheet generation from existing Magic123 outputs.
- Elapsed time: <1 min generation after mesh export was available
- Result: SUCCESS: generated Object C R5 CSV and contact sheet; auto_mask selected for Phase6 fusion.
- Metrics: raw: vertices=18695 faces=34856 geometry=3 texture=2 fusion=3; auto_mask: vertices=25874 faces=50000 geometry=4 texture=4 fusion=4 selected; refined_mask: vertices=27090 faces=50000 geometry=3 texture=3 fusion=3.
- Figure paths: reports/figures/object_c_magic123_r5_contact_sheet.jpg; data/scene/object_c/raw/object_c.jpg; data/scene/object_c/masked/object_c_auto.png; data/scene/object_c/masked/object_c_refined.png
- Video paths: runs/scene/image3d_raw/raw/fine/results/fine_ep0050_lambertian.mp4; runs/scene/image3d_auto_mask/auto_mask/fine/results/fine_ep0050_lambertian.mp4; runs/scene/image3d_refined_mask/refined_mask/fine/results/fine_ep0050_lambertian.mp4
- Cause analysis: Raw input kept background contamination; refined mask had the cleanest input crop but produced more inflated/distorted geometry and a more fragmented texture atlas. Auto-mask preserved the object silhouette and eye/detail better across views, giving the best fusion-readiness tradeoff.
- Next step: Use runs/scene/image3d_auto_mask/auto_mask/fine/mesh/mesh.obj with albedo.png as Object C in Phase6 Blender fusion.

## Phase6_object_a_mesh_export - object_a_2dgs_mesh_cluster1_export

- Date: 2026-06-01 11:25:42 UTC
- Phase: Phase6_object_a_mesh_export
- Run ID: object_a_2dgs_mesh_cluster1_export
- Goal: Export Object A 2DGS mesh variants for Phase6 fusion scale/orientation tuning.
- Command: `CUDA_VISIBLE_DEVICES=1 python external/2d-gaussian-splatting/render.py -s runs/scene/object_a_colmap_medium/colmap/dense -m runs/scene/object_a_2dgs_half --iteration 15000 --skip_train --skip_test --quiet --mesh_res 256 --num_cluster 1`
- Config: `runs/scene/object_a_2dgs_half/cfg_args; configs/scene/fusion_main.yaml`
- Hardware: cuda:1 RTX A6000; no training, render/TSDF mesh extraction only
- Elapsed time: under 1 minute
- Result: SUCCESS: preserved cluster10 mesh as fuse_post_clusters10.ply, exported largest connected cluster mesh as fuse_post_cluster1.ply for fusion.
- Metrics: cluster10 vertices=337535 faces=644899; cluster1 vertices=263429 faces=509456; cluster1 selected for smaller connected footprint; py_compile later passed
- Figure paths: runs/scene/fusion_main/layout_topdown_v1.jpg; runs/scene/fusion_main/layout_topdown.jpg
- Video paths: none
- Cause analysis: The v1 Object A footprint was too large because the 2DGS TSDF export kept multiple connected components and the y-up conversion treated depth as ground extent. Keeping only the largest component and using z-up gives a more plausible table-top footprint for Blender composition.
- Next step: Use fuse_post_cluster1.ply with z-up and target_height=0.60 in Phase6 main fusion and compare scale/position/shadow variants before final Blender render.

## Phase6_R6_fusion_transform_shadow - fusion_r6_layout_closeout_v2

- Date: 2026-06-01 11:25:59 UTC
- Phase: Phase6_R6_fusion_transform_shadow
- Run ID: fusion_r6_layout_closeout_v2
- Goal: Generate Phase6 main fusion scene and R6 scale/position/shadow ablation materials from selected A/B/C assets.
- Command: `bash scripts/run_fusion.sh --config configs/scene/fusion_main.yaml --run_id fusion_main_layout_v2 --device cpu; bash scripts/run_fusion.sh --config configs/scene/fusion_scale_ablation.yaml --run_id fusion_scale_small_layout_v2 --device cpu; bash scripts/run_fusion.sh --config configs/scene/fusion_position_ablation.yaml --run_id fusion_position_shift_layout_v2 --device cpu; bash scripts/run_fusion.sh --config configs/scene/fusion_shadow_ablation.yaml --run_id fusion_shadow_off_layout_v2 --device cpu; python scripts/finalize_fusion_r6.py`
- Config: `configs/scene/fusion_main.yaml; configs/scene/fusion_scale_ablation.yaml; configs/scene/fusion_position_ablation.yaml; configs/scene/fusion_shadow_ablation.yaml; configs/scene/final_video.yaml`
- Hardware: CPU layout/manifest generation; final-video dry-run checked Blender path only
- Elapsed time: under 3 minutes
- Result: SUCCESS with render blocker: wrote fusion scene manifests, transform tables, top-down previews, R6 CSV/contact sheet; final video dry-run now only reports missing Blender executable.
- Metrics: main_v2 selected: A extent=0.589x0.377x0.735, B extent=0.563x0.592x0.498, C extent=0.497x0.443x0.542; scores main_v2 scale/contact/lighting/view=4/4/3/4; R6 table reports/tables/fusion_r6_transform_shadow_ablation.csv
- Figure paths: reports/figures/fusion_r6_layout_contact_sheet.jpg; runs/scene/fusion_main/layout_topdown.jpg; runs/scene/fusion_scale_ablation/layout_topdown.jpg; runs/scene/fusion_position_ablation/layout_topdown.jpg; runs/scene/fusion_shadow_ablation/layout_topdown.jpg
- Video paths: none; Blender render pending
- Cause analysis: The best pre-render layout uses the cleaned Object A cluster1 mesh and z-up conversion. Scale-small reduces asset legibility, position-shift compresses spacing toward the right, and shadow-off is a control that requires Blender render to judge final grounding. Blender is not installed/on PATH, so Phase6 has layout/manifest ablation materials but not final rendered fusion frames yet.
- Next step: Install/provide Blender path, then run final still/video render from runs/scene/fusion_main/scene.json and update R6 lighting/video scores plus fusion_scene/fusion_video checklist items.

## Phase6_fusion_render_harness - fusion_blender_transform_matrix_fix

- Date: 2026-06-01 11:29:16 UTC
- Phase: Phase6_fusion_render_harness
- Run ID: fusion_blender_transform_matrix_fix
- Goal: Ensure Blender rendering reuses the exact Phase6 fusion transforms used by layout previews.
- Command: `Patched scripts/fuse_scene.py to store center/source_to_fusion/rotation/contact_shift matrices and scripts/blender_fusion_scene.py to apply the full matrix per imported asset; regenerated fusion scene JSONs with --resume.`
- Config: `configs/scene/fusion_main.yaml; configs/scene/final_video.yaml`
- Hardware: CPU code/config regeneration only; Blender still missing on PATH
- Elapsed time: under 3 minutes
- Result: SUCCESS: scene.json assets now include full transform matrices; smoke_data for fusion_main passed; tests/scene passed 59/59.
- Metrics: object_a/object_b/object_c transform keys present=true; pytest tests/scene 59 passed; final-video dry-run blocker remains missing tools.blender
- Figure paths: runs/scene/fusion_main/layout_topdown.jpg; reports/figures/fusion_r6_layout_contact_sheet.jpg
- Video paths: none
- Cause analysis: The earlier Blender harness would not exactly match the layout preview because it applied only scale/location/source-up. Storing and applying the full matrix prevents scale/orientation drift when final render starts.
- Next step: Provide or install Blender, then run scripts/run_final_video.sh from the selected runs/scene/fusion_main/scene.json and score final shadow/video quality.

## Phase6_blender_install_smoke - blender_454_install_and_fusion_smoke

- Date: 2026-06-01 11:36:45 UTC
- Phase: Phase6_blender_install_smoke
- Run ID: blender_454_install_and_fusion_smoke
- Goal: Install portable Blender and verify Phase6 A/B/C fused scene can be imported and rendered headlessly.
- Command: `wget https://download.blender.org/release/Blender4.5/blender-4.5.4-linux-x64.tar.xz; tar -xf; external/blender/blender-4.5.4-linux-x64/blender --version; blender -b --python scripts/blender_fusion_scene.py -- --scene runs/scene/fusion_main/scene.json --output runs/scene/fusion_main/smoke_still.png --frames 1 --resolution 640x360 --samples 8`
- Config: `configs/scene/fusion_main.yaml; configs/scene/final_video.yaml`
- Hardware: CPU headless Blender 4.5.4 LTS; no CUDA training; official portable Linux x64 build
- Elapsed time: download about 43s; render about 1s after startup
- Result: SUCCESS: Blender 4.5.4 LTS installed under external/blender, configs updated, final_video dry-run validation ok, single-frame fusion smoke rendered.
- Metrics: Blender imports: object_a PLY 238.53ms, object_b OBJ 66.72ms, object_c OBJ 51.89ms; render samples=8, resolution=640x360, output exists; dry-run validation ok
- Figure paths: runs/scene/fusion_main/smoke_still.png
- Video paths: none
- Cause analysis: Portable Blender avoids sudo/system package changes and works with the server GLIBC. The smoke confirms A/B/C transforms, textures, and Cycles headless rendering work, but the current smoke uses a neutral floor and still needs real background/backplate integration before final Phase6 scoring.
- Next step: Add/render real background context, then run still-frame scale/contact/shadow micro-ablation before final video.

## Phase6_garden_insertion_tuning - fusion_garden_insert_v2_to_v3

- Date: 2026-06-01 11:51:42 UTC
- Phase: Phase6_garden_insertion_tuning
- Run ID: fusion_garden_insert_v2_to_v3
- Goal: Tune A/B/C insertion scale and spacing inside the fixed garden 2DGS background scene.
- Command: `Render transparent Blender stills for fusion_main v2, scale_small, and table_insert_v3; composite each over runs/scene/background_2dgs_high/test/ours_30000/renders/00000.png; generate fusion_garden_insertion_tuning.csv and contact sheet.`
- Config: `configs/scene/fusion_main.yaml; configs/scene/fusion_scale_ablation.yaml; configs/scene/fusion_table_insert_v3.yaml; configs/scene/final_video.yaml`
- Hardware: CPU Blender 4.5.4 LTS headless; samples=8; resolution=640x360; no training
- Elapsed time: about 1 second per still render after startup; table/contact-sheet generation under 1 minute
- Result: SUCCESS: saved non-overwritten foreground/composite frames for each iteration; table_insert_v3 selected as current best garden insertion candidate and final_video now points to its scene.json.
- Metrics: v2_main scores scale/contact/lighting/view=2/3/3/2 artifacts=4; scale_small=3/3/3/3 artifacts=3; table_insert_v3=4/4/3/4 artifacts=2 selected_for_next_video_smoke
- Figure paths: runs/scene/fusion_main/garden_v2_composite.jpg; runs/scene/fusion_scale_ablation/garden_scale_small_composite.jpg; runs/scene/fusion_table_insert_v3/garden_v3_composite.jpg; reports/figures/fusion_garden_insertion_iterations.jpg
- Video paths: none yet; still-frame iteration material saved for report
- Cause analysis: The initial fusion looked too large when composited into the real garden 2DGS frame. A simple global scale reduction helped but still spread objects near table edges. The table_insert_v3 candidate uses stronger global scaling and tighter x spacing, making A/B/C sit more plausibly on the fixed garden table while preserving all three asset identities.
- Next step: Run a short multi-frame/video smoke from table_insert_v3, then tune camera path and shadow/light before final 720-frame render.

## Phase6_pre_final_composition_review - fusion_pre_final_review_pack

- Date: 2026-06-01 11:56:16 UTC
- Phase: Phase6_pre_final_composition_review
- Run ID: fusion_pre_final_review_pack
- Goal: Generate pre-final composition review stills before any final video render, per user request.
- Command: `Preserved review_default from table_insert_v3, rendered review_soft, review_side_light, review_no_shadow at 640x360 samples=8, composited over garden 2DGS render 00000, and generated review CSV/contact sheet. One side-light command failed due argparse parsing of negative vector and was rerun with --light-location=...`
- Config: `configs/scene/fusion_table_insert_v3.yaml; runs/scene/fusion_table_insert_v3/scene.json; runs/scene/fusion_table_insert_v3/scene_no_shadow.json`
- Hardware: CPU Blender 4.5.4 LTS headless; no training; low-resolution still frames only
- Elapsed time: about 1 second per still render after startup
- Result: SUCCESS: pre-final review pack generated; soft_shadow is recommended for user review; final video was not run.
- Metrics: default_shadow scale/contact/lighting/view=4/4/3/4 artifacts=2; soft_shadow=4/4/4/4 artifacts=2; side_light=4/4/3/4 artifacts=2; no_shadow=4/2/2/3 artifacts=3
- Figure paths: reports/figures/fusion_pre_final_review_sheet.jpg; runs/scene/fusion_table_insert_v3/review_default_composite.jpg; runs/scene/fusion_table_insert_v3/review_soft_composite.jpg; runs/scene/fusion_table_insert_v3/review_side_light_composite.jpg; runs/scene/fusion_table_insert_v3/review_no_shadow_composite.jpg
- Video paths: none; stopped before final video for user review
- Cause analysis: The still review confirms table_insert_v3 is a reasonable fixed-garden insertion layout. Soft shadow best matches diffuse garden lighting while preserving contact cues. No-shadow looks less grounded and is useful as R6 negative control.
- Next step: Wait for user to approve a composition variant before any final video or long render; if approved, update final render settings to the selected light/shadow variant and run only a short video smoke first.

## Phase6_fusion_quality_fix - fusion_table_insert_v4_color_orientation_fix

- Date: 2026-06-01 12:02:51 UTC
- Phase: Phase6_fusion_quality_fix
- Run ID: fusion_table_insert_v4_color_orientation_fix
- Goal: Respond to visual review concerns: Object A looked colorless and Object C appeared fallen over in the pre-final fusion stills.
- Command: `Inspect mesh visuals/bounds; patch Blender PLY material to use vertex color attribute Col; create configs/scene/fusion_table_insert_v4.yaml with Object C source_up=z; render 640x360 soft-shadow still and composite over fixed garden 2DGS background; update final_video fused_scene to v4 but do not run final video.`
- Config: `configs/scene/fusion_table_insert_v4.yaml; configs/scene/final_video.yaml; scripts/blender_fusion_scene.py`
- Hardware: CPU Blender 4.5.4 LTS headless; samples=8; no training; no final video
- Elapsed time: under 5 minutes
- Result: SUCCESS: Object A now displays vertex colors; Object C is upright in v4 composite; final_video dry-run validation ok and points to v4 scene.
- Metrics: A PLY has vertex_colors shape=(263429,4); C mesh texture OBJ with z-up v4; tests test_pipeline_dry_run.py and test_required_configs.py passed 11/11; final_video dry-run VALIDATION ok
- Figure paths: runs/scene/fusion_table_insert_v4/review_soft_composite.jpg; reports/figures/fusion_v3_v4_orientation_color_fix.jpg
- Video paths: none; intentionally stopped before final video
- Cause analysis: A looked colorless because the Blender material script overwrote imported PLY vertex colors with a uniform material. C looked fallen over because Magic123 auto-mask OBJ was treated as y-up; using z-up preserves its generated upright orientation in the fixed garden composition.
- Next step: User should review v4 still. If accepted, run a short multi-view video smoke from v4 before any final 720-frame render.

## Phase6_asset_quality_audit - fusion_asset_quality_regression_audit

- Date: 2026-06-01 12:07:18 UTC
- Phase: Phase6_asset_quality_audit
- Run ID: fusion_asset_quality_regression_audit
- Goal: Audit user-reported visual failures before final video: Object A color/geometry mismatch, Object B mug orientation, Object C malformed shape.
- Command: `Inspect Object A source frames and 2DGS renders, B SDS preview, C Magic123 R5 contact sheet, and current fusion v4; generate reports/figures/fusion_asset_quality_audit.jpg.`
- Config: `configs/scene/fusion_table_insert_v4.yaml; reports/figures/object_a_2dgs_half_preview.jpg; reports/figures/object_c_magic123_r5_contact_sheet.jpg`
- Hardware: CPU/PIL inspection only; no training and no video render
- Elapsed time: under 1 minute
- Result: FAILURE/QUALITY REGRESSION CONFIRMED: current fusion still is not final-quality. A 2DGS image render preserves the original object, but A TSDF mesh export is broken and loses recognizable color/shape; B SDS mug is upright in preview but poorly displayed in current composition; C Magic123 output is intrinsically malformed/rounded with fragmented texture.
- Metrics: A 2DGS half preview visually matches source frames better than A mesh; A mesh vertex colors exist but geometry/color transfer is not report-quality; C R5 contact sheet shows auto/refined/raw all have shape artifacts; current fusion should not be used for final video.
- Figure paths: reports/figures/fusion_asset_quality_audit.jpg; reports/figures/object_a_frames_medium_contact.jpg; reports/figures/object_a_2dgs_half_preview.jpg; reports/figures/object_c_magic123_r5_contact_sheet.jpg
- Video paths: none
- Cause analysis: The main issue is asset representation mismatch, not only scale/light. 2DGS render quality for Object A is acceptable, but mesh extraction from 2DGS creates a poor TSDF surface for Blender. Magic123 single-image output has inherent geometry errors. Continuing to tune current mesh composition would hide rather than solve the failure.
- Next step: Switch Phase6 recovery plan: use Object A 2DGS rendered billboard/plane or smaller proxy instead of bad TSDF mesh; rotate/show B handle with better camera/rotation; either reduce C prominence or test raw/refined C variants and choose least-bad before short video smoke.

## Phase3_failure_diagnosis - object_a_backview_sampling_failure

- Date: 2026-06-01 12:21:50 UTC
- Phase: Phase3_failure_diagnosis
- Run ID: object_a_backview_sampling_failure
- Goal: Mark the original Object A Phase3 run as invalid before rerun because frame sampling covered mostly rear views and produced non-report-quality reconstruction/fusion.
- Command: `User visual review of Object A source frames, 2DGS previews, and Phase6 fusion; no new training or rendering executed in this diagnostic note.`
- Config: `configs/scene/object_a_frames_*.yaml; configs/scene/object_a_colmap_*.yaml; configs/scene/object_a_2dgs_*.yaml; configs/scene/fusion_table_insert_v4.yaml`
- Hardware: manual/CPU visual diagnosis only; no GPU
- Elapsed time: not applicable
- Result: FAILURE CONFIRMED AND PHASE3 INVALIDATED: the original Object A data selection is not acceptable for final submission. The sampled views are dominated by the object's back side, and one full-resolution preview view is visibly blurred; downstream COLMAP, 2DGS, mesh export, and fusion inherit this viewpoint bias.
- Metrics: Previous metrics are retained only as failed-iteration evidence: full PSNR 21.8139 SSIM 0.8420 LPIPS 0.3549; half PSNR 22.4246 SSIM 0.7944 LPIPS 0.3245; these do not certify semantic/front-view quality because the input view distribution is biased.
- Figure paths: reports/figures/object_a_frames_medium_contact.jpg; reports/figures/object_a_2dgs_full_preview.jpg; reports/figures/object_a_2dgs_half_preview.jpg; reports/figures/fusion_asset_quality_audit.jpg
- Video paths: none
- Cause analysis: Root cause is data selection, not only fusion scale or Blender material. The first Phase3 frame extraction sampled too many rear-view frames, so the trained Object A representation did not contain enough front/side appearance for recognizable final insertion. Mesh extraction then amplified the issue by producing weak geometry/color. This failure must be included in the final report as an iteration lesson and replaced by a new 7-image curated front/side/top set.
- Next step: User will provide seven curated Object A images, with the seventh as top view. Rebuild Phase3 from scratch: verify contact sheet, create sparse/medium/dense subsets, rerun COLMAP, rerun 2DGS full/half or selected resolution experiments, regenerate metrics/previews, and update Phase6 to use the new Object A asset.

## Phase3_rerun_colmap - object_a_reselect_7_colmap_failed

- Date: 2026-06-01 12:33:51 UTC
- Phase: Phase3_rerun_colmap
- Run ID: object_a_reselect_7_colmap_failed
- Goal: Rerun Object A Phase3 using the newly uploaded curated 7-view image set and rebuild sparse/medium/dense COLMAP inputs.
- Command: `python scripts/stage_object_a_reselect.py --src . --root data/scene/object_a --contact_sheet reports/figures/object_a_reselect_7_contact.jpg; bash scripts/run_object_a_colmap.sh for sparse/medium/dense; diagnostic relaxed COLMAP dense run with lower SIFT thresholds and relaxed matcher/mapper.`
- Config: `configs/scene/object_a_colmap_sparse.yaml; configs/scene/object_a_colmap_medium.yaml; configs/scene/object_a_colmap_dense.yaml; single_camera=false; reports/tables/object_a_colmap_r2_stats.csv`
- Hardware: CPU COLMAP 3.6.0 through scripts/colmap_clean_env.sh; num_threads=90; no GPU training
- Elapsed time: under 10 minutes including diagnostics
- Result: FAILED/BLOCKED: New 7-image Object A set was staged correctly and fixed the previous back-view bias, but COLMAP did not produce a usable sparse model. Default sparse/medium/dense runs produced no verified matches; relaxed dense diagnostic produced only disconnected two-image submodels.
- Metrics: sparse 3 images: 0 registered; medium 5 images: 0 registered; dense 7 images: 0 registered under default settings. Relaxed dense diagnostic: best submodel 2/7 registered images, 32 points, 64 observations, mean reprojection error 2.813792px; insufficient for 2DGS.
- Figure paths: reports/figures/object_a_reselect_7_contact.jpg
- Video paths: none
- Cause analysis: The curated views solve the semantic coverage problem, but the images are not COLMAP-friendly: dimensions/crops differ, the object has low repeated texture and specular/plastic surfaces, background overlap changes strongly, and viewpoint jumps are large. COLMAP therefore cannot establish a connected multi-view graph. A 2DGS render cannot be run without valid COLMAP camera poses.
- Next step: Provide a more COLMAP-friendly Object A capture: preferably 30-80 consecutive frames or photos from the same camera resolution, with smaller viewpoint steps, stable textured background/table, no cropping/resizing between frames, and front/side/back/top coverage. Then rerun the same staging/COLMAP/2DGS pipeline and overwrite the failed Object A tables.

## Phase3_rollback - object_a_original_phase3_retained

- Date: 2026-06-01 12:42:37 UTC
- Phase: Phase3_rollback
- Run ID: object_a_original_phase3_retained
- Goal: Rollback the failed 7-image Object A rerun as final evidence and retain the original video-based Phase3 metrics and checklist status.
- Command: `Restore Object A COLMAP/2DGS CSV tables from original recorded metrics; set checklist Object A/R2/R3 back to true; reset COLMAP configs to single_camera=true; rerun original video frame extraction for sparse/medium/dense and rebuild medium contact sheet.`
- Config: `configs/scene/object_a_colmap_sparse.yaml; configs/scene/object_a_colmap_medium.yaml; configs/scene/object_a_colmap_dense.yaml; reports/tables/object_a_colmap_r2_stats.csv; reports/tables/object_a_2dgs_metrics.csv`
- Hardware: CPU ffmpeg/docs rollback only; no GPU training
- Elapsed time: under 5 minutes
- Result: SUCCESS WITH ARTIFACT LIMITATION: report tables, checklist, configs, and extracted input frames now retain the original video-based Phase3. Previously deleted local 2DGS checkpoints/old COLMAP workspaces cannot be restored without rerunning because they were generated artifacts, not tracked source files.
- Metrics: Restored original R2/R3 metrics: sparse 35/4130, medium 87/9837, dense 148/24125; full PSNR 21.8139 SSIM 0.8420 LPIPS 0.3549; half PSNR 22.4246 SSIM 0.7944 LPIPS 0.3245. Re-extracted frames total=296 across sparse/medium/dense.
- Figure paths: reports/figures/object_a_frames_medium_contact.jpg; reports/figures/object_a_2dgs_full_preview.jpg; reports/figures/object_a_2dgs_half_preview.jpg
- Video paths: data/scene/object_a/raw/object_a.mp4
- Cause analysis: The 7-image rerun should be treated only as a failed diagnostic, not replacement Phase3. Original video-based Phase3 remains the final recorded baseline because continuous frames were COLMAP-friendly and produced complete metrics. Remaining concern is qualitative/fusion usage, especially preview-view selection and mesh export quality, not the existence of a valid original Phase3 experiment.
- Next step: If local checkpoints are needed for Phase6, rerun Object A 2DGS/mesh export from the original video-based COLMAP commands. Otherwise use retained report tables/previews as Phase3 evidence and avoid claiming the 7-image rerun superseded it.

## Phase3_rerun_2dgs - object_a_2dgs_dense

- Date: 2026-06-01 13:14:37 UTC
- Phase: Phase3_rerun_2dgs
- Run ID: object_a_2dgs_dense
- Goal: Rerun Object A from original video-based dense COLMAP using half-resolution 2DGS to recover a higher-quality A asset for Phase6.
- Command: `bash scripts/run_object_a_colmap.sh --config configs/scene/object_a_colmap_dense.yaml --run_id object_a_colmap_dense --device cpu --resume; bash scripts/run_object_a_2dgs.sh --config configs/scene/object_a_2dgs_dense.yaml --run_id object_a_2dgs_dense --device cuda:1 --resume; bash scripts/run_2dgs_eval.sh --config configs/scene/object_a_2dgs_dense.yaml --run_id object_a_2dgs_dense_eval --device cuda:1`
- Config: `configs/scene/object_a_colmap_dense.yaml; configs/scene/object_a_2dgs_dense.yaml; external/2d-gaussian-splatting/utils/general_utils.py`
- Hardware: GPU cuda:1 NVIDIA RTX A6000 for 2DGS train/eval; CPU COLMAP dense rerun; GPU0 occupied by unrelated user process
- Elapsed time: COLMAP under 10 minutes; 2DGS train about 5 minutes; eval about 2 minutes including first-time VGG/LPIPS weight download
- Result: SUCCESS: dense COLMAP registered 148/148 images and dense-half 2DGS trained to 15000 iterations, rendered 19 eval views, and produced complete metrics/preview.
- Metrics: COLMAP registered_images=148 sparse_points=24269 mean_reprojection_error=1.126696px; 2DGS PSNR=24.765415 SSIM=0.822762 LPIPS=0.300010 checkpoint_size=75425324 render_count=19 gt_count=19 final_points=309114
- Figure paths: reports/figures/object_a_2dgs_dense_preview.jpg; reports/tables/object_a_2dgs_metrics.csv
- Video paths: none
- Cause analysis: The earlier dense-half attempt failed at iteration 7000 only because Matplotlib removed FigureCanvasAgg.tostring_rgb; patched colormap to use buffer_rgba fallback. The new 2DGS render preserves Object A color, so any later gray/odd appearance in Blender should be attributed to mesh/export/material conversion rather than Phase3 2DGS training.
- Next step: Use dense-half as the preferred Object A 2DGS result; before final fusion, audit mesh export or use 2DGS-rendered billboard/proxy if TSDF mesh still loses color/shape.

## Phase6_asset_recovery - fusion_table_insert_v6

- Date: 2026-06-01 13:28:15 UTC
- Phase: Phase6_asset_recovery
- Run ID: fusion_table_insert_v6
- Goal: Recover Phase6 after Object A mesh quality failure by exporting dense-half 2DGS mesh, reducing mesh resolution, splitting connected components, and selecting a cleaner Object A component for fusion.
- Command: `bash scripts/run_2dgs_eval.sh --config configs/scene/object_a_2dgs_mesh_export.yaml --run_id object_a_2dgs_dense_mesh_export_256 --device cuda:1 --resume; python scripts/export_mesh_components.py --input runs/scene/object_a_2dgs_dense/train/ours_15000/fuse_post.ply --out_dir runs/scene/object_a_2dgs_dense/train/ours_15000/components --count 8; bash scripts/run_fusion.sh --config configs/scene/fusion_table_insert_v6.yaml --run_id fusion_table_insert_v6 --device cpu; blender still render and composite.`
- Config: `configs/scene/object_a_2dgs_mesh_export.yaml; configs/scene/fusion_table_insert_v5.yaml; configs/scene/fusion_table_insert_v6.yaml; scripts/eval_2dgs.py; scripts/blender_fusion_scene.py; scripts/export_mesh_components.py`
- Hardware: Object A mesh export on cuda:1 RTX A6000; Blender CPU headless low-sample stills; no final video
- Elapsed time: mesh export under 5 minutes including failed wrapper attempt; Blender previews under 5 seconds each
- Result: PARTIAL SUCCESS: v5 proved full A TSDF mesh is unusable because it includes table/background shell; connected component analysis selected component_02 as the least-bad Object A mesh; v6 removes the background shell and produces a usable but visibly imperfect A fusion still. C is scaled down to reduce Magic123 artifacts.
- Metrics: dense A mesh_res1024 fuse_post=611MB 12275319 vertices 23782653 faces; mesh_res256 fuse_post=31MB 618102 vertices 1171796 faces; selected component_02=19922 vertices 38636 faces; v6 transformed extents A=0.260x0.288x0.474 B=0.349x0.367x0.309 C=0.258x0.305x0.219
- Figure paths: runs/scene/fusion_table_insert_v5/review_v5_composite.jpg; runs/scene/fusion_object_a_components/components_preview.png; runs/scene/fusion_table_insert_v6/review_v6_composite.jpg; runs/scene/fusion_table_insert_v6/transform_table.csv
- Video paths: none; final video intentionally not rendered before user review
- Cause analysis: Object A 2DGS image render is high quality, but TSDF mesh extraction reconstructs surrounding table/background surfaces together with the object. Lower mesh_res improves runtime but not semantics. Connected component filtering removes the largest background shells, but the selected object component remains incomplete, so report should describe mesh-export/fusion as a limitation of converting 2DGS to mesh for object insertion.
- Next step: Run small Phase6 ablations around v6: A scale/rotation, C smaller/larger scale, object spacing/shadow, then prepare a still review pack and stop before final video.

## Phase6_R6_ablation - fusion_table_insert_v6_position

- Date: 2026-06-01 13:31:49 UTC
- Phase: Phase6_R6_ablation
- Run ID: fusion_table_insert_v6_position
- Goal: Run scale, position, and shadow ablations around the recovered Object A component-based fusion and choose a candidate still for user review before final video.
- Command: `Generate fusion_table_insert_v6_scale/position/no_shadow configs; run scripts/run_fusion.sh for each; render low-sample Blender stills; composite over the same garden 2DGS backplate; create reports/figures/fusion_v6_ablation_contact.jpg and reports/tables/fusion_v6_r6_ablation.csv.`
- Config: `configs/scene/fusion_table_insert_v6.yaml; configs/scene/fusion_table_insert_v6_scale.yaml; configs/scene/fusion_table_insert_v6_position.yaml; configs/scene/fusion_table_insert_v6_no_shadow.yaml`
- Hardware: CPU Blender 4.5.4 headless, samples=8, 960x540 stills; no training and no final video
- Elapsed time: under 5 minutes for all stills
- Result: SUCCESS: R6 scale/position/shadow ablation completed. Wider spacing variant is the current selected candidate for user review; no-shadow rejected because contact with table becomes weak.
- Metrics: manual rubric 1-5: base scale/contact=3 position=3 shadow=4; scale variant scale/contact=4 position=3 shadow=4; position variant scale/contact=4 position=4 shadow=4 selected; no-shadow scale/contact=2 shadow=1 rejected. Artifact count remains 4 due Object A and C generation artifacts.
- Figure paths: reports/figures/fusion_v6_ablation_contact.jpg; runs/scene/fusion_table_insert_v6_position/review_composite.jpg; reports/tables/fusion_v6_r6_ablation.csv
- Video paths: none; final video intentionally paused before user review
- Cause analysis: Scale and spacing changes improve readability, but cannot solve Object A TSDF incompleteness. Keeping shadows is necessary for contact plausibility. C should remain small because Magic123 geometry artifacts become more obvious at larger size.
- Next step: Ask user to review the v6 ablation contact sheet. If accepted, update final_video to the selected position variant and run only a short video smoke before final render.

## Phase6_orientation_evidence - fusion_table_insert_v7_a_flip_y_b_flip_x

- Date: 2026-06-01 13:43:20 UTC
- Phase: Phase6_orientation_evidence
- Run ID: fusion_table_insert_v7_a_flip_y_b_flip_x
- Goal: Fix Object A/B orientation before final video and generate report evidence for Object A failure localization and Object C limitations.
- Command: `python scripts/make_fusion_variant.py for A/B rotation variants; bash scripts/run_fusion.sh for new variants; blender still render; python scripts/composite_backplate.py; python scripts/make_report_evidence.py --all`
- Config: `configs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x.yaml; configs/scene/fusion_table_insert_v7_a_flip_y_b_flip_y.yaml; reports/tables/object_a_stage_failure_evidence.csv; reports/tables/object_c_limitations_summary.csv; reports/tables/fusion_v7_orientation_ablation.csv`
- Hardware: CPU Blender still render; no final video; no long training
- Elapsed time: lightweight still-render/evidence generation
- Result: SUCCESS: generated two extra orientation candidates with A flip Y plus B flip X/Y, selected A flip Y + B flip X for user review, and produced A/C evidence tables and figures for the paper.
- Metrics: A evidence: COLMAP 148/148 registered, 24125 sparse points, reprojection 1.1296px; 2DGS PSNR 24.7654 SSIM 0.8228 LPIPS 0.3000; mesh export failure starts at TSDF/mesh stage with mesh_res1024 12275319 vertices/23782653 faces/610.9MB; component_02 selected with 19922 vertices/38636 faces. C evidence: raw/auto/refined geometry-texture-fusion scores are 3/2/3, 4/4/4, 3/3/3; auto_mask remains selected.
- Figure paths: reports/figures/fusion_v7_orientation_contact.jpg; reports/figures/object_a_2dgs_render_gt_samples.jpg; reports/figures/object_a_stage_evidence.jpg; reports/figures/object_c_magic123_r5_contact_sheet.jpg
- Video paths: none
- Cause analysis: Object A is now demonstrably good through COLMAP and 2DGS rendering, while the visible failure begins when converting the 2DGS representation into a mesh for Blender fusion: table/background shells and connected-component ambiguity remain. Orientation correction fixes the upside-down placement better, but it cannot recover missing mesh geometry. Object C limitations are caused by single-image ambiguity and mask sensitivity; auto mask is best but still needs small scale in fusion.
- Next step: Ask user to review the selected still before updating final_video and rendering the final multiview video.

## Phase6_object_a_diagnostic - object_a_per_view_quality_diagnostic

- Date: 2026-06-01 13:48:16 UTC
- Phase: Phase6_object_a_diagnostic
- Run ID: object_a_per_view_quality_diagnostic
- Goal: Diagnose why Object A's umbrella-facing side looks acceptable while the non-umbrella side looks blurry/unstable in fusion.
- Command: `python lightweight per-view GT/render PSNR and gradient-energy diagnostic over runs/scene/object_a_2dgs_dense/test/ours_15000`
- Config: `reports/tables/object_a_2dgs_per_view_quality.csv; reports/figures/object_a_2dgs_per_view_quality_sorted.jpg`
- Hardware: CPU image analysis only; no training
- Elapsed time: under 1 minute
- Result: SUCCESS: generated per-view quality CSV/contact sheet. The lowest views include severe foreground/background ambiguity and table/occluder influence, while several back/side views still have high PSNR, indicating the final blur is mainly amplified during mesh extraction/fusion rather than pure 2DGS rendering.
- Metrics: Worst PSNR views: 00009=16.50, 00013=20.29, 00017=20.43, 00000=21.02, 00012=22.43, 00005=23.56. Best PSNR views: 00011=27.15, 00007=27.33, 00004=27.33, 00003=27.54, 00016=27.96, 00010=29.41.
- Figure paths: reports/figures/object_a_2dgs_per_view_quality_sorted.jpg
- Video paths: none
- Cause analysis: The umbrella-facing side has stronger silhouette/color cues and remains recognizable after mesh filtering. The non-umbrella side is mostly white low-texture material with self-occlusion and table/background entanglement; when the 2DGS field is converted to a mesh, TSDF/component filtering removes or smears weaker surfaces and preserves high-contrast umbrella-side surfaces more reliably.
- Next step: Use this evidence in the report; continue Phase6 with orientation candidate A flip Y + B flip X unless user requests a different composition.

## Phase6_gaussian_fusion_extension - fusion_gaussian_a_smoke

- Date: 2026-06-01 13:59:19 UTC
- Phase: Phase6_gaussian_fusion_extension
- Run ID: fusion_gaussian_a_smoke
- Goal: Test whether bypassing Object A mesh extraction with Gaussian-level fusion could preserve the non-umbrella side better.
- Command: `python scripts/gaussian_fusion_smoke.py --scene runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json --object_gaussian runs/scene/object_a_2dgs_dense/point_cloud/iteration_15000/point_cloud.ply --component_mesh runs/scene/object_a_2dgs_dense/train/ours_15000/components/component_02.ply --out_dir runs/scene/fusion_gaussian_a_smoke --bbox_margin 0.15; CUDA_VISIBLE_DEVICES=1 python external/2d-gaussian-splatting/render.py -m runs/scene/object_a_2dgs_dense_gaussian_crop --iteration 15000 --skip_train --skip_mesh`
- Config: `runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json; scripts/gaussian_fusion_smoke.py; reports/tables/object_a_gaussian_fusion_smoke.csv`
- Hardware: GPU1 for lightweight render smoke only; no training; no mesh export
- Elapsed time: under 1 minute render after setup
- Result: SUCCESS: produced Object A Gaussian crop PLYs, transformed fusion diagnostic views, and rendered 19 original-camera test views for the crop model. The crop removes most table/background splats and preserves color/detail on some non-umbrella difficult views, but bbox cropping is too aggressive for all views.
- Metrics: Source A checkpoint has 309114 gaussians and is 71.93MB; crop keeps 12067 gaussians (3.90%) and is 2.81MB. Render smoke completed for 19 test views. View 00017 shows the non-umbrella side is much cleaner in Gaussian crop than in mesh fusion; views such as 00000 become empty/too sparse because the component bbox crop is conservative.
- Figure paths: runs/scene/fusion_gaussian_a_smoke/object_a_gaussian_crop_source_views.jpg; runs/scene/fusion_gaussian_a_smoke/object_a_gaussian_crop_fusion_views.jpg; reports/figures/object_a_gaussian_crop_render_smoke.jpg
- Video paths: none
- Cause analysis: Gaussian-level handling is promising because it bypasses the TSDF mesh extraction step that caused Object A shell/blur artifacts. It does not yet replace the main Blender fusion because the current Phase6 coordinates are a compositing coordinate system, not the garden COLMAP world coordinate system; true 3D Gaussian insertion requires alignment into background 2DGS coordinates and a better crop/mask than component bbox alone.
- Next step: Keep Blender mesh fusion as the main deliverable; cite Gaussian crop as an extension/diagnostic, and only continue to true Gaussian scene merge if there is extra time after final video approval.

## Phase6_gaussian_crop_debug - object_a_gaussian_crop_00004_debug

- Date: 2026-06-01 14:02:42 UTC
- Phase: Phase6_gaussian_crop_debug
- Run ID: object_a_gaussian_crop_00004_debug
- Goal: Debug why Gaussian crop render view 00004 appears almost black.
- Command: `Pixel-stat diagnostic for margin=0.15 crop; rerun gaussian_fusion_smoke.py with bbox_margin=1.0; render crop model with 2DGS renderer on GPU1; compare nonblack pixel ratios for view 00004.`
- Config: `runs/scene/fusion_gaussian_a_smoke/gaussian_fusion_smoke_stats.csv; runs/scene/fusion_gaussian_a_smoke_margin100/gaussian_fusion_smoke_stats.csv`
- Hardware: GPU1 lightweight render smoke; no training
- Elapsed time: under 1 minute render
- Result: SUCCESS: view 00004 is not pure black; it has a tiny rendered Gaussian cluster. Increasing crop margin improves coverage but remains sparse.
- Metrics: margin=0.15: 1528 nonblack pixels, 0.6810% of image, bbox 220 223 287 272, mean RGB 0.447. margin=1.0: 3876 nonblack pixels, 1.7276% of image, bbox 192 184 287 308, mean RGB 1.104. Full 2DGS render: 224068 nonblack pixels, 99.87% of image, mean RGB 111.58.
- Figure paths: runs/scene/object_a_2dgs_dense_gaussian_crop/test/ours_15000/renders/00004.png; runs/scene/object_a_2dgs_dense_gaussian_crop_margin100/test/ours_15000/renders/00004.png
- Video paths: none
- Cause analysis: The crop render uses a black background and only the retained Object A Gaussian subset. For view 00004, most full-model visible content lies outside the component-derived bbox or belongs to table/background/context splats; after cropping, only a small object-side cluster projects into the image. The issue is crop/mask coverage, not a failed renderer.
- Next step: Do not treat this crop as final fusion. If Gaussian fusion is continued, replace bbox cropping with a mask/frustum-aware Gaussian selection or a softer opacity/color-based crop.

## Phase7_preview - final_video_preview_8f_wide

- Date: 2026-06-01 14:09:19 UTC
- Phase: Phase7_preview
- Run ID: final_video_preview_8f_wide
- Goal: Prepare final multiview video pipeline after Phase6 composition approval checkpoint.
- Command: `Patch scripts/blender_fusion_scene.py with phase7_walkthrough camera path; update configs/scene/final_video.yaml to v7 scene; dry-run render-video; render 8 transparent foreground frames; composite over 2DGS garden backplate; make preview contact sheet.`
- Config: `configs/scene/final_video.yaml; runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json; scripts/blender_fusion_scene.py`
- Hardware: CPU Blender preview render, 640x360, 8 frames, samples=8
- Elapsed time: lightweight preview under 1 minute
- Result: SUCCESS: final_video config now points to the selected v7 fusion scene and the Phase7 camera path renders/composites correctly. First direct Blender backplate preview exposed gray floor occlusion, so the pipeline was corrected to transparent foreground plus 2DGS backplate compositing.
- Metrics: Preview video 640x360, 8 frames, 24 fps; final dry-run command validates 720 frames at 1920x1080 with samples=48 and phase7_walkthrough camera path.
- Figure paths: reports/figures/final_video_phase7_preview_8f_wide_contact.jpg
- Video paths: runs/scene/final_video_preview/phase7_preview_8f_wide_composite.mp4
- Cause analysis: Directly rendering the background image inside Blender caused the contact floor to cover the table/background. The corrected video path renders assets and shadow as transparent foreground, then composites over the garden 2DGS backplate frame. Camera path now includes background/wide view, approach, object-focused passes, and final wide shot.
- Next step: User should review the 8-frame preview contact sheet/video before launching the full 1080p final video render.

## Phase7_viewpoint_correction - final_video_preview_8f_moving_bg

- Date: 2026-06-01 14:13:26 UTC
- Phase: Phase7_viewpoint_correction
- Run ID: final_video_preview_8f_moving_bg
- Goal: Correct final video preview so the person/camera viewpoint moves through the reconstructed background while objects remain inserted on the table.
- Command: `Render transparent foreground frames with fixed A/B/C object transforms; composite them over a sequence of background 2DGS test renders instead of a single fixed backplate; add scripts/composite_foreground_sequence.py.`
- Config: `scripts/composite_foreground_sequence.py; scripts/blender_fusion_scene.py; runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json`
- Hardware: CPU compositing only after prior 8-frame Blender preview
- Elapsed time: under 1 minute
- Result: SUCCESS: fixed the preview logic. The prior fixed-backplate preview was rejected because it made the background static. The corrected preview uses moving 2DGS background frames 00000,00003,00006,00009,00012,00015,00018,00021 and composites the stationary inserted assets onto each view.
- Metrics: 8 foreground frames, 8 moving background frames, output fps=8; contact sheet shows background viewpoint changes across frames.
- Figure paths: reports/figures/final_video_phase7_preview_8f_moving_bg_script_contact.jpg
- Video paths: runs/scene/final_video_preview/phase7_preview_8f_moving_bg_script.mp4
- Cause analysis: The object transforms are fixed in the fusion scene, but compositing them over a single static backplate violates the required camera-motion video. Correct video generation must render or select a sequence of background 2DGS views and composite each foreground frame onto the corresponding moving background frame.
- Next step: For final output, render a smooth background 2DGS camera path or use ordered background frames, render matching transparent foreground frames, and composite frame by frame; do not use a single fixed backplate for the final video.

## Phase7_orbit_fix - final_video_preview_8f_orbit_moving_bg

- Date: 2026-06-01 14:16:57 UTC
- Phase: Phase7_orbit_fix
- Run ID: final_video_preview_8f_orbit_moving_bg
- Goal: Fix Phase7 preview so A/B/C remain fixed in the scene but their visible sides change as the viewer/camera orbits the table.
- Command: `Patch scripts/blender_fusion_scene.py with phase7_table_orbit camera path; render 8 transparent orbit foreground frames; composite with moving background 2DGS frames via scripts/composite_foreground_sequence.py.`
- Config: `scripts/blender_fusion_scene.py; scripts/composite_foreground_sequence.py; runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json`
- Hardware: CPU Blender preview render, 640x360, 8 frames, samples=8
- Elapsed time: lightweight preview
- Result: SUCCESS: corrected the previous failure where object appearance stayed nearly constant. The new orbit preview changes the visible sides of A/B/C while using moving background views.
- Metrics: 8 orbit foreground frames composited with background frames 00000,00003,00006,00009,00012,00015,00018,00021; output video fps=8.
- Figure paths: reports/figures/final_video_phase7_preview_8f_orbit_moving_bg_contact.jpg
- Video paths: runs/scene/final_video_preview/phase7_preview_8f_orbit_moving_bg.mp4
- Cause analysis: The previous moving-background preview changed only the background strongly; the foreground camera path did not produce sufficient viewpoint change for the inserted objects. The new table-orbit path keeps object transforms fixed but rotates the rendering camera around the fused assets, so their visible sides change as expected for a moving viewer.
- Next step: Ask user to review this orbit preview; if accepted, launch full-resolution/fuller-frame final render with phase7_table_orbit and frame-by-frame moving-background compositing.

## Phase7_unified_proxy_preview - final_video_unified_proxy_8f

- Date: 2026-06-01 14:25:28 UTC
- Phase: Phase7_unified_proxy_preview
- Run ID: final_video_unified_proxy_8f
- Goal: Replace failed cross-coordinate compositing preview with a single unified Blender scene for stable object/background relative motion.
- Command: `external/blender/blender-4.5.4-linux-x64/blender -b --python scripts/blender_fusion_scene.py -- --scene runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json --output runs/scene/final_video_preview/unified_proxy_8f/phase7_unified_proxy_8f.mp4 --frames 8 --resolution 640x360 --samples 16 --camera-path phase7_table_orbit --background-image runs/scene/background_2dgs_high/test/ours_30000/renders/00000.png --environment proxy_garden_table`
- Config: `configs/scene/final_video.yaml; scripts/blender_fusion_scene.py; runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json`
- Hardware: CPU Blender preview render, 640x360, 8 frames, samples=16
- Elapsed time: lightweight preview
- Result: SUCCESS: generated a unified Blender proxy-garden-table preview. A/B/C remain fixed on one table while the camera orbits; visible object sides change consistently and relative positions no longer drift against the background.
- Metrics: Video: 640x360, 8 frames, 24 fps, 82422 bytes. Dry-run final command validates 720 frames at 1920x1080, samples=48, camera_path=phase7_table_orbit, environment=proxy_garden_table.
- Figure paths: reports/figures/final_video_unified_proxy_8f_contact.jpg
- Video paths: runs/scene/final_video_preview/unified_proxy_8f/phase7_unified_proxy_8f.mp4
- Cause analysis: The previous moving-background compositing previews were rejected because foreground Blender assets and 2DGS background renders were not in the same 3D coordinate system, so relative scale/orientation/contact could not be physically trusted. The unified proxy scene uses the 2DGS garden render as reference panels and renders A/B/C/table with one Blender camera, matching the assignment wording of exporting AIGC objects as textured meshes and combining them in Blender.
- Next step: User should review the unified proxy contact sheet/video. If composition is accepted, render the full final video; otherwise tune object scale, rotation, or table positions before final rendering.

## Phase7_background_ablation - final_video_backplate_only_8f

- Date: 2026-06-01 14:28:34 UTC
- Phase: Phase7_background_ablation
- Run ID: final_video_backplate_only_8f
- Goal: Reduce garden background distortion in the final video preview.
- Command: `external/blender/blender-4.5.4-linux-x64/blender -b --python scripts/blender_fusion_scene.py -- --scene runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json --output runs/scene/final_video_preview/backplate_only_8f/phase7_backplate_only_8f.mp4 --frames 8 --resolution 640x360 --samples 16 --camera-path phase7_table_orbit --background-image runs/scene/background_2dgs_high/test/ours_30000/renders/00000.png --environment backplate_only`
- Config: `scripts/blender_fusion_scene.py; runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json`
- Hardware: CPU Blender preview render, 640x360, 8 frames, samples=16
- Elapsed time: lightweight preview
- Result: SUCCESS: added and tested backplate_only environment. The garden now uses the original 2DGS render as a camera backplate, avoiding the strong perspective distortion and wall-like appearance of proxy_garden_table panels.
- Metrics: Video: 640x360, 8 frames. Visual comparison: backplate_only preserves original garden/table appearance better, while proxy_garden_table gives stronger unified 3D spatial consistency but distorts the garden into reference panels.
- Figure paths: reports/figures/final_video_backplate_only_8f_contact.jpg
- Video paths: runs/scene/final_video_preview/backplate_only_8f/phase7_backplate_only_8f.mp4
- Cause analysis: The distortion came from projecting one 2DGS garden render onto four vertical proxy panels. That representation is not the true reconstructed 3D garden, so wide/orbit camera views expose seams and perspective stretching. Using the original 2DGS render as a camera backplate preserves the reconstruction appearance but sacrifices background parallax.
- Next step: User should choose between backplate_only for best visual fidelity to the original garden render or proxy_garden_table for stronger but less realistic 3D spatial reference. If backplate_only is selected, update final_video.yaml before the final render.

## Phase7_background_ablation - final_video_proxy_ring_8f

- Date: 2026-06-01 14:34:42 UTC
- Phase: Phase7_background_ablation
- Run ID: final_video_proxy_ring_8f
- Goal: Try to preserve proxy_garden_table camera/viewpoint correctness while reducing garden distortion using directional 2DGS background panels.
- Command: `external/blender/blender-4.5.4-linux-x64/blender -b --python scripts/blender_fusion_scene.py -- --scene runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json --output runs/scene/final_video_preview/proxy_ring_8f/phase7_proxy_ring_8f.mp4 --frames 8 --resolution 640x360 --samples 16 --camera-path phase7_table_orbit --background-image runs/scene/background_2dgs_high/test/ours_30000/renders/00000.png --background-sequence-dir runs/scene/background_2dgs_high/test/ours_30000/renders --environment proxy_garden_ring`
- Config: `scripts/blender_fusion_scene.py; runs/scene/fusion_table_insert_v7_a_flip_y_b_flip_x/scene.json`
- Hardware: CPU Blender preview render, 640x360, 8 frames, samples=16
- Elapsed time: lightweight preview
- Result: FAILED_VISUAL: camera/object orientation remains correct, but the ring background creates obvious seams, dark bands, and panel-strip artifacts. It is not suitable for the final video.
- Metrics: Video: 640x360, 8 frames. Qualitative result: direction correctness preserved, background fidelity worse than backplate_only and not clearly better than proxy_garden_table.
- Figure paths: reports/figures/final_video_proxy_ring_8f_contact.jpg
- Video paths: runs/scene/final_video_preview/proxy_ring_8f/phase7_proxy_ring_8f.mp4
- Cause analysis: The directional ring still represents a complex 2DGS garden with flat image panels. Panel height/coverage and image discontinuities make seams visible during orbit; using multiple images without geometric alignment does not solve the lack of true shared 3D coordinates.
- Next step: Discard proxy_garden_ring for final. Return to proxy_garden_table as the orientation-correct baseline and tune panel placement/scale only if needed; otherwise use backplate_only when background fidelity is prioritized.

## Phase6_closeout - final_fusion_asset_pose_and_quality

- Date: 2026-06-02 03:01:00 UTC
- Phase: Phase6_closeout
- Run ID: final_fusion_asset_pose_and_quality
- Goal: Close out the final scene-fusion result after the long sequence of layout, scale, orientation, object-quality, and representation experiments.
- Command: `Use selected scene runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json; preserve final tuned object transforms with --placement scene; summarize Phase6 ablations and final qualitative result.`
- Config: `runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json; reports/tables/fusion_v6_r6_ablation.csv; reports/tables/fusion_v7_orientation_ablation.csv; reports/tables/object_a_stage_failure_evidence.csv; reports/tables/object_c_limitations_summary.csv; reports/tables/object_b_sds_prompt_ablation.csv`
- Hardware: no new training; final assessment based on completed GPU/CPU fusion experiments and final 900-frame render outputs
- Elapsed time: Phase6 included many short Blender still/video previews plus earlier A/B/C asset generation and export; this entry is the final synthesis rather than a new render run
- Result: final fusion candidate accepted for report use. The final placement keeps A/B/C fixed in world coordinates and uses the user-tuned transforms from the synchronized scene. The object angles and tabletop positions are now substantially better than the earlier v3-v7 attempts: Object A no longer lies flat in the original wrong orientation, Object B is rotated into a mostly plausible horizontal tabletop pose, and Object C is visible with a smaller scale that reduces its single-image reconstruction artifacts.
- Metrics: final tuned parameters encoded in the selected scene rather than rerun from pixel placement. The last manually tuned command used placement pixels A=(610,480), B=(815,500), C=(990,480), scale multipliers A=1.55, B=1.20, C=1.70, rotation deltas A=(-30,90,0), B=(130,10,0), C=(90,-20,20), and height offsets A=-0.65, B=-0.60, C=-0.60 before being frozen into `scene_2dgs_table_aligned.json`.
- Figure paths: reports/figures/check_b_z90_1f_contact.jpg; reports/figures/fusion_v7_orientation_contact.jpg; reports/figures/fusion_v6_ablation_contact.jpg; reports/figures/object_a_stage_evidence.jpg; reports/figures/object_c_magic123_r5_contact_sheet.jpg
- Video paths: none for this closeout entry; final video recorded in Phase7_closeout
- Cause analysis: The final fusion uses a Mesh-rendering plus 2DGS backplate-compositing strategy rather than code-level conversion of all inserted objects into Gaussian splats. A/B/C remain textured meshes in Blender, while the garden/table background remains a 2DGS model rendered separately. The two representations are unified through the same pinhole camera intrinsics/extrinsics and then merged in image space with alpha plus depth-based occlusion. This choice avoided the unstable coordinate and quality problems observed in the Gaussian-crop smoke tests, while retaining the high-quality 2DGS background.
- Cause analysis: The main strength of the final version is not that every asset is perfect, but that the spatial relationship is now credible. The objects sit on the table at reasonable scale, their rotations are no longer obviously wrong, and flat surfaces in the background/table remain sharp because the selected 30k 2DGS background was the best of the 5k/15k/30k ablation. The moving 2DGS background also preserves the changing outdoor illumination and sun/view-dependent appearance better than the earlier fixed-backplate previews.
- Cause analysis: The remaining asset distortions should be reported explicitly. Object A is limited by the original video capture and by its high-frequency, corner-rich geometry. Although the video-based COLMAP/2DGS run gave usable metrics, the training views were not equally stable for all sides, and TSDF mesh extraction amplified shell and component artifacts. Component selection made A usable for fusion but did not recover perfect geometry. Object B is the best SDS prompt/export variant, but text-to-3D SDS still leaves some local shape and orientation defects. Object C is constrained by the assignment requirement of single-image reconstruction; Magic123 can infer a textured mesh, but thickness, silhouette, and back/side geometry are partly hallucinated, so C still has visible structural distortion.
- Next step: Do not launch more Phase6 repair experiments. Use this final version in the report with a balanced statement: the final scene succeeds at camera-synchronized placement and overall composition, but the A/B/C meshes expose the limits of video-derived 2DGS mesh extraction, SDS text-to-3D geometry, and single-image Magic123 reconstruction.

## Phase7_closeout - final_scene_fusion_900f_1080p

- Date: 2026-06-02 03:01:00 UTC
- Phase: Phase7_closeout
- Run ID: final_scene_fusion_900f_1080p
- Goal: Produce the final multi-view walkthrough render as a continuous 15-second 1080p video and close out the rendering/occlusion analysis.
- Command: `CUDA_VISIBLE_DEVICES=1 python external/2d-gaussian-splatting/render.py -s data/scene/background/garden/colmap -m runs/scene/background_2dgs_high --iteration 30000 --skip_train --skip_test --skip_mesh --render_path --render_path_frames 900 --export_camera_json runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json --resolution 1920; CUDA_VISIBLE_DEVICES=1 python scripts/render_synchronized_composite.py --scene runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json --placement scene --frames 900 --samples 16 --fps 60 --resolution 1920x1080 --output_dir runs/scene/final_video_900f_1080p --output_video runs/scene/final_video_900f_1080p/final_scene_fusion_15s_60fps_1080p.mp4 --contact_sheet reports/figures/final_scene_fusion_900f_1080p_contact.jpg --camera_json runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json --camera_indices 0:900 --background_dir runs/scene/background_2dgs_high/traj/ours_30000/renders --background_depth_dir runs/scene/background_2dgs_high/traj/ours_30000/vis`
- Config: `external/2d-gaussian-splatting/render.py; scripts/render_synchronized_composite.py; scripts/blender_fusion_scene.py; scripts/composite_foreground_sequence.py; runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json`
- Hardware: GPU1 RTX A6000 selected with CUDA_VISIBLE_DEVICES=1 for 2DGS path rendering and Blender/composite pipeline; no network access required
- Elapsed time: final run completed before 2026-06-02 02:51 UTC; no active Blender/composite process remained at verification time
- Result: success. The final video is a real 900-frame synchronized trajectory, not a repeated or interpolated 24-frame preview. 2DGS first rendered a continuous camera path and exported the same 900 pinhole cameras to JSON. Blender then used the same camera sequence to render the transparent mesh foreground, and the compositor merged each foreground frame with the matching 2DGS RGB/depth background.
- Metrics: foreground_frames=900; composite_frames=900; video_width=1920; video_height=1080; fps=60; duration=15.000000s; nb_frames=900; output_size=19500904 bytes; contact_sheet_size=906172 bytes
- Figure paths: reports/figures/final_scene_fusion_900f_1080p_contact.jpg
- Video paths: runs/scene/final_video_900f_1080p/final_scene_fusion_15s_60fps_1080p.mp4
- Cause analysis: This final pipeline fixes the earlier synchronization failure. The 24-frame preview was useful only for pose debugging because adjacent views changed too much for interpolation or frame repetition to be acceptable. The final result instead renders a continuous 2DGS trajectory with 900 cameras and uses the same camera JSON for Blender, so the background and inserted meshes move together under a shared pinhole camera model.
- Cause analysis: The background vase can now create a real-looking foreground/background occlusion in some views, which is an improvement over the earlier renders where the inserted objects always appeared pasted on top. However, the occlusion is still an image-space approximation rather than a physically unified 3D scene. The compositor uses the 2DGS background depth and an approximate foreground depth from the inserted mesh locations. This allows the vase to hide objects when it should, but it also produces false occlusion in some frames. In the final seconds, some regions that are not true background surfaces appear as transparent texture-like masks over the objects. This is a limitation of the depth-compositing approximation and of imperfect 2DGS depth around thin/ambiguous structures.
- Cause analysis: This limitation is intentionally left in the final result and should be discussed rather than hidden. Fixing it properly would require a more faithful foreground depth pass, better object-level masks, or true code-level insertion of the assets into a unified Gaussian/geometry representation. Given the assignment timeline, the chosen result is more defensible: it demonstrates synchronized multi-view rendering and partial real occlusion, while clearly documenting where image-space compositing breaks down.
- Next step: Use this video as the final Phase7 artifact. In the report, describe the final method as synchronized Mesh foreground rendering plus 2DGS RGB/depth backplate compositing; list the final strengths as credible tabletop placement, clearer flat background/table details, synchronized camera motion, and changing outdoor illumination; list the final weaknesses as A/B/C reconstruction distortion and approximate depth occlusion artifacts around the vase/background in late frames.

## Phase6_alignment_addendum - final_fusion_transform_tuning_rationale

- Date: 2026-06-02 03:09:24 UTC
- Phase: Phase6_alignment_addendum
- Run ID: final_fusion_transform_tuning_rationale
- Goal: Record the rationale for final A/B/C pose tuning and explain why enumerated transform search was used instead of solving a single linear transform across asset and 2DGS coordinate systems.
- Command: `Manual review and final frozen transform analysis for runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json; no new render or training.`
- Config: `runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json; scripts/render_synchronized_composite.py; reports/figures/final_scene_fusion_900f_1080p_contact.jpg`
- Hardware: No new compute; documentation addendum based on completed Phase6/Phase7 outputs
- Elapsed time: documentation update only
- Result: Recorded final transform-tuning rationale. A/B/C already had pose and axis inconsistencies after their independent reconstruction/generation stages, so final fusion required many manual/enum trials over rotation, scale, and height to recover plausible real-world tabletop placement. The accepted frozen scene uses A rotation [-30,270,-8], scale 0.73536, location [-0.3984,1.5744,0.2943]; B rotation [310,10,12], scale 0.45473, location [0.3582,1.7979,-0.0836]; C rotation [90,-20,4], scale 0.54453, location [0.9876,1.6392,0.1995].
- Metrics: Pre-freeze tuning used placement pixels A=(610,480), B=(815,500), C=(990,480); scale multipliers A=1.55, B=1.20, C=1.70; rotation deltas A=(-30,90,0), B=(130,10,0), C=(90,-20,20); height offsets A=-0.65, B=-0.60, C=-0.60. Final video remains 900 frames, 1920x1080, 60fps, 15s.
- Figure paths: reports/figures/final_scene_fusion_900f_1080p_contact.jpg; reports/figures/check_b_z90_1f_contact.jpg
- Video paths: runs/scene/final_video_900f_1080p/final_scene_fusion_15s_60fps_1080p.mp4
- Cause analysis: In theory, a clean solution would estimate a linear/similarity transform from each asset coordinate system into the garden/2DGS coordinate system. I did not choose that route because the necessary constraints were not reliable: A, B, and C come from separate pipelines with different up axes, arbitrary origin/scale conventions, and imperfect reconstructed geometry; the garden 2DGS coordinate system is COLMAP-defined; and we did not have enough trusted 3D correspondences between each object mesh and the target tabletop scene. Solving a single linear transform under these conditions would give a numerically neat but physically brittle result, especially because A and C contain reconstruction artifacts and B/C have AIGC hallucinated geometry. The safer engineering choice was therefore a constrained enumerated search: use 2DGS depth/table pixels for approximate placement, then enumerate rotation/scale/height around that placement, render short previews, and select the transform with the best visual contact, orientation, and scale consistency. This is less mathematically elegant than a solved transform, but it is more robust for corrupted/generated assets and is easier to justify with visible ablation evidence.
- Next step: Use this rationale in the report: state that the final fusion uses manually enumerated pose/scale/height tuning because cross-pipeline coordinate axes and asset geometry were not reliable enough for a single estimated linear transform; present the final frozen parameters and note this as a limitation/future work.

## Phase8_method_comparison - asset_generation_method_comparison

- Date: 2026-06-02 03:13:24 UTC
- Phase: Phase8_method_comparison
- Run ID: asset_generation_method_comparison
- Goal: Compare multiview reconstruction, text-to-3D generation, and single-image-to-3D generation on geometry accuracy, texture detail, compute time, distortion sources, and fusion usability.
- Command: `Summarize existing Phase3/4/5 evidence tables into reports/tables/asset_generation_method_comparison.csv and docs/phase8_method_comparison.md; no new training.`
- Config: `reports/tables/object_a_stage_failure_evidence.csv; reports/tables/object_a_2dgs_metrics.csv; reports/tables/object_a_colmap_r2_stats.csv; reports/tables/object_b_sds_prompt_ablation.csv; reports/tables/object_c_magic123_r5_ablation.csv; reports/tables/object_c_limitations_summary.csv`
- Hardware: No new compute; evidence synthesis only
- Elapsed time: documentation/table generation only
- Result: SUCCESS: created a dedicated comparison artifact for the PDF requirement. The table and report notes show that all three generation methods have distortion, but their failure modes differ: multiview A is accurate through COLMAP/2DGS but degrades during mesh extraction; text-generated B is prompt-flexible but affected by SDS hallucination/Janus/roughness; single-image C preserves front appearance but hallucinates side/back geometry and is mask-sensitive.
- Metrics: A: COLMAP 148/148 registered, 24125 sparse points, reprojection 1.1296px; 2DGS PSNR 24.7654 SSIM 0.8228 LPIPS 0.3000; component_02 has 19922 vertices/38636 faces. B selected style-constrained run: 10000 steps, 47:39, 39314 vertices/78628 faces, 1024x1024 texture, scores geometry=4 texture=3 fusion=4. C selected auto-mask run: fine training 16.724 min, 25874 vertices/50000 faces, 2048x2048 texture, scores geometry=4 texture=4 fusion=4.
- Figure paths: reports/figures/object_a_stage_evidence.jpg; reports/figures/object_c_magic123_r5_contact_sheet.jpg; reports/figures/final_scene_fusion_900f_1080p_contact.jpg
- Video paths: runs/scene/final_video_900f_1080p/final_scene_fusion_15s_60fps_1080p.mp4
- Cause analysis: The comparison should not claim any method is distortion-free. Multiview reconstruction has the strongest physical/camera evidence but becomes fragile when converting 2DGS to a mesh for Blender. Text-to-3D requires no real object input and is flexible, but the geometry and texture are prior-driven and can be locally unstable. Single-image generation preserves visible front texture but has unavoidable ambiguity for hidden sides and thickness, so mask quality directly changes geometry. These differences explain why final fusion required per-object scale/rotation/height tuning rather than one uniform transform rule.
- Next step: Use reports/tables/asset_generation_method_comparison.csv as the comparison table and docs/phase8_method_comparison.md as the report paragraph source. Then continue Phase8/Phase9 deliverable audit and README/checklist cleanup.

## Phase5_failure_audit - object_c_magic123_reconstruction_failed

- Date: 2026-06-02 05:55:00 UTC
- Phase: Phase5_failure_audit
- Run ID: object_c_magic123_reconstruction_failed
- Goal: Re-audit Object C Magic123 outputs after manual inspection of the 50 validation views showed that the previously selected auto-mask reconstruction does not form a reliable 3D object.
- Command: `Manual inspection of runs/scene/image3d_auto_mask/auto_mask/fine/validation/fine_ep0050_*.jpg, runs/scene/image3d_auto_mask/auto_mask/fine/results/fine_ep0050_lambertian.mp4, and reports/figures/object_c_magic123_r5_contact_sheet.jpg; update failure markers and R5 evidence tables.`
- Config: `configs/scene/image3d_auto_mask.yaml; configs/scene/image3d_raw.yaml; configs/scene/image3d_refined_mask.yaml`
- Hardware: No new training or GPU run; manual artifact audit plus table update.
- Elapsed time: audit/update only.
- Result: FAILED / REJECTED: Object C should no longer be described as successfully reconstructed. The auto-mask run preserves a recognizable front view, but the side/back validation views are hallucinated and unstable, and the albedo atlas is fragmented. Raw and refined-mask variants do not solve this issue.
- Metrics: Updated `reports/tables/object_c_magic123_r5_ablation.csv` and `reports/tables/object_c_limitations_summary.csv`: raw geometry=2 texture=2 fusion=1; auto_mask geometry=1 texture=2 fusion=1; refined_mask geometry=1 texture=2 fusion=1. All three are now marked `rejected_reconstruction_failed`.
- Figure paths: `reports/figures/object_c_magic123_r5_contact_sheet.jpg`
- Video paths: `runs/scene/image3d_auto_mask/auto_mask/fine/results/fine_ep0050_lambertian.mp4`
- Cause analysis: The previous scoring overvalued front-view recognizability and underweighted multi-view consistency. Single-image Magic123 inferred plausible front appearance from the input image, but hidden side/back geometry was not constrained by observations and did not converge to a stable 3D asset. This is a stronger failure than a minor artifact: it invalidates Object C as a clean fusion-ready mesh.
- Next step: Do not use the current Object C auto-mask mesh as a successful reconstruction claim. Either rerun Phase5 with a better single-image input/mask/framing, or keep the current outputs only as failure evidence for the limitations of single-image-to-3D generation.

## Phase5_restart - object_c_mouse_magic123_final_dirty

- Date: 2026-06-02 06:20:22 UTC
- Phase: Phase5_restart
- Run ID: object_c_mouse_magic123_final_dirty
- Goal: Restart Object C clean final run after accidental duplicate Magic123 launch
- Command: `setsid bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_mouse_final_clean.yaml --run_id object_c_mouse_magic123_final_clean --device cuda:0`
- Config: `configs/scene/image3d_mouse_final_clean.yaml`
- Hardware: RTX A6000 cuda:0; old dirty run stopped after duplicate process was detected
- Elapsed time: old dirty run about 6 minutes before forced stop; clean run restarted at 2026-06-02 06:18 UTC
- Result: Dirty run rejected; clean run object_c_mouse_magic123_final_clean is now the only valid Object C final run
- Metrics: dirty run reached early coarse checkpoints only; not used for final
- Figure paths: none
- Video paths: none
- Cause analysis: The first C process was still alive but was not matched by the initial process grep; a second --resume process was started and both wrote to the same workspace. To avoid ambiguous checkpoints and event files, the contaminated workspace runs/scene/image3d_mouse_final is excluded.
- Next step: Use runs/scene/image3d_mouse_final_clean and outputs/object_c_mouse_magic123_final_clean only for Object C final evidence.

## Phase4_mesh_repair - text3d_simple_object_mesh_repair_t4

- Date: 2026-06-02 07:22:28 UTC
- Phase: Phase4_mesh_repair
- Run ID: text3d_simple_object_mesh_repair_t4
- Goal: Repair B text-to-3D mesh export after default threestudio OBJ export produced an empty/fragmented mesh.
- Command: `Export from epoch=0-step=10000.ckpt; sweep isosurface_threshold in {auto,0.8,1.2,2.0,4.0,8.0}; select threshold=4.0; create solid-blue MTL for fusion.`
- Config: `configs/scene/text3d_simple_object_final.yaml`
- Hardware: GPU1 NVIDIA RTX A6000 for export; C Magic123 continued on GPU0.
- Elapsed time: about 2 min lightweight export/sweep
- Result: Fixed B mesh for fusion: runs/scene/text3d_simple_object_final/mesh_final_t4_solid_blue/save/it10000-export/model.obj and model.mtl. Final mesh has 1498 vertices, 2996 faces, one connected component, watertight=True. Textured t4 OBJ also exported but was not selected because UV/texture export fragmented the surface into many components.
- Metrics: threshold sweep: auto fragmented 325 components; t0.8 4 comps/6356 faces; t1.2 2 comps/4876 faces; t2.0 9 comps/3568 faces; t4.0 1 comp/2996 faces; t8.0 1 comp/1160 faces.
- Figure paths: reports/figures/object_b_simple_mesh_final_t4_solid_blue_preview.jpg
- Video paths: none
- Cause analysis: Default threshold 25.0 produced an empty isosurface; auto threshold 0.406 recovered surfaces but fragmented them. Higher threshold 4.0 suppresses low-density fragments and gives a closed single component. Because B is a matte blue ball, solid material is more stable for Blender fusion and shadows than the fragmented UV texture export.
- Next step: Use mesh_final_t4_solid_blue as Object B in Phase6 fusion; keep textured and sweep outputs as ablation/evidence of SDS mesh extraction sensitivity.

## Phase5_final - object_c_mouse_magic123_final_clean

- Date: 2026-06-02 07:35:34 UTC
- Phase: Phase5_final
- Run ID: object_c_mouse_magic123_final_clean
- Goal: Finish Object C single-image-to-3D reconstruction using Magic123 with auto-masked mouse image.
- Command: `nohup bash scripts/run_object_c_magic123.sh --config configs/scene/image3d_mouse_final_clean.yaml --run_id object_c_mouse_magic123_final_clean --device cuda:0 > outputs/object_c_mouse_magic123_final_clean/nohup.log 2>&1 &`
- Config: `configs/scene/image3d_mouse_final_clean.yaml`
- Hardware: GPU0 NVIDIA RTX A6000; peak log memory about 13.2GB during fine stage.
- Elapsed time: fine stage 17.18 minutes logged; full coarse+fine run completed after prior monitoring interval
- Result: Completed coarse and fine stages, exported fine OBJ/MTL/albedo and 100-view render set. Final selected mesh path: runs/scene/image3d_mouse_final_clean/auto_mask/fine/mesh/mesh.obj. Visual preview shows front and oblique-front views preserve mouse buttons/scroll wheel, while side/back views collapse into a smooth white blob.
- Metrics: fine mesh: 23019 OBJ vertices, 24340 UV vertices, 46038 faces, bbox x[-0.4842,0.4864] y[-0.4811,0.4361] z[-0.2793,0.3444]; trimesh split reports 39 components and watertight=False; mesh cleaning log: raw 110248 vertices/220496 faces -> cleaned 23019 vertices/46038 faces; final average fine loss around 0.0022.
- Figure paths: reports/figures/object_c_mouse_magic123_final_preview.jpg
- Video paths: none
- Cause analysis: Single-image Magic123 is strongly view-prior limited. The input only provides one visible mouse side, so the generated geometry keeps recognizable details near the input view but hallucinated unseen sides are oversmoothed and partially fragmented. For fusion, use orientations that expose the preserved front/oblique-front side and treat side/back artifacts as evidence of single-image reconstruction limitations.
- Next step: Use C fine mesh for Phase6 fusion with constrained orientation and scale; if quality is unacceptable in final composition, document the limitation rather than retraining more variants.

## Phase6_coordinate_preview - fusion_coordinate_final_preview

- Date: 2026-06-02 07:40:35 UTC
- Phase: Phase6_coordinate_preview
- Run ID: fusion_coordinate_final_preview
- Goal: Fuse final A/B/C assets into the garden 2DGS background using 2DGS camera/depth coordinate unprojection rather than purely manual XYZ placement.
- Command: `python scripts/render_synchronized_composite.py --scene runs/scene/fusion_coordinate_final/scene.json --camera_json runs/scene/background_2dgs_high/cameras.json --background_dir runs/scene/background_2dgs_high/test/ours_30000/renders --output_dir runs/scene/fusion_coordinate_final_preview --frames 8 --placement image_table --placement_pixels object_a:560,540;object_b:800,540;object_c:1040,540 --use_foreground_depth`
- Config: `configs/scene/fusion_coordinate_final.yaml`
- Hardware: CPU/Blender background render; no long GPU training; all GPUs idle except small residual allocation on GPU0.
- Elapsed time: about 20 seconds for 8-frame Blender foreground plus composite
- Result: Coordinate-based fusion preview completed. A/B/C are placed on the table by unprojecting table pixels through 2DGS camera 0 and depth_00000, then rendered with 2DGS cameras 0:8 and composited over garden renders. Outputs: runs/scene/fusion_coordinate_final_preview/preview.mp4 and reports/figures/fusion_coordinate_final_preview_contact.jpg.
- Metrics: Placement pixels: A(560,540), B(800,540), C(1040,540). Unprojected locations before asset contact shift: A[-0.3923,2.0435,0.1058], B[0.3097,2.0754,0.0207], C[0.9764,2.1032,-0.0517]. Preview video: 1600x1036, 8 frames, 2.0s. Asset extents after normalization: A[0.3775,0.5215,0.3619], B[0.1720,0.1677,0.2053], C[0.3805,0.2953,0.2303].
- Figure paths: reports/figures/fusion_coordinate_final_preview_contact.jpg; runs/scene/fusion_coordinate_final/layout_topdown.jpg
- Video paths: runs/scene/fusion_coordinate_final_preview/preview.mp4
- Cause analysis: The coordinate placement now uses the reconstructed 2DGS background camera and depth map, so the object locations are tied to table pixels and remain synchronized across multiple background views. The preview shows camera-dependent placement rather than a static 2D overlay. However Blender foreground depth output was all zeros, so depth occlusion was skipped; shadows are also not physically matched to the 2DGS table yet. This preview should be treated as a coordinate-fusion checkpoint before final video generation.
- Next step: Ask user to inspect the contact sheet/video. If composition is acceptable, fix foreground depth/shadow handling or render a shadow receiver pass; otherwise adjust only placement pixels/scale multipliers and rerun the same coordinate pipeline.

## Phase6_debug - fusion_coordinate_final_a_proxy_v4_emission

- Date: 2026-06-02 08:00:35 UTC
- Phase: Phase6_debug
- Run ID: fusion_coordinate_final_a_proxy_v4_emission
- Goal: Diagnose why Object A 2DGS render looks normal but fused Blender result is distorted, and test a Gaussian proxy fallback.
- Command: `python scripts/export_mesh_components.py ...; python scripts/build_gaussian_proxy_mesh.py --opacity_threshold 0.85 --crop_bounds 1.15,4.15,0.15,2.45,5.15,8.65; python scripts/render_synchronized_composite.py --frames 1/8 --samples 8`
- Config: `configs/scene/fusion_coordinate_final_a_proxy.yaml`
- Hardware: CPU/Blender smoke render, no long training; peak Blender memory about 320 MB in 1-frame checks.
- Elapsed time: about 10 min lightweight diagnostics
- Result: Failed as final-quality fusion but useful as evidence. Object A training-view 2DGS render remains good, while TSDF mesh export and Gaussian proxy both fail to produce a clean standalone mouse. fuse_post.ply has 142156 vertices, 263088 faces, 27 connected components and is not watertight; largest component still contains broad table/background surfaces. Gaussian proxy v4 selected 4392/113502 points and produced a 52704-vertex proxy mesh, but the composite frame still shows fragmented geometry rather than a solid mouse.
- Metrics: A 2DGS render metrics: PSNR 31.9452, SSIM 0.9511, LPIPS 0.1455. fuse_post: 142156 vertices, 263088 faces, 27 components, not watertight. proxy_v4: selected_points=4392, mesh_vertices=52704, mesh_faces=87840, target_height=0.22, final fused extent=(0.371,0.307,0.330).
- Figure paths: reports/figures/object_a_mouse_final_gt_render_preview.jpg; reports/figures/object_a_mouse_mesh_component_preview.jpg; runs/scene/fusion_coordinate_final/object_a_gaussian_crop_check/object_a_gaussian_crop_source_views.jpg; runs/scene/object_a_mouse_2dgs_final/gaussian_proxy_mouse_crop_v4_solid/proxy_contact.jpg; runs/scene/fusion_coordinate_final_a_proxy_v4_emission_1f/composite_frames/frame_0001.jpg
- Video paths: runs/scene/fusion_coordinate_final_a_proxy_v4_preview/preview.mp4
- Cause analysis: The A 2DGS optimization reconstructed the full local capture scene (mouse plus white table and background) rather than an isolated object. Novel-view image rendering can still look correct because splatting is evaluated near trained camera trajectories, but exporting a mesh or placing the Gaussian subset in Blender requires stable object-level geometry and foreground separation. The mesh extraction produces non-watertight multi-component surfaces dominated by table/background fragments, and opacity/DBSCAN Gaussian cropping cannot isolate a dense complete mouse body. Therefore coordinate placement was not the main failure; the object-level geometry exported from A is the bottleneck.
- Next step: For the final deliverable, keep the A training-render evidence and failure analysis. If a clean A insertion is required, retrain A with object masks or a cleaner turntable capture/background-removal pipeline before 2DGS; otherwise use the proxy only as a documented failed Gaussian-level extension, not as final-quality fusion.

## Phase6_debug - fusion_old_ab_mouse_c_final_params_1f

- Date: 2026-06-02 08:06:33 UTC
- Phase: Phase6_debug
- Run ID: fusion_old_ab_mouse_c_final_params_1f
- Goal: Render one diagnostic frame with the previously dirty/old Object A and Object B while replacing Object C with the current mouse Magic123 mesh, using the previously frozen fusion parameters.
- Command: `python scripts/fuse_scene.py --config configs/scene/fusion_old_ab_mouse_c.yaml --run_id fusion_old_ab_mouse_c --device cpu; python scripts/render_synchronized_composite.py --frames 1 --placement image_table --placement_pixels object_a:610,480;object_b:815,500;object_c:990,480 --asset_scale_multipliers object_a:1.55;object_b:1.20;object_c:1.70 --asset_rotation_delta_deg object_a:-30,90,0;object_b:130,10,0;object_c:90,-20,20 --asset_height_offsets object_a:-0.65;object_b:-0.60;object_c:-0.60`
- Config: `configs/scene/fusion_old_ab_mouse_c.yaml`
- Hardware: CPU Blender 4.5.4 headless, 1 frame, 1600x1036, samples=12; no training.
- Elapsed time: under 5 min including Blender import-format fixes
- Result: SUCCESS: generated one diagnostic frame with old A/B and current mouse C. Old A component PLY was not accepted by Blender 4.5, so it was converted to an OBJ-compatible copy without changing geometry. The resulting frame shows old A on the left, old B in the center, and the new mouse C on the right under the previously frozen placement/scale/rotation/height parameters.
- Metrics: output_frame=runs/scene/fusion_old_ab_mouse_c/final_params_1f/composite_frames/frame_0001.jpg; resolution=1600x1036; samples=12; foreground depth invalid so depth occlusion was skipped; C uses current mesh runs/scene/image3d_mouse_final_clean/auto_mask/fine/mesh/mesh.obj.
- Figure paths: runs/scene/fusion_old_ab_mouse_c/final_params_1f/composite_frames/frame_0001.jpg; reports/figures/fusion_old_ab_mouse_c_final_params_1f_contact.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/final_params_1f/preview.mp4
- Cause analysis: This diagnostic intentionally reuses the old final tuning parameters rather than optimizing for the new mouse C. Therefore C is visible and recognizable but not physically retuned for contact/scale; it appears large and floating. The earlier PLY import failure was a format compatibility issue with Blender's current importer, resolved by exporting the same old A component to OBJ.
- Next step: Use this as a comparison/diagnostic frame only. If this old-A/B plus new-C combination is needed as a candidate final composition, run a small pose/scale/height retune for C rather than reusing the old C transform parameters blindly.

## Phase6_debug_correction - fusion_old_ab_mouse_c_final_trajectory_frame001_plyfix

- Date: 2026-06-02 08:10:43 UTC
- Phase: Phase6_debug_correction
- Run ID: fusion_old_ab_mouse_c_final_trajectory_frame001_plyfix
- Goal: Correct the old-A/B plus current-mouse-C diagnostic frame so that A/B exactly follow the previously frozen final parameters and camera trajectory.
- Command: `Regenerated Object A as Blender-compatible PLY, used placement=scene with the final 900-frame trajectory camera/background, and rendered one 1920x1080 diagnostic frame.`
- Config: `configs/scene/fusion_old_ab_mouse_c.yaml`
- Hardware: CPU Blender 4.5.4 headless, 1 frame, 1920x1080, samples=12; no training.
- Elapsed time: under 5 min
- Result: SUCCESS correction. The earlier diagnostic frame is superseded because converting Object A to OBJ changed Blender import axis conventions and because the test-view background was not the final trajectory view. The corrected frame uses a PLY-compatible copy of old A, preserves frozen A/B transforms, renders with final trajectory camera 0, and replaces only C with the current mouse mesh.
- Metrics: output_frame=runs/scene/fusion_old_ab_mouse_c/final_trajectory_frame001_plyfix/composite_frames/frame_0001.jpg; A path=runs/scene/object_a_2dgs_dense/train/ours_15000/components/component_02_blender_compat.ply; B path unchanged; C path=runs/scene/image3d_mouse_final_clean/auto_mask/fine/mesh/mesh.obj; placement=scene; camera_json=runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json.
- Figure paths: runs/scene/fusion_old_ab_mouse_c/final_trajectory_frame001_plyfix/composite_frames/frame_0001.jpg; reports/figures/fusion_old_ab_mouse_c_final_trajectory_frame001_plyfix_contact.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/final_trajectory_frame001_plyfix/preview.mp4
- Cause analysis: The previous diagnostic did not faithfully preserve the old visual setup. OBJ import introduced axis-convention differences for A, while the static test background made frozen final transforms appear spatially wrong. Keeping A as PLY and rendering against the final trajectory background restores the old A/B layout; only C is swapped to the current mouse asset.
- Next step: Use the plyfix frame as the valid diagnostic comparison. Treat the previous fusion_old_ab_mouse_c_final_params_1f entry as superseded.

## Phase6_debug_correction - fusion_old_ab_mouse_c_final_trajectory_24f_plyfix

- Date: 2026-06-02 08:13:22 UTC
- Phase: Phase6_debug_correction
- Run ID: fusion_old_ab_mouse_c_final_trajectory_24f_plyfix
- Goal: Generate a 24-frame trajectory preview using the corrected old A/B placement and the current mouse C asset.
- Command: `python scripts/render_synchronized_composite.py --scene runs/scene/fusion_old_ab_mouse_c/final_params_ply_aligned_1f/scene_2dgs_table_aligned.json --placement scene --camera_json runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json --background_dir runs/scene/background_2dgs_high/traj/ours_30000/renders --background_glob '*.png' --background_depth_dir runs/scene/background_2dgs_high/traj/ours_30000/vis --background_depth_glob 'depth_*.tiff' --output_dir runs/scene/fusion_old_ab_mouse_c/final_trajectory_24f_plyfix --output_video runs/scene/fusion_old_ab_mouse_c/final_trajectory_24f_plyfix/preview_24f.mp4 --contact_sheet reports/figures/fusion_old_ab_mouse_c_final_trajectory_24f_plyfix_contact.jpg --blender external/blender/blender-4.5.4-linux-x64/blender --frames 24 --fps 12 --samples 12 --resolution 1920x1080 --camera_indices 0:24 --background_indices 0:24 --use_foreground_depth --depth_bias 0.08`
- Config: `configs/scene/fusion_old_ab_mouse_c.yaml`
- Hardware: CPU Blender 4.5.4 headless; 24 frames at 1920x1080, samples=12; no long training.
- Elapsed time: about 1 minute for foreground rendering plus 4 seconds for compositing
- Result: SUCCESS: generated 24 composite frames and a 12-fps preview video using the corrected PLY Object A, old Object B, and current mouse Object C. A/B remain fixed in the previously recovered final-trajectory layout, while the background camera moves through the first 24 final trajectory views.
- Metrics: composite_frames=24; foreground_frames=24; foreground_depth_files=24; mp4_size=1.2MB; contact_sheet_size=896KB; resolution=1920x1080; fps=12; samples=12; depth occlusion requested but skipped for all frames because foreground depth was invalid.
- Figure paths: reports/figures/fusion_old_ab_mouse_c_final_trajectory_24f_plyfix_contact.jpg; runs/scene/fusion_old_ab_mouse_c/final_trajectory_24f_plyfix/composite_frames/frame_0001.jpg ... frame_0024.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/final_trajectory_24f_plyfix/preview_24f.mp4
- Cause analysis: This run validates the corrected coordinate/format setup over a short trajectory rather than a single frame. Keeping Object A as a Blender-compatible PLY avoids the OBJ axis-convention error, and using placement=scene preserves the frozen final transforms. The remaining limitation is compositing: Blender produced invalid foreground depth arrays in this path, so the output is alpha/RGB compositing without table-object depth occlusion.
- Next step: Ask the user to inspect the 24-frame preview/contact sheet. If composition is acceptable, fix the foreground depth export or shadow/occlusion pass before any longer final video; if not, adjust only the scene-space transforms and rerun another short 24-frame preview.

## Phase6_debug_correction - fusion_old_ab_mouse_c_final_trajectory_24view_plyfix

- Date: 2026-06-02 08:16:06 UTC
- Phase: Phase6_debug_correction
- Run ID: fusion_old_ab_mouse_c_final_trajectory_24view_plyfix
- Goal: Regenerate the 24-frame preview with true multi-view sampling across the full 900-frame trajectory after the previous 0:24 consecutive-frame preview was judged too static.
- Command: `python scripts/render_synchronized_composite.py --scene runs/scene/fusion_old_ab_mouse_c/final_params_ply_aligned_1f/scene_2dgs_table_aligned.json --placement scene --camera_json runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json --background_dir runs/scene/background_2dgs_high/traj/ours_30000/renders --background_glob '*.png' --background_depth_dir runs/scene/background_2dgs_high/traj/ours_30000/vis --background_depth_glob 'depth_*.tiff' --output_dir runs/scene/fusion_old_ab_mouse_c/final_trajectory_24view_plyfix --output_video runs/scene/fusion_old_ab_mouse_c/final_trajectory_24view_plyfix/preview_24view.mp4 --contact_sheet reports/figures/fusion_old_ab_mouse_c_final_trajectory_24view_plyfix_contact.jpg --blender external/blender/blender-4.5.4-linux-x64/blender --frames 24 --fps 12 --samples 12 --resolution 1920x1080 --camera_indices 0,39,78,117,156,195,235,274,313,352,391,430,469,508,547,586,625,665,704,743,782,821,860,899 --background_indices 0,39,78,117,156,195,235,274,313,352,391,430,469,508,547,586,625,665,704,743,782,821,860,899 --use_foreground_depth --depth_bias 0.08`
- Config: `configs/scene/fusion_old_ab_mouse_c.yaml`
- Hardware: CPU Blender 4.5.4 headless; 24 frames at 1920x1080, samples=12; no long training.
- Elapsed time: about 1 minute foreground render plus 4 seconds compositing
- Result: SUCCESS: regenerated 24 frames with sparse trajectory indices covering bg_00000 to bg_00899. This supersedes fusion_old_ab_mouse_c_final_trajectory_24f_plyfix, which used consecutive indices 0:24 and therefore showed almost identical viewing angles.
- Metrics: camera_indices=0,39,78,117,156,195,235,274,313,352,391,430,469,508,547,586,625,665,704,743,782,821,860,899; composite_frames=24; resolution=1920x1080; fps=12; samples=12; depth occlusion requested but skipped because foreground depth was invalid.
- Figure paths: reports/figures/fusion_old_ab_mouse_c_final_trajectory_24view_plyfix_contact.jpg; runs/scene/fusion_old_ab_mouse_c/final_trajectory_24view_plyfix/composite_frames/frame_0001.jpg ... frame_0024.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/final_trajectory_24view_plyfix/preview_24view.mp4
- Cause analysis: The previous 24-frame preview was not a true 24-view diagnostic because the command used camera_indices/background_indices 0:24, i.e. the first 24 adjacent frames of a 900-frame smooth trajectory. Adjacent frames have only tiny camera displacement, so the contact sheet appeared to have repeated angles. The corrected run samples the entire trajectory uniformly, which exposes the object layout from substantially different viewpoints.
- Next step: Use the 24view contact sheet for user inspection. Do not generate a longer final video until the user accepts this multi-view composition; if rejected, adjust scene-space object transforms and rerun a sparse 24-view preview first.

## Phase6_debug_correction - fusion_old_ab_mouse_c_24view_approx_occlusion_restore

- Date: 2026-06-02 08:22:27 UTC
- Phase: Phase6_debug_correction
- Run ID: fusion_old_ab_mouse_c_24view_approx_occlusion_restore
- Goal: Diagnose why the old 900-frame final video had apparent vase/background occlusion while the new old-A/B plus mouse-C 24-view preview lost it, then restore the same occlusion path.
- Command: `Patched scripts/composite_foreground_sequence.py duplicate _parse_indices signature; recomposited existing foreground frames with --approx_scene runs/scene/fusion_old_ab_mouse_c/final_params_ply_aligned_1f/scene_2dgs_table_aligned.json, 2DGS background depth, and sparse camera/background indices 0..899.`
- Config: `scripts/composite_foreground_sequence.py; runs/scene/fusion_old_ab_mouse_c/final_params_ply_aligned_1f/scene_2dgs_table_aligned.json; runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json`
- Hardware: CPU/PIL/imageio compositing only; no Blender rerender and no training.
- Elapsed time: under 1 minute
- Result: SUCCESS: identified that the new preview used --use_foreground_depth, whose Blender depth files were all zeros, so depth occlusion was skipped. The old 900-frame result relied on approximate foreground depth from asset centers and 2DGS background depth. Recomposition restored that approximate occlusion path for the 24-view old-A/B plus mouse-C preview.
- Metrics: A/B transforms match old final scene exactly; C location/rotation match old final scene, but C mesh path changed and extent changed from old C [0.5127,0.3770,0.4995] to mouse C [0.5454,0.3850,0.6334]. Foreground depth files from --use_foreground_depth: valid_ratio=0.0 for checked frames. Approx recomposite: 24 frames, 1920x1080, fps=12, indices=0,39,...,899.
- Figure paths: reports/figures/fusion_old_ab_mouse_c_final_trajectory_24view_plyfix_approx_occ_contact.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/final_trajectory_24view_plyfix_approx_occ/preview_24view_approx_occ.mp4
- Cause analysis: The failure was not caused by changing Object C alone. The root cause was an execution/method mismatch: the replacement-C preview switched to Blender foreground-depth compositing, but the current Blender depth export under transparent/backplate_only rendered 256x256 zero-valued depth arrays, triggering the compositor's skip-depth-occlusion branch. The old final video's apparent vase occlusion was produced by an approximate image-space depth compositor using projected asset centers and 2DGS background depth. A secondary code bug, a duplicate _parse_indices definition, prevented reusing that approximate path until patched.
- Next step: Use the approx_occ 24-view preview as the correct comparison for user inspection. If final quality is required, run the replacement-C video with --approx_depth_occlusion or repair the true Blender Z-depth export before using --use_foreground_depth.

## Phase6_debug_correction - c_swap_preserve_old_ab_6f

- Date: 2026-06-02 08:44:08 UTC
- Phase: Phase6_debug_correction
- Run ID: c_swap_preserve_old_ab_6f
- Goal: Diagnose Object A color regression after replacing old C with mouse C, and test a C-only swap that preserves the old final A/B foreground.
- Command: `Rendered six diagnostic views; compared old final A crops with newly rerendered A; attempted Blender vertex-color material fixes; then rendered old-C-only and new-C-only passes and composited by removing old C from the old final foreground and overlaying new mouse C.`
- Config: `scripts/blender_fusion_scene.py; scripts/composite_foreground_sequence.py; runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json; runs/scene/fusion_old_ab_mouse_c/a_color_compare_origA_6f/scene_2dgs_table_aligned.json`
- Hardware: CPU Blender/PIL only, 6 frames at 1920x1080 samples=8; no training.
- Elapsed time: under 5 minutes
- Result: SUCCESS diagnostic: direct rerendering with the current Blender material path turns old Object A gray/white, while the original final_video_900f foreground still contains the correct red/blue A appearance. The working replacement strategy is to preserve old final A/B foreground and only swap C: old C is erased with a C-only alpha mask and current mouse C is overlaid, then the combined foreground is composited with the 2DGS background/depth.
- Metrics: A crop comparison before workaround: bg00000 old mean RGB [149.27,137.80,125.87] vs rerendered new [85.96,80.75,74.61]; bg00156 old [198.48,189.63,159.86] vs new [83.98,77.52,68.66]. Output frames=6; indices=0,156,313,469,625,899.
- Figure paths: reports/figures/object_a_old_final_vs_new_mouse_c_color_compare.jpg; reports/figures/c_swap_preserve_old_ab_6f_contact.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_6f/preview.mp4
- Cause analysis: The issue is not the mouse C placement parameters. The replacement-C pipeline rerendered Object A through the current Blender PLY material path, which fails to reproduce the colored old A appearance seen in final_video_900f foreground. Since the old final foreground already contains correct A/B colors, the most reliable way to replace only C is not to rerender A/B at all. The C-only swap preserves old A/B exactly and limits new rendering to the changed C asset.
- Next step: Use the C-only swap pipeline for any final replacement-C video, unless a separate robust PLY vertex-color/material reconstruction is fixed and visually verified.

## Phase6_final_video - c_swap_preserve_old_ab_900f_1080p

- Date: 2026-06-02 08:59:01 UTC
- Phase: Phase6_final_video
- Run ID: c_swap_preserve_old_ab_900f_1080p
- Goal: Generate one 15-second 1080p/60fps final replacement-C video while preserving the old accepted A/B foreground colors and placement.
- Command: `python scripts/run_c_only_swap_video.py --old_scene runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json --new_scene runs/scene/fusion_old_ab_mouse_c/a_color_compare_origA_6f/scene_2dgs_table_aligned.json --old_final_foreground_dir runs/scene/final_video_900f_1080p/foreground --background_dir runs/scene/background_2dgs_high/traj/ours_30000/renders --background_depth_dir runs/scene/background_2dgs_high/traj/ours_30000/vis --camera_json runs/scene/background_2dgs_high/traj/ours_30000/cameras_900.json --output_dir runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p --output_video runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p/final_scene_fusion_c_swap_15s_60fps_1080p.mp4 --contact_sheet reports/figures/c_swap_preserve_old_ab_900f_1080p_contact.jpg --frames 900 --fps 60 --samples 16 --resolution 1920x1080`
- Config: `scripts/run_c_only_swap_video.py; runs/scene/check_b_z90_1f/scene_2dgs_table_aligned.json; runs/scene/fusion_old_ab_mouse_c/a_color_compare_origA_6f/scene_2dgs_table_aligned.json`
- Hardware: CPU Blender/PIL/imageio rendering and compositing; no training; 900 frames, 1920x1080, samples=16.
- Elapsed time: in progress
- Result: RUNNING. The run entered Blender and is producing `old_c_mask/frame_*.png`; final success will be recorded only after the MP4, contact sheet, and composite frame sequence exist.
- Metrics: target_frames=900; target_fps=60; target_duration=15s; resolution=1920x1080; old_C_mask_frames_started=true; old_final_A_B_foreground=runs/scene/final_video_900f_1080p/foreground; approximate_depth_occlusion=true.
- Figure paths: reports/figures/c_swap_preserve_old_ab_900f_1080p_contact.jpg; reports/figures/object_c_old_vs_mouse_reconstruction_comparison.jpg; reports/figures/object_a_old_final_vs_new_mouse_c_color_compare.jpg
- Video paths: runs/scene/fusion_old_ab_mouse_c/c_swap_preserve_old_ab_900f_1080p/final_scene_fusion_c_swap_15s_60fps_1080p.mp4
- Cause analysis: This run is a corrective final-video attempt after direct rerendering of old A caused a visible color regression. Object A was not replaced in this video. Instead, A/B are preserved from the old final foreground, old C is removed by an old-C-only alpha mask, and the tuned mouse C is overlaid as the only newly rendered object. This also provides report evidence that Object A's failure mode comes from the reconstruction/export/fusion chain: the original A capture contained a large fraction of table/background pixels, so COLMAP and 2DGS optimized both the target object and nearby static background as scene content. In image-space, some A views remained recognizable, especially the umbrella-side texture, but mesh conversion introduced table/background shells, incomplete opposite-side geometry, and unstable color transfer. Object C is documented as a single-image generation limitation: the old C reconstruction had plausible validation views but was visually unusable in the final context, while the new mouse C has a simpler input shape and a cleaner tuned placement, but it still remains an image-space replacement rather than a physically unified 3D insertion.
- Next step: Monitor completion, inspect the 900-frame contact sheet and MP4, then append final status with file sizes and visual acceptance/failure notes.
