from pathlib import Path

from cvhw3_scene.config import SceneConfig


def test_required_task1_configs_load():
    config_paths = [
        "configs/scene/background_2dgs_low.yaml",
        "configs/scene/background_2dgs_smoke.yaml",
        "configs/scene/background_2dgs_mid.yaml",
        "configs/scene/background_2dgs_high.yaml",
        "configs/scene/object_a_colmap_dense.yaml",
        "configs/scene/object_a_colmap_smoke.yaml",
        "configs/scene/object_a_colmap_smoke_head.yaml",
        "configs/scene/object_a_colmap_medium.yaml",
        "configs/scene/object_a_colmap_sparse.yaml",
        "configs/scene/object_a_frames_dense.yaml",
        "configs/scene/object_a_frames_medium.yaml",
        "configs/scene/object_a_frames_sparse.yaml",
        "configs/scene/object_a_2dgs_full.yaml",
        "configs/scene/object_a_2dgs_half.yaml",
        "configs/scene/text3d_simple.yaml",
        "configs/scene/text3d_detailed.yaml",
        "configs/scene/text3d_style_constrained.yaml",
        "configs/scene/image3d_raw.yaml",
        "configs/scene/image3d_auto_mask.yaml",
        "configs/scene/image3d_refined_mask.yaml",
        "configs/scene/fusion_main.yaml",
        "configs/scene/fusion_scale_ablation.yaml",
        "configs/scene/fusion_position_ablation.yaml",
        "configs/scene/fusion_shadow_ablation.yaml",
        "configs/scene/final_video.yaml",
    ]

    for path in config_paths:
        assert Path(path).exists(), path
        config = SceneConfig.from_yaml(path)
        assert config.command
        assert config.name
