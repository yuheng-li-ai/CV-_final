from pathlib import Path


COMMON_FLAGS = ["--config", "--run_id", "--device", "--dry_run", "--resume"]


def test_required_smoke_scripts_exist_with_common_flags():
    for name in [
        "smoke_env.sh",
        "smoke_data.sh",
        "smoke_train.sh",
        "smoke_render.sh",
        "smoke_export.sh",
    ]:
        script = Path("scripts") / name
        text = script.read_text(encoding="utf-8")
        for flag in COMMON_FLAGS:
            assert flag in text


def test_required_formal_run_scripts_exist_with_common_flags():
    for name in [
        "run_background_2dgs.sh",
        "run_2dgs_eval.sh",
        "run_object_a_frames.sh",
        "run_object_a_colmap.sh",
        "run_object_a_2dgs.sh",
        "run_object_b_sds.sh",
        "run_object_c_magic123.sh",
        "run_fusion.sh",
        "run_final_video.sh",
    ]:
        script = Path("scripts") / name
        text = script.read_text(encoding="utf-8")
        for flag in COMMON_FLAGS:
            assert flag in text
