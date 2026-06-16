from pathlib import Path

from cvhw3_scene.inputs import InputStagingPlan, stage_local_inputs


def test_stage_local_inputs_copies_to_canonical_paths(tmp_path):
    source_video = tmp_path / "任务A.mp4"
    source_image = tmp_path / "任务C.jpg"
    source_video.write_bytes(b"video")
    source_image.write_bytes(b"image")

    plan = stage_local_inputs(
        InputStagingPlan(
            object_a_video=source_video,
            object_c_image=source_image,
            data_root=tmp_path / "data",
        )
    )

    assert plan.object_a_target == tmp_path / "data/scene/object_a/raw/object_a.mp4"
    assert plan.object_c_target == tmp_path / "data/scene/object_c/raw/object_c.jpg"
    assert plan.object_a_target.read_bytes() == b"video"
    assert plan.object_c_target.read_bytes() == b"image"


def test_stage_local_inputs_dry_run_does_not_copy(tmp_path):
    source_video = tmp_path / "任务A.mp4"
    source_image = tmp_path / "任务C.jpg"
    source_video.write_bytes(b"video")
    source_image.write_bytes(b"image")

    plan = stage_local_inputs(
        InputStagingPlan(
            object_a_video=source_video,
            object_c_image=source_image,
            data_root=tmp_path / "data",
            dry_run=True,
        )
    )

    assert not plan.object_a_target.exists()
    assert not plan.object_c_target.exists()
