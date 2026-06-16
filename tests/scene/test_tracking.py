from cvhw3_scene.tracking import TrackingPlan, build_tracking_plan


def test_tracking_plan_is_disabled_by_default():
    assert build_tracking_plan({}) == TrackingPlan(enabled=False, backend="none")


def test_tracking_plan_accepts_wandb_and_swanlab_without_importing_sdks():
    wandb = build_tracking_plan(
        {"logging": {"enabled": True, "backend": "wandb", "project": "cvhw3-task1"}}
    )
    swanlab = build_tracking_plan(
        {"logging": {"enabled": True, "backend": "swanlab", "project": "cvhw3-task1"}}
    )

    assert wandb.backend == "wandb"
    assert swanlab.backend == "swanlab"
    assert wandb.command_env()["WANDB_PROJECT"] == "cvhw3-task1"
    assert swanlab.command_env()["SWANLAB_PROJECT"] == "cvhw3-task1"
