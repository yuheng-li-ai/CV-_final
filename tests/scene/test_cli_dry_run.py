from cvhw3_scene.cli import build_parser


def test_cli_help_lists_scene_pipeline_commands():
    parser = build_parser()
    help_text = parser.format_help()

    for command in [
        "prepare-object",
        "run-colmap",
        "train-2dgs",
        "generate-text3d",
        "generate-image3d",
        "fuse-scene",
        "render-video",
        "record-experiment",
    ]:
        assert command in help_text
