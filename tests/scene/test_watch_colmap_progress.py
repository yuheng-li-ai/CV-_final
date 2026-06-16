from scripts.watch_colmap_progress import parse_progress, render_bar


def test_parse_colmap_feature_progress():
    progress = parse_progress("Processed file [7/49]\n", images=49)

    assert progress["stage"] == "feature"
    assert progress["processed"] == 7
    assert 0 < progress["percent"] < 0.2


def test_parse_colmap_mapping_progress_and_render_bar():
    text = "Registering image #31 (20)\nBundle adjustment report\n"
    progress = parse_progress(text, images=49)
    bar = render_bar(progress, images=49, width=10)

    assert progress["stage"] == "mapping"
    assert progress["registered"] == 20
    assert "COLMAP [" in bar
    assert "registered~=20/49" in bar


def test_parse_colmap_done():
    progress = parse_progress("Writing reconstruction...\nElapsed time: 2.3 [minutes]\n", images=49)

    assert progress["stage"] == "done"
    assert progress["done"] is True


def test_parse_colmap_mid_bundle_adjustment_is_not_done():
    text = (
        "Global bundle adjustment\n"
        "Elapsed time: 0.2 [minutes]\n"
        "Registering image #80 (42)\n"
        "Bundle adjustment report\n"
    )
    progress = parse_progress(text, images=148)

    assert progress["stage"] == "mapping"
    assert progress["done"] is False
