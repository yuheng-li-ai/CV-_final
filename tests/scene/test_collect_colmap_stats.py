from scripts.collect_colmap_stats import parse_stats


def test_parse_colmap_model_analyzer_stats():
    text = """Cameras: 1
Images: 12
Registered images: 12
Points: 671
Observations: 1962
Mean track length: 2.923994
Mean observations per image: 163.500000
Mean reprojection error: 1.014050px
"""

    stats = parse_stats(text)

    assert stats["registered_images"] == "12"
    assert stats["points"] == "671"
    assert stats["mean_reprojection_error_px"] == "1.014050"
