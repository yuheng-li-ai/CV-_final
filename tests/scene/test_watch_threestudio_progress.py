from scripts.watch_threestudio_progress import parse_progress, render_bar


def test_parse_progress_file_percentage():
    progress = parse_progress("Generation progress: 42.50%", max_steps=100)

    assert progress["stage"] == "training"
    assert progress["percent"] == 0.425
    assert progress["done"] is False


def test_parse_log_step_progress():
    progress = parse_progress("Epoch 0:  50%|#####     | 5/10", max_steps=10)
    bar = render_bar(progress, width=10)

    assert progress["stage"] == "training"
    assert progress["step"] == 5
    assert "THREESTUDIO [" in bar


def test_parse_export_stage():
    progress = parse_progress("Exporting mesh assets ...", max_steps=100)

    assert progress["stage"] == "exporting"
    assert progress["done"] is False
