from scripts.probe_threestudio_env import classify_import


def test_classify_import_marks_missing_required_dependency():
    result = classify_import("definitely_missing_package_for_cvhw3", required=True)

    assert result["name"] == "definitely_missing_package_for_cvhw3"
    assert result["required"] is True
    assert result["ok"] is False


def test_classify_import_marks_available_dependency():
    result = classify_import("json", required=True)

    assert result["name"] == "json"
    assert result["ok"] is True
