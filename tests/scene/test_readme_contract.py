from pathlib import Path


def test_readme_contains_submission_required_sections():
    text = Path("README.md").read_text(encoding="utf-8")
    for required in [
        "Requirements",
        "Data Preparation",
        "Train",
        "Test And Evaluation",
        "Render",
        "Reproduce Task 1",
        "GitHub",
        "Model Weights",
    ]:
        assert required in text
