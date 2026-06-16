from pathlib import Path

import yaml

from cvhw3_scene.checklist import ChecklistStatus, find_false_items, load_checklist_status


def test_find_false_items_flattens_nested_checklist():
    false_items = find_false_items(
        {
            "main_pipeline": {"object_A_colmap": False, "object_A_2dgs": True},
            "harness": {"run_manager": True},
        }
    )

    assert false_items == ["main_pipeline.object_A_colmap"]


def test_load_checklist_status_reports_required_false_items(tmp_path):
    path = tmp_path / "checklist.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "required_keys": {
                    "object_A_colmap": False,
                    "object_A_2dgs": True,
                    "weights_link": False,
                }
            }
        ),
        encoding="utf-8",
    )

    status = load_checklist_status(path, required_section="required_keys")

    assert status == ChecklistStatus(
        path=path,
        required_false=["required_keys.object_A_colmap", "required_keys.weights_link"],
        all_false=["required_keys.object_A_colmap", "required_keys.weights_link"],
    )
