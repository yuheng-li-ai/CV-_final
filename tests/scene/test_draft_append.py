from cvhw3_scene.draft import DraftEntry, build_draft_entry


def test_build_draft_entry_contains_required_fields():
    entry = build_draft_entry(
        DraftEntry(
            date="2026-05-30",
            phase="Phase1",
            run_id="smoke_env",
            goal="Validate environment smoke harness.",
            command="bash scripts/smoke_env.sh --dry_run",
            config="configs/smoke/env.yaml",
            hardware="cuda:0",
            elapsed_time="3s",
            result="passed",
            metrics='{"status": "ok"}',
            figure_paths="none",
            video_paths="none",
            cause_analysis="Dependencies import correctly.",
            next_step="Run smoke_data.",
        )
    )

    for field in [
        "Date",
        "Phase",
        "Run ID",
        "Goal",
        "Command",
        "Config",
        "Hardware",
        "Elapsed time",
        "Result",
        "Metrics",
        "Figure paths",
        "Video paths",
        "Cause analysis",
        "Next step",
    ]:
        assert f"- {field}:" in entry
