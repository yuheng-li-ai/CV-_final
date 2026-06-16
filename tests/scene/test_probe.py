import json

from cvhw3_scene.probe import parse_ffprobe_json, summarize_tool


def test_parse_ffprobe_json_extracts_media_summary():
    raw = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "nb_frames": "120",
                    "duration": "4.0",
                    "avg_frame_rate": "30/1",
                }
            ],
            "format": {"duration": "4.0", "size": "1234"},
        }
    )

    summary = parse_ffprobe_json(raw)

    assert summary.codec == "h264"
    assert summary.width == 1920
    assert summary.height == 1080
    assert summary.frames == 120
    assert summary.duration_seconds == 4.0
    assert summary.frame_rate == 30.0
    assert summary.size_bytes == 1234


def test_summarize_tool_reports_missing_tool():
    summary = summarize_tool("definitely_missing_cvhw3_tool")

    assert summary.name == "definitely_missing_cvhw3_tool"
    assert not summary.found
    assert not summary.runnable
    assert summary.path == ""


def test_summarize_tool_accepts_explicit_relative_path(tmp_path):
    tool = tmp_path / "tool.sh"
    tool.write_text("#!/usr/bin/env bash\necho tool-version\n", encoding="utf-8")
    tool.chmod(0o755)

    summary = summarize_tool(str(tool))

    assert summary.found
    assert summary.runnable
    assert summary.version == "tool-version"
