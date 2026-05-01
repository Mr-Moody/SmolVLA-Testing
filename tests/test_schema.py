"""Tests for annotation schema and parser."""
import json
import pytest
from src.annotation.schema import PhaseSegment, EpisodeAnnotation, parse_qwen_output


def _valid_annotation(duration=5.0):
    return {
        "episode_id": "ep_001",
        "task": "pick_place",
        "overall_confidence": 0.9,
        "notes": "test",
        "segments": [
            {"phase_id": 0, "phase_name": "free_motion",
             "start_t": 0.0, "end_t": 1.0, "confidence": 0.95, "evidence": "arm moving"},
            {"phase_id": 1, "phase_name": "fine_align",
             "start_t": 1.0, "end_t": 2.5, "confidence": 0.9, "evidence": "approaching"},
            {"phase_id": 2, "phase_name": "contact_establish",
             "start_t": 2.5, "end_t": 3.0, "confidence": 0.88, "evidence": "gripper closes"},
            {"phase_id": 3, "phase_name": "constrained_motion",
             "start_t": 3.0, "end_t": 4.5, "confidence": 0.92, "evidence": "lifting"},
            {"phase_id": 4, "phase_name": "verify_release",
             "start_t": 4.5, "end_t": duration, "confidence": 0.87, "evidence": "placing"},
        ],
    }


def test_valid_annotation_parses():
    ann = EpisodeAnnotation.model_validate(_valid_annotation())
    ann.validate_against_duration(5.0)
    assert ann.episode_id == "ep_001"


def test_invalid_phase_id():
    data = _valid_annotation()
    data["segments"][0]["phase_id"] = 9
    data["segments"][0]["phase_name"] = "free_motion"
    with pytest.raises(Exception):
        EpisodeAnnotation.model_validate(data)


def test_start_t_not_less_than_end_t():
    data = _valid_annotation()
    data["segments"][0]["start_t"] = 1.5
    data["segments"][0]["end_t"] = 0.5
    with pytest.raises(Exception):
        EpisodeAnnotation.model_validate(data)


def test_phase_name_mismatch():
    data = _valid_annotation()
    data["segments"][0]["phase_name"] = "fine_align"  # wrong for phase_id=0
    with pytest.raises(Exception):
        EpisodeAnnotation.model_validate(data)


def test_confidence_out_of_range():
    data = _valid_annotation()
    data["segments"][0]["confidence"] = 1.5
    with pytest.raises(Exception):
        EpisodeAnnotation.model_validate(data)


def test_invalid_task():
    data = _valid_annotation()
    data["task"] = "robot_dance"
    with pytest.raises(Exception):
        EpisodeAnnotation.model_validate(data)


def test_parse_qwen_output_plain_json():
    data = _valid_annotation()
    raw = json.dumps(data)
    ann = parse_qwen_output(raw, episode_duration=5.0)
    assert ann.episode_id == "ep_001"


def test_parse_qwen_output_strips_markdown_fences():
    data = _valid_annotation()
    raw = f"```json\n{json.dumps(data)}\n```"
    ann = parse_qwen_output(raw, episode_duration=5.0)
    assert ann.task == "pick_place"


def test_parse_qwen_output_strips_plain_fences():
    data = _valid_annotation()
    raw = f"```\n{json.dumps(data)}\n```"
    ann = parse_qwen_output(raw, episode_duration=5.0)
    assert ann.overall_confidence == 0.9


def test_parse_qwen_output_no_json_raises():
    with pytest.raises(ValueError, match="No JSON"):
        parse_qwen_output("Just some commentary with no JSON.", 5.0)


def test_validate_against_duration_wrong_end():
    data = _valid_annotation(duration=5.0)
    # Change last end_t to be wrong (must be > start_t=4.5 to pass PhaseSegment validation)
    data["segments"][-1]["end_t"] = 4.8  # valid segment but wrong vs duration 5.0
    ann = EpisodeAnnotation.model_validate(data)
    with pytest.raises(ValueError, match="episode_duration"):
        ann.validate_against_duration(5.0)


def test_validate_against_duration_gap():
    data = _valid_annotation()
    data["segments"][1]["start_t"] = 1.5  # gap from 1.0 to 1.5
    ann = EpisodeAnnotation.model_validate(data)
    with pytest.raises(ValueError, match="contiguous"):
        ann.validate_against_duration(5.0)
