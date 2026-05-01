"""Tests for the annotation Validator."""
import json
import pytest
import numpy as np

from src.annotation.schema import EpisodeAnnotation
from src.annotation.validator import Validator, ValidationResult


def _make_annotation(segments=None, task="pick_place"):
    if segments is None:
        segments = [
            {"phase_id": 0, "phase_name": "free_motion",
             "start_t": 0.0, "end_t": 1.0, "confidence": 0.9, "evidence": "moving"},
            {"phase_id": 1, "phase_name": "fine_align",
             "start_t": 1.0, "end_t": 2.0, "confidence": 0.9, "evidence": "aligning"},
            {"phase_id": 2, "phase_name": "contact_establish",
             "start_t": 2.0, "end_t": 2.5, "confidence": 0.9, "evidence": "contact"},
            {"phase_id": 3, "phase_name": "constrained_motion",
             "start_t": 2.5, "end_t": 4.0, "confidence": 0.9, "evidence": "lifting"},
            {"phase_id": 4, "phase_name": "verify_release",
             "start_t": 4.0, "end_t": 5.0, "confidence": 0.9, "evidence": "placing"},
        ]
    return EpisodeAnnotation.model_validate({
        "episode_id": "test_ep",
        "task": task,
        "overall_confidence": 0.9,
        "notes": None,
        "segments": segments,
    })


def _make_episode(n=150, fps=30.0, contact_frame=60):
    timestamps = np.arange(n) / fps
    wrench = np.zeros((n, 6))
    wrench[contact_frame:, 2] = 5.0  # Fz spike at contact_frame
    gripper = np.zeros(n)
    gripper[contact_frame:] = 1.0
    return {"timestamps": timestamps, "wrench": wrench, "gripper_state": gripper}


def test_valid_annotation_passes():
    v = Validator()
    ann = _make_annotation()
    result = v.validate(ann, episode_duration=5.0)
    assert result.passed
    assert len(result.errors()) == 0


def test_illegal_transition_detected():
    # Skip from free_motion directly to contact_establish
    segs = [
        {"phase_id": 0, "phase_name": "free_motion",
         "start_t": 0.0, "end_t": 2.0, "confidence": 0.9, "evidence": "moving"},
        {"phase_id": 2, "phase_name": "contact_establish",
         "start_t": 2.0, "end_t": 5.0, "confidence": 0.9, "evidence": "contact"},
    ]
    ann = _make_annotation(segments=segs)
    v = Validator()
    result = v.validate(ann, episode_duration=5.0)
    assert not result.passed
    errors = result.errors()
    assert any("transition" in e.rule for e in errors)


def test_short_segment_low_confidence_is_warning():
    segs = [
        {"phase_id": 0, "phase_name": "free_motion",
         "start_t": 0.0, "end_t": 1.0, "confidence": 0.9, "evidence": "moving"},
        {"phase_id": 1, "phase_name": "fine_align",
         "start_t": 1.0, "end_t": 1.05, "confidence": 0.7, "evidence": "short"},  # 50ms, low conf
        {"phase_id": 2, "phase_name": "contact_establish",
         "start_t": 1.05, "end_t": 2.0, "confidence": 0.9, "evidence": "contact"},
        {"phase_id": 3, "phase_name": "constrained_motion",
         "start_t": 2.0, "end_t": 4.0, "confidence": 0.9, "evidence": "lifting"},
        {"phase_id": 4, "phase_name": "verify_release",
         "start_t": 4.0, "end_t": 5.0, "confidence": 0.9, "evidence": "placing"},
    ]
    ann = _make_annotation(segments=segs)
    v = Validator()
    result = v.validate(ann, episode_duration=5.0)
    warnings = result.warnings()
    assert any("duration" in w.rule for w in warnings)


def test_fsm_crosscheck_agreement():
    """Qwen and FSM agree on contact_establish → no cross-check warning."""
    # contact_frame=60 at fps=30 → t=2.0s, matching our annotation
    ep = _make_episode(contact_frame=60)
    ann = _make_annotation()
    v = Validator()
    result = v.validate(ann, episode=ep, episode_duration=5.0)
    crosscheck_warnings = [w for w in result.warnings() if w.rule == "fsm_crosscheck"]
    assert len(crosscheck_warnings) == 0


def test_fsm_crosscheck_disagreement():
    """FSM contact at t=1.0s but Qwen says t=2.0s → warning."""
    ep = _make_episode(contact_frame=30)  # t=1.0s
    ann = _make_annotation()  # contact_establish starts at t=2.0s
    v = Validator()
    result = v.validate(ann, episode=ep, episode_duration=5.0)
    crosscheck_warnings = [w for w in result.warnings() if w.rule == "fsm_crosscheck"]
    assert len(crosscheck_warnings) == 1


def test_coverage_missing_phase_is_warning():
    # Missing fine_align
    segs = [
        {"phase_id": 0, "phase_name": "free_motion",
         "start_t": 0.0, "end_t": 2.0, "confidence": 0.9, "evidence": "moving"},
        {"phase_id": 2, "phase_name": "contact_establish",
         "start_t": 2.0, "end_t": 3.0, "confidence": 0.9, "evidence": "contact"},
        {"phase_id": 3, "phase_name": "constrained_motion",
         "start_t": 3.0, "end_t": 4.5, "confidence": 0.9, "evidence": "lifting"},
        {"phase_id": 4, "phase_name": "verify_release",
         "start_t": 4.5, "end_t": 5.0, "confidence": 0.9, "evidence": "placing"},
    ]
    # This annotation has an illegal transition (0→2), so we manually validate
    # only the coverage rule here by constructing without the transition check
    ann = EpisodeAnnotation.model_validate({
        "episode_id": "ep",
        "task": "pick_place",
        "overall_confidence": 0.85,
        "notes": "fine_align skipped",
        "segments": segs,
    })
    v = Validator()
    result = v.validate(ann)
    coverage_issues = [i for i in result.issues if i.rule == "coverage"]
    assert any("fine_align" in i.message for i in coverage_issues)


def test_schema_contiguity_error():
    """Non-contiguous segments caught as error."""
    segs = [
        {"phase_id": 0, "phase_name": "free_motion",
         "start_t": 0.0, "end_t": 1.0, "confidence": 0.9, "evidence": "moving"},
        {"phase_id": 1, "phase_name": "fine_align",
         "start_t": 1.5, "end_t": 5.0, "confidence": 0.9, "evidence": "gap here"},
    ]
    ann = EpisodeAnnotation.model_validate({
        "episode_id": "ep",
        "task": "pick_place",
        "overall_confidence": 0.85,
        "notes": None,
        "segments": segs,
    })
    v = Validator()
    result = v.validate(ann, episode_duration=5.0)
    errors = result.errors()
    assert any("contiguous" in e.message.lower() or "schema" in e.rule for e in errors)
