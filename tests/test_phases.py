"""Unit tests for src/common/phases.py."""
import pytest
from src.common.phases import Phase, PHASE_NAMES, PHASE_DESCRIPTIONS, LEGAL_TRANSITIONS, is_legal_transition


def test_all_phases_have_names():
    for phase in Phase:
        assert phase in PHASE_NAMES
        assert isinstance(PHASE_NAMES[phase], str)
        assert len(PHASE_NAMES[phase]) > 0


def test_all_phases_have_descriptions_for_both_tasks():
    for task in ("pick_place", "msd_plug"):
        assert task in PHASE_DESCRIPTIONS
        for phase in Phase:
            assert phase in PHASE_DESCRIPTIONS[task]
            assert len(PHASE_DESCRIPTIONS[task][phase]) > 0


def test_forward_transitions_are_legal():
    assert is_legal_transition(Phase.FREE_MOTION, Phase.FINE_ALIGN)
    assert is_legal_transition(Phase.FINE_ALIGN, Phase.CONTACT_ESTABLISH)
    assert is_legal_transition(Phase.CONTACT_ESTABLISH, Phase.CONSTRAINED_MOTION)
    assert is_legal_transition(Phase.CONSTRAINED_MOTION, Phase.VERIFY_RELEASE)


def test_self_loops_are_legal():
    for phase in Phase:
        assert is_legal_transition(phase, phase), f"Self-loop not legal for {phase}"


def test_fine_align_fallback_to_free_motion_is_legal():
    assert is_legal_transition(Phase.FINE_ALIGN, Phase.FREE_MOTION)


def test_verify_release_to_free_motion_is_not_legal():
    assert not is_legal_transition(Phase.VERIFY_RELEASE, Phase.FREE_MOTION)


def test_illegal_backward_jumps():
    # No large backward jumps allowed
    assert not is_legal_transition(Phase.CONTACT_ESTABLISH, Phase.FREE_MOTION)
    assert not is_legal_transition(Phase.CONSTRAINED_MOTION, Phase.FINE_ALIGN)
    assert not is_legal_transition(Phase.VERIFY_RELEASE, Phase.CONTACT_ESTABLISH)


def test_skip_forward_transitions_are_not_legal():
    # Skipping phases is not allowed
    assert not is_legal_transition(Phase.FREE_MOTION, Phase.CONTACT_ESTABLISH)
    assert not is_legal_transition(Phase.FREE_MOTION, Phase.CONSTRAINED_MOTION)
    assert not is_legal_transition(Phase.FINE_ALIGN, Phase.CONSTRAINED_MOTION)


def test_legal_transitions_cover_all_phases():
    for phase in Phase:
        assert phase in LEGAL_TRANSITIONS
