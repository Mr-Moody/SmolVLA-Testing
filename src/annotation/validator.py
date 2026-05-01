"""Validation layer for Qwen3-VL annotation outputs.

Checks schema, transition legality, duration sanity, FSM cross-check,
and phase coverage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.annotation.schema import EpisodeAnnotation
from src.common.phases import Phase, PHASE_NAMES, is_legal_transition

logger = logging.getLogger(__name__)

# Thresholds
MIN_SEGMENT_DURATION_S = 0.1  # 100 ms
MIN_SEGMENT_CONFIDENCE_FOR_SHORT = 0.95
FSM_CROSSCHECK_TOL_S = 0.2  # 200 ms

# Phases expected per task
_EXPECTED_PHASES = {
    "pick_place": {Phase.FREE_MOTION, Phase.FINE_ALIGN, Phase.CONTACT_ESTABLISH,
                   Phase.CONSTRAINED_MOTION, Phase.VERIFY_RELEASE},
    "msd_plug": {Phase.FREE_MOTION, Phase.FINE_ALIGN, Phase.CONTACT_ESTABLISH,
                 Phase.CONSTRAINED_MOTION, Phase.VERIFY_RELEASE},
}


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    rule: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    issues: List[Issue] = field(default_factory=list)

    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self):
        return [i for i in self.issues if i.severity == "warning"]


class Validator:
    """Validate an EpisodeAnnotation against the LeRobot episode.

    Args:
        fsm_gripper_force_threshold: Force threshold (N) used by the simple
            proprioceptive FSM for contact_establish detection.
    """

    def __init__(self, fsm_gripper_force_threshold: float = 3.0):
        self.fsm_threshold = fsm_gripper_force_threshold

    def validate(
        self,
        annotation: EpisodeAnnotation,
        episode: Optional[Dict[str, Any]] = None,
        episode_duration: Optional[float] = None,
    ) -> ValidationResult:
        """Run all validation checks.

        Args:
            annotation: The parsed EpisodeAnnotation.
            episode: Optional raw episode dict (needed for FSM cross-check).
            episode_duration: Episode duration for contiguity check.

        Returns:
            ValidationResult with issues classified as error/warning.
        """
        issues: List[Issue] = []

        # 1. Schema re-check (contiguity)
        if episode_duration is not None:
            try:
                annotation.validate_against_duration(episode_duration)
            except ValueError as exc:
                issues.append(Issue("error", "schema", str(exc)))

        # 2. Transition legality
        issues.extend(self._check_transitions(annotation))

        # 3. Duration sanity
        issues.extend(self._check_durations(annotation))

        # 4. FSM cross-check (if episode data available)
        if episode is not None:
            issues.extend(self._fsm_crosscheck(annotation, episode))

        # 5. Coverage
        issues.extend(self._check_coverage(annotation))

        errors = [i for i in issues if i.severity == "error"]
        return ValidationResult(passed=(len(errors) == 0), issues=issues)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_transitions(self, annotation: EpisodeAnnotation) -> List[Issue]:
        issues = []
        segs = annotation.segments
        for i in range(1, len(segs)):
            prev = Phase(segs[i - 1].phase_id)
            curr = Phase(segs[i].phase_id)
            if not is_legal_transition(prev, curr):
                issues.append(Issue(
                    "error",
                    "transition_legality",
                    f"Illegal transition: {PHASE_NAMES[prev]} → {PHASE_NAMES[curr]} "
                    f"at t={segs[i].start_t:.3f}s",
                ))
        return issues

    def _check_durations(self, annotation: EpisodeAnnotation) -> List[Issue]:
        issues = []
        for seg in annotation.segments:
            dur = seg.end_t - seg.start_t
            if dur < MIN_SEGMENT_DURATION_S:
                if seg.confidence < MIN_SEGMENT_CONFIDENCE_FOR_SHORT:
                    issues.append(Issue(
                        "warning",
                        "duration_sanity",
                        f"Segment {PHASE_NAMES[Phase(seg.phase_id)]} at "
                        f"t={seg.start_t:.3f}–{seg.end_t:.3f}s is only "
                        f"{dur*1000:.0f} ms (confidence={seg.confidence:.2f} < "
                        f"{MIN_SEGMENT_CONFIDENCE_FOR_SHORT})",
                    ))
        return issues

    def _fsm_crosscheck(
        self, annotation: EpisodeAnnotation, episode: Dict[str, Any]
    ) -> List[Issue]:
        """Compare Qwen boundaries to a simple proprioceptive FSM.

        Only checks contact_establish and constrained_motion start times.
        """
        issues = []
        timestamps = np.asarray(episode.get("timestamps", []), dtype=float)
        wrench = episode.get("wrench")
        gripper = episode.get("gripper_state")

        if len(timestamps) == 0 or (wrench is None and gripper is None):
            return issues  # Not enough data for FSM cross-check

        # Estimate contact_establish start via force threshold
        fsm_contact_t = self._estimate_contact_start(timestamps, wrench, gripper)
        if fsm_contact_t is None:
            return issues

        # Find Qwen's contact_establish start
        qwen_contact_seg = next(
            (s for s in annotation.segments if s.phase_id == Phase.CONTACT_ESTABLISH.value),
            None,
        )
        if qwen_contact_seg is None:
            return issues

        diff = abs(qwen_contact_seg.start_t - fsm_contact_t)
        if diff > FSM_CROSSCHECK_TOL_S:
            issues.append(Issue(
                "warning",
                "fsm_crosscheck",
                f"contact_establish start disagreement: Qwen={qwen_contact_seg.start_t:.3f}s "
                f"FSM={fsm_contact_t:.3f}s (diff={diff*1000:.0f} ms > "
                f"{FSM_CROSSCHECK_TOL_S*1000:.0f} ms tolerance)",
            ))

        return issues

    def _estimate_contact_start(
        self,
        timestamps: np.ndarray,
        wrench: Optional[Any],
        gripper: Optional[Any],
    ) -> Optional[float]:
        """Simple proprioceptive estimate of contact_establish onset."""
        if wrench is not None:
            w = np.asarray(wrench, dtype=float)
            if w.ndim == 2 and w.shape[1] >= 3:
                force_mag = np.linalg.norm(w[:, :3], axis=1)
                above = np.where(force_mag > self.fsm_threshold)[0]
                if len(above) > 0:
                    return float(timestamps[above[0]])
        if gripper is not None:
            g = np.asarray(gripper, dtype=float)
            binary = (g > 0.5).astype(int)
            for i in range(1, len(binary)):
                if binary[i - 1] == 0 and binary[i] == 1:
                    return float(timestamps[i])
        return None

    def _check_coverage(self, annotation: EpisodeAnnotation) -> List[Issue]:
        issues = []
        present = {Phase(s.phase_id) for s in annotation.segments}
        expected = _EXPECTED_PHASES.get(annotation.task, set())
        missing = expected - present
        if missing:
            missing_names = [PHASE_NAMES[p] for p in sorted(missing)]
            # It's a warning, not error — justification may be in notes
            issues.append(Issue(
                "warning",
                "coverage",
                f"Expected phases missing: {missing_names}. "
                f"If intentional, document in 'notes'.",
            ))
        return issues
