"""Pydantic models for Qwen3-VL phase annotation output.

Use ``parse_qwen_output(raw, episode_duration)`` to turn raw model text into a
validated ``EpisodeAnnotation``.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator
from pydantic import ConfigDict

from src.common.phases import Phase, PHASE_NAMES


class PhaseSegment(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    phase_id: int
    phase_name: str
    start_t: float
    end_t: float
    confidence: float
    evidence: str

    @field_validator("phase_id")
    @classmethod
    def validate_phase_id(cls, v):
        if v not in [p.value for p in Phase]:
            raise ValueError(f"phase_id must be 0–4, got {v}")
        return v

    @field_validator("phase_name")
    @classmethod
    def validate_phase_name(cls, v):
        valid = set(PHASE_NAMES.values())
        if v not in valid:
            raise ValueError(f"phase_name '{v}' not in {valid}")
        return v

    @model_validator(mode="after")
    def check_time_order(self):
        if self.start_t >= self.end_t:
            raise ValueError(
                f"start_t ({self.start_t}) must be < end_t ({self.end_t})"
            )
        return self

    @model_validator(mode="after")
    def check_name_matches_id(self):
        expected = PHASE_NAMES[Phase(self.phase_id)]
        if self.phase_name != expected:
            raise ValueError(
                f"phase_name '{self.phase_name}' does not match phase_id "
                f"{self.phase_id} (expected '{expected}')"
            )
        return self

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class EpisodeAnnotation(BaseModel):
    episode_id: str
    task: str
    segments: List[PhaseSegment]
    overall_confidence: float
    notes: Optional[str] = None

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        if v not in ("pick_place", "msd_plug"):
            raise ValueError(f"task must be 'pick_place' or 'msd_plug', got '{v}'")
        return v

    @field_validator("overall_confidence")
    @classmethod
    def validate_overall_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"overall_confidence must be in [0, 1], got {v}")
        return v

    @field_validator("segments")
    @classmethod
    def validate_segments_non_empty(cls, v):
        if len(v) == 0:
            raise ValueError("segments must not be empty")
        return v

    def validate_against_duration(self, episode_duration: float, tol: float = 0.05):
        """Check contiguity, first start = 0, last end = episode_duration.

        Args:
            episode_duration: Total episode length in seconds.
            tol: Allowed floating-point tolerance in seconds.

        Raises:
            ValueError: If any constraint is violated.
        """
        segs = self.segments
        if abs(segs[0].start_t) > tol:
            raise ValueError(
                f"First segment start_t ({segs[0].start_t}) must be ~0.0"
            )
        if abs(segs[-1].end_t - episode_duration) > tol:
            raise ValueError(
                f"Last segment end_t ({segs[-1].end_t}) must equal "
                f"episode_duration ({episode_duration})"
            )
        for i in range(1, len(segs)):
            gap = abs(segs[i].start_t - segs[i - 1].end_t)
            if gap > tol:
                raise ValueError(
                    f"Segments not contiguous: seg[{i-1}].end_t={segs[i-1].end_t} "
                    f"!= seg[{i}].start_t={segs[i].start_t}"
                )
            if segs[i].start_t < segs[i - 1].start_t:
                raise ValueError(
                    f"Segments overlap: seg[{i}].start_t={segs[i].start_t} "
                    f"< seg[{i-1}].end_t={segs[i-1].end_t}"
                )


def parse_qwen_output(raw: str, episode_duration: float) -> EpisodeAnnotation:
    """Parse raw Qwen3-VL text into a validated EpisodeAnnotation.

    Strips markdown fences, extracts the first JSON object, validates against
    the pydantic schema and episode duration.

    Args:
        raw: Raw text output from QwenAnnotator.annotate_episode().
        episode_duration: Total episode length in seconds for contiguity check.

    Returns:
        Validated EpisodeAnnotation.

    Raises:
        ValueError: If parsing or validation fails.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Extract the first JSON object
    brace_start = cleaned.find("{")
    if brace_start == -1:
        raise ValueError(f"No JSON object found in model output:\n{raw[:500]}")
    brace_end = cleaned.rfind("}")
    if brace_end == -1:
        raise ValueError(f"Unclosed JSON object in model output:\n{raw[:500]}")
    json_str = cleaned[brace_start : brace_end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}\nRaw JSON:\n{json_str[:500]}") from exc

    try:
        annotation = EpisodeAnnotation.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc

    annotation.validate_against_duration(episode_duration)
    return annotation
