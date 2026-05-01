"""Prompt builder for Qwen3-VL phase annotation.

Composes a system message (taxonomy + schema) and a user message (video + task),
optionally inserting few-shot examples loaded from data/gold/few_shot.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.phases import Phase, PHASE_NAMES, PHASE_DESCRIPTIONS

logger = logging.getLogger(__name__)

_GOLD_PATH = Path(__file__).resolve().parents[2] / "data" / "gold" / "few_shot.json"

# Explicit observable cues per phase (supplement the taxonomy descriptions)
_OBSERVABLE_CUES: Dict[Phase, str] = {
    Phase.FREE_MOTION: (
        "Gripper/fingers NOT touching any surface. Arm moving smoothly through air. "
        "No deformation of object or gripper fingers visible."
    ),
    Phase.FINE_ALIGN: (
        "Gripper/connector very close to target but NOT yet touching. "
        "No contact yet — gripper fingers not compressed."
    ),
    Phase.CONTACT_ESTABLISH: (
        "Gripper fingers visibly compressing onto the object OR pin tips touching "
        "the connector face. First measurable force spike. "
        "Key discriminator from fine_align: physical contact is visible."
    ),
    Phase.CONSTRAINED_MOTION: (
        "Object is grasped AND moving (lift, in-hand motion, or insertion stroke). "
        "Gripper remains closed throughout this phase."
    ),
    Phase.VERIFY_RELEASE: (
        "Object is at target location (placed or fully inserted). "
        "Gripper opening visible. Arm beginning to retract."
    ),
}

_OUTPUT_SCHEMA_SUMMARY = """
Output ONLY valid JSON matching this schema exactly:
{
  "episode_id": "<string>",
  "task": "pick_place" | "msd_plug",
  "segments": [
    {
      "phase_id": <int 0-4>,
      "phase_name": "<canonical name>",
      "start_t": <float seconds>,
      "end_t": <float seconds>,
      "confidence": <float 0-1>,
      "evidence": "<one sentence describing what you see>"
    }
  ],
  "overall_confidence": <float 0-1>,
  "notes": "<optional annotator commentary>"
}
Rules:
- Segments must be contiguous and non-overlapping.
- First segment start_t = 0.0; last segment end_t = episode duration.
- phase_name must exactly match the canonical name (free_motion, fine_align,
  contact_establish, constrained_motion, verify_release).
- Do NOT output markdown fences or any text outside the JSON object.
"""


class PromptBuilder:
    """Build Qwen3-VL prompts for phase annotation.

    Args:
        task: ``"pick_place"`` or ``"msd_plug"``.
        gold_path: Path to few-shot JSON. If None, uses the default repo path.
    """

    def __init__(self, task: str, gold_path: Optional[Path] = None):
        assert task in ("pick_place", "msd_plug"), f"Unknown task: {task}"
        self.task = task
        self._gold_path = Path(gold_path) if gold_path else _GOLD_PATH
        self._few_shot: List[Dict[str, Any]] = []
        self._load_few_shot()

    def _load_few_shot(self):
        if self._gold_path.exists():
            try:
                data = json.loads(self._gold_path.read_text())
                self._few_shot = [ex for ex in data if ex.get("task") == self.task]
                logger.info(
                    "Loaded %d few-shot examples for task '%s' from %s",
                    len(self._few_shot),
                    self.task,
                    self._gold_path,
                )
            except Exception as exc:
                logger.warning("Failed to load few-shot examples: %s", exc)
        else:
            logger.warning(
                "Gold few-shot file not found at %s — proceeding without few-shot examples. "
                "Run scripts/build_gold_set.py to create it.",
                self._gold_path,
            )

    def build_system_message(self) -> str:
        """Return the system message string."""
        lines = [
            "You are an expert annotator of robot manipulation videos.",
            "",
            "## Five-Phase Taxonomy",
            "",
        ]
        for phase in Phase:
            name = PHASE_NAMES[phase]
            pick_desc = PHASE_DESCRIPTIONS["pick_place"][phase]
            msd_desc = PHASE_DESCRIPTIONS["msd_plug"][phase]
            cue = _OBSERVABLE_CUES[phase]
            lines += [
                f"### Phase {int(phase)}: {name}",
                f"- Pick-and-place: {pick_desc}",
                f"- MSD plugging:   {msd_desc}",
                f"- Observable cue: {cue}",
                "",
            ]

        lines += [
            "## Output Schema",
            _OUTPUT_SCHEMA_SUMMARY,
        ]
        return "\n".join(lines)

    def build_user_message(self, episode_id: str, episode_duration: float) -> str:
        """Return the user message (without the video multimodal content)."""
        return (
            f"Task: {self.task}\n"
            f"Episode ID: {episode_id}\n"
            f"Episode duration: {episode_duration:.2f} seconds\n\n"
            f"Watch the video and annotate every phase segment. "
            f"Output only the JSON object — no other text."
        )

    def build_few_shot_turns(self) -> List[Dict[str, str]]:
        """Return a list of (user_text, assistant_json) dicts for up to 3 examples."""
        turns = []
        for ex in self._few_shot[:3]:
            user_text = (
                f"Task: {ex.get('task', self.task)}\n"
                f"Episode ID: {ex.get('episode_id', 'gold_example')}\n"
                f"Episode duration: {ex.get('duration_s', 0.0):.2f} seconds\n\n"
                f"Watch the video and annotate every phase segment. "
                f"Output only the JSON object."
            )
            assistant_json = json.dumps(ex.get("annotation", ex), indent=2)
            turns.append({"user": user_text, "assistant": assistant_json})
        return turns
