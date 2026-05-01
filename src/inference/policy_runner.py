"""PolicyRunner — closed-loop episode execution for all three policy variants.

The user-facing language prompt is constant for the whole episode. Phase
signals are strictly internal (from RuntimeFSM or dataset subtask).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.common.phases import Phase, PHASE_NAMES
from src.fsm.runtime_fsm import RuntimeFSM, Observation

logger = logging.getLogger(__name__)

VALID_VARIANTS = ("smolvla_baseline", "smolvla_fsm", "pi0_subtask")


@dataclass
class StepLog:
    step: int
    timestamp: float
    phase: Optional[str]
    action: Optional[list]
    obs_tcp_pose: Optional[list]
    obs_gripper: Optional[float]


class PolicyRunner:
    """Executes a closed-loop episode for any of the three policy variants.

    Args:
        checkpoint_path: Path to the policy checkpoint directory.
        variant: One of ``"smolvla_baseline"``, ``"smolvla_fsm"``,
            ``"pi0_subtask"``.
        robot_interface: Object implementing ``get_observation() → Observation``
            and ``execute_action(action, phase) → None``.
        task: ``"pick_place"`` or ``"msd_plug"`` (needed for FSM and subtask).
        fsm_config: Dict of FSM thresholds (loaded from configs/fsm/).
        log_dir: Root directory for run logs. Defaults to ``outputs/runs/``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        variant: str,
        robot_interface: Any,
        task: str = "pick_place",
        fsm_config: Optional[Dict[str, Any]] = None,
        log_dir: str = "outputs/runs",
    ):
        assert variant in VALID_VARIANTS, f"variant must be one of {VALID_VARIANTS}"
        self.checkpoint_path = Path(checkpoint_path)
        self.variant = variant
        self.robot = robot_interface
        self.task = task
        self._log_dir = Path(log_dir)
        self._policy = None
        self._fsm: Optional[RuntimeFSM] = None
        self._fsm_config = fsm_config or {}

        if variant in ("smolvla_fsm", "pi0_subtask"):
            self._fsm = RuntimeFSM(task=task, config=self._fsm_config)

    def _load_policy(self):
        """Lazy-load policy from checkpoint. Raises if not implemented for variant."""
        if self._policy is not None:
            return
        logger.info("Loading policy from %s (variant=%s)", self.checkpoint_path, self.variant)
        if self.variant == "smolvla_baseline":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            self._policy = SmolVLAPolicy.from_pretrained(str(self.checkpoint_path))
        elif self.variant == "smolvla_fsm":
            from src.smolvla_fork.modeling_smolvla import SmolVLAPhasedPolicy
            self._policy = SmolVLAPhasedPolicy.from_pretrained(str(self.checkpoint_path))
        elif self.variant == "pi0_subtask":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            self._policy = PI0Policy.from_pretrained(str(self.checkpoint_path))
        logger.info("Policy loaded.")

    def run_episode(
        self,
        prompt: str,
        max_steps: int = 300,
    ) -> Dict[str, Any]:
        """Execute one episode.

        Args:
            prompt: User-facing language prompt (constant for the whole episode).
            max_steps: Maximum number of control steps.

        Returns:
            Dict with ``success``, ``steps``, ``phase_trace``, and ``log_path``.
        """
        self._load_policy()
        if self._fsm is not None:
            self._fsm.reset()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{self.variant}"
        run_dir = self._log_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "step_log.jsonl"

        phase_trace = []
        step_logs = []
        t_start = time.time()

        for step in range(max_steps):
            obs = self.robot.get_observation()
            current_phase: Optional[Phase] = None

            # Determine phase signal
            if self.variant == "smolvla_fsm":
                current_phase = self._fsm.step(obs)
            elif self.variant == "pi0_subtask":
                current_phase = self._fsm.step(obs)

            # Build batch for policy
            batch = self._obs_to_batch(obs, prompt, current_phase)

            # Get action from policy
            with __import__("torch").no_grad():
                if self.variant == "smolvla_fsm" and current_phase is not None:
                    import torch
                    phase_id = torch.tensor([int(current_phase)], dtype=torch.long)
                    action_tensor = self._policy.select_action(batch, phase_id=phase_id)
                else:
                    action_tensor = self._policy.select_action(batch)

            action = action_tensor.squeeze(0).cpu().numpy()

            # Execute on robot
            self.robot.execute_action(action, current_phase or Phase.FREE_MOTION)

            # Log
            phase_name = PHASE_NAMES[current_phase] if current_phase is not None else "none"
            phase_trace.append(phase_name)
            log = StepLog(
                step=step,
                timestamp=time.time() - t_start,
                phase=phase_name,
                action=action.tolist(),
                obs_tcp_pose=obs.tcp_pose.tolist(),
                obs_gripper=float(obs.gripper_state),
            )
            step_logs.append(log)

        # Write log
        with open(log_file, "w") as f:
            for log in step_logs:
                f.write(json.dumps(asdict(log)) + "\n")

        return {
            "success": None,  # caller determines success from robot state
            "steps": len(step_logs),
            "phase_trace": phase_trace,
            "log_path": str(log_file),
        }

    def _obs_to_batch(self, obs: Observation, prompt: str, phase: Optional[Phase]) -> Dict:
        """Convert Observation to a policy input batch dict."""
        import torch
        state = torch.tensor(obs.tcp_pose, dtype=torch.float32).unsqueeze(0)
        batch = {
            "observation.state": state,
            "observation.language_tokens": self._encode_prompt(prompt),
            "observation.language_attention_mask": torch.ones(1, 48, dtype=torch.long),
        }
        if phase is not None and self.variant == "pi0_subtask":
            # For π0: modify prompt to include subtask string
            subtask = f"{self.task}.{PHASE_NAMES[phase]}"
            batch["observation.language_tokens"] = self._encode_prompt(subtask + " " + prompt)
        return batch

    def _encode_prompt(self, text: str):
        """Encode a text prompt. Placeholder — real tokenization done by policy."""
        import torch
        return torch.zeros(1, 48, dtype=torch.long)
