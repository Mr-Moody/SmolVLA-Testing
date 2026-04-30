"""
Patch lerobot tokenizer processor to avoid hard-failing when task is None.

Experimental behavior:
- If a sample has no task string, substitute a fallback task label.
- Log a warning instead of raising ValueError.

Safe to re-run (idempotent).
"""

from __future__ import annotations

import os
from pathlib import Path


FALLBACK_TASK_TEXT = "Picking up a soup can and placing it on a cardboard surface"
PATCH_MARKER = "# SmolVLA-Testing patch: allow missing task with fallback label."


def resolve_target() -> Path:
    env_root = os.getenv("LEROBOT_ROOT")
    if env_root:
        return Path(env_root).expanduser() / "src" / "lerobot" / "processor" / "tokenizer_processor.py"
    return Path.home() / "smolvla_project" / "lerobot" / "src" / "lerobot" / "processor" / "tokenizer_processor.py"


TARGET = resolve_target()


def build_replacement_block() -> str:
    return f'''            # SmolVLA-Testing patch: allow missing task with fallback label.
            # Build a fallback with the correct batch length so language tokens
            # align with image/state tensors at train time.
            batch_size = 1
            for obs_value in observation.values():
                if hasattr(obs_value, "shape") and len(obs_value.shape) > 0:
                    batch_size = int(obs_value.shape[0])
                    break
                if isinstance(obs_value, list) and len(obs_value) > 0:
                    batch_size = len(obs_value)
                    break
            logging.warning(
                "Task is None in observation; substituting fallback label for batch_size=%s.",
                batch_size,
            )
            task = ["{FALLBACK_TASK_TEXT}"] * batch_size
'''


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Could not find target file: {TARGET}")

    source = TARGET.read_text()
    replacement = build_replacement_block()

    if PATCH_MARKER in source:
        candidates = [
            '            task = ["unknown task"] * batch_size',
            f'            task = ["{FALLBACK_TASK_TEXT}"] * batch_size',
            '            task = [FALLBACK_TASK_TEXT] * batch_size',
        ]
        for candidate in candidates:
            if candidate in source:
                updated = source.replace(candidate, f'            task = ["{FALLBACK_TASK_TEXT}"] * batch_size', 1)
                TARGET.write_text(updated)
                print(f"Patched successfully (updated existing patch): {TARGET}")
                print("Missing tasks will now use a per-batch fallback label list.")
                return

        print("ERROR: Found patch marker but could not locate the fallback task assignment line.")
        print(f"Target: {TARGET}")
        raise SystemExit(1)

    original = '            raise ValueError("Task cannot be None")'
    if original not in source:
        print("ERROR: Could not find expected strict task check in tokenizer_processor.py")
        print(f"Target: {TARGET}")
        raise SystemExit(1)

    patched = source.replace(original, replacement, 1)
    TARGET.write_text(patched)
    print(f"Patched successfully: {TARGET}")
    print(f"Missing tasks will now use fallback label '{FALLBACK_TASK_TEXT}'.")


if __name__ == "__main__":
    main()
