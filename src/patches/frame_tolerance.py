"""
Patch lerobot's decode_video_frames_torchcodec to warn (not raise) when
timestamp tolerance is violated, and proceed with nearest-frame fallback.

This is intended for experimentation on imperfectly synchronized datasets.
Safe to re-run (idempotent).
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def resolve_target() -> Path:
    env_root = os.getenv("LEROBOT_ROOT")
    if env_root:
        return Path(env_root).expanduser() / "src" / "lerobot" / "datasets" / "video_utils.py"
    return Path.home() / "smolvla_project" / "lerobot" / "src" / "lerobot" / "datasets" / "video_utils.py"


TARGET = resolve_target()

PATCH_MARKER = "# SmolVLA-Testing patch: tolerate timestamp violations by using nearest frame."

# Match the strict-tolerance block (the one that raises FrameTimestampError).
# We constrain replacement to decode_video_frames_torchcodec only.
STRICT_BLOCK_RE = re.compile(
    r"    is_within_tol = min_ < tolerance_s\n"
    r"    if not is_within_tol\.all\(\):\n"
    r"(?:        .*\n)*?"
    r"\s*raise FrameTimestampError\(\n"
    r"(?:        .*\n)*?"
    r"        \)\n",
    re.MULTILINE,
)

TORCHCODEC_START_RE = re.compile(r"^def\s+decode_video_frames_torchcodec\s*\(", re.MULTILINE)

PATCH = """    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        # SmolVLA-Testing patch: tolerate timestamp violations by using nearest frame.
        logging.warning(
            "Timestamp tolerance violated (%s > %s). Using nearest frame fallback for %s.",
            min_[~is_within_tol],
            tolerance_s,
            video_path,
        )
"""


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Could not find target file: {TARGET}")

    source = TARGET.read_text()

    start_match = TORCHCODEC_START_RE.search(source)
    if not start_match:
        print("ERROR: Could not find decode_video_frames_torchcodec in video_utils.py")
        print(f"Target: {TARGET}")
        raise SystemExit(1)

    fn_start = start_match.start()
    next_def = re.search(r"^def\s+", source[fn_start + 1 :], re.MULTILINE)
    if next_def:
        fn_end = fn_start + 1 + next_def.start()
    else:
        fn_end = len(source)

    fn_src = source[fn_start:fn_end]

    if PATCH_MARKER in fn_src:
        # Even when already patched, verify strict raise block is absent
        if STRICT_BLOCK_RE.search(fn_src):
            print("ERROR: Patch marker exists but strict tolerance raise still present in decode_video_frames_torchcodec.")
            print("Refusing to continue; please re-sync this script and re-run it.")
            raise SystemExit(1)
        print(f"Patch already applied: {TARGET}")
        return

    strict_match = STRICT_BLOCK_RE.search(fn_src)
    if not strict_match:
        print("ERROR: Could not find strict tolerance raise block inside decode_video_frames_torchcodec")
        print(f"Target: {TARGET}")
        raise SystemExit(1)

    patched_fn = fn_src[: strict_match.start()] + PATCH + fn_src[strict_match.end() :]

    # Safety check: ensure strict raise block no longer exists in patched function.
    if STRICT_BLOCK_RE.search(patched_fn):
        print("ERROR: Failed to remove strict tolerance raise block from decode_video_frames_torchcodec")
        raise SystemExit(1)

    patched = source[:fn_start] + patched_fn + source[fn_end:]
    TARGET.write_text(patched)

    print(f"Patched successfully: {TARGET}")
    print("FrameTimestampError tolerance guard now logs warning and falls back to nearest frame.")


if __name__ == "__main__":
    main()
