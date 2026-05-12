#!/usr/bin/env python3
"""
Post-process per-frame VLM labels using the canonical task sequence as a
constraint. The 8B model's per-frame outputs are unreliable, but the
*sequence* of subtasks in a successful episode is highly constrained. By
projecting the noisy per-frame predictions onto a state machine matching
the canonical order, we recover most of the true label sequence.

Algorithm (batch, per episode):

  1. Walk the per-frame labels in order.
  2. For each frame's predicted label:
     a. If the label is the current state (self-loop) -> accept, reset jump
        accumulator.
     b. If the label is a valid forward transition from the current state
        (next allowed state, or skip-allowed state like nudge/align which
        are optional) -> advance to that state, reset accumulator.
     c. If the label is the explicit backward-loop transition
        (grasp -> approach during failed grasp) -> advance to that state,
        reset accumulator.
     d. Otherwise (unreachable) -> emit the current state, increment a
        counter for the unreachable label. If the SAME unreachable label
        has been observed for `--jump-threshold` consecutive frames, accept
        it as a forward jump and update the state. If a different label
        breaks the streak, reset the counter for that prior label.

  3. Emit the corrected label for each frame.
  4. Optionally collapse consecutive identical labels into segments
     (matches the existing collapse_labels_to_segments format).

The state machine:

    None
     -> approach_MSD_plug                          [self-loop ok]
     -> approach_MSD_plug,grasp_the_plug           [self-loop ok]
     -> grasp_the_plug                             [self-loop ok; back to approach_MSD_plug allowed (failed grasp)]
     -> move_the_plug_to_the_socket                [self-loop ok]
     -> place_the_plug_in_the_socket               [self-loop ok]
     -> nudging_the_plug_into_the_socket           [optional, self-loop ok]
     -> align_handle                               [optional, self-loop ok]
     -> push_down_on_the_plug                      [self-loop ok, terminal]

Optional states (nudge, align) can be skipped — the decoder allows
transitioning past them.

Usage:
    python postprocess_subtasks.py --input raw_subtasks.jsonl --output cleaned_subtasks.jsonl
    python postprocess_subtasks.py --input raw_subtasks.jsonl --output cleaned.jsonl --jump-threshold 3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# Canonical sequence in order. Optional states marked separately so we know
# they can be skipped over.
CANONICAL_SEQUENCE = [
    "approach_MSD_plug",
    "approach_MSD_plug,grasp_the_plug",
    "grasp_the_plug",
    "move_the_plug_to_the_socket",
    "place_the_plug_in_the_socket",
    "nudging_the_plug_into_the_socket",
    "align_handle",
    "push_down_on_the_plug",
]

OPTIONAL_STATES = {
    "nudging_the_plug_into_the_socket",
    "align_handle",
}

# State indices for fast comparison.
STATE_INDEX = {state: idx for idx, state in enumerate(CANONICAL_SEQUENCE)}

# Failed-grasp backward loop: from grasp_the_plug we can return to
# approach_MSD_plug (or the merged label) if the grasp is released.
BACKWARD_TRANSITIONS = {
    "grasp_the_plug": {"approach_MSD_plug", "approach_MSD_plug,grasp_the_plug"},
    "approach_MSD_plug,grasp_the_plug": {"approach_MSD_plug"},
}


def normalize_label(raw: str) -> str:
    """
    Canonicalize a model output: lowercase, strip whitespace, sort the
    members of a multi-label so 'grasp,approach' == 'approach,grasp'.
    """
    if not raw:
        return ""
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    parts = sorted(set(parts))
    return ",".join(parts)


# Build a normalized->canonical lookup so we accept either ordering.
_CANONICAL_LOOKUP: dict[str, str] = {}
for state in CANONICAL_SEQUENCE:
    _CANONICAL_LOOKUP[normalize_label(state)] = state


def to_canonical(raw: str) -> str | None:
    """Map a raw model label string to a canonical state, or None if unknown."""
    norm = normalize_label(raw)
    return _CANONICAL_LOOKUP.get(norm)


def is_forward_transition(current: str | None, candidate: str) -> bool:
    """
    True if `candidate` is the next allowed state from `current`. Optional
    states can be skipped, so the next state may be 1, 2, or 3 positions
    ahead in CANONICAL_SEQUENCE if the intervening states are all optional.
    """
    if candidate not in STATE_INDEX:
        return False
    if current is None:
        # Only valid first state is approach_MSD_plug or the merge label.
        return candidate in ("approach_MSD_plug", "approach_MSD_plug,grasp_the_plug")
    cur_idx = STATE_INDEX[current]
    cand_idx = STATE_INDEX[candidate]
    if cand_idx <= cur_idx:
        return False
    # Allow skipping over consecutive optional states.
    for between_idx in range(cur_idx + 1, cand_idx):
        if CANONICAL_SEQUENCE[between_idx] not in OPTIONAL_STATES:
            return False
    return True


def is_backward_transition(current: str | None, candidate: str) -> bool:
    if current is None:
        return False
    return candidate in BACKWARD_TRANSITIONS.get(current, set())


def decode_episode(
    raw_labels: list[str],
    jump_threshold: int = 3,
) -> tuple[list[str], dict[str, int]]:
    """
    Run the constrained decoder over an episode's per-frame labels.

    Returns (corrected_labels, stats).
    """
    state: str | None = None
    out: list[str] = []
    pending_label: str | None = None
    pending_count = 0

    stats = defaultdict(int)

    for raw in raw_labels:
        canonical = to_canonical(raw)
        emitted: str

        if canonical is None:
            # Unknown / unparseable -> stay in current state
            emitted = state or "approach_MSD_plug"
            stats["unparseable"] += 1
            pending_label = None
            pending_count = 0

        elif canonical == state:
            # Self-loop -> accept
            emitted = state
            stats["self_loop"] += 1
            pending_label = None
            pending_count = 0

        elif is_forward_transition(state, canonical):
            # Valid forward step -> advance
            state = canonical
            emitted = state
            stats["forward"] += 1
            pending_label = None
            pending_count = 0

        elif is_backward_transition(state, canonical):
            # Valid failed-grasp loop -> advance (well, retreat)
            state = canonical
            emitted = state
            stats["backward_loop"] += 1
            pending_label = None
            pending_count = 0

        else:
            # Unreachable label. Accumulate; if the same unreachable label
            # repeats for `jump_threshold` consecutive frames, accept the
            # jump as a forward correction.
            if canonical == pending_label:
                pending_count += 1
            else:
                pending_label = canonical
                pending_count = 1

            if pending_count >= jump_threshold and STATE_INDEX.get(canonical, -1) > STATE_INDEX.get(state or "", -1):
                # Accept the jump.
                state = canonical
                emitted = state
                stats["accepted_jump"] += 1
                pending_label = None
                pending_count = 0
                # Backfill the previous (jump_threshold - 1) emissions to
                # this label too, since they were the same unreachable
                # observation. This avoids a sudden flip.
                backfill_n = min(jump_threshold - 1, len(out))
                for k in range(1, backfill_n + 1):
                    out[-k] = state
                    stats["backfilled"] += 1
            else:
                emitted = state or "approach_MSD_plug"
                stats["unreachable_held"] += 1

        out.append(emitted)

    if state is None:
        # Episode never advanced past the start. Default everything to the
        # initial state so we emit something usable.
        out = ["approach_MSD_plug"] * len(raw_labels)

    return out, dict(stats)


def collapse_to_segments(
    corrected_labels: list[str],
    timestamps_s: list[float],
    episode_start_s: float,
    episode_end_s: float,
) -> list[dict[str, Any]]:
    """
    Same logic as the production collapse_labels_to_segments, but operates
    on already-corrected single labels per frame (no per-label-set merging
    needed).
    """
    if not corrected_labels or not timestamps_s:
        return []
    if len(corrected_labels) != len(timestamps_s):
        raise ValueError("corrected_labels and timestamps_s must be same length")

    segments: list[dict[str, Any]] = []
    current_label = corrected_labels[0]
    current_start = episode_start_s

    for idx in range(1, len(corrected_labels)):
        if corrected_labels[idx] != current_label:
            segments.append(
                {
                    "phase": current_label,
                    "start_s": round(current_start, 4),
                    "end_s": round(timestamps_s[idx], 4),
                }
            )
            current_label = corrected_labels[idx]
            current_start = timestamps_s[idx]

    segments.append(
        {
            "phase": current_label,
            "start_s": round(current_start, 4),
            "end_s": round(episode_end_s, 4),
        }
    )
    return segments


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process VLM subtask labels with sequence constraints")
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with per-frame raw labels. Two formats supported: "
             "(a) the production segments format with 'subtasks' field, "
             "(b) a per-frame format with 'frame_labels' and 'timestamps_s' "
             "lists. See README in this script for details.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--jump-threshold", type=int, default=3,
                        help="Consecutive unreachable-label frames before accepting a forward jump (default: 3)")
    parser.add_argument("--print-stats", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))

    out_rows: list[dict[str, Any]] = []
    total_stats: dict[str, int] = defaultdict(int)

    for row in rows:
        episode_index = row.get("episode_index", 0)

        # Format (b): per-frame labels + timestamps.
        if "frame_labels" in row and "timestamps_s" in row:
            raw_labels = [
                ",".join(item) if isinstance(item, list) else str(item)
                for item in row["frame_labels"]
            ]
            timestamps_s = list(row["timestamps_s"])
            episode_start_s = float(row.get("episode_start_s", timestamps_s[0]))
            episode_end_s = float(row.get("episode_end_s", timestamps_s[-1]))

            corrected, stats = decode_episode(raw_labels, args.jump_threshold)
            for k, v in stats.items():
                total_stats[k] += v

            segments = collapse_to_segments(corrected, timestamps_s, episode_start_s, episode_end_s)
            out_rows.append({"episode_index": episode_index, "subtasks": segments})

        # Format (a): segments — re-expand to per-frame at the segment
        # boundaries, decode, then re-collapse. Less precise than format (b)
        # because we lose per-frame information, but useful for cleaning up
        # already-collapsed outputs.
        elif "subtasks" in row:
            print(f"WARNING: episode {episode_index} is in collapsed segment format; "
                  f"per-frame correction is approximate. Re-run inference with "
                  f"per-frame output for best results.")
            out_rows.append(row)

        else:
            print(f"WARNING: episode {episode_index} has unrecognized format; passing through unchanged.")
            out_rows.append(row)

    write_jsonl(Path(args.output), out_rows)
    print(f"Wrote {len(out_rows)} episode(s) to {args.output}")

    if args.print_stats:
        print()
        print("Decoder stats (across all episodes):")
        for k in sorted(total_stats):
            print(f"  {k}: {total_stats[k]}")


if __name__ == "__main__":
    main()