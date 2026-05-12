#!/usr/bin/env python3
"""
v3 debug: address the "model thinks plug is always at socket" failure mode
by:

1. Forcing a binary question about plug attachment FIRST (held vs not held),
   before asking for fine-grained location.
2. Explicitly directing which camera to use for which question. The blue
   socket block is always visible in third-person, which biases the model
   toward "plug is at socket" answers. Telling the model to use the wrist
   camera for held/not-held disambiguation removes that confound.
3. Adding a --wrist-only flag for an A/B test where third-person is dropped
   entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from label_subtasks import (  # type: ignore
        extract_frame_at_index,
        find_video_sources,
        get_dataset_root,
        get_video_fps,
        load_episode_windows,
        load_frame_metadata,
        parse_subtasks,
        save_frame_temp,
    )
except ImportError as exc:
    print(f"ERROR: Could not import production script helpers: {exc}")
    sys.exit(1)


MOTION_PROMPT_BOTH_CAMS = """You are analyzing a sequence of robot manipulation frames to extract MOTION
information. The frames show a green two-finger gripper that picks up a small
black plug from a table and inserts it into a socket on a blue block.

You will receive frame pairs at different points in time. Each pair contains:
  - a third-person view (wider scene, shows table AND blue block together)
  - a wrist-mounted view (looks down from the gripper, shows what is directly
    below the gripper)

CAMERA USAGE — IMPORTANT
The third-person view ALWAYS shows the blue socket block somewhere in frame.
This does NOT mean the plug is at the socket. To determine where the plug
actually is, use the WRIST CAMERA as the primary signal:
  - If the wrist camera shows the table surface with the plug resting on it,
    the plug is on_table (regardless of what is visible in third-person).
  - If the wrist camera shows the plug between the gripper fingers with no
    table surface visible underneath, the plug is held_by_gripper.
  - If the wrist camera shows the blue socket block close-up underneath,
    the plug is at_socket or in_socket.
Use third-person to judge gripper translation across the workspace.

Output EXACTLY these five lines, in this order, no extra text:

PLUG_HELD: <one of: no | yes>
  - Answer this FIRST, using the wrist camera. Is the plug being held by the
    gripper fingers (off the table, between the fingers)? Yes or no.

PLUG_LOCATION: <one of: on_table | held_in_air | at_socket | in_socket>
  - on_table: PLUG_HELD=no AND plug rests on the table
  - held_in_air: PLUG_HELD=yes AND plug is in transit, not at the socket
  - at_socket: plug is touching or directly above the socket
  - in_socket: plug is inserted into the socket cavity
  - This MUST be consistent with PLUG_HELD.

GRIPPER_TRANSLATION: <one of: stationary | small | large>
  - stationary: gripper position barely changes across the sequence
  - small: gripper moves a few cm (micro-adjustment range)
  - large: gripper moves substantially across the workspace

FINGER_MOTION: <one of: stationary_open | stationary_closed | closing | opening>
  - Use the wrist camera. Compare finger spacing across frames.
  - stationary_open: fingers held apart with no change in spacing
  - stationary_closed: fingers held together / on plug with no change
  - closing: finger spacing decreases across the sequence
  - opening: finger spacing increases across the sequence

PLUG_LOCATION_CHANGE: <one of: no_change | table_to_held | held_to_socket | settling_in_socket>
  - no_change: plug stays in the same place across the sequence
  - table_to_held: plug starts on table, ends held by gripper
  - held_to_socket: plug starts held in air, ends at/in socket
  - settling_in_socket: plug is at/in socket throughout, possibly being seated"""


MOTION_PROMPT_WRIST_ONLY = """You are analyzing a sequence of robot manipulation frames to extract MOTION
information. The frames show a green two-finger gripper that picks up a small
black plug from a table and inserts it into a socket on a blue block.

You are seeing only the WRIST-MOUNTED camera, which looks down from the
gripper and shows what is directly below it. The third-person view has been
withheld intentionally so you can focus on what the gripper is doing.

What you should see:
  - During pickup: the table surface with the plug on it.
  - When held: the plug between the fingers, no table visible underneath.
  - During insertion: the blue socket block close-up.

Output EXACTLY these five lines, in this order, no extra text:

PLUG_HELD: <one of: no | yes>
  - Is the plug being held by the gripper fingers (off the table, between
    the fingers)?

PLUG_LOCATION: <one of: on_table | held_in_air | at_socket | in_socket>
  - on_table: PLUG_HELD=no AND plug rests on the table
  - held_in_air: PLUG_HELD=yes AND plug is in transit, not at the socket
  - at_socket: plug is touching or directly above the socket
  - in_socket: plug is inserted into the socket cavity

GRIPPER_TRANSLATION: <one of: stationary | small | large>
  - Judge from how the scene under the gripper changes across frames.
  - stationary: scene barely changes
  - small: scene shifts a small amount (micro-adjustment)
  - large: scene shifts substantially (gripper translating workspace)

FINGER_MOTION: <one of: stationary_open | stationary_closed | closing | opening>
  - Compare finger spacing across frames.

PLUG_LOCATION_CHANGE: <one of: no_change | table_to_held | held_to_socket | settling_in_socket>"""


def build_label_prompt(motion_summary: str, history: list[str]) -> str:
    history_text = (
        " -> ".join(history) if history else "(empty — assume start of approach_MSD_plug)"
    )
    return f"""You are labeling a robot manipulation subtask from a single moment in time.
Task: pick up an MSD plug from the table and insert it into a socket on a blue block.

You will receive ONE pair of synchronized images and a pre-computed MOTION
SUMMARY for the moment around these images. Trust the motion summary — it
was computed from a wider time window than you can see, with explicit
direction on which camera to trust for which question.

LABELS:
- approach_MSD_plug: gripper is empty, fingers open, moving toward or
  positioning over the plug on the table.
- grasp_the_plug: fingers are actively closing around the plug. Plug is
  still on the table or just being lifted.
- move_the_plug_to_the_socket: plug is held by closed fingers and being
  carried through space toward the socket. Plug not yet at the socket.
- place_the_plug_in_the_socket: plug is at/near the socket, being inserted
  while still held. Spans final approach, contact, insertion, pre-release.
- nudging_the_plug_into_the_socket: plug is in the socket, gripper closed,
  applying lateral pressure to seat deeper. UNCOMMON.
- align_handle: plug is mostly seated, gripper is pushing a tilted handle
  upright. UNCOMMON.
- push_down_on_the_plug: gripper has RELEASED the plug (fingers open) and
  is pressing down on the top to lock it in. Always last in the sequence.

ALLOWED MULTI-LABEL: approach_MSD_plug,grasp_the_plug
  Use this when fingers are actively closing AND gripper is still making
  small position adjustments.

DECISION RUBRIC — branch on PLUG_HELD first, then PLUG_LOCATION.

If PLUG_HELD = no:
  - PLUG_LOCATION must be on_table.
  - FINGER_MOTION = closing -> grasp_the_plug
    (or approach_MSD_plug,grasp_the_plug if GRIPPER_TRANSLATION = small)
  - otherwise -> approach_MSD_plug

If PLUG_HELD = yes:
  - PLUG_LOCATION = held_in_air -> move_the_plug_to_the_socket
  - PLUG_LOCATION = at_socket or in_socket:
    * lateral pressure visible in active frame -> nudging_the_plug_into_the_socket
    * handle visibly tilted -> align_handle
    * FINGER_MOTION = opening, gripper pressing down -> push_down_on_the_plug
    * otherwise -> place_the_plug_in_the_socket

MOTION SUMMARY:
{motion_summary}

HISTORY (recent labels, oldest first):
{history_text}

Output ONLY the label string, comma-separated if multi-label, no spaces."""


def find_frame_indices_for_timestamp(
    frame_metadata: list[dict],
    target_time_s: float,
    context_window: int,
    sampling_hz: float,
    video_fps: float,
) -> tuple[list[int], int]:
    if not frame_metadata:
        return [], 0
    target_ns = int(target_time_s * 1e9)
    best_idx = 0
    best_delta = float("inf")
    for idx, meta in enumerate(frame_metadata):
        ts_ns = meta.get("host_timestamp_ns")
        if ts_ns is None:
            continue
        delta = abs(int(ts_ns) - target_ns)
        if delta < best_delta:
            best_delta = delta
            best_idx = idx
    step = max(1, int(round(video_fps / max(sampling_hz, 1e-6))))
    indices: list[int] = []
    for offset in range(-context_window, context_window + 1):
        meta_idx = best_idx + offset * step
        if 0 <= meta_idx < len(frame_metadata):
            indices.append(meta_idx)
    if not indices:
        indices = [best_idx]
    active_pos = indices.index(best_idx) if best_idx in indices else len(indices) // 2
    return indices, active_pos


def is_wrist_camera(video_source: Path) -> bool:
    """
    Heuristic: wrist cameras typically have 'wrist' or 'd405' in the path.
    Adjust if your naming differs.
    """
    name = video_source.parent.name.lower()
    parent_name = video_source.parent.parent.name.lower() if video_source.parent.parent else ""
    return "wrist" in name or "wrist" in parent_name


def build_motion_call_content(
    frame_paths_by_position: list[list[str]],
    active_pos: int,
    wrist_only: bool,
) -> list[dict]:
    content: list[dict] = []
    for ctx_idx, paths in enumerate(frame_paths_by_position):
        offset = ctx_idx - active_pos
        if offset < 0:
            marker = f"Frame {offset} (before ACTIVE):"
        elif offset == 0:
            marker = "ACTIVE pair (the moment to label):"
        else:
            marker = f"Frame +{offset} (after ACTIVE):"
        content.append({"type": "text", "text": marker})
        for path in paths:
            if path and Path(path).exists():
                content.append({"type": "image", "image": path})
    prompt = MOTION_PROMPT_WRIST_ONLY if wrist_only else MOTION_PROMPT_BOTH_CAMS
    content.append({"type": "text", "text": prompt})
    return content


def build_label_call_content(
    active_paths: list[str], motion_summary: str, history: list[str]
) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": "ACTIVE pair (this is the moment to label):"}
    ]
    for path in active_paths:
        if path and Path(path).exists():
            content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": build_label_prompt(motion_summary, history)})
    return content


def run_vlm_call(llm, tokenizer, sampling_params, content: list[dict]) -> str:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return "[qwen_vl_utils not available]"
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    outputs = llm.generate(
        [{"prompt": text, "multi_modal_data": {"image": image_inputs}}],
        sampling_params=sampling_params,
    )
    if outputs and outputs[0].outputs:
        return outputs[0].outputs[0].text.strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 two-call debug for subtask labelling")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--data-name", default="double_d405")
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--sampling-hz", type=float, default=5.0)
    parser.add_argument("--subtasks", default="")
    parser.add_argument("--subtasks-file", default=None)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--history", default="")
    parser.add_argument("--save-frames", default=None)
    parser.add_argument(
        "--wrist-only",
        action="store_true",
        help="Drop the third-person camera entirely. Tests whether removing "
             "the always-visible blue block fixes the at_socket bias.",
    )
    args = parser.parse_args()

    if args.context_window < 1:
        args.context_window = 1

    subtasks = parse_subtasks(args.subtasks, args.subtasks_file)
    video_sources = find_video_sources(args.video_path, args.data_name)
    if not video_sources:
        print(f"ERROR: No video sources for {args.video_path or args.data_name}")
        sys.exit(1)

    if args.wrist_only:
        wrist_sources = [vs for vs in video_sources if is_wrist_camera(vs)]
        if not wrist_sources:
            print("WARNING: --wrist-only set but no wrist cam detected by heuristic.")
            print("         Falling back to using the LAST video source as wrist.")
            wrist_sources = [video_sources[-1]]
        video_sources = wrist_sources
        print(f"WRIST-ONLY MODE: using {len(video_sources)} camera(s)")

    print("Video sources:")
    for vs in video_sources:
        print(f"  {vs}")
    print()

    dataset_root = get_dataset_root(video_sources, args.data_name)
    primary_camera = video_sources[0].parent.name
    frame_metadata = load_frame_metadata(dataset_root, primary_camera)
    video_fps = get_video_fps(video_sources[0])

    try:
        episodes = load_episode_windows(dataset_root)
        episode_start_s = int(episodes[0]["start_ns"]) / 1e9
        absolute_target_s = episode_start_s + args.time
        print(f"Episode 0 starts at absolute t={episode_start_s:.3f}s")
        print(f"Target: {args.time:.3f}s into episode 0 (absolute {absolute_target_s:.3f}s)")
    except Exception:
        absolute_target_s = args.time
    print()

    frame_indices, active_pos = find_frame_indices_for_timestamp(
        frame_metadata, absolute_target_s, args.context_window, args.sampling_hz, video_fps
    )
    if not frame_indices:
        print("ERROR: Could not resolve frame index")
        sys.exit(1)

    print("Frame indices (active marked with *): ", end="")
    for i, fi in enumerate(frame_indices):
        marker = "*" if i == active_pos else " "
        print(f"{marker}{fi}", end=" ")
    print()
    print()

    save_dir = Path(args.save_frames) if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_paths_by_position: list[list[str]] = []
    all_temp_files: list[str] = []
    try:
        for ctx_idx, frame_index in enumerate(frame_indices):
            position_paths: list[str] = []
            for video_source in video_sources:
                frame = extract_frame_at_index(video_source, frame_index)
                if frame is None:
                    continue
                tmp_path = save_frame_temp(frame)
                if not tmp_path:
                    continue
                position_paths.append(tmp_path)
                all_temp_files.append(tmp_path)
                if save_dir:
                    import shutil
                    label = "active" if ctx_idx == active_pos else f"ctx{ctx_idx - active_pos:+d}"
                    cam_name = video_source.parent.name
                    shutil.copy(tmp_path, save_dir / f"{label}_{cam_name}.jpg")
            frame_paths_by_position.append(position_paths)

        active_paths = frame_paths_by_position[active_pos]
        if not active_paths:
            print("ERROR: Active pair frames could not be extracted")
            sys.exit(1)

        from vllm import LLM, SamplingParams

        total_images = sum(len(p) for p in frame_paths_by_position)
        llm = LLM(
            model="Qwen/Qwen3-VL-8B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": max(1, total_images)},
        )
        tokenizer = llm.get_tokenizer()

        print("=" * 70)
        print(f"CALL 1: MOTION ANALYSIS  ({'WRIST-ONLY' if args.wrist_only else 'BOTH CAMERAS'})")
        print("=" * 70)
        motion_params = SamplingParams(max_tokens=250, temperature=0.0)
        motion_content = build_motion_call_content(
            frame_paths_by_position, active_pos, args.wrist_only
        )
        motion_summary = run_vlm_call(llm, tokenizer, motion_params, motion_content)
        print(motion_summary)
        print()

        print("=" * 70)
        print("CALL 2: LABEL DECISION")
        print("=" * 70)
        history_list = [h.strip() for h in args.history.split(",") if h.strip()] if args.history else []
        label_params = SamplingParams(max_tokens=32, temperature=0.0, repetition_penalty=1.05)
        label_content = build_label_call_content(active_paths, motion_summary, history_list)
        label_output = run_vlm_call(llm, tokenizer, label_params, label_content)
        print(f"Raw label: {label_output!r}")
        print()

    finally:
        for f in all_temp_files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()