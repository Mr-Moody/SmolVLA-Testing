#!/usr/bin/env python3
"""
Label robot manipulation subtasks in cleaned dataset videos with Qwen3-VL.

The script mirrors the input style of 07_inference_qwen_test.py:
- accept a --video-path or auto-discover videos under --data-name
- use synchronized camera feeds
- sample frames at a configurable rate

Output:
- JSONL in the same format as cleaned_datasets/example/subtasks.jsonl
- one line per episode:
  {"episode_index": 0, "subtasks": [{"phase": "...", "start_s": ..., "end_s": ...}, ...]}

Subtasks are configurable via --subtasks or --subtasks-file so this script can
be reused for different tasks later.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_SUBTASKS = [
    "approach_MSD_plug",
    "positioning_the_gripper",
    "grasp_the_plug",
    "move_the_plug_to_the_socket",
    "place_the_plug_in_the_socket",
    "nudging_the_plug_into_the_socket",
    "align_handle",
    "push_down_on_the_plug",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_subtasks(subtasks_arg: str | None, subtasks_file: str | None) -> list[str]:
    if subtasks_file:
        path = Path(subtasks_file)
        if not path.exists():
            raise FileNotFoundError(f"Subtasks file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = load_jsonl(path)
            values: list[str] = []
            for row in rows:
                value = row.get("phase") or row.get("name") or row.get("subtask")
                if value:
                    values.append(str(value))
            if values:
                return values
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                values = []
                for item in data:
                    if isinstance(item, str):
                        values.append(item)
                    elif isinstance(item, dict):
                        value = item.get("phase") or item.get("name") or item.get("subtask")
                        if value:
                            values.append(str(value))
                if values:
                    return values
            raise ValueError(f"Could not parse subtasks from {path}")

    if subtasks_arg:
        values = [item.strip() for item in subtasks_arg.split(",") if item.strip()]
        if values:
            return values

    return list(DEFAULT_SUBTASKS)


def find_video_sources(video_path: str | None = None, data_name: str = "double_d405") -> list[Path]:
    if video_path:
        path = Path(video_path)
        if path.is_dir():
            import glob

            return sorted(Path(p) for p in glob.glob(str(path / "**" / "rgb.mp4"), recursive=True))
        return [path]

    data_dir = Path(f"/scratch0/xparker/cleaned_datasets/{data_name}/cameras")
    if not data_dir.exists():
        return []

    videos = sorted(
        camera_dir / "rgb.mp4"
        for camera_dir in data_dir.iterdir()
        if camera_dir.is_dir() and (camera_dir / "rgb.mp4").exists()
    )
    if videos:
        return videos

    import glob

    return sorted(Path(p) for p in glob.glob(str(data_dir / "**" / "rgb.mp4"), recursive=True))


def get_dataset_root(video_sources: list[Path], data_name: str) -> Path:
    for video_source in video_sources:
        for parent in [video_source.parent, *video_source.parents]:
            if (parent / "episode_events.jsonl").exists():
                return parent
    return Path(f"/scratch0/xparker/cleaned_datasets/{data_name}")


def get_video_frame_count(video_path: Path) -> int:
    try:
        import cv2
    except ImportError:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def get_video_fps(video_path: Path) -> float:
    try:
        import cv2
    except ImportError:
        return 30.0

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30.0
    finally:
        cap.release()


def extract_frame_at_index(video_path: Path, frame_index: int):
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed")
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    finally:
        cap.release()


def save_frame_temp(frame, temp_dir: str = "/tmp") -> str | None:
    try:
        import cv2
        import os
        import tempfile
    except ImportError:
        return None

    try:
        fd, fpath = tempfile.mkstemp(suffix=".jpg", dir=temp_dir)
        os.close(fd)
        cv2.imwrite(fpath, frame)
        return fpath
    except Exception as exc:
        print(f"Error saving frame: {exc}")
        return None


def load_episode_windows(dataset_dir: Path) -> list[dict[str, Any]]:
    events_path = dataset_dir / "episode_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Could not find episode_events.jsonl at {events_path}")

    events = load_jsonl(events_path)
    episodes: list[dict[str, Any]] = []
    pending_start: dict[str, Any] | None = None
    episode_index = 0

    for event in events:
        event_name = event.get("event")
        if event_name == "episode_start":
            pending_start = event
            continue
        if event_name == "episode_end" and pending_start is not None:
            start_ns = int(pending_start["robot_timestamp_ns"])
            end_ns = int(event["robot_timestamp_ns"])
            if end_ns > start_ns:
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                    }
                )
                episode_index += 1
            pending_start = None

    if not episodes:
        raise ValueError(f"No complete episode windows found in {events_path}")

    return episodes


def load_frame_metadata(dataset_dir: Path, primary_camera: str) -> list[dict[str, Any]]:
    frames_path = dataset_dir / "cameras" / primary_camera / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"Could not find frames.jsonl at {frames_path}")
    return load_jsonl(frames_path)


def load_robot_data(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Load robot.jsonl and return (records, sorted_timestamps_ns) for bisect lookup."""
    robot_path = dataset_dir / "robot.jsonl"
    if not robot_path.exists():
        return [], []
    records = load_jsonl(robot_path)
    records.sort(key=lambda r: r["timestamp_ns"])
    ts = [r["timestamp_ns"] for r in records]
    return records, ts


def get_robot_record_at_ns(
    robot_data: list[dict[str, Any]],
    robot_ts: list[int],
    query_ns: int,
    max_gap_ms: float = 200.0,
) -> dict[str, Any] | None:
    if not robot_data:
        return None
    idx = bisect.bisect_left(robot_ts, query_ns)
    candidates = [c for c in (idx, idx - 1) if 0 <= c < len(robot_ts)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(robot_ts[i] - query_ns))
    if abs(robot_ts[best] - query_ns) > max_gap_ms * 1e6:
        return None
    return robot_data[best]


def format_robot_states_for_prompt(
    records: list[dict[str, Any] | None],
    center_idx: int,
) -> str:
    lines = ["ROBOT STATE (aligned to camera frames, center = NOW):"]
    for i, rec in enumerate(records):
        offset = i - center_idx
        marker = " ← NOW (label this)" if offset == 0 else ""
        if rec is None:
            lines.append(f"  [{offset:+d}] no data{marker}")
            continue
        rs = rec["robot_state"]
        ea = rec["executed_action"]
        gs = rs["gripper_state"]
        gw = rs["gripper_width"] * 1000
        gc = ea["gripper_command"]
        d = ea["cartesian_delta_translation"]
        mag = math.sqrt(sum(x**2 for x in d)) * 1000
        dz = d[2] * 1000
        tcp_z = rs["tcp_position_xyz"][2] * 1000
        if gc == 1.0 and gs == "OPEN":
            cmd_str = "cmd=CLOSE"
        elif gc == 1.0 and gs == "CLOSE":
            cmd_str = "cmd=CLOSE"
        else:
            cmd_str = "cmd=open"
        dz_str = f"{dz:+.1f}"
        lines.append(
            f"  [{offset:+d}] gripper={gs}({gw:.1f}mm) {cmd_str}  "
            f"speed={mag:.1f}mm  dz={dz_str}mm  tcp_z={tcp_z:.0f}mm{marker}"
        )
    lines += [
        "",
        "GRIPPER HINTS:",
        "  gripper=OPEN + cmd=CLOSE + width falling   → grasp_the_plug (fingers actively closing)",
        "  gripper=CLOSE + width<20mm                 → plug in hand",
        "  gripper=CLOSE + speed>8mm                  → arm transporting plug",
        "  gripper=CLOSE + speed<4mm + dz<0           → inserting or nudging",
        "  gripper=OPEN + speed<3mm                   → positioning_the_gripper or released",
        "  gripper=OPEN + speed>8mm                   → approach_MSD_plug or recovering",
        "  gripper=OPEN + cmd=open + plug in socket   → push_down_on_the_plug",
    ]
    return "\n".join(lines)


def sample_episode_frame_indices(
    frame_metadata: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
    video_fps: float,
    sampling_hz: float,
) -> tuple[list[int], list[float]]:
    episode_indices: list[int] = []
    episode_timestamps_s: list[float] = []

    for frame_index, frame_meta in enumerate(frame_metadata):
        frame_ns = frame_meta.get("host_timestamp_ns")
        if frame_ns is None:
            continue
        frame_ns = int(frame_ns)
        if start_ns <= frame_ns <= end_ns:
            episode_indices.append(frame_index)
            episode_timestamps_s.append(frame_ns / 1e9)

    if not episode_indices:
        return [], []

    step = max(1, int(round(video_fps / max(sampling_hz, 1e-6))))
    sampled_indices = episode_indices[::step]
    sampled_timestamps_s = episode_timestamps_s[::step]

    if sampled_indices[0] != episode_indices[0]:
        sampled_indices.insert(0, episode_indices[0])
        sampled_timestamps_s.insert(0, episode_timestamps_s[0])

    if sampled_indices[-1] != episode_indices[-1]:
        sampled_indices.append(episode_indices[-1])
        sampled_timestamps_s.append(episode_timestamps_s[-1])

    return sampled_indices, sampled_timestamps_s


def create_subtask_prompt(
    subtasks: list[str],
    args,
    history: list[str] | None = None,
    robot_context: str | None = None,
) -> str:
    n = 3
    if history:
        recent = history[-n:]
        history_text = (
            f"Sequence so far (most recent last): {' → '.join(recent)}\n"
            f"Label the center frame as the current phase, the next phase, OR an earlier phase "
            f"if the plug has been dropped and the robot is recovering."
        )
    else:
        history_text = "Sequence so far: [episode start] — first label should be approach_MSD_plug."

    context_note = (
        f"You are given {2 * args.context_window + 1} sequential frames. "
        f"The CENTER frame is the one to label. Use surrounding frames only to judge motion direction."
    ) if args.context_window > 0 else (
        "You are given a single frame to label."
    )

    robot_section = f"\n{robot_context}\n" if robot_context else ""

    return f"""\
You are labeling robot arm subtasks for MSD plug insertion.
TASK: Insert a Micro-D Subminiature (MSD) plug into a socket on a blue block. The plug has a rectangular connector body and a thumb-screw handle.
KEY TERMS — PLUG = object held by gripper. SOCKET = fixed receptor on the blue block. These are different objects.

{context_note}
{history_text}
{robot_section}
CANONICAL SEQUENCE (normal progression — can restart from any earlier step if plug is dropped):
  1. approach_MSD_plug
  2. positioning_the_gripper
  3. grasp_the_plug
  4. move_the_plug_to_the_socket
  5. place_the_plug_in_the_socket
  6. nudging_the_plug_into_the_socket  [uncommon]
  7. align_handle                       [rare]
  8. push_down_on_the_plug

RECOVERY (plug dropped — look for plug on table, gripper empty or re-opening):
  If plug is visible on the table and gripper is empty → label as approach_MSD_plug (restart)
  If gripper is repositioning above a dropped plug → label as positioning_the_gripper
  Any step after grasp_the_plug can restart to approach_MSD_plug if plug is lost

SUBTASK DEFINITIONS:

approach_MSD_plug
  START: arm begins moving from rest toward the plug lying on the table
  END:   gripper fingertips reach the height of the plug handle OR settle directly above it
  VISUAL: broad fast arm motion; gripper open; plug unconstrained on table; gripper far from plug
  ≠ positioning_the_gripper: approach is fast/coarse; positioning is slow/fine at plug level

positioning_the_gripper
  START: gripper slows to make fine adjustments around the plug
  END:   fingers begin to close
  VISUAL: gripper open; fingertips at same height as plug handle; plug visible between open jaws; slow precise movements
  ≠ approach_MSD_plug: if arm is still in fast broad motion, use approach instead

grasp_the_plug
  START: fingers begin visibly narrowing toward the plug
  END:   fingers fully closed and stationary around plug
  VISUAL: finger gap shrinking; plug handle starting to be squeezed
  DURATION: typically under 1 second

move_the_plug_to_the_socket
  START: plug is firmly held, arm begins transporting it toward socket
  END:   any part of plug is directly above any part of the socket on the blue block
  VISUAL: arm in broad motion; plug in closed gripper; socket visible at a distance; clear transit motion
  ≠ place_the_plug_in_the_socket: once plug tip is above/adjacent to socket, use place instead

place_the_plug_in_the_socket
  START: plug tip is directly above or touching the socket entry
  END:   plug body is fully or mostly inside socket cavity
  VISUAL: plug tip entering socket hole; slow careful downward motion; plug and socket in close proximity

nudging_the_plug_into_the_socket  [UNCOMMON — most insertions skip this]
  START: plug is partially in socket but not fully seated
  END:   plug fully seated (no gap between connector and socket face)
  VISUAL: gripper fingers closed around plug that is already in socket; small lateral or pressing adjustments

align_handle  [RARE]
  START: plug handle is visibly tilted off-vertical after partial insertion
  END:   handle restored to upright position
  VISUAL: thumb-screw handle clearly leaning; gripper pushing handle upright

push_down_on_the_plug  [ALWAYS LAST]
  START: gripper OPENS and presses down on top face of already-inserted plug
  END:   arm begins to retract away
  VISUAL: gripper open; finger pads pressing on top surface of plug body; plug already seated in socket
  ≠ place_the_plug_in_the_socket: here the gripper is OPEN pressing from above, not inserting

VALID CO-OCCURRENCES (the only pairs that can be active simultaneously):
  positioning_the_gripper + grasp_the_plug      (fingers closing while still adjusting)
  move_the_plug_to_the_socket + place_the_plug_in_the_socket  (plug arriving at socket)
  nudging_the_plug_into_the_socket + align_handle              (simultaneous seating corrections)
All other combinations are INVALID. Do not output more than 2 labels at once.

TEMPORAL STABILITY: A subtask typically lasts several seconds. If you labeled the previous frame as X and nothing has clearly changed, output X again. Only switch when you see an unambiguous visual change.

OUTPUT FORMAT: Output one or two subtask names separated by a comma. No explanation. No punctuation other than the comma separator. No extra words.
Examples of valid output:
  approach_MSD_plug
  positioning_the_gripper, grasp_the_plug
  nudging_the_plug_into_the_socket"""


def label_frame_with_context(
    llm,
    tokenizer,
    sampling_params,
    frames: list[str],
    subtasks: list[str],
    args,
    history: list[str] | None = None,
    robot_context: str | None = None,
) -> list[str]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("WARNING: qwen_vl_utils not available")
        return [subtasks[-1]] if subtasks else ["idle"]

    prompt = create_subtask_prompt(subtasks, args, history, robot_context)
    content: list[dict[str, str]] = []
    for frame_path in frames:
        if frame_path and Path(frame_path).exists():
            content.append({"type": "image", "image": frame_path})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    outputs = llm.generate(
        [{"prompt": text, "multi_modal_data": {"image": image_inputs}}],
        sampling_params=sampling_params,
    )

    result_text = ""
    if outputs and outputs[0].outputs:
        result_text = outputs[0].outputs[0].text.strip().lower()

    labels: list[str] = []

    def add_label(label: str) -> None:
        if label not in labels:
            labels.append(label)

    raw_candidates = [
        piece.strip().strip('"\'')
        for piece in result_text.replace("\n", ",").split(",")
        if piece.strip()
    ]

    for candidate in raw_candidates:
        normalized_candidate = candidate.strip(". :;!()[]{}").strip()
        for subtask in subtasks:
            if normalized_candidate == subtask.lower() or subtask.lower() in normalized_candidate:
                add_label(subtask)

    if labels:
        return labels

    if result_text:
        first_token = result_text.split()[0].strip(".,:;!()[]{}")
        for subtask in subtasks:
            if first_token == subtask.lower():
                return [subtask]

    return [subtasks[-1]] if subtasks else ["idle"]


def collapse_labels_to_segments(
    labels: list[list[str]],
    timestamps_s: list[float],
    episode_start_s: float,
    episode_end_s: float,
) -> list[dict[str, Any]]:
    if not labels or not timestamps_s:
        return []

    intervals_by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for idx, frame_labels in enumerate(labels):
        if not frame_labels:
            continue
        interval_start = episode_start_s if idx == 0 else timestamps_s[idx]
        interval_end = timestamps_s[idx + 1] if idx + 1 < len(timestamps_s) else episode_end_s
        if interval_end <= interval_start:
            continue
        for label in frame_labels:
            intervals_by_label[label].append((interval_start, interval_end))

    segments: list[dict[str, Any]] = []
    for label in sorted(intervals_by_label):
        merged: list[tuple[float, float]] = []
        for start_s, end_s in sorted(intervals_by_label[label]):
            if not merged or start_s > merged[-1][1]:
                merged.append((start_s, end_s))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
        for start_s, end_s in merged:
            segments.append(
                {
                    "phase": label,
                    "start_s": round(start_s, 4),
                    "end_s": round(end_s, 4),
                }
            )

    segments.sort(key=lambda row: (row["start_s"], row["end_s"], row["phase"]))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Label subtasks in video using Qwen3-VL")
    parser.add_argument(
        "--video-path",
        default=None,
        help="Path to video file or directory (auto-detect if not provided)",
    )
    parser.add_argument(
        "--data-name",
        default="double_d405",
        help="Dataset folder name under /scratch0/<user>/cleaned_datasets (default: double_d405)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file. Defaults to <dataset_root>/subtasks.jsonl if omitted.",
    )
    parser.add_argument(
        "--subtasks",
        default="",
        help="Comma-separated subtask names to use as the label set.",
    )
    parser.add_argument(
        "--subtasks-file",
        default=None,
        help="Optional JSON or JSONL file containing subtask names.",
    )
    parser.add_argument(
        "--sampling-hz",
        type=float,
        default=5.0,
        help="Frame sampling rate in Hz (default: 5.0)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Number of neighboring sampled frames to include on each side.",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=3000,
        help="Max model sequence length",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=0,
        help="Optional cap on the number of episodes to process (0 means all).",
    )
    args = parser.parse_args()

    subtasks = parse_subtasks(args.subtasks, args.subtasks_file)
    video_sources = find_video_sources(args.video_path, args.data_name)
    if not video_sources:
        print(f"ERROR: No video sources found for {args.video_path or args.data_name}")
        sys.exit(1)

    print("Auto-detected video sources:")
    for video_source in video_sources:
        print(f" - {video_source}")
    print("")

    dataset_root = get_dataset_root(video_sources, args.data_name)
    output_path = Path(args.output) if args.output else dataset_root / "subtasks.jsonl"

    try:
        episodes = load_episode_windows(dataset_root)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.limit_episodes > 0:
        episodes = episodes[: args.limit_episodes]

    primary_camera = video_sources[0].parent.name
    try:
        frame_metadata = load_frame_metadata(dataset_root, primary_camera)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    robot_data, robot_ts = load_robot_data(dataset_root)
    if robot_data:
        print(f"Loaded {len(robot_data)} robot records for proprioceptive context.")
    else:
        print("WARNING: No robot.jsonl found — labeling without proprioceptive context.")

    frame_count = min(get_video_frame_count(video_source) for video_source in video_sources)
    if frame_count <= 0:
        print("ERROR: Could not determine video frame count")
        sys.exit(1)

    video_fps = get_video_fps(video_sources[0])
    frame_interval = max(1, int(round(video_fps / max(args.sampling_hz, 1e-6))))
    print(f"Video FPS: {video_fps:.2f} | sampling at {args.sampling_hz:.2f} Hz | step {frame_interval}")
    print(f"Using {len(subtasks)} subtasks: {', '.join(subtasks)}")
    print("")

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model="Qwen/Qwen3-VL-4B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": max(1, (2 * args.context_window + 1) * len(video_sources))},
        )
    except Exception as exc:
        print(f"ERROR: Failed to initialize model: {exc}")
        sys.exit(1)

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=24, temperature=0.0, repetition_penalty=1.05)

    results: list[dict[str, Any]] = []

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        start_ns = int(episode["start_ns"])
        end_ns = int(episode["end_ns"])
        episode_start_s = start_ns / 1e9
        episode_end_s = end_ns / 1e9
        history = deque(maxlen=3)

        sampled_indices, sampled_timestamps_s = sample_episode_frame_indices(
            frame_metadata,
            start_ns,
            end_ns,
            video_fps,
            args.sampling_hz,
        )

        print(
            f"Episode {episode_index}: {len(sampled_indices)} sampled frame(s) "
            f"from {episode_start_s:.2f}s to {episode_end_s:.2f}s"
        )

        if not sampled_indices:
            results.append({"episode_index": episode_index, "subtasks": []})
            continue

        frame_labels: list[list[str]] = []

        for sample_pos, frame_index in enumerate(sampled_indices):
            context_start = max(0, sample_pos - args.context_window)
            context_end = min(len(sampled_indices), sample_pos + args.context_window + 1)
            context_positions = list(range(context_start, context_end))
            center_idx_in_context = sample_pos - context_start

            # Gather proprioceptive data for each context frame.
            robot_records: list[dict[str, Any] | None] = []
            for context_pos in context_positions:
                ts_ns = int(sampled_timestamps_s[context_pos] * 1e9)
                robot_records.append(get_robot_record_at_ns(robot_data, robot_ts, ts_ns))
            robot_context = (
                format_robot_states_for_prompt(robot_records, center_idx_in_context)
                if robot_data else None
            )

            temp_files: list[str] = []
            try:
                for context_pos in context_positions:
                    context_frame_index = sampled_indices[context_pos]
                    for video_source in video_sources:
                        frame = extract_frame_at_index(video_source, context_frame_index)
                        if frame is None:
                            temp_files.append("")
                            continue
                        temp_path = save_frame_temp(frame)
                        temp_files.append(temp_path or "")

                valid_temp_files = [item for item in temp_files if item]
                if not valid_temp_files:
                    label_set = [subtasks[-1]] if subtasks else ["idle"]
                else:
                    label_set = label_frame_with_context(
                        llm,
                        tokenizer,
                        sampling_params,
                        valid_temp_files,
                        subtasks,
                        args,
                        list(history),
                        robot_context,
                    )

                frame_labels.append(label_set)
                history.append(",".join(label_set))
                print(f"  Frame {frame_index + 1}/{frame_count}: {', '.join(label_set)}")
            finally:
                for temp_file in temp_files:
                    if temp_file:
                        try:
                            Path(temp_file).unlink(missing_ok=True)
                        except Exception:
                            pass

        subtasks_rows = collapse_labels_to_segments(
            frame_labels,
            sampled_timestamps_s,
            episode_start_s,
            episode_end_s,
        )
        results.append({"episode_index": episode_index, "subtasks": subtasks_rows})

    write_jsonl(output_path, results)
    print("")
    print(f"Saved {len(results)} episode(s) to {output_path}")


if __name__ == "__main__":
    main()
