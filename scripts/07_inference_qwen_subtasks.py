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
import json
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


def create_subtask_prompt(subtasks: list[str], history: list[str] | None = None) -> str:
    subtask_list = ", ".join(subtasks)
    history_text = ""
    if history:
        history_text = f"\nPrevious labels: {' -> '.join(history[-3:])}"

    return (
f"""You are labeling robot manipulation subtasks from synchronized camera frames.
The task involves inserting an MSD (Micro-D Subminiature) plug into its corresponding socket connector.
The MSD plug has a specific keyed orientation that must be respected, and precise microadjustments are required at the insertion point to seat it properly.
A frame may contain ONE or MULTIPLE simultaneous subtasks. It is common for frames to have 2 or more labels.
Output all active subtasks occurring in the current frame.

Important distinction: The PLUG is the object being held by the gripper. The SOCKET is the fixed connector receiving the plug. These are separate objects.

Subtask definitions:
- approach_MSD_plug: Gripper moving toward the MSD plug WITHOUT yet holding it. The arm is in motion to reach the plug for grasping. Only active when the gripper does NOT currently possess the plug. CANNOT occur with positioning_the_gripper. Do NOT use this label if the gripper is already holding the plug.
- positioning_the_gripper: Micro-adjustments of gripper finger position and orientation to align properly around the plug. Small precise movements to achieve optimal grasp points before closing fingers. CANNOT occur with approach_MSD_plug.
- grasp_the_plug: Gripper fingers actively closing and squeezing around the plug to secure it. The grasping action is in progress (not before, not after). Label while fingers are applying force to hold the plug. MUST BE LABELED ALONE.
- move_the_plug_to_the_socket: Arm is in motion transporting the held plug (already grasped) toward the socket location. Large arm movements carrying the plug through space to approach the SOCKET. Only active while the arm is actively moving the plug toward the SOCKET. The gripper must be holding the plug during this action.
- place_the_plug_in_the_socket: The plug is being inserted into the socket opening with correct orientation. Includes initial contact with the socket and partial insertion as the plug enters the socket cavity while maintaining the keyed alignment.
- nudging_the_plug_into_the_socket: The plug is already in the socket and the gripper is pushing on the sides of the plug to nudge it further into the hole. Small side-to-side or lateral pressure adjustments to seat the plug deeper into the socket cavity. MUST BE LABELED ALONE.
- align_handle: The plug handle is not upright and the gripper is pushing it back up into the raised position. This occurs after the plug is mostly seated and before the final locking push. MUST BE LABELED ALONE.
- push_down_on_the_plug: Applying downward force on the plug at the end to lock it in place into the socket. Final pressing action that secures and locks the plug fully. MUST BE LABELED ALONE.

Important: Multiple subtasks often occur together in a single frame, but some subtasks must be labeled alone.
Examples of valid multi-label frames:
  - "move_the_plug_to_the_socket,place_the_plug_in_the_socket" (arm moving plug while beginning insertion)
  - "positioning_the_gripper,move_the_plug_to_the_socket" (adjusting grip position while moving toward socket)

Examples of single-label frames (do not combine these):
  - "grasp_the_plug" (only this action)
  - "nudging_the_plug_into_the_socket" (only this action)
  - "align_handle" (only this action)
  - "push_down_on_the_plug" (only this action)

Only output multiple labels if you truly observe multiple simultaneous actions. If only one action is clearly occurring, output that label alone. Do not overthink—be honest about what you observe."""

)


def label_frame_with_context(
    llm,
    tokenizer,
    sampling_params,
    frames: list[str],
    subtasks: list[str],
    history: list[str] | None = None,
) -> list[str]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("WARNING: qwen_vl_utils not available")
        return [subtasks[-1]] if subtasks else ["idle"]

    prompt = create_subtask_prompt(subtasks, history)
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
        default=0.98,
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
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0, repetition_penalty=1.05)

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
                        list(history),
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
