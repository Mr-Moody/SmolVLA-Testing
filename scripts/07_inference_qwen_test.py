#!/usr/bin/env python3
"""
Test script: Load Qwen3-VL-4B and identify objects frame by frame.
The model is run on each extracted frame and the results are aggregated.

Usage (on remote):
  python3 scripts/07_inference_qwen_test.py \
    --video-path /scratch0/xparker/cleaned_datasets/qwen_data/video.mp4 \
    --frames-to-extract 10 \
    --gpu-mem-util 0.98 \
    --max-model-len 1024
"""

import os
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def extract_frames(video_path, num_frames=None):
    """Extract frames from a video file.

    If num_frames is None, extract every frame.
    Otherwise, sample evenly spaced frames.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed. Install with: pip install opencv-python")
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"ERROR: Could not read video {video_path}")
            cap.release()
            return []
        
        print(f"  Total frames in video: {total_frames}")
        
        frames = []

        if num_frames is None or num_frames >= total_frames:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return []

def save_frames_temp(frames, temp_dir="/tmp"):
    """Save frames to temp files for vLLM to access."""
    import cv2
    import os
    import tempfile
    
    temp_files = []
    try:
        for i, frame in enumerate(frames):
            fd, fpath = tempfile.mkstemp(suffix=".jpg", dir=temp_dir)
            os.close(fd)
            cv2.imwrite(fpath, frame)
            temp_files.append(fpath)
    except Exception as e:
        print(f"Error saving frames: {e}")
        return []
    
    return temp_files


def get_video_frame_count(video_path):
    try:
        import cv2
    except ImportError:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def extract_frame_at_index(video_path, frame_index):
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed. Install with: pip install opencv-python")
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


def find_video_sources(video_path=None, data_name: str = "double_d405"):
    if video_path:
        video_path = Path(video_path)
        if video_path.is_dir():
            import glob

            return sorted(Path(p) for p in glob.glob(str(video_path / "**" / "rgb.mp4"), recursive=True))
        return [video_path]

    data_dir = Path(f"/scratch0/xparker/cleaned_datasets/{data_name}/cameras")
    videos = sorted(
        camera_dir / "rgb.mp4"
        for camera_dir in data_dir.iterdir()
        if camera_dir.is_dir() and (camera_dir / "rgb.mp4").exists()
    )

    if videos:
        return videos

    import glob

    fallback = sorted(Path(p) for p in glob.glob(str(data_dir / "**" / "rgb.mp4"), recursive=True))
    return fallback


def get_first_episode_frame_range(dataset_dir, primary_camera="d405_rgb"):
    """Get the frame index range for the first episode.
    
    Returns tuple of (start_frame_idx, end_frame_idx) or None if not available.
    """
    # Load episode boundaries from episode_events.jsonl
    episode_events_path = Path(dataset_dir) / "episode_events.jsonl"
    if not episode_events_path.exists():
        print(f"WARNING: Could not find episode_events.jsonl at {episode_events_path}")
        return None
    
    events = []
    with open(episode_events_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    
    if len(events) < 1:
        print("WARNING: No episode events found")
        return None
    
    # Get the first episode start timestamp
    first_event = events[0]
    if first_event.get("event") != "episode_start":
        print(f"WARNING: First event is not episode_start: {first_event}")
        return None
    
    first_ep_start_ns = first_event.get("robot_timestamp_ns")
    
    # Get the second episode start timestamp (or use end of video if only one episode)
    first_ep_end_ns = None
    if len(events) > 1:
        second_event = events[1]
        if second_event.get("event") == "episode_start":
            first_ep_end_ns = second_event.get("robot_timestamp_ns")
    
    # Load camera frames metadata to map timestamps to frame indices
    frames_jsonl_path = Path(dataset_dir) / "cameras" / primary_camera / "frames.jsonl"
    if not frames_jsonl_path.exists():
        print(f"WARNING: Could not find frames.jsonl at {frames_jsonl_path}")
        return None
    
    frame_metas = []
    with open(frames_jsonl_path) as f:
        for line in f:
            if line.strip():
                frame_metas.append(json.loads(line))
    
    if not frame_metas:
        print("WARNING: No frame metadata found")
        return None
    
    # Find frames within the first episode's time range
    start_frame_idx = None
    end_frame_idx = None
    
    for i, frame_meta in enumerate(frame_metas):
        frame_ns = frame_meta.get("host_timestamp_ns")
        if frame_ns is None:
            continue
        
        # Find first frame at or after episode start
        if start_frame_idx is None and frame_ns >= first_ep_start_ns:
            start_frame_idx = i
        
        # Find first frame at or after episode end (if we have one)
        if first_ep_end_ns is not None and end_frame_idx is None and frame_ns >= first_ep_end_ns:
            end_frame_idx = i
            break
    
    # If we didn't find an end frame, use the last frame
    if end_frame_idx is None:
        end_frame_idx = len(frame_metas)
    
    if start_frame_idx is None:
        print("WARNING: Could not map episode timestamps to frame indices")
        return None
    
    print(f"First episode frames: {start_frame_idx} to {end_frame_idx - 1}")
    return (start_frame_idx, end_frame_idx)


def main():
    parser = argparse.ArgumentParser(description="Identify objects from video using Qwen3-VL")
    parser.add_argument("--video-path", 
                        default=None,
                        help="Path to video file (auto-detect if not provided)")
    parser.add_argument("--data-name",
                        default="double_d405",
                        help="Dataset folder name under /scratch0/<user>/cleaned_datasets (default: double_d405)")
    parser.add_argument("--frames-to-extract", type=int, default=0,
                        help="Number of frames to extract from video; 0 means all frames")
    parser.add_argument("--sampling-hz", type=float, default=5.0,
                        help="Frame sampling rate in Hz (default: 5.0 Hz, approximately every 6th frame at 30Hz)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.98,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model sequence length")
    args = parser.parse_args()

    # If a specific video path wasn't provided, discover camera feeds under the chosen dataset
    video_paths = find_video_sources(args.video_path, args.data_name)
    if not video_paths:
        print(f"ERROR: No video sources found under /scratch0/xparker/cleaned_datasets/{args.data_name}/cameras")
        sys.exit(1)

    print("Auto-detected video sources:")
    for video_path in video_paths:
        print(f" - {video_path}")
    print("")

    print("Loading Qwen model and running frame-by-frame object identification...")

    user_prompt = (
        "You will receive synchronized images from both cameras at the same timestamp. "
        "Inspect both views together and list every distinct object visible in either view. "
        "Reply with one short object name per line, no extra commentary."
    )

    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    try:
        llm = LLM(
            model="Qwen/Qwen3-VL-4B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 3},
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        sys.exit(1)

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=64, temperature=0.0, repetition_penalty=1.05)

    frame_count = min(get_video_frame_count(video_path) for video_path in video_paths)
    if frame_count <= 0:
        print("ERROR: Could not determine video frame count")
        sys.exit(1)

    if args.frames_to_extract <= 0 or args.frames_to_extract >= frame_count:
        frame_indices = list(range(frame_count))
        print(f"Processing all synchronized timestamps: {frame_count}\n")
    else:
        frame_indices = [int(i * frame_count / args.frames_to_extract) for i in range(args.frames_to_extract)]
        print(f"Processing {len(frame_indices)} synchronized timestamps sampled from {frame_count} total\n")

    # Downsample to target sampling rate (Hz)
    # Try to get video FPS from first video source
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_paths[0]))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if video_fps <= 0:
            video_fps = 30.0  # Default fallback
    except Exception:
        video_fps = 30.0  # Default fallback
    
    target_sampling_hz = args.sampling_hz
    frame_interval = max(1, int(round(video_fps / target_sampling_hz)))
    frame_indices = frame_indices[::frame_interval]
    print(f"Downsampling to {target_sampling_hz:.1f} Hz (video fps: {video_fps:.1f}, keeping every {frame_interval}th frame)")
    print(f"Final frame count after downsampling: {len(frame_indices)}\n")

    # Filter to only first episode frames
    dataset_dir = video_paths[0].parent.parent.parent  # cameras/<camera>/rgb.mp4 → cleaned_datasets/<name>
    primary_camera = video_paths[0].parent.name  # e.g., "d405_rgb"
    episode_range = get_first_episode_frame_range(dataset_dir, primary_camera)
    
    if episode_range:
        start_frame, end_frame = episode_range
        frame_indices = [i for i in frame_indices if start_frame <= i < end_frame]
        print(f"Filtered to first episode: {len(frame_indices)} frame(s) to process\n")
    else:
        print("WARNING: Could not determine first episode range; processing all frames\n")

    all_objects = []

    for frame_index in frame_indices:
        frames_at_timestamp = []
        temp_files = []

        for video_path in video_paths:
            frame = extract_frame_at_index(video_path, frame_index)
            if frame is None:
                print(f"[Frame {frame_index + 1}/{frame_count}] WARNING: missing frame in {video_path}")
                continue
            frames_at_timestamp.append(frame)

        expected_feeds = len(video_paths)
        if len(frames_at_timestamp) != expected_feeds:
            print(f"[Frame {frame_index + 1}/{frame_count}] Skipping incomplete camera set (found {len(frames_at_timestamp)}/{expected_feeds})")
            continue

        temp_files = save_frames_temp(frames_at_timestamp)
        if len(temp_files) != len(video_paths):
            print(f"[Frame {frame_index + 1}/{frame_count}] ERROR: Could not save frames for inference")
            for fpath in temp_files:
                try:
                    Path(fpath).unlink(missing_ok=True)
                except Exception:
                    pass
            sys.exit(1)

        print(f"[Frame {frame_index + 1}/{frame_count}] " + " | ".join(temp_files))

        try:
            # Build a multimodal message containing all available images at this timestamp
            image_contents = [{"type": "image", "image": p} for p in temp_files]
            image_contents.append({"type": "text", "text": user_prompt})

            messages = [{"role": "user", "content": image_contents}]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            outputs = llm.generate(
                [{"prompt": text, "multi_modal_data": {"image": image_inputs}}],
                sampling_params=sampling_params,
            )

            result_text = ""
            if outputs and outputs[0].outputs:
                result_text = outputs[0].outputs[0].text.strip()

            print(result_text if result_text else "(no output)")

            lines = [l.strip() for l in result_text.replace(',', '\n').splitlines() if l.strip()]
            for line in lines:
                obj = line.lstrip('- ').lstrip('0123456789. ').strip()
                if obj and obj.lower() not in [u.lower() for u in all_objects]:
                    all_objects.append(obj)
        finally:
            for fpath in temp_files:
                try:
                    Path(fpath).unlink(missing_ok=True)
                except Exception:
                    pass

    print("=" * 60)
    print("Aggregated objects across all frames and camera sources:")
    for obj in all_objects:
        print(f" - {obj}")
    print("=" * 60)
    print("✓ Object identification complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
