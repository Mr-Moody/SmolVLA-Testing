#!/usr/bin/env python3
"""
Test script: Load Qwen3-VL-4B and identify objects from a single video.
Video shows objects one-by-one; model describes/identifies each.

Usage (on remote):
  python3 scripts/07_inference_qwen_test.py \
    --video-path /scratch0/xparker/cleaned_datasets/qwen_data/video.mp4 \
    --frames-to-extract 10 \
    --gpu-mem-util 0.98 \
    --max-model-len 1024
"""

import os
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def extract_frames(video_path, num_frames=10):
    """Extract evenly-spaced frames from a video file."""
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
        
        # Extract evenly-spaced frames
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = []
        frame_paths = []
        
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
    import tempfile
    
    temp_files = []
    try:
        for i, frame in enumerate(frames):
            fd, fpath = tempfile.mkstemp(suffix=".jpg", dir=temp_dir)
            cv2.imwrite(fpath, frame)
            temp_files.append(fpath)
    except Exception as e:
        print(f"Error saving frames: {e}")
        return []
    
    return temp_files


def find_video_sources(video_path=None):
    if video_path:
        return [Path(video_path)]

    data_dir = Path("/scratch0/xparker/cleaned_datasets/qwen_data/cameras")
    camera_names = ["ee_zed_m_left", "ee_zed_m_right", "third_person_d405"]
    videos = []

    for camera_name in camera_names:
        candidate = data_dir / camera_name / "rgb.mp4"
        if candidate.exists():
            videos.append(candidate)

    if videos:
        return videos

    import glob

    fallback = sorted(Path(p) for p in glob.glob(str(data_dir / "**" / "rgb.mp4"), recursive=True))
    return fallback

def main():
    parser = argparse.ArgumentParser(description="Identify objects from video using Qwen3-VL")
    parser.add_argument("--video-path", 
                        default=None,
                        help="Path to video file (auto-detect if not provided)")
    parser.add_argument("--frames-to-extract", type=int, default=10,
                        help="Number of frames to extract from video")
    parser.add_argument("--gpu-mem-util", type=float, default=0.98,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=1024,
                        help="Max model sequence length")
    args = parser.parse_args()

    video_paths = find_video_sources(args.video_path)
    if not video_paths:
        print("ERROR: No video sources found under /scratch0/xparker/cleaned_datasets/qwen_data/cameras")
        sys.exit(1)

    print("Auto-detected video sources:")
    for video_path in video_paths:
        print(f" - {video_path}")
    print("")

    print("Loading Qwen annotator and running video-level object identification...")
    from annotation.serve_qwen import QwenAnnotator

    try:
        annotator = QwenAnnotator(
            model_id=None,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize annotator: {e}")
        sys.exit(1)

    user_prompt = (
        "List every distinct object visible in this camera view. "
        "Reply with one short object name per line, no extra commentary."
    )

    all_objects = []

    for source_index, video_path in enumerate(video_paths, start=1):
        print(f"Video {source_index}/{len(video_paths)}: {video_path}")
        print(f"Extracting {args.frames_to_extract} frames...\n")

        frames = extract_frames(str(video_path), num_frames=args.frames_to_extract)
        if not frames:
            print("Could not extract frames")
            sys.exit(1)

        print(f"✓ Extracted {len(frames)} frames\n")

        try:
            result_text = annotator.annotate_episode(str(video_path), user_prompt, max_tokens=512)
            print("Model output:\n")
            print(result_text)

            lines = [l.strip() for l in result_text.replace(',', '\n').splitlines() if l.strip()]
            for line in lines:
                obj = line.lstrip('- ').lstrip('0123456789. ').strip()
                if obj and obj.lower() not in [u.lower() for u in all_objects]:
                    all_objects.append(obj)
        except Exception as e:
            print(f"ERROR during annotation for {video_path}: {e}")
            sys.exit(1)

    print("=" * 60)
    print("Aggregated objects across all camera sources:")
    for obj in all_objects:
        print(f" - {obj}")
    print("=" * 60)
    print("✓ Object identification complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
