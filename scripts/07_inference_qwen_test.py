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

    # Find video if not provided
    video_path = args.video_path
    if not video_path:
        import glob
        data_dir = Path("/scratch0/xparker/cleaned_datasets/qwen_data")
        video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        videos = []
        for pattern in video_patterns:
            videos.extend(glob.glob(str(data_dir / "**" / pattern), recursive=True))
        
        if not videos:
            print(f"ERROR: No video found in {data_dir}")
            sys.exit(1)
        
        video_path = sorted(videos)[0]
        print(f"Auto-detected video: {video_path}\n")

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video: {video_path}")
    print(f"Extracting {args.frames_to_extract} frames...\n")
    
    frames = extract_frames(str(video_path), num_frames=args.frames_to_extract)
    
    if not frames:
        print("Could not extract frames")
        sys.exit(1)
    
    print(f"✓ Extracted {len(frames)} frames\n")
    
    # Load QwenAnnotator which handles video processing and multi-modal input
    print("Loading Qwen annotator and running video-level object identification...")
    try:
        from src.annotation.serve_qwen import QwenAnnotator
    except Exception:
        # Fallback import if running from package layout
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

    # Prompt asking for distinct object names across the whole video
    user_prompt = (
        "List every distinct object visible in the video. "
        "Reply with one short object name per line, no extra commentary."
    )

    try:
        result_text = annotator.annotate_episode(str(video_path), user_prompt, max_tokens=512)
        print("Model output:\n")
        print(result_text)

        # Parse lines into deduplicated object list
        lines = [l.strip() for l in result_text.replace(',', '\n').splitlines() if l.strip()]
        unique = []
        for l in lines:
            obj = l.lstrip('- ').lstrip('0123456789. ').strip()
            if obj and obj.lower() not in [u.lower() for u in unique]:
                unique.append(obj)

        print("\nAggregated objects seen in video:")
        for o in unique:
            print(f" - {o}")

    except Exception as e:
        print(f"ERROR during annotation: {e}")
        sys.exit(1)

    # Cleanup temp files
    import os as os_module
    for fpath in frame_paths:
        try:
            os_module.remove(fpath)
        except:
            pass

    print("="*60)
    print("✓ Object identification complete")
    print("="*60)

if __name__ == "__main__":
    main()
