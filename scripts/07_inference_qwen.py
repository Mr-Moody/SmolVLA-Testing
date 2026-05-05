#!/usr/bin/env python3
"""
Subtask labeler: Use Qwen3-VL to label robot manipulation frames with temporal context.
Processes frames with a sliding window of previous/future context for consistency.

Usage:
  python3 scripts/qwen_subtask_labeler.py \
    --video-path /path/to/video.mp4 \
    --output /path/to/labels.json \
    --context-window 3 \
    --gpu-mem-util 0.98 \
    --max-model-len 2048
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Define subtasks (update based on your domain)
SUBTASKS = [
    "approach",           # Robot moving toward target
    "positioning",        # Fine gripper alignment
    "grasp",             # Closing gripper on plug
    "insertion",         # Moving plug toward socket
    "insertion_complete", # Seating the plug fully
    "verification",      # Checking plug is seated
    "idle"               # No meaningful action
]

class TemporalLabelSmoother:
    """Post-process labels to enforce temporal consistency."""
    
    def __init__(self, min_dwell_frames=2, smoothing_window=3):
        """
        Args:
            min_dwell_frames: Minimum frames a subtask must persist
            smoothing_window: Median filter window size
        """
        self.min_dwell_frames = min_dwell_frames
        self.smoothing_window = smoothing_window
    
    def smooth(self, labels: List[str], confidences: List[float]) -> Tuple[List[str], List[float]]:
        """Apply temporal smoothing to label sequence."""
        if not labels:
            return labels, confidences
        
        # Step 1: Median filtering
        smoothed = self._median_filter(labels)
        
        # Step 2: Enforce minimum dwell time
        smoothed = self._enforce_min_dwell(smoothed)
        
        # Step 3: Smooth confidences to match labels
        smooth_confidences = self._smooth_confidences(smoothed, confidences)
        
        return smoothed, smooth_confidences
    
    def _median_filter(self, labels: List[str]) -> List[str]:
        """Apply median filter to remove single-frame spikes."""
        if len(labels) < self.smoothing_window:
            return labels
        
        filtered = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(labels)):
            start = max(0, i - half_window)
            end = min(len(labels), i + half_window + 1)
            window = labels[start:end]
            
            # Count occurrences and pick most frequent
            from collections import Counter
            most_common = Counter(window).most_common(1)[0][0]
            filtered.append(most_common)
        
        return filtered
    
    def _enforce_min_dwell(self, labels: List[str]) -> List[str]:
        """Merge segments shorter than min_dwell_frames."""
        if not labels:
            return labels
        
        result = []
        i = 0
        
        while i < len(labels):
            current_label = labels[i]
            run_length = 1
            
            # Count consecutive occurrences
            while i + run_length < len(labels) and labels[i + run_length] == current_label:
                run_length += 1
            
            if run_length >= self.min_dwell_frames:
                # Keep this run
                result.extend([current_label] * run_length)
            else:
                # Too short, replace with previous or next label
                if result:
                    replacement = result[-1]
                else:
                    # Look ahead to find next label
                    replacement = labels[i + run_length] if i + run_length < len(labels) else current_label
                
                result.extend([replacement] * run_length)
            
            i += run_length
        
        return result
    
    def _smooth_confidences(self, smoothed_labels: List[str], original_confidences: List[float]) -> List[float]:
        """Adjust confidences to match smoothed labels.
        
        Penalizes labels that changed during smoothing by reducing their confidence.
        """
        if len(smoothed_labels) != len(original_confidences):
            return original_confidences
        
        new_confidences = []
        for i, (smoothed_label, orig_conf) in enumerate(zip(smoothed_labels, original_confidences)):
            # If label was changed during smoothing, penalize confidence
            # We approximate this by checking if the label is stable in its local window
            if i > 0 and i < len(smoothed_labels) - 1:
                if smoothed_labels[i-1] == smoothed_label == smoothed_labels[i+1]:
                    # Label is stable in context, keep confidence
                    new_confidences.append(orig_conf)
                else:
                    # Label boundaries changed, reduce confidence by 20%
                    new_confidences.append(max(0.0, orig_conf * 0.8))
            else:
                new_confidences.append(orig_conf)
        return new_confidences


def extract_frames(video_path, num_frames=None):
    """Extract frames from a video file."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed")
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


def save_frame_temp(frame, temp_dir="/tmp"):
    """Save single frame to temp file."""
    import cv2
    import tempfile
    
    try:
        fd, fpath = tempfile.mkstemp(suffix=".jpg", dir=temp_dir)
        os.close(fd)
        cv2.imwrite(fpath, frame)
        return fpath
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None


def get_video_frame_count(video_path):
    """Get total frame count from video."""
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
    """Extract single frame at given index."""
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


def find_video_sources(video_path=None, data_name: str = "double_d405"):
    """Find video sources (either user-specified or auto-detect).

    Args:
        video_path: explicit path to a single video file.
        data_name: dataset folder name under `/scratch0/<user>/cleaned_datasets`.
    """
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


def create_subtask_prompt(context_frames: List[str], current_idx: int, subtask_history: List[str] = None) -> str:
    """
    Create a prompt with temporal context.
    
    Args:
        context_frames: List of temp file paths (ordered in time)
        current_idx: Index of the "current" frame in context_frames
        subtask_history: Previous 2-3 subtask labels for context
    
    Returns:
        Formatted prompt string
    """
    subtask_list = ", ".join(SUBTASKS)
    
    history_text = ""
    if subtask_history and len(subtask_history) > 0:
        history_text = f"\nPrevious subtask sequence (last 3 frames): {' → '.join(subtask_history[-3:])}"
    
    prompt = f"""You are labeling a robot manipulation video frame-by-frame.
The frames below show a sequence of moments in time (ordered left to right).
Frame at position [{current_idx}] (marked with *) is the current frame you must label.{history_text}

The robot is performing an MSD plug insertion task.
At the CURRENT FRAME, which subtask is the robot executing?

Subtask options: {subtask_list}

Respond with ONLY the subtask name, no other text. Example: "positioning"
"""
    return prompt


def label_frame_with_context(
    llm,
    tokenizer,
    sampling_params,
    context_frames: List[str],
    current_frame_idx: int,
    subtask_history: List[str] = None
) -> Tuple[str, float]:
    """
    Label a frame using Qwen with temporal context.
    
    Args:
        llm: Loaded Qwen model
        tokenizer: Model tokenizer
        sampling_params: Sampling parameters
        context_frames: List of temp file paths (window around current frame)
        current_frame_idx: Index of current frame in context_frames
        subtask_history: Previous subtask labels
    
    Returns:
        (subtask_label, confidence_score)
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("WARNING: qwen_vl_utils not available, using fallback")
        return "unknown", 0.0
    
    prompt = create_subtask_prompt(context_frames, current_frame_idx, subtask_history)
    
    # Build message with all context frames
    content = []
    for i, fpath in enumerate(context_frames):
        if fpath and Path(fpath).exists():
            content.append({"type": "image", "image": fpath})
    
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    
    try:
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
            result_text = outputs[0].outputs[0].text.strip().lower()
        
        # Extract subtask from response using word-boundary matching
        import re
        subtask = "idle"
        confidence = 0.0
        
        for candidate in SUBTASKS:
            if re.search(r'\b' + re.escape(candidate) + r'\b', result_text):
                subtask = candidate
                confidence = 0.8  # Default confidence if found
                break
        
        # If no match, try to parse the response more liberally
        if subtask == "idle" and result_text:
            words = result_text.split()
            if words:
                closest = find_closest_subtask(words[0])
                if closest:
                    subtask = closest
                    confidence = 0.5  # Lower confidence for fuzzy match
        
        return subtask, confidence
    
    except Exception as e:
        print(f"    ERROR during inference: {e}")
        return "idle", 0.0


def find_closest_subtask(word: str) -> Optional[str]:
    """Find closest subtask match using string similarity."""
    from difflib import SequenceMatcher
    
    matches = [(t, SequenceMatcher(None, word, t).ratio()) for t in SUBTASKS]
    matches.sort(key=lambda x: x[1], reverse=True)
    
    best_match, score = matches[0]
    if score > 0.6:  # Similarity threshold
        return best_match
    return None


def main():
    parser = argparse.ArgumentParser(description="Label video frames with subtasks using Qwen3-VL")
    parser.add_argument("--video-path", default=None,
                        help="Path to video file (auto-detect if not provided)")
    parser.add_argument("--data-name",
                        default="double_d405",
                        help="Dataset folder name under /scratch0/<user>/cleaned_datasets (default: double_d405)")
    parser.add_argument("--output", type=str, default="subtask_labels.json",
                        help="Output JSON file for labels")
    parser.add_argument("--context-window", type=int, default=2,
                        help="Number of frames of context (before and after current); default 2 = 5 total images")
    parser.add_argument("--frames-to-extract", type=int, default=0,
                        help="Number of frames to process; 0 means all frames")
    parser.add_argument("--gpu-mem-util", type=float, default=0.98,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Max model sequence length")
    parser.add_argument("--skip-smoothing", action="store_true",
                        help="Skip temporal smoothing post-processing")
    
    args = parser.parse_args()

    video_paths = find_video_sources(args.video_path, args.data_name)
    if not video_paths:
        print(f"ERROR: No video sources found under /scratch0/xparker/cleaned_datasets/{args.data_name}/cameras")
        sys.exit(1)

    print(f"Processing synchronized feeds from {len(video_paths)} cameras:")
    for vp in video_paths:
        print(f"  - {vp}")
    print()

    # Use minimum frame count across all videos
    frame_count = min(get_video_frame_count(vp) for vp in video_paths)
    if frame_count <= 0:
        print("ERROR: Could not determine video frame count")
        sys.exit(1)

    if args.frames_to_extract <= 0 or args.frames_to_extract >= frame_count:
        frame_indices = list(range(frame_count))
        print(f"Processing all frames: {frame_count}\n")
    else:
        frame_indices = [int(i * frame_count / args.frames_to_extract) for i in range(args.frames_to_extract)]
        print(f"Processing {len(frame_indices)} sampled frames from {frame_count} total\n")

    print("Loading Qwen3-VL-4B-Instruct model...")
    from vllm import LLM, SamplingParams

    try:
        # Image limit needs to accommodate: context_frames * num_cameras
        # Default: context_window=2 gives 5 frames, 3 cameras = 15 images
        max_images = (2 * args.context_window + 1) * len(video_paths)
        llm = LLM(
            model="Qwen/Qwen3-VL-4B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": max_images},
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        sys.exit(1)

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0, repetition_penalty=1.05)

    print("=" * 70)
    print("SUBTASK LABELING")
    print("=" * 70)

    labels = []
    confidences = []
    subtask_history = deque(maxlen=3)

    for frame_idx, frame_index in enumerate(frame_indices):
        # Build context window
        context_start = max(0, frame_index - args.context_window)
        context_end = min(frame_count, frame_index + args.context_window + 1)
        context_indices = list(range(context_start, context_end))
        current_in_context = frame_index - context_start
        
        print(f"\n[Frame {frame_index + 1}/{frame_count}] "
              f"(window: {context_start}-{context_end-1}, current at pos {current_in_context})")
        
        # Extract and save all context frames from all cameras
        # Structure: List of temp file paths ordered as [ctx0_cam0, ctx0_cam1, ctx0_cam2, ctx1_cam0, ...]
        temp_files = []
        
        for ctx_offset, ctx_idx in enumerate(context_indices):
            for video_path in video_paths:
                frame = extract_frame_at_index(video_path, ctx_idx)
                if frame is None:
                    print(f"  WARNING: Could not extract frame {ctx_idx} from {video_path.name}")
                    temp_files.append(None)
                    continue
                
                temp_file = save_frame_temp(frame)
                temp_files.append(temp_file)
        
        # Skip if we couldn't get a complete current frame set (all cameras for current timestamp)
        current_frame_start = current_in_context * len(video_paths)
        current_frame_end = current_frame_start + len(video_paths)
        current_frame_set = temp_files[current_frame_start:current_frame_end]

        if not all(current_frame_set):
            print(f"  ERROR: Could not extract complete current frame set from all cameras (found {sum(1 for x in current_frame_set if x)} / {len(current_frame_set)})")
            labels.append("unknown")
            confidences.append(0.0)
            for fpath in temp_files:
                if fpath and Path(fpath).exists():
                    try:
                        Path(fpath).unlink()
                    except Exception:
                        pass
            continue
        
        if not any(temp_files):
            print(f"  ERROR: No valid context frames extracted")
            labels.append("unknown")
            confidences.append(0.0)
            continue
        
        # Run inference
        subtask, confidence = label_frame_with_context(
            llm, tokenizer, sampling_params,
            temp_files, current_in_context,
            list(subtask_history)
        )
        
        labels.append(subtask)
        confidences.append(confidence)
        subtask_history.append(subtask)
        
        print(f"  Label: {subtask} (confidence: {confidence:.2f})")
        
        # Clean up temp files
        for fpath in temp_files:
            if fpath and Path(fpath).exists():
                try:
                    Path(fpath).unlink()
                except Exception:
                    pass

    print("\n" + "=" * 70)
    print("POST-PROCESSING: Temporal Smoothing")
    print("=" * 70)
    
    if not args.skip_smoothing:
        smoother = TemporalLabelSmoother(min_dwell_frames=2, smoothing_window=3)
        labels, confidences = smoother.smooth(labels, confidences)
        print("Smoothing applied")
    else:
        print("Smoothing skipped")

    # Save results
    output_data = {
        "video_paths": [str(v) for v in video_paths],
        "frame_count": frame_count,
        "frames_processed": len(frame_indices),
        "context_window": args.context_window,
        "subtasks": SUBTASKS,
        "labels": labels,
        "confidences": confidences,
        "frame_indices": frame_indices,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Labels saved to: {output_path}")
    print("=" * 70)
    
    # Summary statistics
    from collections import Counter
    label_counts = Counter(labels)
    print("Label distribution:")
    for subtask, count in label_counts.most_common():
        percentage = 100 * count / len(labels)
        print(f"  {subtask}: {count} frames ({percentage:.1f}%)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()