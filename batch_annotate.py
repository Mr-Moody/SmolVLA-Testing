"""
Batch annotation script for overnight processing with Qwen3-VL.

Simpler alternative to the existing annotate_cleaned_dataset.py, optimized for
batch overnight runs with robust error handling.

Usage:
    python batch_annotate.py \\
        --dataset-root cleaned_datasets/001 \\
        --task "pick_and_place" \\
        --output-dir annotations_output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

logger = logging.getLogger(__name__)


def batch_annotate_dataset(
    dataset_root: Path,
    task_name: str = "robot_manipulation",
    output_dir: Optional[Path] = None,
    batch_size: int = 4,
    max_episodes: Optional[int] = None,
):
    """
    Batch-annotate a cleaned dataset with Qwen3-VL.
    
    Args:
        dataset_root: Path to cleaned dataset
        task_name: Task description
        output_dir: Output directory for annotations
        batch_size: Batch size for vLLM
        max_episodes: Limit episodes (None = all)
    """
    if output_dir is None:
        output_dir = dataset_root / "qwen_annotations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find episodes
    episodes_meta_file = dataset_root / "session_metadata.json"
    if episodes_meta_file.exists():
        try:
            meta = json.loads(episodes_meta_file.read_text())
            num_episodes = meta.get("num_episodes", 0)
        except Exception as e:
            logger.warning(f"Could not read session metadata: {e}")
            num_episodes = 0
    else:
        num_episodes = 0
    
    if num_episodes == 0:
        # Try to count video files
        video_dir = dataset_root / "videos"
        if video_dir.exists():
            videos = sorted(video_dir.glob("*.mp4"))
            num_episodes = len(videos)
    
    if num_episodes == 0:
        logger.warning(f"No episodes found in {dataset_root}")
        return {"status": "failed", "reason": "No episodes found"}
    
    if max_episodes:
        num_episodes = min(num_episodes, max_episodes)
    
    logger.info(f"Annotating {num_episodes} episodes from {dataset_root}")
    
    # Try to load Qwen
    try:
        from annotation.serve_qwen import QwenAnnotator
    except ImportError as e:
        logger.error(f"Could not import QwenAnnotator: {e}")
        return {"status": "failed", "reason": "vLLM/Qwen not available"}
    
    try:
        annotator = QwenAnnotator(tensor_parallel_size=1)
    except Exception as e:
        logger.error(f"Failed to initialize QwenAnnotator: {e}")
        return {"status": "failed", "reason": str(e)}
    
    # Process episodes
    annotations_jsonl = output_dir / "annotations.jsonl"
    success_count = 0
    failed_episodes = []
    
    with open(annotations_jsonl, "w") as f:
        for ep_idx in range(num_episodes):
            try:
                # Get video path
                video_dir = dataset_root / "videos"
                video_candidates = sorted(
                    video_dir.glob(f"*episode*{ep_idx:06d}*.mp4")
                    + video_dir.glob(f"*episode*{ep_idx:03d}*.mp4")
                    + video_dir.glob(f"*_{ep_idx:06d}.mp4")
                )
                
                if not video_candidates:
                    logger.warning(f"Episode {ep_idx}: no video found")
                    failed_episodes.append(ep_idx)
                    continue
                
                video_path = video_candidates[0]
                
                # Build prompt
                prompt = f"""Analyze this robot manipulation episode and provide:
1. Phase: (reach/grasp/lift/place/release)
2. Success: (yes/no/uncertain)
3. Key observations: (2-3 bullet points)

Task: {task_name}
Episode: {ep_idx}

Provide response as JSON."""
                
                # Annotate
                result_text = annotator.annotate_episode(
                    video_path=str(video_path),
                    prompt=prompt,
                    max_tokens=512,
                )
                
                # Try to parse as JSON
                try:
                    result_json = json.loads(result_text)
                except (json.JSONDecodeError, ValueError):
                    result_json = {"raw_response": result_text}
                
                result_json["episode_idx"] = ep_idx
                result_json["task"] = task_name
                
                f.write(json.dumps(result_json) + "\n")
                success_count += 1
                
                if (ep_idx + 1) % 10 == 0:
                    logger.info(f"Annotated {ep_idx + 1}/{num_episodes} episodes")
                    
            except Exception as e:
                logger.warning(f"Failed to annotate episode {ep_idx}: {e}")
                failed_episodes.append(ep_idx)
    
    # Summary
    summary = {
        "dataset": str(dataset_root),
        "task": task_name,
        "total_episodes": num_episodes,
        "successful": success_count,
        "failed": len(failed_episodes),
        "failed_episode_ids": failed_episodes,
        "output_file": str(annotations_jsonl),
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Annotation complete: {success_count}/{num_episodes} episodes")
    logger.info(f"Results: {annotations_jsonl}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch annotate cleaned dataset with Qwen3-VL"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", type=str, default="robot_manipulation")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    result = batch_annotate_dataset(
        dataset_root=args.dataset_root,
        task_name=args.task,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
    )
    
    print(json.dumps(result, indent=2))
