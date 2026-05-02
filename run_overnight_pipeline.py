#!/usr/bin/env python3
"""
Overnight batch pipeline: Clean → Annotate → Convert datasets with Qwen3-VL.

Usage:
    python run_overnight_pipeline.py \\
        --raw-datasets raw_datasets \\
        --dataset-names 001 002 003 \\
        --qwen-model "Qwen/Qwen3-VL-30B-A3B-Instruct" \\
        --output-dir overnight_output \\
        --skip-annotation  # optional: skip Qwen, just clean and convert \\
        --num-gpus 1

Features:
  • Processes multiple datasets sequentially
  • Robust error handling with per-dataset logging
  • Qwen3-VL annotation with batching and progress tracking
  • Automatic checkpoint/resume on failure
  • Summary report at end
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

console = Console()

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(output_dir: Path, pipeline_name: str = "overnight_pipeline"):
    """Configure logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create a logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO level, with Rich formatting)
    console_handler = RichHandler(console=console, show_time=True)
    console_handler.setLevel(logging.INFO)

    # File handler (DEBUG level, full details)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    raw_path: Path
    cleaned_path: Path
    lerobot_path: Path
    annotations_path: Path
    status: str = "pending"  # pending, cleaning, cleaning_done, annotating, annotating_done, converting, done, failed
    error: Optional[str] = None
    timings: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    raw_root: Path
    cleaned_root: Path
    lerobot_root: Path
    output_root: Path
    dataset_names: list[str]
    qwen_model: Optional[str] = None
    skip_annotation: bool = False
    num_gpus: int = 1
    batch_size_annotation: int = 4
    max_episodes_per_dataset: Optional[int] = None
    checkpoint_file: Optional[Path] = None

    def datasets(self) -> list[DatasetConfig]:
        """Generate DatasetConfig for each dataset."""
        configs = []
        for name in self.dataset_names:
            raw = self.raw_root / name
            configs.append(
                DatasetConfig(
                    name=name,
                    raw_path=raw,
                    cleaned_path=self.cleaned_root / name,
                    lerobot_path=self.lerobot_root / name,
                    annotations_path=self.output_root / f"annotations_{name}",
                )
            )
        return configs


# ============================================================================
# Pipeline Components
# ============================================================================


class DatasetCleaner:
    """Wrapper around data_cleaner.py for batch processing."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def clean(
        self,
        config: DatasetConfig,
        force: bool = False,
        camera_tolerance_ms: float = 150.0,
        joint_threshold: float = 5e-4,
        gripper_threshold: float = 2e-4,
    ) -> bool:
        """Clean a single dataset."""
        try:
            from data_cleaner import DatasetCleaner as DC

            self.logger.info(f"[{config.name}] Starting dataset cleaning...")
            cleaner = DC(
                raw_root=config.raw_path,
                output_root=config.cleaned_path,
                camera_tolerance_ns=camera_tolerance_ms * 1e6,
                joint_motion_threshold=joint_threshold,
                gripper_motion_threshold=gripper_threshold,
                force=force,
            )
            cleaner.run()
            self.logger.info(f"[{config.name}] Cleaning completed successfully.")
            config.status = "cleaning_done"
            config.timings["cleaning"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"[{config.name}] Cleaning failed: {e}", exc_info=True)
            config.status = "failed"
            config.error = f"Cleaning failed: {str(e)}"
            return False


class QwenAnnotationEngine:
    """Batch annotation with Qwen3-VL using vLLM."""

    def __init__(self, logger: logging.Logger, model_id: Optional[str] = None, num_gpus: int = 1):
        self.logger = logger
        self.model_id = model_id
        self.num_gpus = num_gpus
        self._qwen = None

    def _load_qwen(self):
        """Lazy-load Qwen model."""
        if self._qwen is not None:
            return

        try:
            from annotation.serve_qwen import QwenAnnotator

            self.logger.info(f"Loading Qwen3-VL model (tensor_parallel={self.num_gpus})...")
            self._qwen = QwenAnnotator(
                model_id=self.model_id,
                tensor_parallel_size=self.num_gpus,
                gpu_memory_utilization=0.85,
            )
            self.logger.info("Qwen3-VL model loaded and ready.")
        except Exception as e:
            self.logger.error(f"Failed to load Qwen model: {e}", exc_info=True)
            raise

    def annotate_dataset(
        self,
        config: DatasetConfig,
        task_name: str = "robot_task",
        batch_size: int = 4,
        max_episodes: Optional[int] = None,
    ) -> bool:
        """Annotate all episodes in a cleaned dataset."""
        try:
            self._load_qwen()
            self.logger.info(f"[{config.name}] Starting Qwen3-VL annotation...")

            # Use the existing annotate_dataset logic
            from scripts.annotate_cleaned_dataset import annotate_cleaned_dataset

            config.annotations_path.mkdir(parents=True, exist_ok=True)

            annotate_cleaned_dataset(
                dataset_root=config.cleaned_path,
                task_name=task_name,
                output_dir=config.annotations_path,
                qwen_annotator=self._qwen,
                batch_size=batch_size,
                max_episodes=max_episodes,
                logger=self.logger,
            )

            self.logger.info(f"[{config.name}] Annotation completed successfully.")
            config.status = "annotating_done"
            config.timings["annotation"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"[{config.name}] Annotation failed: {e}", exc_info=True)
            config.status = "failed"
            config.error = f"Annotation failed: {str(e)}"
            return False


class DatasetConverter:
    """Convert cleaned dataset to lerobot format."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def convert(
        self,
        config: DatasetConfig,
        primary_camera: str = "default",
        force: bool = False,
    ) -> bool:
        """Convert cleaned dataset to lerobot format."""
        try:
            from data_converter import DataConverter

            self.logger.info(f"[{config.name}] Starting lerobot format conversion...")
            converter = DataConverter(
                cleaned_root=config.cleaned_path,
                output_root=config.lerobot_path,
                primary_camera=primary_camera,
                force=force,
            )
            converter.run()
            self.logger.info(f"[{config.name}] Conversion completed successfully.")
            config.status = "done"
            config.timings["conversion"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"[{config.name}] Conversion failed: {e}", exc_info=True)
            config.status = "failed"
            config.error = f"Conversion failed: {str(e)}"
            return False


# ============================================================================
# Checkpoint System
# ============================================================================


class CheckpointManager:
    """Track and resume pipeline progress."""

    def __init__(self, checkpoint_file: Path, logger: logging.Logger):
        self.checkpoint_file = checkpoint_file
        self.logger = logger

    def save(self, datasets: list[DatasetConfig]) -> None:
        """Save current pipeline state."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "datasets": [
                {
                    "name": d.name,
                    "status": d.status,
                    "error": d.error,
                    "timings": d.timings,
                }
                for d in datasets
            ],
        }
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(state, f, indent=2)
        self.logger.debug(f"Checkpoint saved: {self.checkpoint_file}")

    def load(self) -> Optional[dict]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_file.exists():
            return None
        try:
            with open(self.checkpoint_file) as f:
                state = json.load(f)
            self.logger.info(f"Checkpoint loaded: {self.checkpoint_file}")
            return state
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def restore_status(self, datasets: list[DatasetConfig], checkpoint: dict) -> None:
        """Restore status from checkpoint."""
        status_map = {d["name"]: d for d in checkpoint.get("datasets", [])}
        for ds in datasets:
            if ds.name in status_map:
                ds.status = status_map[ds.name]["status"]
                ds.error = status_map[ds.name].get("error")
                ds.timings = status_map[ds.name].get("timings", {})


# ============================================================================
# Main Pipeline
# ============================================================================


class OvernightPipeline:
    """Main orchestration for multi-dataset overnight processing."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.checkpoint_mgr = CheckpointManager(
            config.checkpoint_file or config.output_root / "checkpoint.json", logger
        )
        self.cleaner = DatasetCleaner(logger)
        self.annotator = (
            QwenAnnotationEngine(logger, config.qwen_model, config.num_gpus)
            if not config.skip_annotation
            else None
        )
        self.converter = DataConverter(logger)

    def run(self) -> dict:
        """Execute the full pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("OVERNIGHT DATASET PIPELINE STARTED")
        self.logger.info("=" * 80)

        datasets = self.config.datasets()
        start_time = time.time()

        # Restore from checkpoint if available
        checkpoint = self.checkpoint_mgr.load()
        if checkpoint:
            self.logger.info("Restoring from checkpoint...")
            self.checkpoint_mgr.restore_status(datasets, checkpoint)

        # Show initial summary
        self._print_summary(datasets)

        # Process each dataset
        for ds in datasets:
            if ds.status == "done":
                self.logger.info(f"[{ds.name}] Skipping (already done)")
                continue

            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Processing dataset: {ds.name}")
            self.logger.info(f"{'=' * 80}")

            # 1. Clean
            ds_start = time.time()
            if ds.status not in ("cleaning_done", "annotating_done", "done"):
                success = self.cleaner.clean(ds, force=True)
                if not success:
                    self.checkpoint_mgr.save(datasets)
                    continue

            # 2. Annotate (if not skipped)
            if not self.config.skip_annotation and ds.status == "cleaning_done":
                success = self.annotator.annotate_dataset(
                    ds,
                    task_name="robot_manipulation",
                    batch_size=self.config.batch_size_annotation,
                    max_episodes=self.config.max_episodes_per_dataset,
                )
                if not success:
                    self.checkpoint_mgr.save(datasets)
                    continue

            # 3. Convert
            if ds.status in ("cleaning_done", "annotating_done"):
                success = self.converter.convert(ds, force=True)
                if not success:
                    self.checkpoint_mgr.save(datasets)
                    continue

            ds.timings["total_dataset"] = time.time() - ds_start
            self.checkpoint_mgr.save(datasets)

        # Final summary
        total_time = time.time() - start_time
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("PIPELINE EXECUTION COMPLETED")
        self.logger.info(f"{'=' * 80}")
        self._print_summary(datasets)
        self._print_timings(datasets, total_time)

        return {
            "success": all(d.status == "done" for d in datasets),
            "datasets": [
                {
                    "name": d.name,
                    "status": d.status,
                    "error": d.error,
                    "timings": d.timings,
                }
                for d in datasets
            ],
            "total_time": total_time,
        }

    def _print_summary(self, datasets: list[DatasetConfig]) -> None:
        """Print status summary table."""
        table = Table(title="Dataset Status Summary")
        table.add_column("Dataset", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Error", style="red")

        for ds in datasets:
            error_text = ds.error[:60] if ds.error else "—"
            table.add_row(ds.name, ds.status, error_text)

        console.print(table)

    def _print_timings(self, datasets: list[DatasetConfig], total_time: float) -> None:
        """Print timing breakdown."""
        table = Table(title="Processing Times")
        table.add_column("Dataset", style="cyan")
        table.add_column("Clean (s)", justify="right")
        table.add_column("Annotate (s)", justify="right")
        table.add_column("Convert (s)", justify="right")
        table.add_column("Total (s)", justify="right", style="green")

        for ds in datasets:
            clean_t = ds.timings.get("cleaning", 0) - ds.timings.get("started", 0)
            ann_t = ds.timings.get("annotation", 0)
            conv_t = ds.timings.get("conversion", 0)
            total_t = ds.timings.get("total_dataset", 0)

            table.add_row(
                ds.name,
                f"{clean_t:.1f}",
                f"{ann_t:.1f}",
                f"{conv_t:.1f}",
                f"{total_t:.1f}",
            )

        console.print(table)
        self.logger.info(f"Total pipeline time: {total_time:.1f}s ({total_time/3600:.2f}h)")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Overnight batch pipeline: Clean → Annotate → Convert datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-datasets",
        type=Path,
        default=Path("raw_datasets"),
        help="Root directory for raw datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--cleaned-datasets",
        type=Path,
        default=Path("cleaned_datasets"),
        help="Root directory for cleaned datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--lerobot-datasets",
        type=Path,
        default=Path("lerobot_datasets"),
        help="Root directory for lerobot-format datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=True,
        help="List of dataset names to process (e.g., 001 002 003)",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=None,
        help="HuggingFace model ID for Qwen (default: auto-detect AWQ/base)",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip Qwen annotation, only clean and convert",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size-annotation",
        type=int,
        default=4,
        help="Batch size for annotation (default: %(default)s)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Max episodes per dataset (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("overnight_output"),
        help="Output directory for logs and checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file path (default: {output_dir}/checkpoint.json)",
    )

    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_logging(args.output_dir, "overnight_pipeline")
    logger.info(f"Log file: {log_file}")

    # Create pipeline config
    config = PipelineConfig(
        raw_root=args.raw_datasets,
        cleaned_root=args.cleaned_datasets,
        lerobot_root=args.lerobot_datasets,
        output_root=args.output_dir,
        dataset_names=args.dataset_names,
        qwen_model=args.qwen_model,
        skip_annotation=args.skip_annotation,
        num_gpus=args.num_gpus,
        batch_size_annotation=args.batch_size_annotation,
        max_episodes_per_dataset=args.max_episodes,
        checkpoint_file=args.checkpoint,
    )

    # Run pipeline
    pipeline = OvernightPipeline(config, logger)
    result = pipeline.run()

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
