#!/usr/bin/env python3
"""
Standalone script for training ACT models on LeRobot datasets.

This script can be run independently without the main.py CLI.
It provides a simple interface for training ACT models with full control
over hyperparameters.

Usage:
    python train_act_standalone.py \
        --dataset-root path/to/lerobot_dataset \
        --output-dir outputs/my_experiment \
        --steps 20000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def setup_lerobot_path(lerobot_root: Path | None = None) -> Path:
    """Find and setup the lerobot repository in sys.path."""
    candidates = []
    
    if lerobot_root is not None:
        candidates.append(lerobot_root.expanduser().resolve())
    
    env_root = os.getenv("LEROBOT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    
    # Try relative to this script
    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / "lerobot")
    candidates.extend(p / "lerobot" for p in script_dir.parents)
    
    for candidate in candidates:
        if (candidate / "src").exists():
            lerobot_src = candidate / "src"
            if str(lerobot_src) not in sys.path:
                sys.path.insert(0, str(lerobot_src))
            LOGGER.info(f"Using lerobot from: {candidate}")
            return candidate
    
    raise FileNotFoundError(
        "Could not find lerobot. Pass --lerobot-root or set LEROBOT_ROOT environment variable."
    )


def train_act_model(
    dataset_root: Path,
    output_dir: Path,
    batch_size: int = 8,
    steps: int = 20000,
    device: str = "cuda",
    num_workers: int = 4,
    log_freq: int = 50,
    save_freq: int = 1000,
    eval_freq: int = 0,
    seed: int = 1000,
    use_amp: bool = False,
    chunk_size: int = 100,
    n_obs_steps: int = 1,
    use_vae: bool = True,
    vision_backbone: str = "resnet18",
    learning_rate: float = 1e-5,
    resume: bool = False,
    episodes: list[int] | None = None,
    lerobot_root: Path | None = None,
) -> Path:
    """Train an ACT model.
    
    Args:
        dataset_root: Path to LeRobotDataset v3 directory
        output_dir: Where to save outputs
        batch_size: Batch size for training
        steps: Number of training steps
        device: "cuda", "cpu", or "mps"
        num_workers: Number of data loading workers
        log_freq: Log metrics every N steps
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluation frequency (0 = disabled)
        seed: Random seed
        use_amp: Enable Automatic Mixed Precision
        chunk_size: Action prediction chunk size
        n_obs_steps: Number of observation steps
        use_vae: Use VAE in model
        vision_backbone: ResNet variant (resnet18, resnet34, resnet50)
        learning_rate: Learning rate for optimizer
        resume: Resume from checkpoint
        episodes: Optional list of episode indices
        lerobot_root: Path to lerobot repository
    
    Returns:
        Path to output directory
    """
    # Setup
    lerobot_root = setup_lerobot_path(lerobot_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Seed for reproducibility
    torch.manual_seed(seed)
    
    # Import after path setup
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act import ACTConfig
    from lerobot.scripts.lerobot_train import train
    from lerobot.configs import FeatureType
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.utils.feature_utils import dataset_to_policy_features
    
    # Verify dataset
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {info_path}")
    
    with open(info_path) as f:
        info = json.load(f)
    
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobotDataset v3, got v{info.get('codebase_version')}"
        )
    
    # Log configuration
    LOGGER.info(f"Dataset: {dataset_root}")
    LOGGER.info(f"  Episodes: {info['total_episodes']}")
    LOGGER.info(f"  Frames: {info['total_frames']}")
    LOGGER.info(f"Output: {output_dir}")
    LOGGER.info(f"Device: {device}")
    LOGGER.info(f"Training config:")
    LOGGER.info(f"  Steps: {steps}")
    LOGGER.info(f"  Batch size: {batch_size}")
    LOGGER.info(f"  Learning rate: {learning_rate}")
    LOGGER.info(f"ACT config:")
    LOGGER.info(f"  Chunk size: {chunk_size}")
    LOGGER.info(f"  Vision backbone: {vision_backbone}")
    LOGGER.info(f"  Use VAE: {use_vae}")
    LOGGER.info(f"  Observation steps: {n_obs_steps}")
    
    # Load metadata and determine features
    repo_id = f"local/{dataset_root.name}"
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # Split into input and output features
    output_features = {k: v for k, v in features.items() 
                      if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() 
                     if k not in output_features}
    
    LOGGER.info(f"Input features: {list(input_features.keys())}")
    LOGGER.info(f"Output features: {list(output_features.keys())}")
    
    # Create ACT config
    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_obs_steps=n_obs_steps,
        n_action_steps=chunk_size,
        use_vae=use_vae,
        vision_backbone=vision_backbone,
        pretrained_backbone_weights=(
            "ResNet18_Weights.IMAGENET1K_V1" 
            if vision_backbone == "resnet18" 
            else None
        ),
        device=device,
        optimizer_lr=learning_rate,
    )
    
    # Create training config
    cfg = TrainPipelineConfig(
        resume=resume,
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=str(dataset_root),
            episodes=episodes,
            use_imagenet_stats=True,
            return_uint8=False,
        ),
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=f"{dataset_root.name}_act",
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        log_freq=log_freq,
        save_freq=save_freq,
        eval_freq=eval_freq,
    )
    
    if resume and (output_dir / "train_config.json").exists():
        cfg.config_path = output_dir / "train_config.json"
        LOGGER.info("Resuming from checkpoint...")
    
    # Train
    LOGGER.info("Starting ACT training...")
    train(cfg)
    LOGGER.info(f"Training complete! Outputs saved to {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train ACT models on LeRobot datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Path to LeRobotDataset v3 root directory",
    )
    
    # Model and training arguments
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: outputs/{dataset_name}_act)",
    )
    parser.add_argument(
        "--lerobot-root", type=Path, default=None,
        help="Path to lerobot repository (auto-detected if not provided)",
    )
    
    # Hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--steps", type=int, default=20000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--seed", type=int, default=1000,
        help="Random seed for reproducibility",
    )
    
    # Device and performance
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to train on",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--use-amp", action="store_true",
        help="Enable Automatic Mixed Precision",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--log-freq", type=int, default=50,
        help="Log frequency (steps)",
    )
    parser.add_argument(
        "--save-freq", type=int, default=1000,
        help="Checkpoint save frequency (steps)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=0,
        help="Evaluation frequency (0 = disabled)",
    )
    
    # ACT-specific arguments
    parser.add_argument(
        "--chunk-size", type=int, default=100,
        help="Number of action steps to predict",
    )
    parser.add_argument(
        "--n-obs-steps", type=int, default=1,
        help="Number of observation steps",
    )
    parser.add_argument(
        "--vision-backbone", type=str, default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Vision backbone architecture",
    )
    parser.add_argument(
        "--use-vae", action="store_true", default=True,
        help="Use VAE in model",
    )
    parser.add_argument(
        "--no-vae", dest="use_vae", action="store_false",
        help="Disable VAE in model",
    )

    # Training control
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--episodes", type=str, default=None,
        help="Comma-separated episode indices to use (e.g., '0,1,2')",
    )
    
    args = parser.parse_args()
    
    # Parse episodes list if provided
    episodes = None
    if args.episodes:
        episodes = [int(e.strip()) for e in args.episodes.split(",")]
    
    # Default output directory
    output_dir = args.output_dir or Path("outputs") / f"{args.dataset_root.name}_act"
    
    # Train
    train_act_model(
        dataset_root=args.dataset_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        device=args.device,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        seed=args.seed,
        use_amp=args.use_amp,
        chunk_size=args.chunk_size,
        n_obs_steps=args.n_obs_steps,
        use_vae=args.use_vae,
        vision_backbone=args.vision_backbone,
        learning_rate=args.learning_rate,
        resume=args.resume,
        episodes=episodes,
        lerobot_root=args.lerobot_root,
    )


if __name__ == "__main__":
    main()
