"""
Train Action Chunking Transformers (ACT) on an exported LeRobotDataset v3 dataset.

ACT (Action Chunking Transformers) is an imitation learning policy that predicts
sequences of actions conditioned on visual observations. It uses a transformer
architecture with a VAE component to learn action sequences.

ACT pretrained base: lerobot/act_base

Key features:
  - Vision backbone: ResNet18 (configurable)
  - Action prediction: Chunk-based (predicts multiple steps at once)
  - Optional VAE: For learning compact action representations
  - Device agnostic: Works on CPU, CUDA, or MPS
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_ACT_POLICY_PATH = "lerobot/act_base"


def train_act(
    *,
    dataset_root: Path,
    lerobot_root: Path,
    policy_path: str,
    output_dir: Path,
    batch_size: int,
    steps: int,
    device: str,
    num_workers: int,
    log_freq: int,
    save_freq: int,
    eval_freq: int,
    seed: int,
    use_amp: bool,
    resume: bool,
    episodes: list[int] | None,
    job_name: str | None,
    push_to_hub: bool,
    policy_repo_id: str | None,
    chunk_size: int,
    n_obs_steps: int,
    use_vae: bool,
    vision_backbone: str,
) -> Path:
    """Train ACT policy on a LeRobotDataset v3 export.
    
    Args:
        dataset_root: Path to the exported LeRobotDataset v3 root directory.
        lerobot_root: Path to the local lerobot repository.
        policy_path: Path or HuggingFace model ID for the pretrained checkpoint.
        output_dir: Directory for checkpoints and logs.
        batch_size: Batch size for training.
        steps: Number of training steps.
        device: Torch device (e.g., "cuda", "cpu", "mps").
        num_workers: Number of workers for data loading.
        log_freq: Logging frequency.
        save_freq: Checkpoint save frequency.
        eval_freq: Evaluation frequency (0 disables evaluation).
        seed: Random seed.
        use_amp: Enable automatic mixed precision.
        resume: Resume from the last checkpoint in output_dir.
        episodes: Optional list of episode indices to use.
        job_name: Name for the training job.
        push_to_hub: Push the trained policy to Hugging Face Hub.
        policy_repo_id: HuggingFace repo ID for push_to_hub.
        chunk_size: Number of action steps to predict in a chunk.
        n_obs_steps: Number of observation steps to use.
        use_vae: Whether to use VAE for action modeling.
        vision_backbone: Vision backbone architecture (e.g., "resnet18", "resnet50").
    
    Returns:
        Path to the output directory.
    
    Raises:
        FileNotFoundError: If lerobot src or dataset metadata is not found.
        ValueError: If dataset is not LeRobotDataset v3.
    """
    lerobot_src = lerobot_root / "src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"Could not find lerobot src directory at {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    if os.getenv("HF_HUB_OFFLINE") == "1":
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
        import lerobot.datasets.utils as dataset_utils

        def _local_safe_version(_repo_id: str, version: str) -> str:
            normalized = str(version).lstrip("v")
            return f"v{normalized}"

        lerobot_dataset.get_safe_version = _local_safe_version
        dataset_utils.get_safe_version = _local_safe_version
        dataset_utils.check_version_compatibility = lambda *args, **kwargs: None

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act import ACTConfig
    from lerobot.scripts.lerobot_train import train
    from lerobot.configs import FeatureType
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.utils.feature_utils import dataset_to_policy_features

    # Load dataset metadata to determine input/output features
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find LeRobot metadata at {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected a LeRobotDataset v3 export, got codebase_version={info.get('codebase_version')!r}"
        )
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]

    repo_id = f"local/{dataset_root.name}"
    root_arg = str(dataset_root)

    LOGGER.info("Policy:    ACT")
    LOGGER.info("Device:    %s", device)
    LOGGER.info("Dataset:   %s", dataset_root)
    LOGGER.info("Total:     %d episode(s), %d frame(s)", total_episodes, total_frames)
    LOGGER.info("Checkpoint:%s", policy_path)
    LOGGER.info("Output:    %s", output_dir)
    LOGGER.info("Steps:     %d  |  batch: %d  |  workers: %d", steps, batch_size, num_workers)
    LOGGER.info("Architecture: chunk_size=%d, n_obs_steps=%d, use_vae=%s, backbone=%s",
                chunk_size, n_obs_steps, use_vae, vision_backbone)
    LOGGER.info("AMP:       %s  |  seed: %d", use_amp, seed)
    if episodes:
        LOGGER.info("Episodes:  %s", episodes)

    # Load metadata to get features
    dataset_metadata = LeRobotDatasetMetadata(repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # Split features into input and output
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    LOGGER.info("Input features: %s", list(input_features.keys()))
    LOGGER.info("Output features: %s", list(output_features.keys()))

    pretrained_path = Path(policy_path) if Path(policy_path).exists() else policy_path

    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_obs_steps=n_obs_steps,
        n_action_steps=chunk_size,  # Number of actions to run per policy invocation
        use_vae=use_vae,
        vision_backbone=vision_backbone,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if vision_backbone == "resnet18" else None,
        device=device,
        push_to_hub=push_to_hub,
        repo_id=policy_repo_id,
    )

    cfg = TrainPipelineConfig(
        resume=resume,
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=root_arg,
            episodes=episodes,
            use_imagenet_stats=True,
            return_uint8=False,
        ),
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=job_name or f"{dataset_root.name}_act",
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        log_freq=log_freq,
        save_freq=save_freq,
        eval_freq=eval_freq,
    )

    if resume:
        cfg.config_path = output_dir / "train_config.json"

    LOGGER.info("Starting ACT training...")
    train(cfg)
    LOGGER.info("Training finished. Outputs written to %s", output_dir)
    return output_dir
