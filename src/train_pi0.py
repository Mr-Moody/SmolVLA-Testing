"""
Train Pi0 or Pi0.5 on an exported LeRobotDataset v3 dataset.

Pi0  pretrained base: lerobot/pi0_base
Pi0.5 pretrained base: lerobot/pi05_base

Key differences from SmolVLA:
  - PaliGemma 2B vision backbone (vs SmolVLM2-500M)
  - Flow-matching action head (same principle, different implementation)
  - Pi0  uses MEAN_STD normalisation; Pi0.5 uses QUANTILES
  - Both use 224x224 images; SmolVLA uses 512x512
  - freeze_vision_encoder / train_expert_only default to False (full fine-tune)
    — pass --freeze-vision-encoder or --train-expert-only to restrict training
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_PI0_POLICY_PATH = "lerobot/pi0_base"
DEFAULT_PI05_POLICY_PATH = "lerobot/pi05_base"


def train_pi0(
    *,
    dataset_root: Path,
    lerobot_root: Path,
    policy_path: str,
    model_type: str,
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
    freeze_vision_encoder: bool,
    train_expert_only: bool,
    gradient_checkpointing: bool,
    dtype: str,
) -> Path:
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
    from lerobot.scripts.lerobot_train import train

    import json
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

    LOGGER.info("Model:     %s", model_type)
    LOGGER.info("Device:    %s", device)
    LOGGER.info("Dataset:   %s", dataset_root)
    LOGGER.info("Total:     %d episode(s), %d frame(s)", total_episodes, total_frames)
    LOGGER.info("Policy:    %s", policy_path)
    LOGGER.info("Output:    %s", output_dir)
    LOGGER.info("Steps:     %d  |  batch: %d  |  workers: %d", steps, batch_size, num_workers)
    LOGGER.info("AMP:       %s  |  seed: %d  |  dtype: %s", use_amp, seed, dtype)
    LOGGER.info(
        "Freeze:    vision_encoder=%s  expert_only=%s  grad_ckpt=%s",
        freeze_vision_encoder, train_expert_only, gradient_checkpointing,
    )
    if episodes:
        LOGGER.info("Episodes:  %s", episodes)

    pretrained_path = Path(policy_path) if Path(policy_path).exists() else policy_path

    if model_type == "pi0":
        from lerobot.policies.pi0 import PI0Config
        policy_cfg = PI0Config(
            pretrained_path=pretrained_path,
            device=device,
            use_amp=use_amp,
            push_to_hub=push_to_hub,
            repo_id=policy_repo_id,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            gradient_checkpointing=gradient_checkpointing,
            dtype=dtype,
        )
    else:
        from lerobot.policies.pi05 import PI05Config
        policy_cfg = PI05Config(
            pretrained_path=pretrained_path,
            device=device,
            use_amp=use_amp,
            push_to_hub=push_to_hub,
            repo_id=policy_repo_id,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            gradient_checkpointing=gradient_checkpointing,
            dtype=dtype,
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
        job_name=job_name or f"{dataset_root.name}_{model_type}",
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

    LOGGER.info("Starting %s training...", model_type)
    train(cfg)
    LOGGER.info("Training finished. Outputs written to %s", output_dir)
    return output_dir
