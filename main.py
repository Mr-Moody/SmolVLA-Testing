r"""
Train SmolVLA, Pi0, or Pi0.5 on an exported LeRobotDataset v3 dataset.

Examples:
    # SmolVLA (default)
    python main.py --dataset-root lerobot_datasets/example --steps 20000 --batch-size 8

    # Pi0
    python main.py --model-type pi0 --dataset-root lerobot_datasets/socket --steps 20000 --batch-size 4

    # Pi0.5
    python main.py --model-type pi05 --dataset-root lerobot_datasets/socket --steps 20000 --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_POLICY_PATHS = {
    "smolvla": "lerobot/smolvla_base",
    "pi0":     "lerobot/pi0_base",
    "pi05":    "lerobot/pi05_base",
}


def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    episodes = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        episodes.append(int(chunk))
    return episodes or None


def ensure_lerobot_importable(lerobot_root: Path) -> None:
    lerobot_src = lerobot_root / "src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"Could not find lerobot src directory at {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def disable_lerobot_hub_lookups() -> None:
    """Force lerobot to stay on the local dataset path when offline mode is requested."""
    if os.getenv("HF_HUB_OFFLINE") != "1":
        return

    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    import lerobot.datasets.utils as dataset_utils

    def _local_safe_version(_repo_id: str, version: str) -> str:
        normalized = str(version)
        if normalized.startswith("v"):
            normalized = normalized[1:]
        return f"v{normalized}"

    lerobot_dataset.get_safe_version = _local_safe_version
    dataset_utils.get_safe_version = _local_safe_version
    dataset_utils.check_version_compatibility = lambda *args, **kwargs: None


def resolve_lerobot_root(explicit_root: Path | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    search_candidates: list[Path] = []

    if explicit_root is not None:
        search_candidates.append(explicit_root.expanduser())

    env_lerobot_root = os.getenv("LEROBOT_ROOT")
    if env_lerobot_root:
        search_candidates.append(Path(env_lerobot_root).expanduser())

    # Most common case: this repo and `lerobot` are siblings.
    search_candidates.append(script_dir.parent / "lerobot")

    # Fallbacks when the script is run from a nested directory layout.
    search_candidates.extend(parent / "lerobot" for parent in script_dir.parents)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in search_candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if (candidate / "src").exists():
            LOGGER.info("Using lerobot root: %s", candidate)
            return candidate

    searched = "\n".join(f" - {path / 'src'}" for path in unique_candidates)
    raise FileNotFoundError(
        "Could not find lerobot src directory.\n"
        "Checked:\n"
        f"{searched}\n"
        "Pass --lerobot-root /path/to/lerobot or set LEROBOT_ROOT."
    )


def load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find LeRobot metadata at {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected a LeRobotDataset v3 export, got codebase_version={info.get('codebase_version')!r}"
        )
    return info


def train_smolvla(
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
) -> Path:
    ensure_lerobot_importable(lerobot_root)
    disable_lerobot_hub_lookups()

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla import SmolVLAConfig
    from lerobot.scripts.lerobot_train import train

    info = load_dataset_info(dataset_root)
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]

    repo_id = f"local/{dataset_root.name}"
    root_arg = str(dataset_root)

    LOGGER.info("Device:    %s", device)
    LOGGER.info("Dataset:   %s", dataset_root)
    LOGGER.info("Total:     %d episode(s), %d frame(s)", total_episodes, total_frames)
    LOGGER.info("Policy:    %s", policy_path)
    LOGGER.info("Output:    %s", output_dir)
    LOGGER.info("Steps:     %d  |  batch: %d  |  workers: %d", steps, batch_size, num_workers)
    LOGGER.info("AMP:       %s  |  seed: %d", use_amp, seed)
    if episodes:
        LOGGER.info("Episodes:  %s", episodes)

    cfg = TrainPipelineConfig(
        resume=resume,
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=root_arg,
            episodes=episodes,
            use_imagenet_stats=True,
            return_uint8=False,
        ),
        policy=SmolVLAConfig(
            pretrained_path=Path(policy_path) if Path(policy_path).exists() else policy_path,
            device=device,
            use_amp=use_amp,
            push_to_hub=push_to_hub,
            repo_id=policy_repo_id,
        ),
        output_dir=output_dir,
        job_name=job_name or f"{dataset_root.name}_smolvla",
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        log_freq=log_freq,
        save_freq=save_freq,
        eval_freq=eval_freq,
    )

    # config_path is not a constructor arg — lerobot expects it set as an
    # attribute when resume=True so validate() can find train_config.json.
    if resume:
        cfg.config_path = output_dir / "train_config.json"

    LOGGER.info("Starting SmolVLA training...")

    train(cfg)
    LOGGER.info("Training finished. Outputs written to %s", output_dir)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a VLA policy (SmolVLA, Pi0, or Pi0.5) on a local LeRobotDataset v3 export."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="smolvla",
        choices=["smolvla", "pi0", "pi05"],
        help="Policy architecture to train (default: smolvla).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Path to an exported LeRobotDataset v3 root. "
            "To train on multiple datasets, merge them first with merge_datasets.py."
        ),
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=None,
        help="Optional path to the local lerobot repository clone (auto-detected if omitted).",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help=(
            "Pretrained checkpoint or HuggingFace model id to fine-tune. "
            "Defaults to lerobot/smolvla_base, lerobot/pi0_base, or lerobot/pi05_base "
            "depending on --model-type."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and logs. Defaults to outputs/<dataset>_<model-type>.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=20_000, help="Number of training steps.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--log-freq", type=int, default=50, help="Training log frequency.")
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency.")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Evaluation frequency. Set 0 to disable env-based evaluation for offline robot datasets.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Training seed.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision if supported.")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional comma-separated subset of episode indices, e.g. 0,1,2",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in --output-dir (requires a train_config.json there).",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Optional run name.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained policy to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default=None,
        help="Required when --push-to-hub is set, e.g. your-user/your-model-name.",
    )
    # Pi0 / Pi0.5 specific flags (ignored when --model-type smolvla)
    parser.add_argument(
        "--freeze-vision-encoder",
        action="store_true",
        help="[Pi0/Pi0.5] Freeze the PaliGemma vision encoder during training.",
    )
    parser.add_argument(
        "--train-expert-only",
        action="store_true",
        help="[Pi0/Pi0.5] Freeze the entire VLM; train only the action expert and projections.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="[Pi0/Pi0.5] Enable gradient checkpointing to reduce VRAM at the cost of speed.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="[Pi0/Pi0.5] Model weight dtype (default: float32).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root: Path = args.dataset_root
    model_type: str = args.model_type
    policy_path: str = args.policy_path or DEFAULT_POLICY_PATHS[model_type]
    output_dir = args.output_dir or Path("outputs") / f"{dataset_root.name}_{model_type}"
    lerobot_root = resolve_lerobot_root(args.lerobot_root)
    episodes = parse_episode_list(args.episodes)

    common = dict(
        dataset_root=dataset_root,
        lerobot_root=lerobot_root,
        policy_path=policy_path,
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
        resume=args.resume,
        episodes=episodes,
        job_name=args.job_name,
        push_to_hub=args.push_to_hub,
        policy_repo_id=args.policy_repo_id,
    )

    if model_type == "smolvla":
        train_smolvla(**common)
    else:
        from train_pi0 import train_pi0
        train_pi0(
            **common,
            model_type=model_type,
            freeze_vision_encoder=args.freeze_vision_encoder,
            train_expert_only=args.train_expert_only,
            gradient_checkpointing=args.gradient_checkpointing,
            dtype=args.dtype,
        )


if __name__ == "__main__":
    main()
