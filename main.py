r"""
Train SmolVLA on an exported LeRobotDataset v3 dataset.

Example:
    /home/thomas/UCL/Industrial\ Project/lerobot/.venv/bin/python main.py \
        --dataset-root lerobot_datasets/example \
        --steps 20000 \
        --batch-size 8
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

DEFAULT_POLICY_PATH = "lerobot/smolvla_base"


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
    dataset_roots: list[Path],
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

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla import SmolVLAConfig
    from lerobot.scripts.lerobot_train import train

    total_episodes = 0
    total_frames = 0
    for root in dataset_roots:
        info = load_dataset_info(root)
        total_episodes += info["total_episodes"]
        total_frames += info["total_frames"]

    if len(dataset_roots) == 1:
        repo_id = f"local/{dataset_roots[0].name}"
        root_arg = str(dataset_roots[0])
    else:
        repo_id = [f"local/{r.name}" for r in dataset_roots]
        root_arg = {f"local/{r.name}": str(r) for r in dataset_roots}

    LOGGER.info("Device:    %s", device)
    for root in dataset_roots:
        LOGGER.info("Dataset:   %s", root)
    LOGGER.info("Total:     %d episode(s), %d frame(s)", total_episodes, total_frames)
    LOGGER.info("Policy:    %s", policy_path)
    LOGGER.info("Output:    %s", output_dir)
    LOGGER.info("Steps:     %d  |  batch: %d  |  workers: %d", steps, batch_size, num_workers)
    LOGGER.info("AMP:       %s  |  seed: %d", use_amp, seed)
    if episodes:
        LOGGER.info("Episodes:  %s", episodes)

    # resume=True requires an existing train_config.json in the output dir.
    # For actual resume runs, pass the config path so lerobot can find it.
    resume_config = output_dir / "train_config.json" if resume else None

    cfg = TrainPipelineConfig(
        resume=resume,
        config_path=resume_config,
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
        job_name=job_name or f"{'_'.join(r.name for r in dataset_roots)}_smolvla",
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        log_freq=log_freq,
        save_freq=save_freq,
        eval_freq=eval_freq,
    )

    LOGGER.info("Starting SmolVLA training...")

    train(cfg)
    LOGGER.info("Training finished. Outputs written to %s", output_dir)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SmolVLA on a local LeRobotDataset v3 export.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        nargs="+",
        help=(
            "Path(s) to exported LeRobotDataset v3 root(s). Pass one or more, e.g. "
            "--dataset-root lerobot_datasets/001 lerobot_datasets/002 lerobot_datasets/003"
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
        default=DEFAULT_POLICY_PATH,
        help="Pretrained SmolVLA checkpoint or model id to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and logs. Defaults to outputs/<dataset>_smolvla.",
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_roots: list[Path] = args.dataset_root  # nargs="+" gives a list
    run_name = "_".join(r.name for r in dataset_roots)
    output_dir = args.output_dir or Path("outputs") / f"{run_name}_smolvla"
    lerobot_root = resolve_lerobot_root(args.lerobot_root)

    train_smolvla(
        dataset_roots=dataset_roots,
        lerobot_root=lerobot_root,
        resume=args.resume,
        policy_path=args.policy_path,
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
        episodes=parse_episode_list(args.episodes),
        job_name=args.job_name,
        push_to_hub=args.push_to_hub,
        policy_repo_id=args.policy_repo_id,
    )


if __name__ == "__main__":
    main()
