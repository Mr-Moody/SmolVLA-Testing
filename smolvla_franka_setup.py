"""
SmolVLA helper for Franka + LeRobot datasets.

This script can:
1. Run a quick SmolVLA inference sanity check.
2. Inspect an exported LeRobotDataset v3 dataset.
3. Start SmolVLA training on that exported dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "lerobot/smolvla_base"
DEFAULT_LEROBOT_ROOT = Path("/home/thomas/UCL/Industrial Project/lerobot")


def ensure_lerobot_importable(lerobot_root: Path) -> None:
    lerobot_src = lerobot_root / "src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"Could not find lerobot src directory at {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata file: {info_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing dataset stats file: {stats_path}")

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    return {"info": info, "stats": stats}


def inspect_exported_dataset(dataset_root: Path) -> None:
    payload = load_dataset_info(dataset_root)
    info = payload["info"]
    stats = payload["stats"]

    print(f"Dataset root      : {dataset_root}")
    print(f"Codebase version  : {info['codebase_version']}")
    print(f"Robot type        : {info['robot_type']}")
    print(f"Episodes / frames : {info['total_episodes']} / {info['total_frames']}")
    print(f"FPS               : {info['fps']}")
    print("Features:")
    for name, feature in info["features"].items():
        print(f"  - {name}: dtype={feature['dtype']} shape={tuple(feature['shape'])}")

    if "observation.state" in stats:
        state_stats = stats["observation.state"]
        print(f"State dim         : {len(state_stats['mean'])}")
    if "action" in stats:
        action_stats = stats["action"]
        print(f"Action dim        : {len(action_stats['mean'])}")


def run_inference_demo(model_id: str, lerobot_root: Path) -> None:
    import torch

    ensure_lerobot_importable(lerobot_root)

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    franka_action_dim = 8
    task = "pick up the red block and place it in the bin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SmolVLA from '{model_id}' ...")
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy = policy.to(device)
    policy.eval()

    print("\nModel config:")
    print(f"  VLM backbone : {policy.config.vlm_model_name}")
    print(f"  chunk_size   : {policy.config.chunk_size}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  max_state_dim : {policy.config.max_state_dim}")
    print(f"  max_action_dim: {policy.config.max_action_dim}")
    print(f"  Image resize  : {policy.config.resize_imgs_with_padding}")

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    batch_size = 1
    height, width = 480, 640
    raw_obs = {
        "observation.images.top": torch.rand(batch_size, 3, height, width, device=device),
        "observation.state": torch.tensor(
            [[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]],
            dtype=torch.float32,
            device=device,
        ),
    }

    print("\nRaw observation keys and shapes:")
    for key, value in raw_obs.items():
        print(f"  {key}: {tuple(value.shape)} dtype={value.dtype}")

    processed_obs = preprocess({**raw_obs, "task": task})
    print("\nProcessed observation keys and shapes:")
    for key, value in processed_obs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__} = {value!r}")

    policy.reset()
    action_tensor = policy.select_action(processed_obs)
    action_denorm = postprocess(action_tensor)
    franka_action = action_denorm[0, :franka_action_dim].cpu()

    print(f"\nRaw action tensor shape : {tuple(action_tensor.shape)}")
    print(f"Denormalised action shape: {tuple(action_denorm.shape)}")
    print("\nFranka action (first step):")
    print(f"  Joint targets (q1-q7) [rad] : {franka_action[:7].tolist()}")
    print(f"  Gripper command       [m]   : {franka_action[7].item():.4f}")


def start_training(args: argparse.Namespace) -> None:
    from main import train_smolvla

    output_dir = args.output_dir or Path("outputs") / f"{args.dataset_root.name}_smolvla"
    train_smolvla(
        dataset_root=args.dataset_root,
        lerobot_root=args.lerobot_root,
        policy_path=args.model_id,
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
        episodes=None,
        job_name=args.job_name,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect datasets, run a SmolVLA demo, or start training.")
    parser.add_argument(
        "--mode",
        choices=("demo", "inspect", "train"),
        default="demo",
        help="demo: random inference sanity check, inspect: print dataset summary, train: start fine-tuning",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to the exported LeRobotDataset v3 root.",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=DEFAULT_LEROBOT_ROOT,
        help="Path to the local lerobot repository clone.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="SmolVLA pretrained checkpoint or model id.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Training output directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=20_000, help="Training steps.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device.")
    parser.add_argument("--num-workers", type=int, default=4, help="Training dataloader workers.")
    parser.add_argument("--log-freq", type=int, default=50, help="Training log frequency.")
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency.")
    parser.add_argument("--eval-freq", type=int, default=0, help="Eval frequency. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=1000, help="Training seed.")
    parser.add_argument("--use-amp", action="store_true", help="Enable AMP if supported.")
    parser.add_argument("--job-name", type=str, default=None, help="Optional training run name.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode in {"inspect", "train"} and args.dataset_root is None:
        raise ValueError("--dataset-root is required for inspect and train modes.")

    if args.mode == "demo":
        run_inference_demo(args.model_id, args.lerobot_root)
        return

    if args.mode == "inspect":
        inspect_exported_dataset(args.dataset_root)
        return

    LOGGER.info("Inspecting dataset before training.")
    inspect_exported_dataset(args.dataset_root)
    start_training(args)


if __name__ == "__main__":
    main()
