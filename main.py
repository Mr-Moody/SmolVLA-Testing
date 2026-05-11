r"""
Unified CLI for the SmolVLA robot data pipeline.

Subcommands
-----------
  clean    Filter a raw recording to motion-only, camera-covered steps.
  label    Launch the browser-based episode labeling UI.
  convert  Export a cleaned dataset to LeRobotDataset v3 format.
  train    Fine-tune SmolVLA, Pi0, Pi0.5, or ACT on an exported dataset.

Examples
--------
  uv --project ../lerobot run python main.py clean 001 --force
  uv --project ../lerobot run python main.py annotate 001
  uv --project ../lerobot run python main.py label --port 5000
  uv --project ../lerobot run python main.py convert 001 --primary-camera ee_zed_m
  uv --project ../lerobot run python main.py train --dataset-root lerobot_datasets/001 --steps 20000
  uv --project ../lerobot run python main.py train --model-type pi0 --dataset-root lerobot_datasets/001 --steps 20000
  uv --project ../lerobot run python main.py train --model-type act --dataset-root lerobot_datasets/001 --steps 20000
  uv --project ../lerobot run python main.py train --model-type act --dataset-root lerobot_datasets/001 --chunk-size 50 --vision-backbone resnet50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

# Ensure src/ modules are importable regardless of invocation directory.
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded defaults — must stay in sync with the corresponding src/ modules.
# They are kept here so that _add_*_parser can register --help text without
# triggering any heavy imports (torch, flask, lerobot) at startup.
# ---------------------------------------------------------------------------

# data_cleaner.py / dataset_utils.py
_CLEAN_DATASETS_ROOT       = Path("raw_datasets")
_CLEAN_OUTPUT_ROOT         = Path("cleaned_datasets")
_CLEAN_CAMERA_TOLERANCE_MS = 150.0          # DEFAULT_CAMERA_TOLERANCE_NS / 1e6
_CLEAN_JOINT_THRESHOLD     = 5e-4
_CLEAN_GRIPPER_THRESHOLD   = 2e-4
_CLEAN_TRANS_THRESHOLD     = 5e-6
_CLEAN_ROT_THRESHOLD       = 5e-5

# scripts/generate_annotations.py
_ANNOT_DATASETS_ROOT       = Path("cleaned_datasets")

# data_converter.py
_CONV_DATASETS_ROOT        = Path("cleaned_datasets")
_CONV_OUTPUT_ROOT          = Path("lerobot_datasets")
_CONV_CAMERA_TOLERANCE_MS  = 150.0          # DEFAULT_CAMERA_TOLERANCE_NS / 1e6
_CONV_TEXT_TOLERANCE_MS    = 2000.0         # DEFAULT_TEXT_TOLERANCE_NS / 1e6
_CONV_VCODEC               = "h264"
_CONV_ENCODER_QUEUE        = 60
_CONV_BLANK_MAX_STEPS      = 1000
_CONV_MIN_GRIPPER_CMD      = 0.1
_CONV_MIN_GRIPPER_SPAN     = 0.002


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

def _add_clean_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("clean", help="Filter a raw recording to motion-only, camera-covered steps.")
    p.add_argument("dataset_name", help="Dataset directory name under --datasets-root.")
    p.add_argument("--datasets-root", type=Path, default=_CLEAN_DATASETS_ROOT,
                   help="Root directory containing raw datasets (default: %(default)s).")
    p.add_argument("--output-root", type=Path, default=_CLEAN_OUTPUT_ROOT,
                   help="Root directory for cleaned output (default: %(default)s).")
    p.add_argument("--camera-tolerance-ms", type=float, default=_CLEAN_CAMERA_TOLERANCE_MS,
                   help="Max robot-to-camera sync error in ms (default: %(default)s).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Limit number of episodes to process.")
    p.add_argument("--joint-motion-threshold", type=float, default=_CLEAN_JOINT_THRESHOLD,
                   help="Max joint delta (rad) considered stationary (default: %(default)s).")
    p.add_argument("--gripper-motion-threshold", type=float, default=_CLEAN_GRIPPER_THRESHOLD,
                   help="Max gripper-width delta (m) considered stationary (default: %(default)s).")
    p.add_argument("--action-translation-threshold", type=float, default=_CLEAN_TRANS_THRESHOLD,
                   help="Min cartesian_delta_translation norm considered movement (default: %(default)s).")
    p.add_argument("--action-rotation-threshold", type=float, default=_CLEAN_ROT_THRESHOLD,
                   help="Min cartesian_delta_rotation norm considered movement (default: %(default)s).")
    p.add_argument("--generate-tasks", action="store_true",
                   help="Auto-assign a task prompt to each kept episode and write annotations.jsonl.")
    p.set_defaults(func=_cmd_clean)


def _cmd_clean(args: argparse.Namespace) -> None:
    from data_cleaner import DatasetCleaner
    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset_name}' not found at {dataset_dir}")
    output_dir = args.output_root / args.dataset_name
    DatasetCleaner(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        camera_tolerance_ns=int(args.camera_tolerance_ms * 1_000_000.0),
        force_overwrite=args.force,
        max_episodes=args.max_episodes,
        joint_motion_threshold=args.joint_motion_threshold,
        gripper_motion_threshold=args.gripper_motion_threshold,
        action_translation_threshold=args.action_translation_threshold,
        action_rotation_threshold=args.action_rotation_threshold,
        generate_tasks=args.generate_tasks,
    ).clean()


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

def _add_label_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("label", help="Launch the browser-based episode labeling UI.")
    p.add_argument("--cleaned-root", type=Path, default=Path("cleaned_datasets"),
                   help="Root of cleaned datasets (default: cleaned_datasets).")
    p.add_argument("--raw-root", type=Path, default=Path("raw_datasets"),
                   help="Root of raw datasets / videos (default: raw_datasets).")
    p.add_argument("--port", type=int, default=5000,
                   help="Port for the labeler server (default: 5000).")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser.")
    p.set_defaults(func=_cmd_label)


def _cmd_label(args: argparse.Namespace) -> None:
    import labeler

    if not args.cleaned_root.exists():
        print(f"Error: cleaned datasets root not found: {args.cleaned_root}", file=sys.stderr)
        sys.exit(1)
    if not args.raw_root.exists():
        print(f"Error: raw datasets root not found: {args.raw_root}", file=sys.stderr)
        sys.exit(1)

    labeler.CLEANED_DATASETS_ROOT = args.cleaned_root
    labeler.RAW_DATASETS_ROOT = args.raw_root

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"Labeler running at {url}")
    print(f"Episodes + annotations root: {args.cleaned_root.resolve()}")
    print(f"Video source root:           {args.raw_root.resolve()}")
    print("Press Ctrl+C to quit.")
    labeler.app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def _add_convert_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("convert", help="Export a cleaned dataset to LeRobotDataset v3 format.")
    p.add_argument("dataset_name", help="Dataset name under --datasets-root.")
    p.add_argument("--datasets-root", type=Path, default=_CONV_DATASETS_ROOT,
                   help="Root containing cleaned datasets (default: %(default)s).")
    p.add_argument("--output-root", type=Path, default=_CONV_OUTPUT_ROOT,
                   help="Root for LeRobotDataset output (default: %(default)s).")
    p.add_argument("--repo-id", type=str, default=None,
                   help="LeRobot repo_id (default: local/<dataset_name>).")
    p.add_argument("--primary-camera", type=str, default=None,
                   help="Camera name to map to observation.images.top.")
    p.add_argument("--cameras", type=str, default=None,
                   help="Comma-separated camera names to include. Omit for all cameras.")
    p.add_argument("--camera-tolerance-ms", type=float, default=_CONV_CAMERA_TOLERANCE_MS,
                   help="Max robot-to-camera sync error in ms (default: %(default)s).")
    p.add_argument("--text-tolerance-ms", type=float, default=_CONV_TEXT_TOLERANCE_MS,
                   help="Max text-to-frame sync error in ms (default: %(default)s).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    p.add_argument("--vcodec", type=str, default=_CONV_VCODEC,
                   help="Video codec (default: %(default)s).")
    p.add_argument("--encoder-threads", type=int, default=None,
                   help="Threads per encoder instance. Omit for codec default.")
    p.add_argument("--encoder-queue-maxsize", type=int, default=_CONV_ENCODER_QUEUE,
                   help="Max buffered frames per camera during encode (default: %(default)s).")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Limit number of episodes to export.")
    p.add_argument("--max-steps-per-episode", type=int, default=None,
                   help="Limit steps exported per episode.")
    p.add_argument("--keep-blank-episodes", action="store_true",
                   help="Keep short episodes with no gripper activity.")
    p.add_argument("--blank-max-steps", type=int, default=_CONV_BLANK_MAX_STEPS,
                   help="Max episode length for blank suppression (default: %(default)s).")
    p.add_argument("--min-gripper-command", type=float, default=_CONV_MIN_GRIPPER_CMD,
                   help="Min absolute gripper command for an active event (default: %(default)s).")
    p.add_argument("--min-gripper-width-span", type=float, default=_CONV_MIN_GRIPPER_SPAN,
                   help="Min gripper width span (m) for an active event (default: %(default)s).")
    p.add_argument("--episode-report", type=Path, default=None,
                   help="Optional JSON path for episode classification report.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (e.g. cuda, cpu). Defaults to cuda if available.")
    p.set_defaults(func=_cmd_convert)


def _cmd_convert(args: argparse.Namespace) -> None:
    from data_converter import SmolVLADatasetConverter, DEFAULT_REPO_OWNER
    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset_name}' not found at {dataset_dir}")
    output_dir = args.output_root / args.dataset_name
    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    SmolVLADatasetConverter(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        repo_id=args.repo_id or f"{DEFAULT_REPO_OWNER}/{args.dataset_name}",
        primary_camera=args.primary_camera,
        cameras=cameras,
        camera_tolerance_ns=int(args.camera_tolerance_ms * 1_000_000.0),
        text_tolerance_ns=int(args.text_tolerance_ms * 1_000_000.0),
        force_overwrite=args.force,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        suppress_blank_episodes=not args.keep_blank_episodes,
        blank_max_steps=args.blank_max_steps,
        min_gripper_command=args.min_gripper_command,
        min_gripper_width_span=args.min_gripper_width_span,
        episode_report_path=args.episode_report,
        device=args.device,
    ).export()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

DEFAULT_POLICY_PATHS = {
    "smolvla": "lerobot/smolvla_base",
    "pi0":     "lerobot/pi0_base",
    "pi05":    "lerobot/pi05_base",
    "act":     "lerobot/act_base",
}


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "train",
        help="Fine-tune SmolVLA, Pi0, Pi0.5, or ACT on an exported LeRobotDataset v3.",
    )
    p.add_argument("--model-type", type=str, default="smolvla",
                   choices=["smolvla", "pi0", "pi05", "act"],
                   help="Policy architecture to train (default: smolvla).")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path to an exported LeRobotDataset v3 root.")
    p.add_argument("--lerobot-root", type=Path, default=None,
                   help="Path to the local lerobot repository (auto-detected if omitted).")
    p.add_argument("--policy-path", type=str, default=None,
                   help="Pretrained checkpoint or HF model id. Defaults to the base for --model-type.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for checkpoints and logs. Defaults to outputs/<dataset>_<model-type>.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--eval-freq", type=int, default=0,
                   help="Evaluation frequency. 0 disables env-based evaluation.")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--use-amp", action="store_true",
                   help="Enable automatic mixed precision.")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated subset of episode indices, e.g. 0,1,2.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last checkpoint in --output-dir.")
    p.add_argument("--job-name", type=str, default=None)
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push the trained policy to the Hugging Face Hub.")
    p.add_argument("--policy-repo-id", type=str, default=None,
                   help="HF repo id for --push-to-hub, e.g. your-user/your-model.")
    # Pi0 / Pi0.5 specific
    p.add_argument("--freeze-vision-encoder", action="store_true",
                   help="[Pi0/Pi0.5] Freeze the PaliGemma vision encoder.")
    p.add_argument("--train-expert-only", action="store_true",
                   help="[Pi0/Pi0.5] Freeze the VLM; train only the action expert.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="[Pi0/Pi0.5] Enable gradient checkpointing to reduce VRAM.")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"],
                   help="[Pi0/Pi0.5] Model weight dtype (default: float32).")
    # ACT specific
    p.add_argument("--chunk-size", type=int, default=100,
                   help="[ACT] Number of action steps to predict in a chunk (default: 100).")
    p.add_argument("--n-obs-steps", type=int, default=1,
                   help="[ACT] Number of observation steps to use (default: 1).")
    p.add_argument("--use-vae", action="store_true", default=True,
                   help="[ACT] Use VAE for action modeling (default: True).")
    p.add_argument("--no-vae", dest="use_vae", action="store_false",
                   help="[ACT] Disable VAE for action modeling.")
    p.add_argument("--vision-backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"],
                   help="[ACT] Vision backbone architecture (default: resnet18).")
    p.set_defaults(func=_cmd_train)


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------

def _add_annotate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "annotate",
        help="Generate per-episode task prompts and write annotations.jsonl into a cleaned dataset.",
    )
    p.add_argument("dataset_name", help="Dataset folder name under --datasets-root.")
    p.add_argument("--datasets-root", type=Path, default=_ANNOT_DATASETS_ROOT,
                   help="Root directory containing cleaned datasets (default: %(default)s).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite an existing annotations.jsonl.")
    p.set_defaults(func=_cmd_annotate)


def _cmd_annotate(args: argparse.Namespace) -> None:
    from src.scripts.generate_annotations import count_episodes, generate_prompts
    from src.dataset_utils import save_annotation

    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    annotations_path = dataset_dir / "annotations.jsonl"
    if annotations_path.exists() and not args.overwrite:
        print(
            f"ERROR: {annotations_path} already exists. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    num_episodes = count_episodes(dataset_dir)
    if num_episodes == 0:
        print("ERROR: No episodes found in episode_events.jsonl.", file=sys.stderr)
        sys.exit(1)

    prompts = generate_prompts(num_episodes)
    for episode_index, task in enumerate(prompts):
        save_annotation(dataset_dir, episode_index, task)

    print(f"Wrote {num_episodes} prompts to {annotations_path}")
    print("\nExamples:")
    for i, p in enumerate(prompts[:5]):
        print(f"  [{i}] {p}")


def _parse_episode_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(c.strip()) for c in value.split(",") if c.strip()] or None


def _ensure_lerobot_importable(lerobot_root: Path) -> None:
    lerobot_src = lerobot_root / "src"
    if not lerobot_src.exists():
        raise FileNotFoundError(f"Could not find lerobot src directory at {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def _disable_lerobot_hub_lookups() -> None:
    if os.getenv("HF_HUB_OFFLINE") != "1":
        return
    import lerobot.datasets.lerobot_dataset as _ds
    import lerobot.datasets.utils as _du

    def _local_safe_version(_repo_id: str, version: str) -> str:
        return f"v{str(version).lstrip('v')}"

    _ds.get_safe_version = _local_safe_version
    _du.get_safe_version = _local_safe_version
    _du.check_version_compatibility = lambda *a, **kw: None


def _resolve_lerobot_root(explicit_root: Path | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root.expanduser())
    env_root = os.getenv("LEROBOT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(script_dir / "lerobot")
    candidates.extend(p / "lerobot" for p in script_dir.parents)

    seen: set[Path] = set()
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            if (r / "src").exists():
                LOGGER.info("Using lerobot root: %s", r)
                return r

    searched = "\n".join(f" - {c / 'src'}" for c in candidates)
    raise FileNotFoundError(
        "Could not find lerobot src directory.\nChecked:\n"
        f"{searched}\n"
        "Pass --lerobot-root /path/to/lerobot or set LEROBOT_ROOT."
    )


def _load_dataset_info(dataset_root: Path) -> dict:
    import json
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find LeRobot metadata at {info_path}")
    with info_path.open("r", encoding="utf-8") as fh:
        info = json.load(fh)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobotDataset v3, got codebase_version={info.get('codebase_version')!r}"
        )
    return info


def _train_smolvla(*, dataset_root, lerobot_root, policy_path, output_dir, batch_size, steps,
                   device, num_workers, log_freq, save_freq, eval_freq, seed, use_amp,
                   resume, episodes, job_name, push_to_hub, policy_repo_id) -> Path:
    _ensure_lerobot_importable(lerobot_root)
    _disable_lerobot_hub_lookups()

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla import SmolVLAConfig
    from lerobot.scripts.lerobot_train import train

    info = _load_dataset_info(dataset_root)
    LOGGER.info("Dataset:   %s  (%d ep, %d frames)",
                dataset_root, info["total_episodes"], info["total_frames"])
    LOGGER.info("Policy:    %s  |  device: %s  |  steps: %d  |  batch: %d",
                policy_path, device, steps, batch_size)

    cfg = TrainPipelineConfig(
        resume=resume,
        dataset=DatasetConfig(
            repo_id=f"local/{dataset_root.name}",
            root=str(dataset_root),
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
    if resume:
        cfg.config_path = output_dir / "train_config.json"

    train(cfg)
    LOGGER.info("Training finished. Outputs: %s", output_dir)
    return output_dir


def _cmd_train(args: argparse.Namespace) -> None:
    dataset_root: Path = args.dataset_root
    model_type: str = args.model_type
    policy_path = args.policy_path or DEFAULT_POLICY_PATHS[model_type]
    output_dir = args.output_dir or Path("outputs") / f"{dataset_root.name}_{model_type}"
    lerobot_root = _resolve_lerobot_root(args.lerobot_root)
    episodes = _parse_episode_list(args.episodes)

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
        _train_smolvla(**common)
    elif model_type == "act":
        from train_act import train_act
        train_act(
            **common,
            chunk_size=args.chunk_size,
            n_obs_steps=args.n_obs_steps,
            use_vae=args.use_vae,
            vision_backbone=args.vision_backbone,
        )
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SmolVLA robot data pipeline — clean, label, convert, train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # Register subparsers up front for --help. All defaults are hardcoded
    # constants above — no src/ module is imported here, so startup is fast
    # regardless of which subcommand is used.
    _add_clean_parser(sub)
    _add_annotate_parser(sub)
    _add_label_parser(sub)
    _add_convert_parser(sub)
    _add_train_parser(sub)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
