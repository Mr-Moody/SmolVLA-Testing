# SmolVLA / Pi0 Dataset + Training Utilities

This repository contains:

- `data_cleaner.py`: filters a raw recorded session to motion-only, camera-covered steps.
- `data_converter.py`: converts a cleaned session to native LeRobotDataset v3 format.
- `main.py`: trains SmolVLA, Pi0, or Pi0.5 on a local LeRobotDataset v3 export.
- `train_pi0.py`: Pi0 / Pi0.5 training backend (called by `main.py`, not run directly).

## Project Layout

Expected sibling layout:

```text
Industrial-Project/
  lerobot/
  SmolVLA-Testing/
```

Data directories created by the pipeline:

```text
SmolVLA-Testing/
  raw_datasets/<dataset_name>/           ← recorded session (input)
  cleaned_datasets/<dataset_name>/       ← after data_cleaner.py
  lerobot_datasets/<dataset_name>/       ← after data_converter.py (LeRobot v3)
  outputs/<dataset_name>_smolvla/        ← after main.py --model-type smolvla
  outputs/<dataset_name>_pi0/            ← after main.py --model-type pi0
  outputs/<dataset_name>_pi05/           ← after main.py --model-type pi05
```

Most commands below are run from `SmolVLA-Testing/`.

## Installation

Install the `lerobot` package with the relevant extras into its own uv environment from within the `lerobot` sibling directory:

```bash
cd ../lerobot

# SmolVLA
uv pip install -e ".[smolvla]"

# Pi0 / Pi0.5 (adds transformers PaliGemma support)
uv pip install -e ".[pi0]"

# Both
uv pip install -e ".[smolvla,pi0]"
```

## Environment Notes

`SmolVLA-Testing` does not have its own `pyproject.toml`, so run Python commands against the `lerobot` project environment:

```bash
uv run --project ../lerobot python ...
```

This ensures imports like `torch` and `lerobot` resolve correctly (including CUDA-enabled torch if installed there).

## Full Pipeline

```bash
# 1. Clean
uv run --project ../lerobot python data_cleaner.py <dataset_name>

# 2. Convert
uv run --project ../lerobot python data_converter.py <dataset_name> --primary-camera <camera_name>

# 3. Train  (SmolVLA is the default; pass --model-type pi0 or --model-type pi05 to switch)
uv run --project ../lerobot python main.py \
  --dataset-root lerobot_datasets/<dataset_name> \
  [--model-type smolvla|pi0|pi05]
```

Example end-to-end for a dataset called `socket` with dual ZED Mini + third-person cameras:

```bash
uv run --project ../lerobot python data_cleaner.py socket --generate-tasks --force
uv run --project ../lerobot python data_converter.py socket --primary-camera ee_zed_m_left --force

# SmolVLA
uv run --project ../lerobot python main.py \
  --dataset-root lerobot_datasets/socket \
  --model-type smolvla \
  --steps 20000 \
  --batch-size 8

# Pi0
uv run --project ../lerobot python main.py \
  --dataset-root lerobot_datasets/socket \
  --model-type pi0 \
  --steps 20000 \
  --batch-size 4
```

## Command Reference

### 1) Clean Raw Dataset

Removes static (non-moving) steps and trims to camera-covered frames only.

```bash
uv run --project ../lerobot python data_cleaner.py <dataset_name> [options]
```

Input: `raw_datasets/<dataset_name>/`  
Output: `cleaned_datasets/<dataset_name>/`

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets-root` | `raw_datasets` | Root containing raw recordings |
| `--output-root` | `cleaned_datasets` | Root for cleaned output |
| `--camera-tolerance-ms` | `150` | Max robot/camera sync error (ms) |
| `--joint-motion-threshold` | `5e-4` | Max joint delta (rad) considered stationary |
| `--gripper-motion-threshold` | `2e-4` | Max gripper delta (m) considered stationary |
| `--action-translation-threshold` | `5e-6` | Min translation norm considered movement |
| `--action-rotation-threshold` | `5e-5` | Min rotation norm considered movement |
| `--max-episodes` | — | Limit number of episodes processed |
| `--force` | — | Overwrite existing output directory |
| `--generate-tasks` | — | Auto-assign a unique global task prompt to each kept episode and write `annotations.jsonl` |

### 2) Convert Cleaned Dataset to LeRobotDataset v3

```bash
uv run --project ../lerobot python data_converter.py <dataset_name> [options]
```

Input: `cleaned_datasets/<dataset_name>/`  
Output: `lerobot_datasets/<dataset_name>/`

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets-root` | `cleaned_datasets` | Root containing cleaned datasets |
| `--output-root` | `lerobot_datasets` | Root for converted output |
| `--repo-id` | `local/<dataset_name>` | LeRobot metadata repo id |
| `--primary-camera` | — | Camera mapped to `observation.images.top` |
| `--camera-tolerance-ms` | `150` | Max robot/camera sync error (ms) |
| `--text-tolerance-ms` | `2000` | Max text/frame sync error (ms) |
| `--vcodec` | `h264` | Output video codec |
| `--max-episodes` | — | Limit exported episodes |
| `--max-steps-per-episode` | — | Limit steps per episode |
| `--force` | — | Overwrite existing output directory |

### 3) Train

```bash
uv run --project ../lerobot python main.py --dataset-root <path> [options]
```

**Common options** (all model types):

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `smolvla` | Policy architecture: `smolvla`, `pi0`, or `pi05` |
| `--dataset-root` | *(required)* | Local LeRobotDataset v3 export directory |
| `--lerobot-root` | *(auto-detected)* | Path to local lerobot clone |
| `--policy-path` | *(model default)* | Pretrained checkpoint or HF model id (see defaults below) |
| `--output-dir` | `outputs/<dataset>_<model-type>` | Directory for checkpoints and logs |
| `--batch-size` | `8` | Training batch size |
| `--steps` | `20000` | Training steps |
| `--device` | `cuda` | Training device (`cuda` or `cpu`) |
| `--num-workers` | `4` | DataLoader worker count |
| `--log-freq` | `50` | Log every N steps |
| `--save-freq` | `1000` | Save checkpoint every N steps |
| `--eval-freq` | `0` | Eval every N steps (`0` disables) |
| `--seed` | `1000` | Training seed |
| `--use-amp` | — | Enable automatic mixed precision |
| `--episodes` | — | Comma-separated episode subset, e.g. `0,1,2` |
| `--resume` | — | Resume from last checkpoint in `--output-dir` |
| `--job-name` | — | Custom run name |
| `--push-to-hub` | — | Push trained policy to Hugging Face Hub |
| `--policy-repo-id` | — | Required with `--push-to-hub`, e.g. `user/model` |

**Default pretrained base checkpoints by model type:**

| `--model-type` | Default `--policy-path` | Normalization |
|----------------|-------------------------|---------------|
| `smolvla` | `lerobot/smolvla_base` | MEAN_STD |
| `pi0` | `lerobot/pi0_base` | MEAN_STD |
| `pi05` | `lerobot/pi05_base` | QUANTILES |

**Pi0 / Pi0.5 only options** (ignored for `smolvla`):

| Flag | Default | Description |
|------|---------|-------------|
| `--freeze-vision-encoder` | — | Freeze the PaliGemma vision encoder |
| `--train-expert-only` | — | Freeze entire VLM; train only the action expert and projections |
| `--gradient-checkpointing` | — | Reduce VRAM at the cost of training speed |
| `--dtype` | `float32` | Model weight dtype: `float32` or `bfloat16` |

> **VRAM guidance:** Pi0 and Pi0.5 use a PaliGemma 2B backbone (~20–24 GB in `float32` for a full fine-tune). Use `--dtype bfloat16` and/or `--gradient-checkpointing` if VRAM-constrained. SmolVLA uses a 500M backbone and fits comfortably in 16 GB.

## Utilities

### Inspect an Exported Dataset

```bash
uv run --project ../lerobot python smolvla_franka_setup.py \
  --mode inspect \
  --dataset-root lerobot_datasets/example
```

### SmolVLA Inference Sanity Check

```bash
uv run --project ../lerobot python smolvla_franka_setup.py --mode demo
```

### Verify CUDA Torch

```bash
uv run --project ../lerobot python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Franka Docker Notes (Optional)

Allow Docker to show windows on host:

```bash
xhost +local:docker
```

Create and enter container:

```bash
docker run -it \
  --name franka_noetic \
  --net=host \
  --privileged \
  -v ~/catkin_ws:/home/thomas/catkin_ws \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  osrf/ros:noetic-desktop-full \
  bash
```

```bash
docker exec -it franka_noetic bash
```

Before robot code:

- Set host ethernet to manual IP `172.16.0.2`.
- Confirm connectivity to robot: `ping 172.16.0.1`.
- In Franka Desk at `https://172.16.0.1`, ensure `FCI` is active (blue), brakes are unlocked, and the external activation device is pressed.

Launch driver in container:

```bash
source devel/setup.bash
roslaunch franka_control franka_control.launch robot_ip:=172.16.0.1 load_gripper:=true
```
