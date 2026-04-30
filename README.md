# SmolVLA / Pi0 Dataset + Training Utilities

Unified pipeline for collecting, labeling, converting, and training robot demonstration data with SmolVLA, Pi0, or Pi0.5 policies.

## Project Layout

```text
Industrial-Project/
  lerobot/              ← sibling lerobot clone (provides torch + policy deps)
  SmolVLA-Testing/
    main.py             ← unified CLI entry point (clean / label / convert / train)
    src/                ← Python source modules
      data_cleaner.py
      data_converter.py
      dataset_utils.py
      labeler.py
      create_labels.py
      merge_datasets.py
      smolvla_franka_setup.py
      train_pi0.py
      patch_frame_tolerance.py
      patch_nvenc.py
      patch_task_none.py
    scripts/            ← bash scripts for batch ops and GPU cluster workflows
      convert_all.sh
      restart_conversion.sh
      run_training.sh
      setup_scratch.sh
      training_status_snapshot.sh
      00_run_params.sh
      01_sync_to_gpu.sh
      02_preflight_gpu.sh
      03_setup_gpu.sh
      04_start_training.sh
      05_extract_from_scratch.sh
    frontend/           ← labeler web UI (templates + static assets)
```

Data directories created by the pipeline:

```text
SmolVLA-Testing/
  raw_datasets/<name>/          ← recorded session (input to clean)
  cleaned_datasets/<name>/      ← output of  main.py clean
  lerobot_datasets/<name>/      ← output of  main.py convert  (LeRobot v3)
  outputs/<name>_smolvla/       ← output of  main.py train --model-type smolvla
  outputs/<name>_pi0/           ← output of  main.py train --model-type pi0
  outputs/<name>_pi05/          ← output of  main.py train --model-type pi05
```

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

## Running Commands

`SmolVLA-Testing` does not have its own `pyproject.toml`. All `main.py` commands must be run against the `lerobot` project environment so that `torch`, `lerobot`, and other heavy deps resolve correctly:

```bash
uv --project ../lerobot run python main.py <subcommand> [args]
```

> **Note:** plain `python` / `python3` won't have `torch` installed, and `uv run` (without `--project ../lerobot`) uses this repo's missing environment. Always use `uv --project ../lerobot run python main.py ...`.

## Full Pipeline

```bash
# 1. Clean raw recording
uv --project ../lerobot run python main.py clean <dataset_name>

# 2. (Optional) Label episodes in the browser UI
uv --project ../lerobot run python main.py label

# 3. Convert to LeRobotDataset v3
uv --project ../lerobot run python main.py convert <dataset_name> --primary-camera <camera_name>

# 4. Train  (SmolVLA is the default; pass --model-type pi0 or --model-type pi05 to switch)
uv --project ../lerobot run python main.py train \
  --dataset-root lerobot_datasets/<dataset_name> \
  [--model-type smolvla|pi0|pi05]
```

Example end-to-end for a dataset called `socket` with dual ZED Mini + third-person cameras:

```bash
uv --project ../lerobot run python main.py clean socket --generate-tasks --force
uv --project ../lerobot run python main.py label
uv --project ../lerobot run python main.py convert socket --primary-camera ee_zed_m_left --force

# SmolVLA
uv --project ../lerobot run python main.py train \
  --dataset-root lerobot_datasets/socket \
  --model-type smolvla \
  --steps 20000 \
  --batch-size 8

# Pi0
uv --project ../lerobot run python main.py train \
  --dataset-root lerobot_datasets/socket \
  --model-type pi0 \
  --steps 20000 \
  --batch-size 4
```

## Command Reference

### `main.py clean` — Filter Raw Dataset

Removes static (non-moving) steps and trims to camera-covered frames only.

```bash
uv --project ../lerobot run python main.py clean <dataset_name> [options]
```

Input: `raw_datasets/<dataset_name>/`  
Output: `cleaned_datasets/<dataset_name>/`

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
| `--generate-tasks` | — | Auto-assign a task prompt to each episode and write `annotations.jsonl` |
| `--force` | — | Overwrite existing output directory |

### `main.py label` — Episode Labeling UI

Launches a local Flask web server for reviewing and labeling episodes in the browser.

```bash
uv --project ../lerobot run python main.py label [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--cleaned-root` | `cleaned_datasets` | Root of cleaned datasets |
| `--raw-root` | `raw_datasets` | Root of raw videos |
| `--port` | `5000` | HTTP port for the labeler server |
| `--no-browser` | — | Don't auto-open the browser |

### `main.py convert` — Convert to LeRobotDataset v3

```bash
uv --project ../lerobot run python main.py convert <dataset_name> [options]
```

Input: `cleaned_datasets/<dataset_name>/`  
Output: `lerobot_datasets/<dataset_name>/`

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets-root` | `cleaned_datasets` | Root containing cleaned datasets |
| `--output-root` | `lerobot_datasets` | Root for converted output |
| `--repo-id` | `local/<dataset_name>` | LeRobot metadata repo id |
| `--primary-camera` | — | Camera mapped to `observation.images.top` |
| `--cameras` | — | Comma-separated camera names to include (default: all) |
| `--camera-tolerance-ms` | `150` | Max robot/camera sync error (ms) |
| `--text-tolerance-ms` | `2000` | Max text/frame sync error (ms) |
| `--vcodec` | `h264` | Output video codec (`h264_nvenc` on GPU cluster) |
| `--max-episodes` | — | Limit exported episodes |
| `--max-steps-per-episode` | — | Limit steps per episode |
| `--episode-report` | — | Write JSON classification report to this path |
| `--force` | — | Overwrite existing output directory |

### `main.py train` — Fine-Tune a Policy

```bash
uv --project ../lerobot run python main.py train --dataset-root <path> [options]
```

**Common options** (all model types):

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `smolvla` | Policy architecture: `smolvla`, `pi0`, or `pi05` |
| `--dataset-root` | *(required)* | Local LeRobotDataset v3 export directory |
| `--lerobot-root` | *(auto-detected)* | Path to local lerobot clone |
| `--policy-path` | *(model default)* | Pretrained checkpoint or HF model id |
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

| `--model-type` | Default `--policy-path` |
|----------------|-------------------------|
| `smolvla` | `lerobot/smolvla_base` |
| `pi0` | `lerobot/pi0_base` |
| `pi05` | `lerobot/pi05_base` |

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
uv --project ../lerobot run python src/smolvla_franka_setup.py \
  --mode inspect \
  --dataset-root lerobot_datasets/example
```

### SmolVLA Inference Sanity Check

```bash
uv --project ../lerobot run python src/smolvla_franka_setup.py --mode demo
```

### Verify CUDA Torch

```bash
uv --project ../lerobot run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
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
