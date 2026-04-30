# SmolVLA / Pi0 / ACT Dataset + Training Utilities

Unified pipeline for collecting, labeling, converting, and training robot demonstration data with SmolVLA, Pi0, Pi0.5, or ACT policies.

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
      train_act.py
      train_pi0.py
      patch_frame_tolerance.py
      patch_nvenc.py
      patch_task_none.py
    train_act_standalone.py ← standalone ACT training script
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
  outputs/<name>_act/           ← output of  main.py train --model-type act
```

## Installation

Install the `lerobot` package with the relevant extras into its own uv environment from within the `lerobot` sibling directory:

```bash
cd ../lerobot

# SmolVLA
uv pip install -e ".[smolvla]"

# Pi0 / Pi0.5 (adds transformers PaliGemma support)
uv pip install -e ".[pi0]"

# ACT / base LeRobot install
uv pip install -e "."

# Both SmolVLA and Pi0/Pi0.5 extras
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

# 4. Train  (SmolVLA is the default; pass --model-type pi0, pi05, or act to switch)
uv --project ../lerobot run python main.py train \
  --dataset-root lerobot_datasets/<dataset_name> \
  [--model-type smolvla|pi0|pi05|act]
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

# ACT
uv --project ../lerobot run python main.py train \
  --dataset-root lerobot_datasets/socket \
  --model-type act \
  --chunk-size 100 \
  --vision-backbone resnet18 \
  --steps 20000 \
  --batch-size 8
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
| `--model-type` | `smolvla` | Policy architecture: `smolvla`, `pi0`, `pi05`, or `act` |
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
| `act` | `lerobot/act_base` |

**Pi0 / Pi0.5 only options** (ignored for `smolvla`):

| Flag | Default | Description |
|------|---------|-------------|
| `--freeze-vision-encoder` | — | Freeze the PaliGemma vision encoder |
| `--train-expert-only` | — | Freeze entire VLM; train only the action expert and projections |
| `--gradient-checkpointing` | — | Reduce VRAM at the cost of training speed |
| `--dtype` | `float32` | Model weight dtype: `float32` or `bfloat16` |

> **VRAM guidance:** Pi0 and Pi0.5 use a PaliGemma 2B backbone (~20–24 GB in `float32` for a full fine-tune). Use `--dtype bfloat16` and/or `--gradient-checkpointing` if VRAM-constrained. SmolVLA uses a 500M backbone and fits comfortably in 16 GB.

**ACT only options** (ignored for SmolVLA / Pi0 / Pi0.5):

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | `100` | Number of future action steps predicted per forward pass |
| `--n-obs-steps` | `1` | Number of observation steps used as input |
| `--vision-backbone` | `resnet18` | ResNet image encoder: `resnet18`, `resnet34`, or `resnet50` |
| `--use-vae` | enabled | Enable ACT's VAE action modeling component |
| `--no-vae` | — | Disable the VAE for simpler/faster training |

ACT is a smaller transformer imitation-learning policy that predicts chunks of actions from visual observations and robot state. It is often faster to train and run than VLM-based policies, and is a good fit for longer manipulation sequences where chunked actions help smooth execution.

| Aspect | SmolVLA | ACT |
|--------|---------|-----|
| Vision model | SmolVLM2-500M | ResNet18/34/50 |
| Approx. size | 500M+ parameters | 10–25M parameters |
| Action prediction | Single action step | Multi-step action chunks |
| Training speed | Medium | Fast |
| Best fit | Language-conditioned or short-horizon tasks | Fast long-horizon imitation learning |

Common ACT configurations:

```bash
# Quick experiment
uv --project ../lerobot run python main.py train \
  --model-type act \
  --dataset-root lerobot_datasets/001 \
  --chunk-size 50 \
  --batch-size 16 \
  --steps 5000 \
  --use-amp

# Higher-capacity training
uv --project ../lerobot run python main.py train \
  --model-type act \
  --dataset-root lerobot_datasets/001 \
  --vision-backbone resnet50 \
  --chunk-size 100 \
  --batch-size 8 \
  --steps 50000 \
  --seed 42

# Limited VRAM
uv --project ../lerobot run python main.py train \
  --model-type act \
  --dataset-root lerobot_datasets/001 \
  --vision-backbone resnet18 \
  --chunk-size 50 \
  --batch-size 2 \
  --num-workers 2 \
  --use-amp
```

### Standalone ACT Training

For ACT-only experimentation, `train_act_standalone.py` exposes a smaller standalone CLI with an additional `--learning-rate` option:

```bash
uv --project ../lerobot run python train_act_standalone.py \
  --dataset-root lerobot_datasets/001 \
  --output-dir outputs/act_experiment \
  --learning-rate 5e-5 \
  --vision-backbone resnet50 \
  --chunk-size 100 \
  --batch-size 8 \
  --steps 20000 \
  --use-amp
```

ACT checkpoints are written under `outputs/<dataset>_act/` by default. The useful artifacts are usually:

```text
outputs/<dataset>_act/
  train_config.json
  checkpoint_*/              ← intermediate checkpoints
  latest_checkpoint/         ← most recent checkpoint
    config.json
    policy_state_dict.pt
    preprocessor/
    postprocessor/
```

Quick ACT troubleshooting:

| Problem | Try |
|---------|-----|
| CUDA out of memory | Reduce `--batch-size`, use `--vision-backbone resnet18`, reduce `--chunk-size`, and enable `--use-amp` |
| Slow training | Use `resnet18`, enable `--use-amp`, or reduce `--num-workers` if loading is unstable |
| Poor loss | Inspect episodes with `main.py label`, train longer, try `resnet50`, or disable VAE with `--no-vae` |
| Weak generalization | Add more varied demonstrations, train longer, or increase backbone capacity |

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
