# SmolVLA Dataset + Training Utilities

This repository contains:

- `data_converter.py`: converts raw recorded sessions to native LeRobotDataset v3 format.
- `main.py`: trains SmolVLA on a local LeRobotDataset v3 export.

## Project Layout

Expected sibling layout:

```text
Industrial-Project/
  lerobot/
  SmolVLA-Testing/
```

Most commands below are run from `SmolVLA-Testing/`.

## Environment Notes

`SmolVLA-Testing` does not have its own `pyproject.toml`, so run Python commands against the `lerobot` project environment:

```bash
uv run --project ../lerobot python ...
```

This ensures imports like `torch` and `lerobot` resolve correctly (including CUDA-enabled torch if installed there).

## Command Usage

### 1) Convert Raw Dataset to LeRobotDataset v3

```bash
uv run --project ../lerobot python data_converter.py <dataset_name> [options]
```

Example:

```bash
uv run --project ../lerobot python data_converter.py example --primary-camera ee_zed_m
```

Common options:

- `--datasets-root`: root containing raw recordings (default: `raw_datasets`)
- `--output-root`: output root for converted datasets (default: `lerobot_datasets`)
- `--repo-id`: metadata repo id (default: `local/<dataset_name>`)
- `--primary-camera`: camera mapped to `observation.images.top`
- `--camera-tolerance-ms`: robot/camera sync tolerance in ms
- `--text-tolerance-ms`: text/frame sync tolerance in ms
- `--force`: overwrite existing output dataset directory
- `--vcodec`: output video codec (default: `h264`)
- `--max-episodes`: limit exported episodes
- `--max-steps-per-episode`: limit steps per episode
- `--keep-blank-episodes`: disable blank-episode suppression
- `--blank-max-steps`: short-episode threshold for blank suppression (default: `1000`)
- `--min-gripper-command`: absolute gripper command threshold for activity detection (default: `0.1`)
- `--min-gripper-width-span`: gripper width span threshold in meters (default: `0.002`)
- `--episode-report`: write JSON report with episode quality metrics and suppression flags

By default, the converter suppresses blank transition episodes that are short and show no gripper activity. This helps when `episode_start` markers are used as separators and include non-grasp repositioning segments.

### 2) Train SmolVLA

```bash
uv run --project ../lerobot python main.py --dataset-root <path> [options]
```

Example (CUDA):

```bash
uv run --project ../lerobot python main.py \
  --dataset-root lerobot_datasets/example \
  --steps 20000 \
  --batch-size 8 \
  --device cuda
```

Common options:

- `--dataset-root` (required): local LeRobotDataset v3 export directory
- `--lerobot-root`: optional local `lerobot` path (auto-detected if omitted)
- `--policy-path`: base SmolVLA checkpoint or HF model id (default: `lerobot/smolvla_base`)
- `--output-dir`: output directory for checkpoints/logs
- `--batch-size`: training batch size (default: `8`)
- `--steps`: training steps (default: `20000`)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--num-workers`: dataloader workers (default: `4`)
- `--log-freq`: training log frequency
- `--save-freq`: checkpoint save frequency
- `--eval-freq`: eval frequency (`0` disables environment eval)
- `--seed`: training seed
- `--use-amp`: enable automatic mixed precision
- `--episodes`: comma-separated subset (example: `0,1,2`)
- `--job-name`: custom run name
- `--push-to-hub`: push trained policy to Hugging Face Hub
- `--policy-repo-id`: required when using `--push-to-hub`

## CUDA Torch Check

To verify CUDA torch in the `lerobot` environment:

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
