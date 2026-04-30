# ACT Model Training Guide

This guide explains how to train and finetune Action Chunking Transformers (ACT) models from the LeRobot framework using the SmolVLA-Testing pipeline.

## Overview

**ACT (Action Chunking Transformers)** is an imitation learning policy that:
- Predicts sequences of actions ("chunks") conditioned on visual observations
- Uses a transformer architecture with ResNet vision backbone
- Optionally includes a VAE component for learning compact action representations
- Is well-suited for manipulation tasks with longer action horizons

## Key Differences from SmolVLA

| Feature | ACT | SmolVLA |
|---------|-----|---------|
| Architecture | Transformer + ResNet backbone | Vision-Language Model (SmolVLM2) |
| Input | Images + proprioceptive state | Images + optional text instructions |
| Output | Action sequences (chunks) | Single action step |
| Training data | Requires action labels | Requires action labels |
| Model size | Smaller (ResNet18-50) | Larger (500M+ parameters) |
| Inference speed | Fast | Moderate |

## Quick Start

### Basic ACT Training

```bash
# Train ACT on a dataset with default settings
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --steps 20000
```

### Train with Custom Chunk Size

```bash
# Smaller chunks (predicts 50 steps per forward pass)
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --chunk-size 50 \
    --steps 20000
```

### Train with Larger Vision Backbone

```bash
# Use ResNet50 instead of ResNet18 for better accuracy (slower training)
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --vision-backbone resnet50 \
    --batch-size 4 \
    --steps 20000
```

### Train Without VAE

```bash
# Disable VAE for simpler, faster training
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --no-vae \
    --steps 20000
```

## Configuration Parameters

### Architecture Parameters (ACT-specific)

- `--chunk-size` (int, default: 100)
  - Number of action steps to predict in each forward pass
  - Smaller chunks = more frequent policy invocations but simpler predictions
  - Larger chunks = less frequent invocations but more complex predictions
  - Typical range: 50-200

- `--n-obs-steps` (int, default: 1)
  - Number of observation steps to include in the policy input
  - Currently, only `n_obs_steps=1` is fully supported
  - Used for temporal context (future support)

- `--vision-backbone` (str, choices: resnet18, resnet34, resnet50, default: resnet18)
  - CNN architecture for extracting visual features
  - `resnet18`: Fast, good for limited VRAM (7M params)
  - `resnet34`: Balanced (21M params)
  - `resnet50`: High capacity but slower (25M params)

- `--use-vae` / `--no-vae` (bool, default: True)
  - VAE learns a compact action representation
  - Helps with mode collapse and exploration
  - Can be disabled for simpler models or faster training

### Training Parameters (shared across all models)

- `--dataset-root` (path, required)
  - Path to exported LeRobotDataset v3 directory
  - Must contain `meta/info.json`

- `--batch-size` (int, default: 8)
  - Batch size for training
  - Reduce if getting CUDA OOM errors
  - Typical range: 4-32

- `--steps` (int, default: 20,000)
  - Number of training steps
  - More steps generally means better performance
  - Start with 10,000-50,000 for quick experiments

- `--device` (str, default: cuda)
  - Training device: `cuda`, `cpu`, or `mps` (Mac)

- `--num-workers` (int, default: 4)
  - Number of data loading workers

- `--log-freq` (int, default: 50)
  - Log metrics every N steps

- `--save-freq` (int, default: 1000)
  - Save checkpoint every N steps

- `--eval-freq` (int, default: 0)
  - Environment evaluation frequency (0 = disabled)

- `--seed` (int, default: 1000)
  - Random seed for reproducibility

- `--use-amp` (flag)
  - Enable Automatic Mixed Precision (faster, uses less VRAM)

- `--resume` (flag)
  - Resume training from last checkpoint in output directory

- `--job-name` (str)
  - Custom name for the training run
  - Default: `{dataset_name}_act`

- `--output-dir` (path)
  - Where to save checkpoints and logs
  - Default: `outputs/{dataset_name}_act/`

- `--push-to-hub` (flag)
  - Push final model to Hugging Face Hub

- `--policy-repo-id` (str)
  - HuggingFace repo ID for `--push-to-hub`
  - Format: `username/model-name`

## Common Training Scenarios

### Scenario 1: Quick Experiment (Fast Iteration)

```bash
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --chunk-size 50 \
    --batch-size 16 \
    --steps 5000 \
    --log-freq 100 \
    --save-freq 500 \
    --use-amp
```

**Expected results:** Trains in ~5-10 minutes on GPU

### Scenario 2: Production Training (Best Accuracy)

```bash
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --vision-backbone resnet50 \
    --chunk-size 100 \
    --batch-size 8 \
    --steps 50000 \
    --log-freq 50 \
    --save-freq 1000 \
    --seed 42
```

**Expected results:** Better generalization, takes longer (~1-2 hours on V100)

### Scenario 3: Multi-Dataset Training

```bash
# Train on multiple datasets by converting and combining them first
# Then train on the combined dataset
uv --project ../lerobot run python main.py convert 002 --primary-camera ee_zed_m
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/combined_001_002 \
    --steps 30000
```

### Scenario 4: Finetuning Pretrained Model

```bash
uv --project ../lerobot run python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/new_task \
    --policy-path path/to/previous/checkpoint \
    --batch-size 16 \
    --steps 5000 \
    --learning-rate 1e-5  # Smaller learning rate for finetuning
```

## Monitoring Training

During training, you'll see output like:

```
2024-12-13 14:32:10 | INFO | Policy:    ACT
2024-12-13 14:32:10 | INFO | Device:    cuda
2024-12-13 14:32:10 | INFO | Dataset:   lerobot_datasets/001  (10 ep, 5000 frames)
2024-12-13 14:32:10 | INFO | Steps:     20000  |  batch: 8  |  workers: 4
2024-12-13 14:32:10 | INFO | Architecture: chunk_size=100, n_obs_steps=1, use_vae=True, backbone=resnet18
...
step: 0 loss: 1.234
step: 50 loss: 0.856
step: 100 loss: 0.742
...
```

Check the output directory for:
- `train_config.json`: Complete training configuration
- `checkpoint_*/`: Model checkpoints at intervals
- `latest_checkpoint/`: Most recent checkpoint
- Logs with training curves (if using wandb integration)

## Model Structure

The trained ACT model consists of:

1. **Vision Backbone**: ResNet encoder for image features
2. **Transformer Encoder**: Processes visual features
3. **Transformer Decoder**: Generates action predictions
4. **VAE (optional)**: Learns latent action distributions
5. **Action Head**: Maps to robot action space

Files saved in output directory:
- `latest_checkpoint/policy_state_dict.pt`: Model weights
- `latest_checkpoint/config.json`: Model configuration
- `preprocessor/`: Input normalization
- `postprocessor/`: Output denormalization

## Using the Trained Model

After training, use the model for inference:

```python
import torch
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors

# Load model and processors
model = ACTPolicy.from_pretrained("path/to/output_dir")
dataset_metadata = ...  # Load metadata for normalization
preprocess, postprocess = make_pre_post_processors(
    model.config, 
    dataset_stats=dataset_metadata.stats
)

# Prepare observation
obs = ...  # Get observation from environment/robot
obs = preprocess(obs)

# Get action prediction
action = model.select_action(obs)
action = postprocess(action)

# Execute action (handle chunk properly)
for step_in_chunk in range(model.config.chunk_size):
    robot.send_action(action[step_in_chunk])
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size` (try 4 or 2)
- Use smaller backbone: `--vision-backbone resnet18`
- Reduce chunk size: `--chunk-size 50`
- Enable AMP: `--use-amp`

### Poor Training Loss

- Check data quality: Run `main.py label` to inspect episodes
- Increase training steps: `--steps 50000`
- Try larger backbone: `--vision-backbone resnet50`
- Disable VAE for simpler model: `--no-vae`

### Slow Training

- Increase batch size (if VRAM allows): `--batch-size 32`
- Reduce number of workers: `--num-workers 2`
- Use faster backbone: `--vision-backbone resnet18`
- Enable AMP: `--use-amp`

### Model Not Generalizing

- Ensure dataset is diverse (multiple episodes, varied tasks)
- Use augmentation (built into LeRobot)
- Try finetuning from pretrained model
- Increase model capacity: `--vision-backbone resnet50`
- Train longer: `--steps 50000`

## References

- **ACT Paper**: "Learning from Demonstrations for Autonomous Manipulation" (Zhao et al., 2023)
- **LeRobot Docs**: https://github.com/huggingface/lerobot
- **ACT in LeRobot**: `lerobot/src/lerobot/policies/act/`

## Example Complete Workflow

```bash
# 1. Clean raw dataset
python main.py clean 001 --force

# 2. Label episodes (optional, for verification)
python main.py label --port 5000

# 3. Convert to LeRobotDataset v3 format
python main.py convert 001 --primary-camera ee_zed_m

# 4. Train ACT model
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --batch-size 8 \
    --steps 20000 \
    --log-freq 100 \
    --save-freq 1000

# 5. Optionally push to Hub
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --steps 20000 \
    --push-to-hub \
    --policy-repo-id your-username/your-model-name
```

## Tips for Best Results

1. **Data Quality**: Clean, consistent demonstrations lead to better models
2. **Chunk Size**: Match task horizon (e.g., 100 for ~3-4 second actions at 30 FPS)
3. **Batch Size**: Larger batches (if VRAM allows) improve stability
4. **Learning Rate**: ACT defaults to 1e-5, which works well for most cases
5. **Evaluation**: Use `--eval-freq` to monitor performance on simulator/real robot
6. **Diverse Data**: Train on varied demonstrations for better generalization

Good luck with ACT training! 🚀
