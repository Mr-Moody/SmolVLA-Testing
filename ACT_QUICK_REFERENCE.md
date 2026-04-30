# ACT Training Quick Reference

## Installation Check

```bash
# Verify lerobot is available
python -c "import lerobot; print(lerobot.__version__)"
```

## 1. Dataset Preparation

```bash
# Clean raw dataset (remove stationary frames)
python main.py clean 001 --force

# Convert to LeRobotDataset v3 format
python main.py convert 001 --primary-camera ee_zed_m

# Result: lerobot_datasets/001/
```

## 2. Training with main.py CLI

### Minimal Training
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --steps 20000
```

### Quick Experiment
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --batch-size 16 \
    --chunk-size 50 \
    --steps 10000 \
    --use-amp
```

### Production Training
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --batch-size 8 \
    --vision-backbone resnet50 \
    --steps 50000 \
    --seed 42
```

### Custom Architecture
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --chunk-size 75 \
    --vision-backbone resnet34 \
    --no-vae \
    --steps 20000
```

## 3. Training with Standalone Script

### Basic
```bash
python train_act_standalone.py \
    --dataset-root lerobot_datasets/001 \
    --steps 20000
```

### With All Options
```bash
python train_act_standalone.py \
    --dataset-root lerobot_datasets/001 \
    --output-dir outputs/my_exp \
    --batch-size 8 \
    --steps 20000 \
    --learning-rate 5e-5 \
    --chunk-size 100 \
    --vision-backbone resnet50 \
    --seed 42 \
    --device cuda \
    --use-amp
```

## 4. Monitoring Training

```bash
# Watch training output
tail -f outputs/001_act/logs.txt

# Check checkpoint directory
ls -lh outputs/001_act/
```

## 5. Resuming Training

### Via main.py
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --output-dir outputs/001_act \
    --resume \
    --steps 40000
```

### Via standalone script
```bash
python train_act_standalone.py \
    --dataset-root lerobot_datasets/001 \
    --output-dir outputs/001_act \
    --resume \
    --steps 40000
```

## 6. Using Pretrained Models

### From HuggingFace Hub
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --policy-path lerobot/act_base \
    --steps 10000
```

### From Local Checkpoint
```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --policy-path outputs/previous_exp/latest_checkpoint \
    --steps 10000
```

## 7. Comparing Different Configurations

```bash
# Training 1: Small, fast
python main.py train --model-type act --dataset-root lerobot_datasets/001 \
    --chunk-size 50 --vision-backbone resnet18 --steps 10000 \
    --output-dir outputs/exp_small

# Training 2: Large, accurate
python main.py train --model-type act --dataset-root lerobot_datasets/001 \
    --chunk-size 100 --vision-backbone resnet50 --steps 50000 \
    --output-dir outputs/exp_large

# Training 3: Balanced
python main.py train --model-type act --dataset-root lerobot_datasets/001 \
    --chunk-size 75 --vision-backbone resnet34 --steps 30000 \
    --output-dir outputs/exp_balanced
```

## 8. Common Parameter Combinations

### For Real-Time Robotics (low latency)
```bash
--chunk-size 20
--vision-backbone resnet18
--batch-size 16
--steps 10000
```

### For Offline Training (best accuracy)
```bash
--chunk-size 100
--vision-backbone resnet50
--batch-size 4
--steps 50000
```

### For Limited VRAM
```bash
--batch-size 4
--vision-backbone resnet18
--chunk-size 50
--use-amp
--num-workers 2
```

### For Experimentation
```bash
--batch-size 32
--chunk-size 50
--steps 5000
--use-amp
--seed 1234
```

## 9. Inference Example

```python
import torch
from pathlib import Path
from lerobot.policies.act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.datasets import LeRobotDatasetMetadata

# Load model
model_path = Path("outputs/001_act/latest_checkpoint")
policy = ACTPolicy.from_pretrained(model_path)
policy.eval()

# Load preprocessor/postprocessor
preprocess = policy.config.preprocessor
postprocess = policy.config.postprocessor

# Get observation from robot
obs = ...  # dict with observation

# Process and predict
obs_proc = preprocess(obs)
with torch.no_grad():
    actions = policy.select_action(obs_proc)
actions = postprocess(actions)

# Execute actions (handle chunk size)
for action in actions[:policy.config.n_action_steps]:
    robot.send_action(action)
```

## 10. Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| CUDA OOM | `--batch-size 4` + `--use-amp` |
| Slow training | `--vision-backbone resnet18` + `--use-amp` |
| Poor accuracy | `--steps 50000` + `--learning-rate 5e-5` |
| Mode collapse | `--use-vae` (default) + more diverse data |
| Overfitting | Reduce `--steps` or add data |

## 11. File Structure After Training

```
outputs/001_act/
├── train_config.json           # Full training config
├── logs/
│   ├── training_curves.json    # Metrics over time
│   └── config.txt              # Human-readable config
├── checkpoint_10000/           # Intermediate checkpoint
│   ├── config.json
│   └── policy_state_dict.pt
├── checkpoint_20000/
├── latest_checkpoint/          # Best checkpoint
│   ├── config.json
│   ├── policy_state_dict.pt
│   ├── preprocessor/
│   └── postprocessor/
└── failed_checkpoints/         # Failed attempts (if any)
```

## 12. Environment Variables

```bash
# Use local models only (no Hub downloads)
export HF_HUB_OFFLINE=1

# Set number of parallel jobs
export OMP_NUM_THREADS=8

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1
```

## 13. Comparing with Other Models

```bash
# SmolVLA (default)
python main.py train --dataset-root lerobot_datasets/001 --steps 20000

# Pi0
python main.py train --model-type pi0 --dataset-root lerobot_datasets/001 --steps 20000

# ACT (new!)
python main.py train --model-type act --dataset-root lerobot_datasets/001 --steps 20000
```

## 14. Pushing to HuggingFace Hub

```bash
python main.py train \
    --model-type act \
    --dataset-root lerobot_datasets/001 \
    --steps 20000 \
    --push-to-hub \
    --policy-repo-id your-username/your-model-name
```

## 15. Environment Setup

```bash
# From main workspace
cd SmolVLA-Testing

# Using uv (recommended)
uv --project ../lerobot run python train_act_standalone.py ...

# Or directly with pip
cd ../lerobot
pip install -e .
cd ../SmolVLA-Testing
python train_act_standalone.py ...
```

---

**Need help?** See `ACT_TRAINING_GUIDE.md` for detailed documentation.
