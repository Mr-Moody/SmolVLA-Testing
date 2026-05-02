# Overnight Annotation Pipeline - Setup & Usage Guide

This guide explains how to set up and run the automatic overnight pipeline for cleaning, annotating, and converting robot datasets to LeRobot format.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Running the Pipeline](#running-the-pipeline)
5. [Monitoring Progress](#monitoring-progress)
6. [Configuration Examples](#configuration-examples)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Minimal Setup (5 minutes)

```bash
# 1. Configure your datasets
edit scripts/06_overnight_params.sh  # Change DATASET_NAMES

# 2. Start the pipeline
./scripts/06_run_overnight.sh

# 3. Monitor (in another terminal)
./scripts/06_monitor_overnight.sh overnight_output/
```

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A100, H100)
  - For Qwen3-VL-30B model annotation
  - 16GB minimum with quantized model (AWQ)
  
- **CPU**: 8+ cores recommended
  
- **RAM**: 32GB+ system RAM
  
- **Storage**: 500GB+ for datasets and outputs
  - Rule of thumb: ~50GB per dataset (raw + cleaned + lerobot output)

### Software

- **Python 3.8+** (tested on 3.10, 3.11)
- **CUDA 11.8+** (if using NVIDIA GPU)
- **cuDNN 8+** (if using NVIDIA GPU)

### Python Packages

Install dependencies in the lerobot environment:

```bash
cd ../lerobot
source .venv/bin/activate  # or conda activate lerobot

# Install phase-specific requirements
pip install -r ../SmolVLA-Testing/requirements_phase.txt

# Verify Qwen3-VL can load
python3 -c "from vllm import LLM; from qwen_vl_utils import *; print('✓ Qwen dependencies OK')"
```

## Setup

### 1. Prepare Raw Datasets

Organize your raw datasets in `raw_datasets/` directory:

```
raw_datasets/
├── 001/
│   ├── session_metadata.json
│   ├── robot.jsonl              # Robot state trajectory
│   ├── episode_events.jsonl     # Episode boundaries
│   ├── text.jsonl               # (optional) Language annotations
│   └── cameras/
│       ├── ee_zed_m/
│       │   ├── frames.jsonl     # Frame timestamps
│       │   └── rgb.mp4          # H.264 video
│       └── gripper/
│           ├── frames.jsonl
│           └── rgb.mp4
├── 002/
│   └── ...
└── ...
```

**Validate your data:**

```bash
python3 << 'EOF'
import json
from pathlib import Path

dataset_root = Path("raw_datasets/001")
assert (dataset_root / "robot.jsonl").exists(), "Missing robot.jsonl"
assert (dataset_root / "episode_events.jsonl").exists(), "Missing episode_events.jsonl"

# Check format
with open(dataset_root / "robot.jsonl") as f:
    first_line = json.loads(f.readline())
    assert "robot_state" in first_line, "Invalid robot.jsonl format"
    print(f"✓ Dataset {dataset_root.name} looks valid")
    print(f"  Keys: {list(first_line.keys())}")
EOF
```

### 2. Configure Parameters

Edit `scripts/06_overnight_params.sh`:

```bash
# Essential settings
DATASET_NAMES="001 002 003"          # Which datasets to process
ENABLE_ANNOTATION=true               # Use Qwen for labeling
NUM_GPUS=1                          # GPU count for Qwen
```

Or use one of the example configs:

```bash
# Small batch with annotation
./scripts/06_run_overnight.sh scripts/06_example_small_batch.sh

# Large batch without annotation (faster)
./scripts/06_run_overnight.sh scripts/06_example_large_no_annotation.sh

# Multi-GPU with quantized model
./scripts/06_run_overnight.sh scripts/06_example_multi_gpu_quantized.sh
```

### 3. Test Environment

Before running overnight, verify everything works:

```bash
# Test GPU availability
nvidia-smi

# Test Python environment
python3 -c "import torch; print(f'PyTorch GPU: {torch.cuda.is_available()}')"

# Test a single dataset (quick 30-second test)
python3 run_overnight_pipeline.py \
    --dataset-names 001 \
    --max-episodes 2 \
    --output-dir test_overnight_output
```

## Running the Pipeline

### Option 1: Using Shell Script (Recommended)

**Linux/macOS:**
```bash
chmod +x scripts/06_run_overnight.sh
./scripts/06_run_overnight.sh scripts/06_overnight_params.sh
```

**Windows:**
```cmd
scripts\06_run_overnight.bat
```

### Option 2: Direct Python

```bash
python3 run_overnight_pipeline.py \
    --dataset-names 001 002 003 \
    --qwen-model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --num-gpus 1 \
    --batch-size-annotation 4
```

### Option 3: Background Process (Overnight)

**Linux/macOS:**
```bash
# Run in background and save output
nohup ./scripts/06_run_overnight.sh scripts/06_overnight_params.sh > overnight.log 2>&1 &
echo $! > overnight.pid
```

**Windows (PowerShell):**
```powershell
$job = Start-Job -ScriptBlock {
    cd C:\path\to\SmolVLA-Testing
    & .\scripts\06_run_overnight.bat
}
$job.Id  # Save this ID
```

## Monitoring Progress

### Real-Time Monitor

In a separate terminal:

```bash
./scripts/06_monitor_overnight.sh overnight_output/ --interval 30
```

Shows:
- Current dataset being processed
- Completion status for each stage
- Recent log entries
- Last checkpoint time

### Check Logs Directly

```bash
# Latest log
tail -100 overnight_output/logs/overnight_run_*.log

# Grep for errors
grep -i error overnight_output/logs/overnight_run_*.log

# Follow live updates
tail -f overnight_output/logs/overnight_run_*.log | grep -E "(episode|COMPLETED|error)"
```

### Checkpoint Status

```bash
# View current progress
python3 << 'EOF'
import json
from pathlib import Path

cp = Path("overnight_output/checkpoint.json")
if cp.exists():
    data = json.loads(cp.read_text())
    for ds in data["datasets"]:
        status = "✓" if ds["status"] == "done" else "✗" if ds["status"] == "failed" else "▸"
        print(f"{status} {ds['name']:10s} {ds['status']:20s}")
EOF
```

## Configuration Examples

### Example 1: Quick Test (1 dataset, no annotation)

```bash
#!/bin/bash
DATASET_NAMES="001"
ENABLE_ANNOTATION=false
MAX_EPISODES_PER_DATASET=5
```

Expected runtime: **2-5 minutes**

### Example 2: Standard Overnight (5 datasets with Qwen)

```bash
#!/bin/bash
DATASET_NAMES="001 002 003 004 005"
ENABLE_ANNOTATION=true
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4
```

Expected runtime: **1-2 hours** (depending on dataset sizes)

### Example 3: Production (20 datasets, 2 GPUs, quantized)

```bash
#!/bin/bash
DATASET_NAMES="001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020"
ENABLE_ANNOTATION=true
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"
NUM_GPUS=2
BATCH_SIZE_ANNOTATION=8
CONTINUE_ON_ERROR=true
ENABLE_CHECKPOINT=true
```

Expected runtime: **4-8 hours**

## Troubleshooting

### Issue: "vllm not found"

**Solution:**
```bash
# Ensure you're in the right environment
source ../lerobot/.venv/bin/activate

# Install vLLM and Qwen dependencies
pip install vllm>=0.7 qwen-vl-utils

# Verify
python3 -c "from vllm import LLM; print('✓ vLLM installed')"
```

### Issue: GPU Out of Memory

**Solutions:**

1. Reduce batch size:
   ```bash
   BATCH_SIZE_ANNOTATION=2  # Instead of 4
   ```

2. Use quantized model:
   ```bash
   QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"
   ```

3. Process fewer datasets at once:
   ```bash
   DATASET_NAMES="001 002"  # Instead of many
   ```

4. Skip annotation:
   ```bash
   ENABLE_ANNOTATION=false
   ```

### Issue: Slow Annotation (10+ min per dataset)

**Diagnose:**
```bash
# Check GPU utilization
nvidia-smi  # Should be >90% during annotation phase

# Check batch size usage
grep "Batch" overnight_output/logs/overnight_run_*.log
```

**Solutions:**

1. Increase batch size:
   ```bash
   BATCH_SIZE_ANNOTATION=8
   ```

2. Use multiple GPUs:
   ```bash
   NUM_GPUS=2  # If available
   ```

3. Use quantized model:
   ```bash
   QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"
   ```

### Issue: Checkpoint Corrupted

**Solution:**
```bash
# Delete checkpoint and restart
rm overnight_output/checkpoint.json

# Re-run pipeline (will start from beginning)
./scripts/06_run_overnight.sh scripts/06_overnight_params.sh
```

### Issue: Specific Dataset Failed

**Solution:**

1. Check error in log:
   ```bash
   grep "error\|failed" overnight_output/logs/*.log | grep dataset_name
   ```

2. Try reprocessing just that dataset:
   ```bash
   python3 run_overnight_pipeline.py \
       --dataset-names 001 \
       --skip-annotation  # Try without annotation first
   ```

3. Validate raw data:
   ```bash
   # Check if files exist and are valid
   python3 << 'EOF'
   import json
   from pathlib import Path
   
   dataset_root = Path("raw_datasets/001")
   robot_lines = (dataset_root / "robot.jsonl").read_text().splitlines()
   print(f"Robot entries: {len(robot_lines)}")
   
   # Check first entry is valid
   first = json.loads(robot_lines[0])
   print(f"First entry keys: {list(first.keys())}")
   EOF
   ```

### Issue: Conversion Fails After Annotation

**Solution:**

Usually indicates misalignment between cameras and robot timestamps.

```bash
# Check camera frame count
python3 << 'EOF'
import json
from pathlib import Path

dataset_root = Path("cleaned_datasets/001")
for cam_dir in (dataset_root / "cameras").iterdir():
    frames_file = cam_dir / "frames.jsonl"
    if frames_file.exists():
        count = len(frames_file.read_text().splitlines())
        print(f"{cam_dir.name}: {count} frames")
EOF

# Re-run with adjusted camera tolerance
# in scripts/06_overnight_params.sh:
CAMERA_TOLERANCE_MS=300  # Increase from default 150
```

## After Successful Overnight Run

### Validate Output

```bash
# Check all datasets converted successfully
ls -lh lerobot_datasets/

# Count episodes in each
python3 << 'EOF'
import json
from pathlib import Path

for ds_dir in sorted(Path("lerobot_datasets").iterdir()):
    meta_file = ds_dir / "meta" / "info.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        episodes = meta.get("total_episodes", "unknown")
        print(f"{ds_dir.name}: {episodes} episodes")
EOF
```

### Next Steps

1. **Merge datasets (optional):**
   ```bash
   python3 src/merge_datasets.py \
       --dataset-roots lerobot_datasets/001 lerobot_datasets/002 lerobot_datasets/003 \
       --output lerobot_datasets/combined_001_002_003
   ```

2. **Train a policy:**
   ```bash
   python3 main.py train \
       --dataset-root lerobot_datasets/001 \
       --model-type act \
       --steps 100000
   ```

3. **Inspect dataset:**
   ```bash
   python3 << 'EOF'
   from lerobot.datasets import LeRobotDataset
   ds = LeRobotDataset("lerobot_datasets/001")
   print(f"Episodes: {len(ds)}")
   sample = ds[0]
   print(f"Sample keys: {list(sample.keys())}")
   EOF
   ```

## Support

- **Error logs**: Check `overnight_output/logs/`
- **Latest run**: `overnight_output/checkpoint.json`
- **Monitor script**: Run `scripts/06_monitor_overnight.sh` in separate terminal
- **Re-run with debug**: Add `--verbose` flag (if implementing)

## Next Iteration

After the overnight pipeline completes, you can:

1. **Review annotations** (if Qwen enabled):
   ```bash
   cat overnight_output/annotations_001/annotations.jsonl | head -5
   ```

2. **Generate reports**:
   ```bash
   python3 << 'EOF'
   import json
   from pathlib import Path
   
   report = Path("overnight_output/summary_report.json")
   if report.exists():
       data = json.loads(report.read_text())
       print(data)
   EOF
   ```

3. **Schedule next run**:
   ```bash
   # Edit schedule if using cron
   crontab -e
   # Add line: 0 22 * * * /path/to/scripts/06_run_overnight.sh
   ```

---

**Estimated Time to First Successful Run**: 30-60 minutes (including setup and first test)

**Estimated Runtime**: 1-8 hours depending on dataset count and configuration
