# ⚡ Overnight Pipeline - Quick Start (5 minutes)

## TL;DR - Get Started Now

```bash
# 1. Install Qwen dependencies
source ../lerobot/.venv/bin/activate
pip install vllm>=0.7 qwen-vl-utils

# 2. Edit configuration
vi scripts/06_overnight_params.sh
# Change: DATASET_NAMES="001 002 003"

# 3. Start pipeline
./scripts/06_run_overnight.sh scripts/06_overnight_params.sh

# 4. Monitor (in another terminal)
./scripts/06_monitor_overnight.sh overnight_output/
```

## What Does It Do?

Automatically processes multiple datasets overnight:

```
raw_datasets/001 ──┐
raw_datasets/002 ──┼──> Clean ──> Annotate (Qwen) ──> Convert ──> lerobot_datasets/001
raw_datasets/003 ──┘                                              lerobot_datasets/002
                                                                  lerobot_datasets/003
```

Each step logs progress and saves checkpoints for resume-ability.

## System Requirements

- **GPU**: NVIDIA with 24GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+ system
- **Storage**: 500GB+ for datasets
- **Python**: 3.8+

## Configuration (60 seconds)

Edit `scripts/06_overnight_params.sh`:

```bash
# Line 9: Which datasets to process
DATASET_NAMES="001 002 003"

# Line 19: Use Qwen for annotation (set false for speed)
ENABLE_ANNOTATION=true

# Line 22-23: GPU settings
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4
```

That's it! Everything else has sensible defaults.

## Run It

**Linux/macOS:**
```bash
chmod +x scripts/06_run_overnight.sh
./scripts/06_run_overnight.sh
```

**Windows:**
```cmd
scripts\06_run_overnight.bat
```

**Or direct Python:**
```bash
python3 run_overnight_pipeline.py --dataset-names 001 002 003
```

## Monitor Progress

```bash
# In another terminal
./scripts/06_monitor_overnight.sh overnight_output/
```

Shows real-time status, latest log entries, and checkpoint info.

## Output Locations

| Output | Location |
|--------|----------|
| Cleaned datasets | `cleaned_datasets/001/`, `002/`, etc. |
| Annotated data | `overnight_output/annotations_001/`, etc. |
| LeRobot format | `lerobot_datasets/001/`, `002/`, etc. |
| Logs | `overnight_output/logs/overnight_run_*.log` |

## Example Configurations

Ready-to-use configs for different scenarios:

```bash
# Quick test (5 min)
./scripts/06_run_overnight.sh scripts/06_example_small_batch.sh

# Large batch without annotation (faster)
./scripts/06_run_overnight.sh scripts/06_example_large_no_annotation.sh

# Multi-GPU with quantized model
./scripts/06_run_overnight.sh scripts/06_example_multi_gpu_quantized.sh
```

## Common Issues

### "vllm not found"
```bash
pip install vllm>=0.7 qwen-vl-utils
```

### GPU out of memory
```bash
# In config: use quantized model
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"
# Or reduce batch size
BATCH_SIZE_ANNOTATION=2
```

### Slow processing
```bash
# Increase batch size (if GPU has room)
BATCH_SIZE_ANNOTATION=8
# Or use multiple GPUs
NUM_GPUS=2
```

### One dataset failed
Check logs then re-run — pipeline resumes from last checkpoint:
```bash
./scripts/06_run_overnight.sh scripts/06_overnight_params.sh
```

## Typical Times

On A100 GPU:
- 1 dataset (30 episodes): 5-8 minutes with Qwen
- 5 datasets: 30-45 minutes
- 10 datasets: 1-1.5 hours
- 20 datasets: 2-3 hours

Without Qwen annotation: 5-10x faster

## Environment Setup (First Time Only)

```bash
# Navigate to lerobot
cd ../lerobot

# Activate environment
source .venv/bin/activate  # Linux/Mac
# OR
conda activate lerobot  # If using conda

# Install extras
pip install vllm>=0.7 qwen-vl-utils

# Verify
python3 -c "from vllm import LLM; print('✓ Ready')"
```

## Advanced: Overnight Scheduling

**Linux/macOS (cron):**
```bash
# Run at 10 PM daily
echo "0 22 * * * /path/to/scripts/06_run_overnight.sh /path/to/scripts/06_overnight_params.sh" | crontab -
```

**Windows (Task Scheduler):**
1. Win+R → `taskschd.msc`
2. Create Basic Task
3. Name: "Overnight Dataset Pipeline"
4. Trigger: Daily, 10:00 PM
5. Action: `C:\path\to\scripts\06_run_overnight.bat`

## Data Format Expected

```
raw_datasets/001/
├── robot.jsonl              ✓ Required
├── episode_events.jsonl     ✓ Required
├── session_metadata.json    (optional)
├── text.jsonl              (optional)
└── cameras/
    └── ee_zed_m/
        ├── rgb.mp4         ✓ Required
        └── frames.jsonl    ✓ Required
```

## After Overnight Completes

### Use the datasets

```bash
# List all datasets created
ls -lh lerobot_datasets/

# Train a model on one
python3 main.py train \
    --dataset-root lerobot_datasets/001 \
    --model-type act \
    --steps 100000

# Merge multiple datasets
python3 src/merge_datasets.py \
    --dataset-roots lerobot_datasets/001 lerobot_datasets/002 \
    --output lerobot_datasets/merged
```

## Summary

| Feature | Details |
|---------|---------|
| **Setup time** | 5 minutes |
| **First run** | 10-60 minutes (depending on dataset size) |
| **Monitoring** | Real-time via separate terminal |
| **Resume** | Automatic from checkpoints |
| **Error handling** | Per-dataset logging, continues on error |
| **Output** | LeRobot v3 format, ready for training |

## Next: Read Full Docs

- **Setup guide**: `OVERNIGHT_SETUP.md` (detailed troubleshooting)
- **Overview**: `OVERNIGHT_PIPELINE.md` (all features)
- **Implementation**: `OVERNIGHT_IMPLEMENTATION_SUMMARY.md` (architecture)

## Quick Test (Recommended First!)

Test with 1 dataset before overnight run:

```bash
python3 run_overnight_pipeline.py \
    --dataset-names 001 \
    --skip-annotation \
    --max-episodes 5
```

Should take 1-2 minutes. If successful, you're ready for overnight!

---

**Ready to process datasets overnight? Start with the TL;DR section above!** 🚀
