# Overnight Annotation & Conversion Pipeline

Automated overnight processing for cleaning, annotating with Qwen3-VL, and converting robot datasets to LeRobot format.

## Quick Start

### 1. Configure Parameters

Edit `scripts/06_overnight_params.sh` to set your dataset names and processing options:

```bash
# List datasets to process
DATASET_NAMES="001 002 003"

# Enable/disable Qwen annotation
ENABLE_ANNOTATION=true

# GPU configuration
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4
```

### 2. Prepare Your Data

Ensure raw datasets are in `raw_datasets/`:
```
raw_datasets/
  001/
    robot.jsonl
    episode_events.jsonl
    cameras/...
    ...
  002/
    ...
```

### 3. Start Overnight Job

**Linux/macOS:**
```bash
chmod +x scripts/06_run_overnight.sh
scripts/06_run_overnight.sh scripts/06_overnight_params.sh
```

**Windows:**
```cmd
scripts\06_run_overnight.bat
```

Or run directly with Python:
```bash
python3 run_overnight_pipeline.py \
    --dataset-names 001 002 003 \
    --qwen-model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --num-gpus 1
```

### 4. Monitor Progress (Optional)

In a separate terminal:
```bash
scripts/06_monitor_overnight.sh overnight_output/
```

## Pipeline Overview

The overnight pipeline automatically:

1. **Clean** each raw dataset
   - Filters to motion-only frames with camera coverage
   - Validates robot state
   - Generates episode boundaries

2. **Annotate** with Qwen3-VL (optional)
   - Processes videos with Qwen3-VL model
   - Extracts phase labels and task descriptions
   - Saves annotations as JSONL

3. **Convert** to LeRobot format
   - Encodes videos with H.264 codec
   - Aligns camera frames with robot state
   - Generates LeRobotDataset v3 output

4. **Resume** on failure
   - Checkpoints after each dataset
   - Automatically resumes from last checkpoint if re-run
   - Detailed error logging per dataset

## Output Structure

```
overnight_output/
  ├── checkpoint.json              # Resumable checkpoint
  ├── logs/
  │   └── overnight_run_*.log      # Detailed logs
  └── annotations_*/               # Per-dataset Qwen outputs

cleaned_datasets/
  ├── 001/
  │   ├── robot.jsonl              # Filtered robot data
  │   ├── episode_events.jsonl     # Episode boundaries
  │   ├── cameras/                 # Camera data
  │   └── annotations/             # (if annotation enabled)
  └── ...

lerobot_datasets/
  ├── 001/                          # LeRobot v3 format
  │   ├── meta/
  │   │   └── info.json
  │   ├── videos/
  │   │   └── episode_*.mp4
  │   └── chunks/
  │       └── episode_*.parquet
  └── ...
```

## Configuration Options

### Core Settings

| Option | Default | Description |
|--------|---------|-------------|
| `DATASET_NAMES` | `001 002 003` | Space-separated dataset IDs |
| `ENABLE_ANNOTATION` | `true` | Enable Qwen3-VL annotation |
| `NUM_GPUS` | `1` | GPUs for tensor parallelism |
| `BATCH_SIZE_ANNOTATION` | `4` | Annotation batch size |

### Processing Controls

| Option | Default | Description |
|--------|---------|-------------|
| `CAMERA_TOLERANCE_MS` | `150` | Max sync error (ms) |
| `JOINT_MOTION_THRESHOLD` | `5e-4` | Joint motion threshold (rad) |
| `GRIPPER_MOTION_THRESHOLD` | `2e-4` | Gripper motion threshold (m) |
| `PRIMARY_CAMERA` | `ee_zed_m` | Camera for video extraction |

### Runtime Controls

| Option | Default | Description |
|--------|---------|-------------|
| `CONTINUE_ON_ERROR` | `false` | Continue on dataset failure |
| `ENABLE_CHECKPOINT` | `true` | Resume from checkpoints |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Advanced Usage

### Skip Annotation (Just Clean + Convert)

```bash
python3 run_overnight_pipeline.py \
    --dataset-names 001 002 003 \
    --skip-annotation
```

### Limit Episodes per Dataset

```bash
python3 run_overnight_pipeline.py \
    --dataset-names 001 002 003 \
    --max-episodes 50
```

### Use Custom Qwen Model

```bash
python3 run_overnight_pipeline.py \
    --dataset-names 001 002 003 \
    --qwen-model "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"
```

### Resume from Checkpoint

Automatically happens if you re-run the same command. To reset:

```bash
rm overnight_output/checkpoint.json
python3 run_overnight_pipeline.py ...
```

## Dependencies

### System Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (for Qwen3-VL)
- **RAM**: 32GB+ total system RAM
- **Disk**: 500GB+ for datasets and outputs

### Python Packages

Install in your lerobot environment:

```bash
cd ../lerobot
source .venv/bin/activate  # or conda activate lerobot

pip install -r ../SmolVLA-Testing/requirements_phase.txt
```

Key dependencies:
- `vllm>=0.7` — for Qwen inference
- `qwen-vl-utils` — Qwen preprocessing
- `torch` — deep learning (from lerobot)
- `lerobot` — dataset handling (from lerobot)

## Troubleshooting

### "vllm is required" Error

Ensure vLLM is installed:
```bash
pip install vllm>=0.7 qwen-vl-utils
```

### GPU Memory Issues

Reduce batch size or enable quantization:
```bash
python3 run_overnight_pipeline.py \
    --qwen-model "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ" \
    --batch-size-annotation 2
```

### Slow Processing

Check GPU utilization during annotation:
```bash
nvidia-smi  # monitor in separate terminal
```

If underutilized, try increasing `BATCH_SIZE_ANNOTATION`.

### Checkpoint Issues

If checkpoint is corrupted:
```bash
rm overnight_output/checkpoint.json
python3 run_overnight_pipeline.py ...  # restarts from beginning
```

## Example: Running 10 Datasets Overnight

1. **Prepare config:**
   ```bash
   cat > scripts/06_overnight_params_prod.sh << 'EOF'
   DATASET_NAMES="001 002 003 004 005 006 007 008 009 010"
   ENABLE_ANNOTATION=true
   NUM_GPUS=1
   BATCH_SIZE_ANNOTATION=4
   CONTINUE_ON_ERROR=false
   ENABLE_CHECKPOINT=true
   EOF
   ```

2. **Start pipeline:**
   ```bash
   nohup ./scripts/06_run_overnight.sh scripts/06_overnight_params_prod.sh > nohup.out 2>&1 &
   ```

3. **Monitor in another terminal:**
   ```bash
   tail -f overnight_output/logs/overnight_run_*.log
   ```

4. **Check status next morning:**
   ```bash
   cat overnight_output/logs/overnight_run_*.log | grep -E "(COMPLETED|FAILED|error)"
   ls -lh lerobot_datasets/  # verify output
   ```

## Performance Estimates

On NVIDIA A100 GPU:

| Stage | ~30 episodes | ~100 episodes |
|-------|-------------|--------------|
| Clean | 5-10s | 20-30s |
| Annotate | 3-5 min | 10-20 min |
| Convert | 5-15 min | 20-60 min |
| **Total** | **~15-25 min** | **~35-90 min** |

## Next Steps After Processing

### Convert to LeBot Training Format

```bash
python3 -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('lerobot_datasets/001'); print(len(ds))"
```

### Train a Policy

```bash
python3 run_overnight_pipeline.py train \
    --dataset-root lerobot_datasets/001 \
    --model-type act \
    --steps 100000
```

### Merge Datasets

```bash
python3 src/merge_datasets.py \
    --dataset-roots lerobot_datasets/001 lerobot_datasets/002 lerobot_datasets/003 \
    --output lerobot_datasets/combined_001_002_003
```

## Support

For issues or improvements:
1. Check logs: `cat overnight_output/logs/overnight_run_*.log`
2. Run with `--skip-annotation` to isolate issues
3. Check GPU with `nvidia-smi` during annotation phase

## License

Same as parent SmolVLA-Testing project.
