# HuggingFace Dataset Workflow

Train directly from datasets hosted on HuggingFace Hub — no manual
clean/convert/annotate/merge steps required.

## Available Datasets

**MSD Connector Plugging (200-series):**

| Dataset | Repo ID | Episodes |
|---------|---------|----------|
| Individual 200-209 | `NexusDwin/msd-connector-200` … `209` | 14-27 each |
| Merged 200-204 | `NexusDwin/msd-connector-200-204` | 121 |
| Merged 205-209 | `NexusDwin/msd-connector-205-209` | 115 |
| Merged 200-209 | `NexusDwin/msd-connector-200-209` | 236 |

**Soup Can Pick-and-Place (100-series):**

| Dataset | Repo ID | Episodes |
|---------|---------|----------|
| Individual 100-103 | `NexusDwin/soup-pick-place-100` … `103` | 24-26 each |
| Merged 100-103 | `NexusDwin/soup-pick-place-100-103` | 101 |

All datasets are cleaned, converted (LeRobotDataset v3), and annotated
with 5 task prompts.

## Quick Start

### Local GPU

1. Edit `scripts_local/hf_params.sh` (or create `hf_params.local.sh`):
   ```bash
   HF_REPO_IDS=("NexusDwin/msd-connector-200-209")
   MODEL_TYPE="smolvla"
   RUN_NAME="hf_msd_200_209"
   STEPS=25000
   ```

2. Run:
   ```bash
   bash scripts_local/03_hf_train.sh
   ```

   This will:
   - Download the dataset to `.hf_cache/` (cached for future runs)
   - Validate it (v3.0, annotated, has data + videos)
   - Apply lerobot patches
   - Launch training

3. Resume a run:
   ```bash
   bash scripts_local/03_hf_train.sh --resume
   ```

4. Force re-download:
   ```bash
   bash scripts_local/03_hf_train.sh --force-download
   ```

### Remote GPU (UCL nodes)

1. Sync code to GPU node:
   ```bash
   bash scripts/01_sync_to_gpu.sh
   ```

2. Set up environment (if first time or scratch wiped):
   ```bash
   bash scripts/03_setup_gpu.sh
   ```

3. Add HF config to `scripts/00_run_params.local.sh`:
   ```bash
   HF_REPO_IDS=("NexusDwin/msd-connector-200-209")
   RUN_NAME="hf_msd_200_209"
   STEPS=25000
   ```

4. SSH to GPU node and run:
   ```bash
   ssh -J $JUMP_HOST $GPU_NODE
   cd ~/smolvla_project/SmolVLA-Testing
   bash scripts/06_hf_prep_and_train.sh
   ```

   The script downloads with HF Hub access, then trains with
   `HF_HUB_OFFLINE=1` to prevent lookups during training.

5. Extract checkpoints back to local:
   ```bash
   bash scripts/05_extract_from_scratch.sh
   ```

### Multi-Dataset Training

Train on multiple HF datasets by listing them:

```bash
# In hf_params.sh or hf_params.local.sh:
HF_REPO_IDS=(
    "NexusDwin/msd-connector-200"
    "NexusDwin/msd-connector-201"
    "NexusDwin/msd-connector-202"
)
```

The pipeline will download each, validate them individually, merge them
into a single dataset, and train on the merged result.

## Python API

### Validate a local dataset

```bash
uv --project ../lerobot run python src/hf/dataset_hub.py validate \
    lerobot_datasets/merged_msd_200_209
```

### Download + validate from HF

```bash
uv --project ../lerobot run python src/hf/dataset_hub.py prepare \
    NexusDwin/msd-connector-200-209 \
    --cache-root .hf_cache
```

### From Python

```python
from src.hf.dataset_hub import prepare_datasets, validate_dataset

# Single dataset
path = prepare_datasets(["NexusDwin/msd-connector-200-209"])

# Multiple → merged
path = prepare_datasets(
    ["NexusDwin/msd-connector-200", "NexusDwin/msd-connector-201"],
    merge_output=Path("lerobot_datasets/merged_custom"),
)

# Validate only
result = validate_dataset(Path("lerobot_datasets/200"))
if not result.valid:
    print(result.errors)
```

## Validation Checks

The pipeline validates each dataset before training:

| Check | Error if... |
|-------|-------------|
| `meta/info.json` exists | Missing → not converted |
| `codebase_version == "v3.0"` | Wrong version → incompatible format |
| `total_episodes > 0` | Zero → empty dataset |
| `total_frames > 0` | Zero → no data |
| `total_tasks > 0` | Zero → not annotated |
| `meta/tasks.parquet` exists | Missing → no task prompts |
| `data/` has `.parquet` files | Missing → not converted |
| `videos/` has `.mp4` files | Missing → not converted |
| `meta/stats.json` exists | Missing → warning only (computed on first train) |

## Cache Behaviour

Downloaded datasets are cached in `.hf_cache/<owner>/<repo>/` (local) or
`/scratch0/$USER/hf_datasets/<owner>/<repo>/` (remote). A cache hit is
detected when `meta/info.json` exists in the cached directory.

Use `--force-download` to re-download.

## Comparison: Old vs New Workflow

**Old workflow (manual data pipeline):**
```
raw_datasets/ → main.py clean → cleaned_datasets/
              → create_labels_*.py → annotations.jsonl
              → main.py convert → lerobot_datasets/
              → merge_datasets.py → merged dataset
              → rsync to GPU → main.py train
```

**New workflow (HuggingFace):**
```
HF Hub → download → validate → merge (if multi) → main.py train
```

The old workflow is still fully supported via `scripts_local/02_run.sh`
and `scripts/04_start_training.sh`. Use the HF workflow when datasets
are already processed and uploaded to HuggingFace.
