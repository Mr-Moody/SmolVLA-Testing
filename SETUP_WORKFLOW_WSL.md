# SmolVLA Qwen Setup Workflow for WSL

All scripts are bash-compatible and work from WSL on Windows.

## Quick Start

```bash
# 1. From local (WSL): Sync code and datasets to remote
bash scripts/01_sync_to_gpu.sh

# 2. From local (WSL): Run preflight checks
bash scripts/02_preflight_gpu.sh

# 3. From local (WSL): Setup remote environment (creates scratch venv)
bash scripts/03_setup_gpu.sh

# 4. From local (WSL): Install Qwen dependencies into scratch venv
INSTALL_DEPS=true bash scripts/03_setup_gpu.sh

# 5. From local (WSL): SSH to remote and test
ssh -i ~/.ssh/ucl_key -J xparker@knuckles.cs.ucl.ac.uk xparker@bluestreak.cs.ucl.ac.uk

# 6. On remote: Run test (this verifies everything works)
bash ~/smolvla_project/SmolVLA-Testing/scripts/06_test_qwen.sh
```

## Script Details

### 00_run_params.sh
Central configuration file. Defines:
- SSH topology, keys, and remote paths
- Dataset names and model hyperparameters
- Remote scratch base: `/scratch0/xparker`

**Run location**: Local (sourced by all other scripts)

### 01_sync_to_gpu.sh
Syncs local code and datasets to remote home and scratch directories.

**Run location**: Local (WSL)

**What it does**:
- Detects remote home directory
- Rsync code to home (SmolVLA-Testing, lerobot source)
- Rsync datasets to scratch (`/scratch0/xparker/lerobot_datasets`)

### 02_preflight_gpu.sh
Verifies local prerequisites and remote connectivity.

**Run location**: Local (WSL)

**What it does**:
- Checks local directories exist
- Verifies SSH connectivity to GPU node via jump host
- Verifies remote code layout and datasets are reachable

### 03_setup_gpu.sh
Sets up the remote environment and installs dependencies.

**Run location**: Local (WSL) — it SSHes to remote and runs setup

**What it does**:
- Runs `setup_scratch.sh` remotely (creates `/scratch0/xparker/smolvla_venv`)
- If `INSTALL_DEPS=true`: Installs vllm and qwen-vl-utils into the scratch venv using pip with `--no-cache-dir`
- All writes stay in `/scratch0` (scratch disk, no quota limit)

**Environment variables created** (on remote):
```
/scratch0/xparker/
├── smolvla_venv/           <- Python venv with all deps
├── lerobot_datasets/       <- Dataset directory (from 01_sync)
└── .cache/
    ├── pip/
    ├── huggingface/
    └── torch/
```

### 06_test_qwen.sh
Tests that Qwen3-VL can load from the scratch venv.

**Run location**: Remote (SSH into server, then run)

**What it does**:
- Verifies scratch venv exists at `/scratch0/xparker/smolvla_venv`
- Checks Python, vLLM, qwen-vl-utils are installed
- Checks GPU availability
- Attempts to load Qwen3-VL model (quick verification)

## Venv Paths

All paths are in scratch (no home quota used):
- Scratch venv: `/scratch0/xparker/smolvla_venv`
- Python executable: `/scratch0/xparker/smolvla_venv/bin/python3`
- Pip executable: `/scratch0/xparker/smolvla_venv/bin/pip`

## Environment Variables (set on remote by setup_scratch.sh)

Cache directories are all on scratch (via setup_scratch.sh):
```bash
export PIP_CACHE_DIR="/scratch0/xparker/.cache/pip"
export HF_HOME="/scratch0/xparker/.cache/huggingface"
export TORCH_HOME="/scratch0/xparker/.cache/torch"
export UV_CACHE_DIR="/scratch0/xparker/.cache/uv"
```

## Troubleshooting

### "Scratch venv not found"
- Run `bash scripts/03_setup_gpu.sh` first (creates the venv)
- Check that `/scratch0/xparker/smolvla_venv` exists on remote

### Pip installing to home instead of scratch
- Make sure you're inside the venv: `source /scratch0/xparker/smolvla_venv/bin/activate`
- Check `which python3` points to `/scratch0/xparker/smolvla_venv/bin/python3`
- Use `pip --no-cache-dir` to avoid cache issues

### Disk quota errors
- All critical packages should be in `/scratch0`, not home
- Check `/scratch0/xparker` free space: `df -h /scratch0`
- If home quota is hit, clean `~/.cache` and `~/.local` on remote

## Notes

- All scripts work from WSL (standard bash, no PowerShell)
- No SSH key needed after first setup (key-based auth installed)
- Datasets are large (100GB+); sync via `01_sync_to_gpu.sh` first
- First `06_test_qwen.sh` run may take 5-10 minutes (model download/cache)
