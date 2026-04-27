# SmolVLA Minimal Workflow (scripts folder)

All workflow scripts now live in `scripts/` and are numbered in execution order.

## Script Order
1. `scripts/00_run_params.sh` (edit only)
2. `scripts/01_sync_to_gpu.sh` (run locally)
3. `scripts/02_preflight_gpu.sh` (run locally, recommended)
4. `scripts/03_setup_gpu.sh` (run on GPU node)
5. `scripts/04_start_training.sh` (run on GPU node)
6. `scripts/05_extract_from_scratch.sh` (run locally after training)

## 1) Edit Parameters
Edit this committed template:
- `scripts/00_run_params.sh`

Then create your personal override (recommended):
```bash
cp scripts/00_run_params.local.example.sh scripts/00_run_params.local.sh
```

Edit local values in:
- `scripts/00_run_params.local.sh` (gitignored)

Template key fields:
```bash
WORKFLOW_USER="your_ucl_username"
REMOTE_PROJECT_DIRNAME="smolvla_project"
RUN_NAME="001_002_003"
DATASET_NAMES=(001 002 003)
DATASET_ROOT="/scratch0/${WORKFLOW_USER}/lerobot_datasets"
SAVE_FREQ=1000
EXTRACT_FOLDER_NAME="${RUN_NAME}_smolvla_full"
```

Notes:
- `LOCAL_PROJECT_ROOT` auto-detects the repository root from script location.
- Override it in `00_run_params.sh` only if needed.

## 2) Local Steps
From repository root (or from inside `scripts/`):
```bash
bash scripts/01_sync_to_gpu.sh
bash scripts/02_preflight_gpu.sh
```

Helper command block (local, copy-paste):
```bash
cp scripts/00_run_params.local.example.sh scripts/00_run_params.local.sh
$EDITOR scripts/00_run_params.local.sh
bash scripts/01_sync_to_gpu.sh
bash scripts/02_preflight_gpu.sh
```

What `01_sync_to_gpu.sh` does:
- code -> remote home project (`~/smolvla_project/SmolVLA-Testing`)
- converted dataset -> remote scratch (`/scratch0/<user>/lerobot_datasets`)

## 3) Connect to GPU
You can use values from params directly:
```bash
source scripts/00_run_params.sh
ssh -l "$REMOTE_USER" -J "$REMOTE_USER@$JUMP_HOST" "$GPU_NODE"
```

## 4) Remote Steps (on GPU)
```bash
cd ~/smolvla_project/SmolVLA-Testing/scripts
bash ./03_setup_gpu.sh
bash ./04_start_training.sh
```

Helper command block (remote, copy-paste):
```bash
cd ~/smolvla_project/SmolVLA-Testing/scripts
bash ./03_setup_gpu.sh
bash ./04_start_training.sh
```

Resume mode:
```bash
bash ./04_start_training.sh --resume
```

Training log path:
```bash
/scratch0/$WORKFLOW_USER/smolvla_outputs/<RUN_NAME>.log
```

Monitor:
```bash
tail -f /scratch0/$WORKFLOW_USER/smolvla_outputs/<RUN_NAME>.log
watch -n 2 nvidia-smi
```

## 5) Extract After Training (Local)
```bash
bash scripts/05_extract_from_scratch.sh
```

Helper command block (post-training local):
```bash
bash scripts/05_extract_from_scratch.sh
```

Safety behavior:
- exits if destination already exists and is non-empty
- destination is `LOCAL_CHECKPOINTS_ROOT/EXTRACT_FOLDER_NAME`
- to avoid overwrite, change `EXTRACT_FOLDER_NAME` in `scripts/00_run_params.sh`
- with local overrides enabled, prefer changing it in `scripts/00_run_params.local.sh`

## Platform Compatibility
- macOS: supported (your current setup).
- Linux: supported.
- Windows: supported only via a Bash environment with `ssh` + `rsync`.
  - Recommended: WSL (Ubuntu)
  - Also possible: Git Bash/MSYS2 if `rsync` and OpenSSH are installed
  - Not supported natively in plain PowerShell/CMD without adaptation

Windows notes:
- In WSL, keep your repo inside the Linux filesystem for best performance.
- In Git Bash/MSYS2, ensure `rsync`, `ssh`, and `bash` are available on PATH.

## Known Constraints
1. One literal `rsync` command cannot copy code to home and data to scratch in the same invocation.
2. `01_sync_to_gpu.sh` gives one-command UX by running two internal `rsync` calls.
3. SSH password prompts depend on your auth setup (keys vs password).
