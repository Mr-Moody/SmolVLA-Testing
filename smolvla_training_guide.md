# ACT / SmolVLA Training Workflow (scripts folder)

All workflow scripts live in `scripts/` and are numbered in execution order.
The current default model is **ACT** (`MODEL_TYPE="act"`). SmolVLA is also
supported — set `MODEL_TYPE="smolvla"` in `00_run_params.sh`.

## Script Order
1. `scripts/00_run_params.sh` — edit parameters (committed template)
2. `scripts/01_sync_to_gpu.sh` — run locally: sync code + datasets to GPU node
3. `scripts/02_preflight_gpu.sh` — run locally: sanity-check all prerequisites
4. `scripts/03_setup_gpu.sh` — run locally: bootstrap scratch env on GPU node
5. `scripts/04_start_training.sh` — run on GPU node: merge datasets and launch training
6. `scripts/05_extract_from_scratch.sh` — run locally after training: pull checkpoints

---

## 0) SSH Key Setup (one-time)

All scripts authenticate using an SSH key rather than passwords. Generate the
key once on your local machine, then install it on the GPU node.

**Generate key (local):**
```bash
ssh-keygen -t ed25519 -f ~/.ssh/ucl_key -N ""
```

**Install on the GPU node** — from an active password-based session on hotrod:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "$(cat ~/.ssh/ucl_key.pub)" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Add SSH config** on your local machine (`~/.ssh/config`):
```
Host ucl-knuckles
    HostName knuckles.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes

Host ucl-hotrod
    HostName hotrod.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes
    ProxyJump ucl-knuckles
```

The `SSH_KEY_FILE` variable in `00_run_params.sh` defaults to `~/.ssh/ucl_key`
and is used by all scripts automatically.

---

## 1) Edit Parameters

Edit the committed template `scripts/00_run_params.sh`. For personal overrides
(different user, paths, or hosts) create a gitignored local file:
```bash
cp scripts/00_run_params.local.example.sh scripts/00_run_params.local.sh
```

**Current key fields:**
```bash
# -------- Workflow identity --------
WORKFLOW_USER="eredhead"

# -------- Core run controls --------
MODEL_TYPE="act"           # smolvla | act
RUN_NAME="test_100_101_102_103"
DATASET_NAMES=(100 101 102 103)
DATASET_ROOT="/scratch0/${WORKFLOW_USER}/lerobot_datasets"
SAVE_FREQ=1000

# -------- Preprocessing controls --------
PREPROCESS_ON_GPU=false    # true = convert on GPU; false = sync pre-converted datasets

# -------- Training hyperparameters --------
STEPS=100000
BATCH_SIZE=4
NUM_WORKERS=4
USE_AMP=true
SEED=1000

# -------- ACT model parameters --------
ACT_POLICY_PATH="lerobot/act_base"
ACT_CHUNK_SIZE=100         # action steps predicted per forward pass
ACT_N_OBS_STEPS=1
ACT_USE_VAE=true           # recommended: VAE is core to ACT
ACT_VISION_BACKBONE="resnet18"  # resnet18 (default) | resnet50 (stronger, more memory)

# -------- SmolVLA model parameters (used when MODEL_TYPE=smolvla) --------
SMOLVLA_POLICY_PATH="lerobot/smolvla_base"

# -------- SSH topology --------
GPU_NODE="hotrod.cs.ucl.ac.uk"
JUMP_HOST="knuckles.cs.ucl.ac.uk"
SSH_KEY_FILE="${HOME}/.ssh/ucl_key"
```

**Parameter notes:**
- `STEPS=100000` — with 94,647 frames and `BATCH_SIZE=4`, one epoch ≈ 23,661 steps.
  100k steps ≈ 4 epochs, which is a reasonable first full run.
- `BATCH_SIZE=4` — limited by GPU VRAM (24 GiB). With `ACT_CHUNK_SIZE=100` the
  attention matrix is large; 8 causes OOM on a 24 GiB card.
- `ACT_CHUNK_SIZE=100` — standard from the ACT paper for manipulation tasks.
- `PREPROCESS_ON_GPU=false` — datasets are converted locally and synced to scratch.

---

## 2) Local Steps

```bash
bash scripts/01_sync_to_gpu.sh   # sync code + datasets
bash scripts/02_preflight_gpu.sh # check everything is in place
```

What `01_sync_to_gpu.sh` does:
- Syncs code → remote home (`~/smolvla_project/SmolVLA-Testing/`)
- If `PREPROCESS_ON_GPU=false`: syncs **only the datasets listed in `DATASET_NAMES`** →
  `/scratch0/<user>/lerobot_datasets/<ds>/` (one rsync per dataset, nothing extra)
- If `PREPROCESS_ON_GPU=true`: syncs **only the cleaned datasets in `DATASET_NAMES`** →
  `/scratch0/<user>/cleaned_datasets/<ds>/`

Preflight note: on a fresh GPU booking, a missing `activate_smolvla.sh` is
expected before setup and is reported as a warning (not a failure). Fix it
by running `03_setup_gpu.sh`.

---

## 3) GPU Environment Setup

Run **once per GPU booking** (scratch is wiped when the booking expires):
```bash
bash scripts/03_setup_gpu.sh
```

This SSHs into the GPU node and runs `scripts/setup_scratch.sh`, which:
- Creates the scratch venv at `/scratch0/<user>/smolvla_venv/`
- Redirects **all** caches to scratch (critical for the 10 GB home quota):
  - `HF_HOME` → `/scratch0/<user>/.cache/huggingface/`
  - `TORCH_HOME` → `/scratch0/<user>/.cache/torch/` (PyTorch/torchvision hub weights)
  - `UV_CACHE_DIR`, `PIP_CACHE_DIR` → scratch
- Writes `activate_smolvla.sh` and `activate_smolvla.csh` shims

> **UCL quota note:** The home NFS share has a hard **10 GB quota**. PyTorch,
> torchvision backbone weights, and HuggingFace model files together exceed
> this easily. Never install packages or download models into `~/.local/` or
> `~/.cache/` directly — always use the scratch-redirected environment.

---

## 4) Connect to GPU

```bash
source scripts/00_run_params.sh
ssh -o IdentityAgent=none -i "${SSH_KEY_FILE}" \
    -o IdentitiesOnly=yes \
    -J "${REMOTE_USER}@${JUMP_HOST}" \
    "${REMOTE_USER}@${GPU_NODE}"
```

Or using the SSH config alias set up in step 0:
```bash
ssh -o IdentityAgent=none ucl-hotrod
```

---

## 5) Launch Training (on GPU node)

```bash
cd ~/smolvla_project/SmolVLA-Testing/scripts
bash ./04_start_training.sh
```

What the script does:
1. Merges datasets listed in `DATASET_NAMES` into a single LeRobotDataset
2. Applies source patches (frame tolerance fallback, task fallback)
3. Launches training via `nohup` with the following environment:
   - `HF_HUB_OFFLINE=1` — no network calls to HuggingFace
   - `TORCH_HOME` → scratch (prevents torchvision downloads hitting home quota)
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces GPU memory fragmentation

Resume a previously interrupted run:
```bash
bash ./04_start_training.sh --resume
```

---

## 6) Monitor Training

**From your Mac (remote):**
```bash
# Current step, percentage, and speed — one-liner
ssh -o IdentityAgent=none ucl-hotrod "tail -20 /scratch0/eredhead/smolvla_outputs/test_100_101_102_103_act.log | grep -oE '[0-9]+/100000.*step/s'"

# Live log tail
ssh -o IdentityAgent=none ucl-hotrod "tail -f /scratch0/eredhead/smolvla_outputs/test_100_101_102_103_act.log"
```

**From hotrod directly:**
```bash
# Current step, percentage, and speed
tail -5 /scratch0/eredhead/smolvla_outputs/test_100_101_102_103_act.log | grep -oE '[0-9]+/100000.*step/s'

# Live tail
tail -f /scratch0/eredhead/smolvla_outputs/test_100_101_102_103_act.log

# GPU utilisation
watch -n 2 nvidia-smi
```

The progress bar in the log looks like:
```
Training:  12%|█▏        | 12000/100000 [45:12<5:31:08,  4.43step/s]
```

At ~4.4 steps/sec, 100k steps takes approximately **6.3 hours**.

Log file path template:
```
/scratch0/<user>/smolvla_outputs/<RUN_NAME>_<MODEL_TYPE>.log
```

---

## 7) Extract Checkpoints (Local)

Pull checkpoints to your local machine at any point during or after training:
```bash
bash scripts/05_extract_from_scratch.sh
```

- Destination: `LOCAL_CHECKPOINTS_ROOT/EXTRACT_FOLDER_NAME`
  (default: `checkpoints/test_100_101_102_103_act_full/`)
- Safe: aborts if destination already exists and is non-empty
- To pull again without overwriting, change `EXTRACT_FOLDER_NAME` in `00_run_params.sh`
- Uses `--partial --inplace` so interrupted transfers can resume

---

## Disk Management

| Location | Purpose | Safe to delete? |
|---|---|---|
| `~/.local/lib/python*/` | Old pip installs leaked into home | Yes — use scratch venv |
| `/scratch0/<user>/lerobot_datasets/<old_run>/` | Datasets not in current run | Yes |
| `/scratch0/<user>/smolvla_outputs/<old_run>/` | Old training outputs | Yes, after extracting |
| `/scratch0/<user>/smolvla_venv/` | Active venv | Only if rebuilding env |
| `~/rl_coursework/` | Academic work | No |

Check quota on hotrod:
```bash
df -h ~
du -sh /scratch0/eredhead/*/
```

---

## Platform Compatibility

- **macOS**: supported (current setup).
- **Linux**: supported.
- **Windows**: supported via WSL (Ubuntu) or Git Bash/MSYS2 with `ssh` and `rsync`.
  Keep the repo inside the Linux filesystem in WSL for best performance.
  Plain PowerShell/CMD is not supported.

---

## Known Constraints

1. One `rsync` invocation cannot send code to home and data to scratch simultaneously —
   `01_sync_to_gpu.sh` handles this with two sequential calls.
2. SSH agent interference: scripts pass `-o IdentitiesOnly=yes -o IdentityAgent=none`
   to prevent "Too many authentication failures" when many keys are loaded in the agent.
3. UCL's home quota is **10 GB**. The `setup_scratch.sh` script redirects all
   large caches to scratch, but be careful not to run pip/uv without the scratch
   environment activated.
