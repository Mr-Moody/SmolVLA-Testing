# ACT / SmolVLA Training Workflow

Both ACT and SmolVLA are supported. Switch between them by setting `MODEL_TYPE`
in `scripts/00_run_params.sh`. Everything else — sync, setup, launch, extract —
uses the same numbered scripts.

## Script Order

1. `scripts/00_run_params.sh` — edit parameters (committed template)
2. `scripts/01_sync_to_gpu.sh` — run locally: sync code + datasets to GPU node
3. `scripts/02_preflight_gpu.sh` — run locally: sanity-check all prerequisites
4. `scripts/03_setup_gpu.sh` — run locally: bootstrap scratch env on GPU node
5. `scripts/04_start_training.sh` — run **on GPU node**: preprocess (optional), merge, launch
6. `scripts/05_extract_from_scratch.sh` — run locally after training: pull checkpoints

---

## 0) SSH Key Setup (one-time per machine)

All scripts authenticate using a key rather than passwords. Generate the key once
on your local machine, then install it on the GPU node.

**Generate key (local):**
```bash
ssh-keygen -t ed25519 -f ~/.ssh/ucl_key -N ""
```

**Install on a new GPU node** — macOS `ssh-copy-id` does not support `-J`, so pipe
the key through SSH directly:
```bash
cat ~/.ssh/ucl_key.pub | ssh -J eredhead@knuckles.cs.ucl.ac.uk \
    eredhead@<gpu-node>.cs.ucl.ac.uk \
    "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
```

**Add SSH config** on your local machine (`~/.ssh/config`). You need **both** the
alias stanza (for interactive use) and the hostname stanza (matched by the scripts
when they construct `ssh ... knuckles.cs.ucl.ac.uk` directly):

```
# --- UCL jump host ---
Host ucl-knuckles
    HostName knuckles.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes

# Hostname stanza — matched when scripts use the full hostname as a jump target.
# Without this, SSH tries all default keys and hits MaxAuthTries on knuckles.
Host knuckles.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes

# --- GPU nodes (add one stanza per node you use) ---
Host ucl-hotrod
    HostName hotrod.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes
    ProxyJump ucl-knuckles

Host hotrod.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes

Host ucl-bumblebee
    HostName bumblebee.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes
    ProxyJump ucl-knuckles

Host bumblebee.cs.ucl.ac.uk
    User eredhead
    IdentityFile ~/.ssh/ucl_key
    IdentitiesOnly yes
```

> **Why two stanzas per host?** The scripts pass `-J eredhead@knuckles.cs.ucl.ac.uk`
> explicitly (not via an alias). When SSH connects to a jump host using the full
> hostname, the `Host ucl-knuckles` alias does not match — SSH falls back to defaults
> and tries all key files in `~/.ssh/`, exhausting `MaxAuthTries` before `ucl_key`
> is reached. The plain `Host knuckles.cs.ucl.ac.uk` stanza with `IdentitiesOnly yes`
> prevents this.

---

## 1) Edit Parameters

Edit `scripts/00_run_params.sh` for each run. For personal overrides create the
gitignored local file:
```bash
cp scripts/00_run_params.local.example.sh scripts/00_run_params.local.sh
```

**All key fields:**
```bash
# -------- Workflow identity --------
WORKFLOW_USER="eredhead"

# -------- Core run controls --------
# Run SmolVLA first, then re-run with MODEL_TYPE="act" for ACT.
MODEL_TYPE="smolvla"           # smolvla | act
RUN_NAME="test_qwen_data"
DATASET_NAMES=(qwen_data)      # array — can be multiple: (100 101 102 103)
DATASET_ROOT="/scratch0/${WORKFLOW_USER}/lerobot_datasets"
SAVE_FREQ=50                   # set low for pipeline validation; use 1000+ for real runs

# -------- Preprocessing controls --------
# true  → sync cleaned_datasets to GPU, convert to LeRobot format there (GPU h264_nvenc)
# false → sync pre-converted lerobot_datasets directly (faster if already converted locally)
PREPROCESS_ON_GPU=true
REMOTE_CLEANED_DATASET_ROOT="/scratch0/${WORKFLOW_USER}/cleaned_datasets"
LOCAL_CLEANED_DATA_SOURCE="${LOCAL_PROJECT_ROOT}/cleaned_datasets"   # synced when true
PREPROCESS_VCODEC="h264_nvenc"
PRIMARY_CAMERA="ee_zed_m_left"        # mapped to observation.images.top in LeRobotDataset
TASK_DESCRIPTION="Looking at objects" # written into annotations.jsonl per episode

# -------- Training hyperparameters --------
STEPS=50                       # use ~100000 for real training runs
BATCH_SIZE=4
NUM_WORKERS=4
USE_AMP=true
SEED=1000

# -------- SmolVLA parameters --------
SMOLVLA_POLICY_PATH="lerobot/smolvla_base"

# -------- ACT parameters --------
ACT_POLICY_PATH="lerobot/act_base"   # private repo; not used by train_act.py (see §ACT)
ACT_CHUNK_SIZE=100
ACT_N_OBS_STEPS=1
ACT_USE_VAE=true
ACT_VISION_BACKBONE="resnet18"       # resnet18 | resnet50

# -------- SSH topology --------
GPU_NODE="bumblebee.cs.ucl.ac.uk"
JUMP_HOST="knuckles.cs.ucl.ac.uk"
SSH_KEY_FILE="${HOME}/.ssh/ucl_key"
```

**Parameter notes:**
- `STEPS` — for pipeline validation use 50–200. For real training with ~5 k frames
  and `BATCH_SIZE=4`, one epoch ≈ 1,250 steps; 100k steps ≈ 80 epochs.
- `SAVE_FREQ` must be ≤ `STEPS`; otherwise no checkpoint is written.
- `PREPROCESS_ON_GPU=true` requires `cleaned_datasets/<ds>/` and
  `cleaned_datasets/<ds>/annotations.jsonl` to exist locally before syncing (see §2).
- `PRIMARY_CAMERA` and `TASK_DESCRIPTION` are defined here for documentation; wire
  them into `04_start_training.sh` if you need to pass `--primary-camera` to the
  converter or automate annotation creation.

---

## 2) Data Pipeline

The full pipeline from raw recordings to a trained checkpoint:

```
raw_datasets/<name>/                ← read-only robot recordings
    main.py clean <name>
cleaned_datasets/<name>/            ← motion-filtered, camera-trimmed episodes
    [create annotations.jsonl]      ← task description per episode (see below)
    (04_start_training.sh runs data_converter.py on GPU when PREPROCESS_ON_GPU=true)
lerobot_datasets/<name>/            ← LeRobotDataset v3 (training-ready)
    04_start_training.sh
smolvla_outputs/<RUN_NAME>_<MODEL>/ ← checkpoints, logs
    05_extract_from_scratch.sh
checkpoints/<RUN_NAME>_<MODEL>_full/ ← local copy
```

**Step 2a — Clean raw data (local):**
```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
    main.py clean <dataset_name> --force
```
> On this Mac, torch and lerobot live under the system Python 3.11 at
> `/Library/Frameworks/Python.framework/Versions/3.11/`. The `uv --project ../lerobot`
> invocation in CLAUDE.md assumes a sibling lerobot checkout at `../lerobot`; use the
> system Python path directly if that directory does not exist.

**Step 2b — Write `annotations.jsonl` (task description per episode):**

The converter reads `cleaned_datasets/<name>/annotations.jsonl` to assign a task
string to each episode. Create it after cleaning. The episode_events.jsonl produced
by the cleaner uses raw event markers, not integer indices, so count episodes from
the clean log output instead:

```bash
# For a single-episode dataset (most common after a fresh clean of one recording):
echo '{"episode_index": 0, "task": "Looking at objects"}' \
    > cleaned_datasets/<name>/annotations.jsonl

# For multiple episodes — replace N with total episode count minus 1:
python3 -c "
import json, pathlib
out = pathlib.Path('cleaned_datasets/<name>/annotations.jsonl')
with out.open('w') as f:
    for i in range(N + 1):
        f.write(json.dumps({'episode_index': i, 'task': 'Looking at objects'}) + '\n')
"
```

> If `annotations.jsonl` is missing the converter writes empty task strings.
> Training still works because `ALLOW_MISSING_TASK_FALLBACK=true` applies a
> per-batch fallback label, but SmolVLA's language conditioning will be meaningless.

**Camera symlink warning (PREPROCESS_ON_GPU=true only):**

`data_cleaner.py` creates `cleaned_datasets/<name>/cameras` as a symlink to
`raw_datasets/<name>/cameras`. `01_sync_to_gpu.sh` transfers the symlink itself,
which breaks on the remote. After running script 01, fix this manually:

```bash
SSH_OPTS="-i ~/.ssh/ucl_key -o IdentitiesOnly=yes -o IdentityAgent=none"
SSH_REMOTE="eredhead@${GPU_NODE}"
SSH_JUMP="eredhead@${JUMP_HOST}"

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" \
    "rm /scratch0/eredhead/cleaned_datasets/<name>/cameras"

rsync -avzP --copy-links \
    -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
    raw_datasets/<name>/cameras/ \
    "${SSH_REMOTE}:/scratch0/eredhead/cleaned_datasets/<name>/cameras/"
```

This is only needed once per dataset. When `PREPROCESS_ON_GPU=false` the pre-converted
lerobot dataset (with videos already embedded) is synced instead, so cameras are
already baked in.

---

## 3) GPU Environment Setup

Run **once per GPU booking** (scratch is wiped when the booking expires):
```bash
bash scripts/03_setup_gpu.sh
```

This SSHes into the GPU node and runs `scripts/setup_scratch.sh`, which:
- Creates the scratch venv at `/scratch0/<user>/smolvla_venv/`
- Installs PyTorch (CUDA), LeRobot, and all dependencies (~2 GB download)
- Redirects **all** caches to scratch (critical for the 10 GB home quota):
  - `HF_HOME` → `/scratch0/<user>/.cache/huggingface/`
  - `TORCH_HOME` → `/scratch0/<user>/.cache/torch/`
  - `UV_CACHE_DIR`, `PIP_CACHE_DIR` → scratch
- Writes `activate_smolvla.sh` and `activate_smolvla.csh` shims
- Patches lerobot's `video_utils.py` and `tokenizer_processor.py`

> **UCL home quota (10 GB):** Never install packages or download models without the
> scratch shim active. `03_setup_gpu.sh` must be re-run at the start of each booking.

---

## 4) Pre-cache HuggingFace Models (once per booking)

Training runs with `HF_HUB_OFFLINE=1`. All required model files must be cached
**before** the first training launch. SSH into the GPU node and run:

```bash
source /scratch0/eredhead/activate_smolvla.sh
VENV=/scratch0/eredhead/smolvla_venv/bin/python

# SmolVLA — base policy weights (~873 MB)
$VENV -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/smolvla_base')
print('smolvla_base cached')
"

# SmolVLM2 — VLM backbone config + tokenizer
# ignore_patterns skips the large weight files; only config/tokenizer are needed
# when load_vlm_weights=False (the training default).
$VENV -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
                  ignore_patterns=['*.bin', '*.safetensors', '*.pt'])
print('SmolVLM2 config cached')
"

# ACT — no HF download needed. Trains from random init.
# ResNet-18 backbone weights are cached by torchvision on first use (TORCH_HOME).
```

> **Snapshot directory pitfall:** `snapshot_download` with an explicit `cache_dir=`
> argument can download blobs without creating the `snapshots/` symlink directory
> that HuggingFace uses to resolve model files. If training fails with
> `LocalEntryNotFoundError` even though the model directory exists under
> `.cache/huggingface/hub/`, re-run `snapshot_download` **without** a `cache_dir`
> argument so the shim's `HF_HOME` is used and the symlink tree is built correctly.

> **Two downloads for SmolVLA:** The policy loads `lerobot/smolvla_base` (the
> fine-tuned checkpoint) and separately loads the VLM backbone config via
> `AutoConfig.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')`.
> Both must be in cache even with `load_vlm_weights=False`.

---

## 5) Connect to GPU Node

```bash
source scripts/00_run_params.sh
ssh -o IdentityAgent=none -i "${SSH_KEY_FILE}" \
    -o IdentitiesOnly=yes \
    -J "${REMOTE_USER}@${JUMP_HOST}" \
    "${REMOTE_USER}@${GPU_NODE}"
```

Or using the SSH config alias:
```bash
ssh ucl-bumblebee   # or ucl-hotrod
```

> **Remote shell is tcsh.** UCL TSG nodes default to tcsh. Multi-line bash passed
> via SSH works with `ssh ... 'bash -s' << 'HEREDOC'`; avoid complex quoting inside
> a single double-quoted remote command string.

---

## 6) Launch Training (on GPU node)

Script 04 must be run **on the GPU node**:
```bash
cd ~/smolvla_project/SmolVLA-Testing/scripts
bash ./04_start_training.sh
```

Or trigger via SSH from local (training runs in nohup background on the remote):
```bash
source scripts/00_run_params.sh
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"
ssh ${SSH_OPTS} -J "${REMOTE_USER}@${JUMP_HOST}" \
    "${REMOTE_USER}@${GPU_NODE}" \
    'bash ~/smolvla_project/SmolVLA-Testing/scripts/04_start_training.sh'
```

What the script does:
1. If `PREPROCESS_ON_GPU=true`: runs `data_converter.py` on each dataset, falling
   back from `h264_nvenc` to software `h264` if NVENC fails.
2. Merges datasets in `DATASET_NAMES` into one LeRobotDataset (if >1).
3. Applies source patches (frame-tolerance fallback, task fallback).
4. Launches training with `nohup`, `HF_HUB_OFFLINE=1`, and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Resume an interrupted run:
```bash
bash ./04_start_training.sh --resume
```

**Running both SmolVLA and ACT sequentially (on GPU node):**
```bash
# SmolVLA pass (MODEL_TYPE=smolvla already in params)
bash ./04_start_training.sh

# ACT pass
sed -i 's/MODEL_TYPE="smolvla"/MODEL_TYPE="act"/' \
    ~/smolvla_project/SmolVLA-Testing/scripts/00_run_params.sh
bash ./04_start_training.sh
```

---

## 7) Monitor Training

**From your Mac:**
```bash
source scripts/00_run_params.sh
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"
LOG="/scratch0/eredhead/smolvla_outputs/${RUN_NAME}_${MODEL_TYPE}.log"

ssh ${SSH_OPTS} -J "${REMOTE_USER}@${JUMP_HOST}" "${REMOTE_USER}@${GPU_NODE}" \
    "bash -c 'tail -f ${LOG}'"
```

**From the GPU node directly:**
```bash
tail -f /scratch0/eredhead/smolvla_outputs/test_qwen_data_smolvla.log
watch -n 2 nvidia-smi
```

Log file path template:
```
/scratch0/<user>/smolvla_outputs/<RUN_NAME>_<MODEL_TYPE>.log
```

Progress bar in log:
```
Training:  12%|█▏        | 6000/50000 [22:14<2:51:08,  4.88step/s]
INFO step:6000 smpl:24000 ep:1 loss:0.412 grdn:3.201 lr:9.8e-05 updt_s:0.180 data_s:0.062
```

**Typical throughput on RTX 4070 Ti SUPER:**
| Model | 50-step validation | Full 100k steps (estimate) |
|---|---|---|
| SmolVLA (`load_vlm_weights=False`) | ~14 s | ~6–8 hours |
| ACT (resnet18 + VAE) | ~22 s | ~4–6 hours |

---

## 8) Extract Checkpoints (Local)

Pull checkpoints at any point during or after training:
```bash
bash scripts/05_extract_from_scratch.sh
```

- Destination: `checkpoints/<EXTRACT_FOLDER_NAME>/` (default: `<RUN_NAME>_<MODEL_TYPE>_full/`)
- Aborts if destination already exists and is non-empty
- Uses `--partial --inplace` so interrupted transfers resume cleanly

**Extracting both SmolVLA and ACT from the same run:**
```bash
# Params set to MODEL_TYPE=act — extract ACT first
bash scripts/05_extract_from_scratch.sh

# Switch to smolvla and extract
sed -i '' 's/MODEL_TYPE="act"/MODEL_TYPE="smolvla"/' scripts/00_run_params.sh
bash scripts/05_extract_from_scratch.sh
sed -i '' 's/MODEL_TYPE="smolvla"/MODEL_TYPE="act"/' scripts/00_run_params.sh  # restore
```

---

## Model-Specific Notes

### SmolVLA

- Base model: `lerobot/smolvla_base` (public HuggingFace, ~873 MB)
- VLM backbone: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (config required even
  when `load_vlm_weights=False`)
- Language conditioning: task string from `annotations.jsonl` is tokenised and fed
  to the VLM at each step — meaningful task descriptions matter for quality training
- `ALLOW_MISSING_TASK_FALLBACK=true` patches the tokenizer to supply a fallback
  label list when a batch contains steps with no task string

### ACT

- No pretrained base model needed — initialises with random weights
- `lerobot/act_base` is a private HuggingFace repo; `train_act.py` accepts the
  `policy_path` argument but does not pass it to `ACTConfig`, so it has no effect
- Vision backbone: ResNet-18 (ImageNet-pretrained, downloaded by torchvision on
  first use into `TORCH_HOME`)
- VAE (`ACT_USE_VAE=true`) is recommended — it is core to the original ACT formulation
- Reduce `ACT_CHUNK_SIZE` from 100 to 50 if you hit OOM on a 24 GiB card

---

## Disk Management

| Location | Purpose | Safe to delete? |
|---|---|---|
| `~/.local/lib/python*/` | Old pip installs leaked into home | Yes — use scratch venv |
| `/scratch0/<user>/lerobot_datasets/<old>/` | Converted datasets not in current run | Yes |
| `/scratch0/<user>/cleaned_datasets/<old>/` | Cleaned datasets not in current run | Yes |
| `/scratch0/<user>/smolvla_outputs/<old>/` | Old training outputs | Yes, after extracting |
| `/scratch0/<user>/.cache/huggingface/hub/` | HF model cache | Only if rebuilding |
| `/scratch0/<user>/smolvla_venv/` | Active venv | Only if rebuilding env |
| `~/rl_coursework/` | Academic work | No |

Check quota:
```bash
df -h ~
du -sh /scratch0/eredhead/*/
```

---

## Known Constraints

1. **Camera symlinks:** `data_cleaner.py` creates `cameras/` as a symlink to
   `raw_datasets/`. When `PREPROCESS_ON_GPU=true`, rsync transfers the symlink not
   the files. Manually re-sync camera files with `--copy-links` after `01_sync_to_gpu.sh`.
2. **SSH `MaxAuthTries`:** UCL GPU nodes reject connections after ~6 failed key
   attempts. Add plain `Host <hostname>` stanzas (not just aliases) to `~/.ssh/config`
   for knuckles and each GPU node, with `IdentitiesOnly yes` (see §0).
3. **HF offline mode:** `04_start_training.sh` sets `HF_HUB_OFFLINE=1`. All required
   model files must be cached before training starts (see §4).
4. **UCL home quota (10 GB):** Re-run `03_setup_gpu.sh` at the start of each new
   GPU booking. Never install packages without the scratch shim activated.
5. **rsync code vs data:** `01_sync_to_gpu.sh` makes two separate rsync calls —
   code to home, datasets to scratch — because they are on different mount points.
6. **Remote shell is tcsh:** Use `ssh ... 'bash -s' << 'HEREDOC'` for multi-line
   bash; avoid double-quoted Python one-liners with complex inner quoting.
