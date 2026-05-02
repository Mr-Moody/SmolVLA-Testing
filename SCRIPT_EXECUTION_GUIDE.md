# 🖥️ Script Execution Guide: Where to Run Each Script

This guide clarifies which scripts run on your **LOCAL MACHINE** and which run on the **REMOTE SERVER**.

---

## Script Execution Matrix

| Script | Purpose | Run On | How |
|--------|---------|--------|-----|
| `00_run_params.sh` | EDIT config for training | LOCAL | `vi scripts/00_run_params.sh` |
| `01_sync_to_gpu.sh` | Sync code to remote | LOCAL | `./scripts/01_sync_to_gpu.sh` |
| `02_preflight_gpu.sh` | Check remote prerequisites | LOCAL | `./scripts/02_preflight_gpu.sh` |
| `03_setup_gpu.sh` | Bootstrap remote env | LOCAL | `./scripts/03_setup_gpu.sh` |
| `04_start_training.sh` | Start training | REMOTE | `ssh server` then `./scripts/04_start_training.sh` |
| `05_extract_from_scratch.sh` | Pull results locally | LOCAL | `./scripts/05_extract_from_scratch.sh` |
| `06_overnight_params.sh` | EDIT config for annotation | LOCAL | `vi scripts/06_overnight_params.sh` |
| `06_run_overnight_LOCAL.sh` | Run annotation pipeline | LOCAL | `./scripts/06_run_overnight_LOCAL.sh` |
| `06_run_overnight_REMOTE.sh` | Run annotation pipeline | REMOTE | `ssh server` then `./scripts/06_run_overnight_REMOTE.sh` |
| `06_monitor_overnight.sh` | Monitor progress live | LOCAL | `./scripts/06_monitor_overnight.sh` |

---

## Training Workflow (ACT/SmolVLA)

These scripts are for **TRAINING** (04_start_training.sh runs ON THE REMOTE SERVER):

```
LOCAL MACHINE                          REMOTE SERVER (GPU)
├─ edit 00_run_params.sh
├─ run 01_sync_to_gpu.sh  ──────────→  [code synced]
├─ run 02_preflight_gpu.sh ──────────→ [checks run]
├─ run 03_setup_gpu.sh     ──────────→ [env setup]
│
└─ SSH and on remote, run:
                                       ├─ 04_start_training.sh  ◄─ RUNS HERE
                                       │  (training happens)
                                       └─ Checkpoint saved
│
├─ Monitor from local: tail logs via SSH
│
└─ run 05_extract_from_scratch.sh ←─── [pull results]
```

---

## Overnight Annotation Workflow (NEW)

These scripts are for **ANNOTATION WITH QWEN3-VL**.

### Option A: Control from Local Machine (Recommended)

```
LOCAL MACHINE                          REMOTE SERVER (GPU)
├─ edit 06_overnight_params.sh
│  Set: REMOTE_EXECUTION=true
│  Set: GPU_NODE="server.com"
│  Set: DATASET_NAMES="001 002 003"
│
├─ run 06_run_overnight_LOCAL.sh ────→ [SSH executes]
│  with REMOTE_EXECUTION=true
│  (pipeline runs remotely)            └─ Qwen annotation happens
│                                       └─ Results saved locally via SSH
│
├─ monitor 06_monitor_overnight.sh (in separate terminal)
│  (watches progress)
│
└─ Results automatically available locally
   in ./overnight_output/
```

**RUN THIS ON YOUR LOCAL MACHINE:**
```bash
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
```

### Option B: Copy Script to Server and Run Directly

```
LOCAL MACHINE                          REMOTE SERVER (GPU)
├─ Copy script to server:
│  scp scripts/06_run_overnight_REMOTE.sh server:~/SmolVLA-Testing/scripts/
│
└─ SSH to server:
                                       ├─ edit 06_run_overnight_REMOTE.sh
                                       │  Set: DATASET_NAMES="001 002 003"
                                       │  Set: ENABLE_ANNOTATION=true
                                       │
                                       ├─ run 06_run_overnight_REMOTE.sh
                                       │  (pipeline runs here)
                                       │  (Qwen annotation happens)
                                       │
                                       └─ Results in ./overnight_output/
│
└─ Retrieve results:
   rsync ... server:~/SmolVLA-Testing/lerobot_datasets/ ./
```

**RUN THIS ON THE REMOTE SERVER:**
```bash
./scripts/06_run_overnight_REMOTE.sh
```

---

## Step-by-Step: Annotation Pipeline (Easiest Method)

### Step 1: Edit Configuration (LOCAL)

On your **LOCAL MACHINE**:

```bash
cd SmolVLA-Testing

# EDIT this file
vi scripts/06_overnight_params.sh

# Change these lines:
DATASET_NAMES="001 002 003"              # Line ~9
ENABLE_ANNOTATION=true                   # Line ~19 (use Qwen)
REMOTE_EXECUTION=true                    # Line ~12 (use remote GPU)
REMOTE_USER="your_username"              # Line ~13
GPU_NODE="your_server.com"               # Line ~14
SSH_KEY_FILE="${HOME}/.ssh/id_ed25519"  # Line ~17
```

### Step 2: Run Pipeline (LOCAL)

On your **LOCAL MACHINE**:

```bash
chmod +x scripts/06_run_overnight_LOCAL.sh
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
```

The script will:
- Check your local dependencies
- SSH to the remote server
- Execute annotation on the remote GPU
- Stream output back to you
- Save logs locally

### Step 3: Monitor Progress (LOCAL, Optional)

In a **separate terminal** on your LOCAL MACHINE:

```bash
./scripts/06_monitor_overnight.sh overnight_output/
```

### Step 4: View Results (LOCAL)

After completion:

```bash
# Logs
tail overnight_output/logs/overnight_run_*.log

# Datasets
ls -lh lerobot_datasets/
```

**That's it! Everything runs from your local machine!**

---

## Step-by-Step: If Running Directly on Server

Only do this if you're comfortable with SSH.

### Step 1: Copy Script to Server (LOCAL)

```bash
scp -i ~/.ssh/id_ed25519 \
    scripts/06_run_overnight_REMOTE.sh \
    your_username@server.example.com:~/SmolVLA-Testing/scripts/
```

### Step 2: SSH to Server and Configure (REMOTE)

```bash
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com

cd ~/SmolVLA-Testing

# Edit configuration in the script
vi scripts/06_run_overnight_REMOTE.sh
# Set DATASET_NAMES, ENABLE_ANNOTATION, NUM_GPUS, etc.
```

### Step 3: Run on Server (REMOTE)

```bash
chmod +x scripts/06_run_overnight_REMOTE.sh

# Run in background (recommended)
nohup ./scripts/06_run_overnight_REMOTE.sh > nohup.out 2>&1 &

# Or run in foreground
./scripts/06_run_overnight_REMOTE.sh

# Check status
tail -f overnight_output/logs/overnight_run_*.log
```

### Step 4: Get Results (LOCAL)

```bash
# From your local machine
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
    your_username@server.example.com:~/SmolVLA-Testing/lerobot_datasets/ \
    ./remote_results/
```

---

## Key Differences

### 06_overnight_params.sh
- **Location**: Edit on LOCAL MACHINE
- **Used by**: Both LOCAL and REMOTE scripts
- **Contains**: All configuration (dataset names, GPU count, etc.)
- **Edit with**: `vi scripts/06_overnight_params.sh`

### 06_run_overnight_LOCAL.sh
- **Location**: Run on LOCAL MACHINE
- **Purpose**: Control everything from your laptop
- **Uses**: SSH to execute on remote (if REMOTE_EXECUTION=true)
- **Run with**: `./scripts/06_run_overnight_LOCAL.sh`

### 06_run_overnight_REMOTE.sh
- **Location**: Copy to REMOTE SERVER, run there
- **Purpose**: Direct execution on GPU server
- **Uses**: Configuration embedded in script
- **Run with**: `ssh server` then `./scripts/06_run_overnight_REMOTE.sh`

### 06_monitor_overnight.sh
- **Location**: Run on LOCAL MACHINE
- **Purpose**: Watch progress in real-time
- **Uses**: Reads checkpoint.json for status
- **Run with**: `./scripts/06_monitor_overnight.sh overnight_output/`

---

## Linux Remote Server Specifics

The pipeline is **fully compatible** with Linux remote servers (GPU clusters, etc.).

### On Remote Server, Install Dependencies:

```bash
# SSH to server
ssh your_username@server.com

# Navigate to project
cd ~/SmolVLA-Testing

# Activate environment
source ../lerobot/.venv/bin/activate

# Install Qwen dependencies
pip install vllm>=0.7 qwen-vl-utils

# Verify
python3 << 'EOF'
from vllm import LLM
from qwen_vl_utils import *
print("✓ Qwen dependencies installed")
EOF
```

### Check GPU on Remote:

```bash
ssh your_username@server.com "nvidia-smi"
```

### SSH Key Setup (First Time Only):

```bash
# On your LOCAL machine
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy to server
ssh-copy-id -i ~/.ssh/id_ed25519 your_username@server.com

# Test connection
ssh -i ~/.ssh/id_ed25519 your_username@server.com "echo 'Connected!'"
```

---

## Troubleshooting

### "I'm not sure if I should run this locally or on server"

**Answer**: When in doubt, run these on LOCAL:
- All `vi` (editing) commands
- `06_run_overnight_LOCAL.sh`
- `06_monitor_overnight.sh`
- Anything that doesn't have "REMOTE" in the filename

### "The script says 'RUN ON: LOCAL' at the top"

That means you run it on your LOCAL MACHINE (your laptop/desktop).

### "The script says 'RUN ON: REMOTE SERVER' at the top"

That means you:
1. SSH to the remote server
2. Copy the script there
3. Run it on the remote server

### "I ran a LOCAL script but nothing happened"

Check:
1. Did you `chmod +x scripts/script.sh` to make it executable?
2. Are you in the right directory? (`cd SmolVLA-Testing`)
3. Is the remote server accessible? (`ssh -i key your_user@server.com` works?)

### "SSH times out"

The network connection might be flaky. For long jobs, use `tmux` on the server:

```bash
ssh your_user@server.com

# Start tmux
tmux new-session -d -s overnight "./scripts/06_run_overnight_REMOTE.sh"

# You can disconnect and check later
tmux attach -t overnight
```

---

## Example: Complete Workflow

```bash
# 1. ON YOUR LOCAL MACHINE
#    Edit configuration
vi scripts/06_overnight_params.sh
# Set: DATASET_NAMES="001 002 003"
# Set: REMOTE_EXECUTION=true
# Set: GPU_NODE="gpu-server.example.com"
# Set: REMOTE_USER="your_username"

# 2. ON YOUR LOCAL MACHINE
#    Run pipeline (it SSH's to server)
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# 3. ON YOUR LOCAL MACHINE
#    In another terminal, monitor progress
./scripts/06_monitor_overnight.sh overnight_output/

# 4. ON YOUR LOCAL MACHINE
#    After completion, check results
ls -lh lerobot_datasets/
```

**Everything was controlled from your laptop!** 🎉

---

## Summary

| Task | Script | Machine | Command |
|------|--------|---------|---------|
| Edit params | `06_overnight_params.sh` | LOCAL | `vi scripts/06_overnight_params.sh` |
| Run overnight | `06_run_overnight_LOCAL.sh` | LOCAL | `./scripts/06_run_overnight_LOCAL.sh` |
| Monitor | `06_monitor_overnight.sh` | LOCAL | `./scripts/06_monitor_overnight.sh` |
| View results | (file system) | LOCAL | `ls lerobot_datasets/` |

Or, if running directly on server:

| Task | Script | Machine | Command |
|------|--------|---------|---------|
| Copy script | `06_run_overnight_REMOTE.sh` | LOCAL (scp) | `scp ... server:...` |
| SSH to server | (terminal) | LOCAL | `ssh server.com` |
| Edit params | `06_run_overnight_REMOTE.sh` | REMOTE | `vi scripts/06_run_overnight_REMOTE.sh` |
| Run overnight | `06_run_overnight_REMOTE.sh` | REMOTE | `./scripts/06_run_overnight_REMOTE.sh` |
| Get results | (rsync) | LOCAL | `rsync ... server:... .` |

---

## Next: Choose Your Method

Pick one:

**Option A (Easier, Recommended):**
→ See [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md) - Workflow 1

**Option B (For Server Admins):**
→ See [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md) - Workflow 2

Both are fully supported! Choose based on your preference.
