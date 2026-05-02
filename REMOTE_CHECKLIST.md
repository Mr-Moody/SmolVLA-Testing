# 🎯 Remote GPU Server Setup - Complete Checklist

This is your **definitive guide** for running the overnight annotation pipeline on a remote Linux GPU server.

---

## Quick Answer: Where Do I Run Each Script?

### 🖥️ On Your LOCAL Machine (Windows/Mac/Linux laptop)

```bash
# EDIT THIS (configuration file)
vi scripts/06_overnight_params.sh

# RUN THIS (executes pipeline either locally or remotely)
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# MONITOR THIS (in another terminal)
./scripts/06_monitor_overnight.sh overnight_output/

# VIEW RESULTS HERE
ls lerobot_datasets/
```

### 🖥️ On REMOTE SERVER (GPU cluster/node)

```bash
# ONLY needed if you want to run directly on server:
# 1. Copy script: scp scripts/06_run_overnight_REMOTE.sh server:...
# 2. SSH to server: ssh user@server.com
# 3. Run: ./scripts/06_run_overnight_REMOTE.sh
```

---

## Two Execution Modes

### Mode 1: Control from Local Machine (Recommended ⭐)

**Everything runs from your laptop. Perfect for most users.**

```
Your Laptop (LOCAL)                    Remote GPU Server
┌─────────────────────┐                ┌──────────────────┐
│ 1. Edit config      │                │                  │
│ 2. Run LOCAL script │───SSH──────→   │ Executes here    │
│ 3. Monitor output   │   ↓            │ (Qwen, etc)      │
│ 4. View results     │←──Stream────   │                  │
└─────────────────────┘                └──────────────────┘
```

**Commands on your LOCAL machine:**

```bash
# Edit config
vi scripts/06_overnight_params.sh

# Set these in the config:
REMOTE_EXECUTION=true
GPU_NODE="server.example.com"
REMOTE_USER="your_username"
DATASET_NAMES="001 002 003"

# Run!
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# Monitor (optional, in another terminal)
./scripts/06_monitor_overnight.sh overnight_output/
```

✅ **Pros:** Full control, real-time monitoring, automatic results retrieval
❌ **Cons:** Need persistent SSH connection (use `nohup` or `tmux` if needed)

---

### Mode 2: Run Directly on Remote Server

**For power users who want to manage jobs on the server directly.**

```
Your Laptop (LOCAL)                    Remote GPU Server
┌─────────────────────┐                ┌──────────────────┐
│ 1. SCP script       │───Copy──────→  │ 1. Edit config   │
│ 2. SSH to server    │                │ 2. Run REMOTE    │
│                     │                │    script        │
│                     │                │ 3. Runs here     │
│ 3. Retrieve results │←──Retrieve──   │ 4. Output saved  │
└─────────────────────┘                └──────────────────┘
```

**Commands:**

```bash
# On LOCAL: Copy script to server
scp scripts/06_run_overnight_REMOTE.sh your_user@server.com:~/SmolVLA-Testing/scripts/

# SSH to server
ssh your_user@server.com

# On REMOTE: Edit config in the script
vi scripts/06_run_overnight_REMOTE.sh
# Set: DATASET_NAMES, ENABLE_ANNOTATION, NUM_GPUS

# On REMOTE: Run (use nohup for persistent execution)
nohup ./scripts/06_run_overnight_REMOTE.sh > nohup.out 2>&1 &

# On REMOTE: Monitor
tail -f overnight_output/logs/overnight_run_*.log

# On LOCAL: Get results when done
rsync -avz your_user@server.com:~/SmolVLA-Testing/lerobot_datasets/ ./
```

✅ **Pros:** Can disconnect from SSH, job keeps running, good for overnight jobs
❌ **Cons:** Must manage job on server, harder to debug issues

---

## Complete Checklist: First Time Setup

### ☐ 1. SSH Key Setup (One-time, ~5 min)

```bash
# On your LOCAL machine:

# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy public key to server
ssh-copy-id -i ~/.ssh/id_ed25519 your_username@server.example.com

# Test connection
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com "nvidia-smi"

# If that works, you're good! ✓
```

### ☐ 2. Install Dependencies on Remote (One-time, ~10 min)

```bash
# SSH to the remote server
ssh your_username@server.example.com

# Navigate to project
cd ~/SmolVLA-Testing

# Activate lerobot environment
source ../lerobot/.venv/bin/activate

# Install Qwen dependencies
pip install vllm>=0.7 qwen-vl-utils

# Verify
python3 -c "from vllm import LLM; from qwen_vl_utils import *; print('✓ Ready')"
```

### ☐ 3. Verify Raw Datasets on Remote

```bash
ssh your_username@server.example.com

cd ~/SmolVLA-Testing

# Check datasets exist
ls -lh raw_datasets/001 raw_datasets/002 raw_datasets/003

# Should show robot.jsonl, episode_events.jsonl, cameras/, etc.
```

### ☐ 4. Test with Small Dataset (LOCAL, ~10 min)

```bash
# On your LOCAL machine:

# Edit config
vi scripts/06_overnight_params.sh

# Set:
REMOTE_EXECUTION=true
GPU_NODE="your_server.example.com"
REMOTE_USER="your_username"
DATASET_NAMES="001"          # Just one dataset
MAX_EPISODES_PER_DATASET=5   # Just 5 episodes
ENABLE_ANNOTATION=false      # Skip Qwen for now

# Run test
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# Should complete in 2-3 minutes if working
```

### ☐ 5. Full Run with Annotation (LOCAL)

```bash
# On your LOCAL machine:

# Edit config for full run
vi scripts/06_overnight_params.sh

# Set:
REMOTE_EXECUTION=true
DATASET_NAMES="001 002 003"          # Your actual datasets
ENABLE_ANNOTATION=true               # Use Qwen
MAX_EPISODES_PER_DATASET=""          # All episodes
NUM_GPUS=1                           # Adjust if you have multiple GPUs
BATCH_SIZE_ANNOTATION=4              # Adjust based on GPU memory

# Run
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# Monitor in another terminal
./scripts/06_monitor_overnight.sh overnight_output/
```

---

## Script Reference

### Scripts You Edit (LOCAL)
- `scripts/06_overnight_params.sh` — Configuration file (edit this!)
- `scripts/06_example_*.sh` — Example configurations

### Scripts You Run (LOCAL)
- `scripts/06_run_overnight_LOCAL.sh` — Main execution (with SSH support)
- `scripts/06_run_overnight_LOCAL.sh` — Real-time monitor
- `scripts/06_run_overnight.bat` — Windows batch version

### Scripts You Run (REMOTE)
- `scripts/06_run_overnight_REMOTE.sh` — Direct execution on server

### Python Scripts (AUTO USED)
- `run_overnight_pipeline.py` — Main orchestration (runs locally or remotely)
- `batch_annotate.py` — Batch Qwen annotation

---

## Configuration Reference

### For LOCAL Execution

```bash
# scripts/06_overnight_params.sh

REMOTE_EXECUTION=false           # ← Run on this machine
DATASET_NAMES="001 002 003"      # Your datasets (space-separated)
ENABLE_ANNOTATION=true           # Use Qwen3-VL
NUM_GPUS=1                       # GPU count
```

### For REMOTE SSH Execution (Recommended)

```bash
# scripts/06_overnight_params.sh

REMOTE_EXECUTION=true            # ← Execute on remote server via SSH
REMOTE_USER="your_username"      # SSH username
GPU_NODE="server.example.com"    # Server hostname
REMOTE_PROJECT_ROOT="/home/your_username/SmolVLA-Testing"
SSH_KEY_FILE="${HOME}/.ssh/id_ed25519"
DATASET_NAMES="001 002 003"
ENABLE_ANNOTATION=true
NUM_GPUS=1
```

### For REMOTE Direct Execution

Edit `/scripts/06_run_overnight_REMOTE.sh` directly on the server:

```bash
# Lines 20-35 of the script

DATASET_NAMES="001 002 003"      # Your datasets
ENABLE_ANNOTATION=true           # Use Qwen
NUM_GPUS=1                       # GPU count
```

---

## Output Locations

### Results are Created in:

```
lerobot_datasets/
├── 001/                    ← Processed dataset 1
│   ├── meta/info.json
│   ├── videos/episode_*.mp4
│   └── chunks/episode_*.parquet
├── 002/                    ← Processed dataset 2
└── ...

overnight_output/
├── checkpoint.json         ← Resume info
└── logs/
    └── overnight_run_*.log ← Detailed logs
```

---

## Troubleshooting

### "SSH connection refused"

```bash
# Check server is accessible
ssh -i ~/.ssh/id_ed25519 your_user@server.com "echo OK"

# If fails, check:
# 1. Hostname is correct
# 2. Username is correct
# 3. SSH key has correct permissions: chmod 600 ~/.ssh/id_ed25519
# 4. Public key is in ~/.ssh/authorized_keys on server
```

### "vllm not found" on Remote

```bash
# SSH to server
ssh your_user@server.com

# Install in lerobot env
source ../lerobot/.venv/bin/activate
pip install vllm>=0.7 qwen-vl-utils
```

### "GPU not found" on Remote

```bash
# Check GPU availability on remote
ssh your_user@server.com "nvidia-smi"

# If not found:
# 1. GPU might not be available
# 2. NVIDIA drivers might not be installed
# 3. Contact server admin
```

### "Out of memory" during Qwen annotation

```bash
# Edit 06_overnight_params.sh:

# Option 1: Reduce batch size
BATCH_SIZE_ANNOTATION=2

# Option 2: Use quantized model
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"

# Option 3: Skip annotation
ENABLE_ANNOTATION=false
```

### "Job seems stuck / SSH connection lost"

Use `tmux` on remote for persistent execution:

```bash
ssh your_user@server.com

# Create tmux session
tmux new-session -d -s overnight "cd ~/SmolVLA-Testing && ./scripts/06_run_overnight_REMOTE.sh"

# You can safely disconnect SSH now
exit

# Check on it later
ssh your_user@server.com "tmux attach -t overnight"
```

---

## Performance Tips

### Speed Up Annotation

```bash
# Use quantized model (40% less VRAM)
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"

# Increase batch size (if GPU has room)
BATCH_SIZE_ANNOTATION=8

# Use multiple GPUs
NUM_GPUS=2
```

### Speed Up Overall

```bash
# Skip annotation entirely (5-10x faster)
ENABLE_ANNOTATION=false

# Limit datasets
DATASET_NAMES="001 002"

# Process fewer episodes
MAX_EPISODES_PER_DATASET=50
```

### Monitor Resources During Run

```bash
ssh your_user@server.com

# Watch GPU usage
watch nvidia-smi

# Watch CPU/memory
top

# Watch disk space
df -h

# Watch process
ps aux | grep python3
```

---

## Next Steps

1. ✅ **Setup SSH keys** (Section "SSH Key Setup")
2. ✅ **Install dependencies** on remote (Section "Install Dependencies on Remote")
3. ✅ **Test with small dataset** (Section "Test with Small Dataset")
4. ✅ **Run full overnight batch** (Section "Full Run with Annotation")

---

## Recommended Workflow

```bash
# Day 1: Setup (30 min)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -i ~/.ssh/id_ed25519 your_user@server.com
ssh your_user@server.com "source ../lerobot/.venv/bin/activate && pip install vllm>=0.7 qwen-vl-utils"

# Day 1-2: Test (5 min + wait time)
vi scripts/06_overnight_params.sh
# Set REMOTE_EXECUTION=true, test with 1 dataset, 5 episodes
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# Day 2: Full Run (5 min setup + overnight execution)
# Edit config for production
vi scripts/06_overnight_params.sh
# Set your datasets, enable Qwen
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
# Monitor
./scripts/06_monitor_overnight.sh overnight_output/

# Next morning: View results
ls -lh lerobot_datasets/
head overnight_output/logs/overnight_run_*.log
```

---

## Summary

| Component | Where It Runs | How You Access It |
|-----------|---------------|-------------------|
| Configuration | Edit on LOCAL | `vi scripts/06_overnight_params.sh` |
| Main script | Runs on LOCAL | `./scripts/06_run_overnight_LOCAL.sh` |
| Execution | Can be LOCAL or REMOTE | Controlled by `REMOTE_EXECUTION` setting |
| GPU/Qwen | Always runs where dataset is | If remote dataset, runs on remote |
| Monitoring | LOCAL machine | `./scripts/06_monitor_overnight.sh` |
| Results | Available locally | `ls lerobot_datasets/` |

**TLDR: Edit config on laptop, run script on laptop, everything else is automatic!** ✨

---

## Support & Issues

For detailed troubleshooting:
- See [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md) for SSH/remote specifics
- See [SCRIPT_EXECUTION_GUIDE.md](SCRIPT_EXECUTION_GUIDE.md) for which scripts run where
- See [OVERNIGHT_SETUP.md](OVERNIGHT_SETUP.md) for general setup

For code issues:
- Check logs: `cat overnight_output/logs/overnight_run_*.log`
- Run test: `./scripts/06_run_overnight_LOCAL.sh --max-episodes 2 --skip-annotation`
- Enable debug: Modify `LOG_LEVEL` in config
