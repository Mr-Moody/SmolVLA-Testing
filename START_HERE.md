# 🎯 START HERE: Overnight Qwen3-VL Annotation Pipeline

Welcome! This is your **complete guide** to running the overnight annotation and dataset conversion pipeline with Qwen3-VL on a remote GPU server.

---

## ⚡ Quick Answer: Where Do I Start?

### 1️⃣ **First Time?** → Read This First

📖 **[REMOTE_CHECKLIST.md](REMOTE_CHECKLIST.md)** (10 min read)
- Answers: "Where do I run each script?"
- Complete setup checklist
- Step-by-step instructions
- Local vs remote workflows

### 2️⃣ **In a Hurry?** → Quick Start

⚡ **[OVERNIGHT_QUICKSTART.md](OVERNIGHT_QUICKSTART.md)** (5 min read)
- TL;DR version
- Essential config in 60 seconds
- Common issues quick fixes

### 3️⃣ **Running on Remote Server?** → Deep Dive

🖥️ **[REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md)** (20 min read)
- SSH setup and configuration
- Two complete workflows (local control vs direct server)
- Connection troubleshooting
- Recommended setup patterns

### 4️⃣ **Need to Know Which Scripts Run Where?** → Reference

📋 **[SCRIPT_EXECUTION_GUIDE.md](SCRIPT_EXECUTION_GUIDE.md)** (10 min read)
- Matrix showing which scripts run locally/remotely
- Step-by-step for both workflows
- Script purpose reference

### 5️⃣ **Want Feature Details?** → Full Reference

📚 **[OVERNIGHT_PIPELINE.md](OVERNIGHT_PIPELINE.md)**
- Feature overview
- Configuration reference
- Advanced usage examples

### 6️⃣ **Detailed Setup?** → Complete Setup Guide

🔧 **[OVERNIGHT_SETUP.md](OVERNIGHT_SETUP.md)**
- Requirements (hardware, software, packages)
- Installation walkthrough
- Performance estimates
- Troubleshooting

---

## 📋 Navigation Quick Reference

| What You Need | Document | Time |
|---------------|----------|------|
| **Main checklist** | [REMOTE_CHECKLIST.md](REMOTE_CHECKLIST.md) | 10 min |
| **Quick start** | [OVERNIGHT_QUICKSTART.md](OVERNIGHT_QUICKSTART.md) | 5 min |
| **Remote setup** | [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md) | 20 min |
| **Which script runs where** | [SCRIPT_EXECUTION_GUIDE.md](SCRIPT_EXECUTION_GUIDE.md) | 10 min |
| **All features** | [OVERNIGHT_PIPELINE.md](OVERNIGHT_PIPELINE.md) | 15 min |
| **Detailed setup** | [OVERNIGHT_SETUP.md](OVERNIGHT_SETUP.md) | 20 min |
| **Implementation details** | [OVERNIGHT_IMPLEMENTATION_SUMMARY.md](OVERNIGHT_IMPLEMENTATION_SUMMARY.md) | 15 min |

---

## 🚀 Fastest Path: Run Your First Batch

### Step 1: SSH Setup (5 min, one-time)

```bash
# On your LOCAL machine:

# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy to server (one command)
ssh-copy-id -i ~/.ssh/id_ed25519 your_username@server.example.com

# Test
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com "nvidia-smi"
```

### Step 2: Install Dependencies (5 min, one-time)

```bash
# SSH to server and install
ssh your_username@server.example.com

cd ~/SmolVLA-Testing
source ../lerobot/.venv/bin/activate
pip install vllm>=0.7 qwen-vl-utils
exit
```

### Step 3: Configure & Run (2 min)

```bash
# On your LOCAL machine:

cd SmolVLA-Testing

# Edit configuration
vi scripts/06_overnight_params.sh

# Change these lines:
# DATASET_NAMES="001 002 003"
# REMOTE_EXECUTION=true
# GPU_NODE="your_server.example.com"
# REMOTE_USER="your_username"

# Run!
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# Monitor in another terminal (optional)
./scripts/06_monitor_overnight.sh overnight_output/
```

### Step 4: Check Results

```bash
# After completion:
ls -lh lerobot_datasets/
```

**That's it!** ✨

---

## 📁 What Was Built?

### Documentation (7 files)
- ✅ **REMOTE_CHECKLIST.md** — Main setup guide  
- ✅ **SCRIPT_EXECUTION_GUIDE.md** — Which scripts run where
- ✅ **REMOTE_SERVER_GUIDE.md** — SSH and remote workflows
- ✅ **OVERNIGHT_PIPELINE.md** — Feature reference
- ✅ **OVERNIGHT_SETUP.md** — Detailed setup guide
- ✅ **OVERNIGHT_QUICKSTART.md** — 5-minute quick start
- ✅ **OVERNIGHT_IMPLEMENTATION_SUMMARY.md** — Technical overview

### Python Scripts (3 files)
- ✅ **run_overnight_pipeline.py** (600+ lines) — Main orchestrator with checkpoint/resume
- ✅ **batch_annotate.py** (180 lines) — Batch Qwen annotation helper
- ✅ **annotate_cleaned_dataset.py** (modified) — Qwen integration

### Shell Scripts (8 files)
- ✅ **06_overnight_params.sh** — Configuration file
- ✅ **06_run_overnight_LOCAL.sh** — Run on local machine (with SSH support)
- ✅ **06_run_overnight_REMOTE.sh** — Run directly on remote server
- ✅ **06_run_overnight.bat** — Windows batch version
- ✅ **06_monitor_overnight.sh** — Progress monitor (Unix)
- ✅ **06_monitor_overnight.bat** — Progress monitor (Windows)
- ✅ **06_example_small_batch.sh** — Example config (5 datasets)
- ✅ **06_example_large_no_annotation.sh** — Example config (20 datasets)
- ✅ **06_example_multi_gpu_quantized.sh** — Example config (multi-GPU)

---

## 🎯 Key Features

✅ **Checkpoint & Resume** — Automatic resume on failure, no re-processing
✅ **Batch Qwen3-VL** — Efficient annotation with configurable batch sizes
✅ **GPU Support** — Single or multi-GPU, quantization support
✅ **Remote SSH** — Full control from your local machine
✅ **Error Handling** — Per-dataset error tracking, detailed logs
✅ **Monitoring** — Real-time progress display, monitor script
✅ **Cross-Platform** — Linux, Windows (WSL), macOS

---

## 🖥️ Two Execution Modes

### Mode 1: Control from Local Machine (Recommended ⭐)

```bash
# Edit config on LOCAL
vi scripts/06_overnight_params.sh

# Run from LOCAL
./scripts/06_run_overnight_LOCAL.sh

# Everything handles remote via SSH automatically
```

**Perfect for:** Most users, interactive development, real-time monitoring

### Mode 2: Direct Server Execution

```bash
# Copy script to SERVER
scp scripts/06_run_overnight_REMOTE.sh server:...

# SSH to server and run
ssh server
./scripts/06_run_overnight_REMOTE.sh
```

**Perfect for:** Production overnight jobs, server admins

---

## 💡 What Each Script Does

| Script | Run On | Purpose |
|--------|--------|---------|
| `06_overnight_params.sh` | LOCAL | Configuration file (edit this!) |
| `06_run_overnight_LOCAL.sh` | LOCAL | Main execution (handles remote via SSH) |
| `06_run_overnight_REMOTE.sh` | REMOTE | Direct server execution |
| `06_monitor_overnight.sh` | LOCAL | Real-time progress monitoring |
| `run_overnight_pipeline.py` | (auto) | Main Python orchestrator |

**Simple rule:** If you're unsure, run on your LOCAL machine.

---

## 📊 Pipeline Flow

```
Raw Datasets
    ↓
Step 1: CLEAN (remove corrupted episodes)
    ↓
Step 2: ANNOTATE (Qwen3-VL generates labels)
    ↓
Step 3: CONVERT (transform to lerobot format)
    ↓
Processed LeRobot Datasets (lerobot_datasets/)

Checkpoint System:
  └─ Tracks progress & enables resume on failure
```

---

## ⚙️ System Requirements

### Local Machine
- Python 3.8+
- SSH access to remote server
- 1GB free disk for logs

### Remote GPU Server
- NVIDIA GPU (24GB+ VRAM recommended, or 16GB with quantization)
- Python 3.8+
- vLLM and Qwen dependencies
- 50GB+ free disk for processing

---

## 🔥 Common First Steps

### "I want to test with a small dataset first"

```bash
vi scripts/06_overnight_params.sh
# Set: DATASET_NAMES="001"
# Set: MAX_EPISODES_PER_DATASET=5
# Set: ENABLE_ANNOTATION=false  # Skip Qwen for now

./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
# Should complete in 2-3 minutes
```

### "I want to use Qwen annotation"

```bash
vi scripts/06_overnight_params.sh
# Set: ENABLE_ANNOTATION=true
# Set: NUM_GPUS=1
# Set: BATCH_SIZE_ANNOTATION=4

# Make sure Qwen is installed on remote:
# ssh your_user@server.com "source ../lerobot/.venv/bin/activate && pip install vllm>=0.7 qwen-vl-utils"
```

### "I want to run multiple datasets overnight"

```bash
vi scripts/06_overnight_params.sh
# Set: DATASET_NAMES="001 002 003 004 005"
# Set: ENABLE_ANNOTATION=true
# Adjust BATCH_SIZE_ANNOTATION if needed for GPU memory

# Use nohup or tmux on server if needed for persistence
nohup ./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh > overnight.log 2>&1 &
```

---

## 📖 Recommended Reading Order

1. **Start:** This file (overview)
2. **Setup:** [REMOTE_CHECKLIST.md](REMOTE_CHECKLIST.md) (main guide)
3. **Execution:** [SCRIPT_EXECUTION_GUIDE.md](SCRIPT_EXECUTION_GUIDE.md) (which script runs where)
4. **Reference:** [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md) (SSH & remote details)
5. **Features:** [OVERNIGHT_PIPELINE.md](OVERNIGHT_PIPELINE.md) (all options)

---

## ✅ Verification Checklist

Before running:

- [ ] SSH keys generated: `ls ~/.ssh/id_ed25519`
- [ ] Remote accessible: `ssh user@server "nvidia-smi"` works
- [ ] Dependencies installed: `ssh user@server "python3 -c 'from vllm import LLM'"`
- [ ] Raw datasets exist: `ssh user@server "ls ~/SmolVLA-Testing/raw_datasets/001"`
- [ ] Disk space adequate: `ssh user@server "df -h"` shows 50GB+ free
- [ ] Configuration edited: `cat scripts/06_overnight_params.sh | grep DATASET_NAMES`

---

## 🐛 Troubleshooting

### "I don't know where to run this"
→ Check [SCRIPT_EXECUTION_GUIDE.md](SCRIPT_EXECUTION_GUIDE.md) — has a matrix showing LOCAL vs REMOTE

### "SSH connection fails"
→ Check [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md) — SSH setup section

### "vllm not found on server"
→ SSH to server and run: `pip install vllm>=0.7 qwen-vl-utils`

### "GPU out of memory"
→ In config, reduce `BATCH_SIZE_ANNOTATION` or use quantized model

### "Script seems stuck"
→ Use `nohup` or `tmux` for persistent execution; check logs in `overnight_output/logs/`

---

## 🎓 Learning Resources

- **5-minute quick start:** [OVERNIGHT_QUICKSTART.md](OVERNIGHT_QUICKSTART.md)
- **Main setup guide:** [REMOTE_CHECKLIST.md](REMOTE_CHECKLIST.md)
- **SSH & remote workflows:** [REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md)
- **Script reference:** [SCRIPT_EXECUTION_GUIDE.md](SCRIPT_EXECUTION_GUIDE.md)
- **All features & options:** [OVERNIGHT_PIPELINE.md](OVERNIGHT_PIPELINE.md)

---

## 🚀 Ready to Start?

1. ✅ Read [REMOTE_CHECKLIST.md](REMOTE_CHECKLIST.md) (10 min)
2. ✅ Follow setup checklist (30 min setup + testing)
3. ✅ Run your first batch (2 min setup + overnight)
4. ✅ Check results (2 min)

**Questions? Check the relevant guide from the table above.** 📚

---

## 💬 Summary

**This system lets you:**
- 🎯 Annotate datasets with Qwen3-VL automatically
- 🖥️ Control everything from your laptop via SSH
- 📊 Convert to LeRobot format in one go
- 🔄 Resume from failures automatically
- 📈 Monitor progress in real-time
- 🌙 Run overnight without keeping your machine on

**Key files you'll edit:**
- `scripts/06_overnight_params.sh` — Configuration

**Key scripts you'll run:**
- `scripts/06_run_overnight_LOCAL.sh` — Execute pipeline
- `scripts/06_monitor_overnight.sh` — Monitor progress

**That's it!** Everything else is automated. 🎉

---

**Start with [REMOTE_CHECKLIST.md](REMOTE_CHECKLIST.md) →**
