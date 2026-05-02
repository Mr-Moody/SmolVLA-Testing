# Remote Server Execution Guide

## Overview

The overnight pipeline can run in **two modes**:

1. **LOCAL MODE** - Run on your local machine (which can SSH to a remote server)
2. **REMOTE MODE** - Run directly on the remote GPU server

This guide explains both workflows and when to use each.

---

## ⚡ Quick Reference

| Task | Script | Run On | When to Use |
|------|--------|--------|------------|
| **Configure** | `scripts/06_overnight_params.sh` | LOCAL (edit locally) | Every run - set DATASET_NAMES, ENABLE_ANNOTATION, etc. |
| **Execute (Local)** | `scripts/06_run_overnight_LOCAL.sh` | LOCAL | When your machine has GPU or for testing |
| **Execute (Remote)** | `scripts/06_run_overnight_LOCAL.sh` with `REMOTE_EXECUTION=true` | LOCAL | When you want to use SSH to remote server |
| **Execute (Direct)** | `scripts/06_run_overnight_REMOTE.sh` | REMOTE SERVER | When you SSH to server and run directly |
| **Monitor** | `scripts/06_monitor_overnight.sh` | LOCAL | Watch progress in separate terminal |

---

## Workflow 1: Run on Remote via SSH (Recommended)

**Best for:** Most users. Control everything from your local machine.

### Step 1: Edit Local Configuration

On your **local machine**:

```bash
cd SmolVLA-Testing
vi scripts/06_overnight_params.sh
```

Set these variables:

```bash
# Which datasets to process
DATASET_NAMES="001 002 003"

# Enable remote execution
REMOTE_EXECUTION=true

# Remote server details
REMOTE_USER="your_username"        # SSH username
GPU_NODE="server.example.com"      # Server hostname
REMOTE_PROJECT_ROOT="/home/your_username/SmolVLA-Testing"  # Where project is on server
SSH_KEY_FILE="${HOME}/.ssh/id_ed25519"  # Path to SSH private key

# Processing options
ENABLE_ANNOTATION=true
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4
```

### Step 2: Verify SSH Configuration

Make sure SSH works:

```bash
# Test connection
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com "echo 'SSH OK'"

# Test GPU availability on server
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com "nvidia-smi"
```

### Step 3: Run Pipeline from Local Machine

On your **local machine**:

```bash
cd SmolVLA-Testing

# Make script executable
chmod +x scripts/06_run_overnight_LOCAL.sh

# Run with SSH to remote
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
```

The script will:
1. Check dependencies on your local machine
2. SSH to the remote server
3. Execute the pipeline on the remote GPU
4. Stream output back to your terminal
5. Save logs locally at `overnight_output/logs/`

### Step 4: Monitor Progress (Optional)

In another **local terminal**:

```bash
# Watch progress live
./scripts/06_monitor_overnight.sh overnight_output/ --interval 30
```

### Step 5: Retrieve Results

After completion:

```bash
# Logs are already local (from SSH output)
cat overnight_output/logs/overnight_run_*.log

# To get remote output files (if using scratch directories):
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
    your_username@server.example.com:/remote/output/dir/ \
    ./local_output/
```

---

## Workflow 2: Run Directly on Remote Server

**Best for:** Server admin or if you want to manage jobs on the server directly.

### Step 1: Copy Script to Server

On your **local machine**:

```bash
# Copy the remote-execution script
scp -i ~/.ssh/id_ed25519 \
    scripts/06_run_overnight_REMOTE.sh \
    your_username@server.example.com:~/SmolVLA-Testing/scripts/

# Or copy entire scripts directory
scp -r -i ~/.ssh/id_ed25519 \
    scripts/ \
    your_username@server.example.com:~/SmolVLA-Testing/
```

### Step 2: Edit Configuration on Server

SSH to the **remote server**:

```bash
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com

# Navigate to project
cd ~/SmolVLA-Testing

# Edit the remote script
vi scripts/06_run_overnight_REMOTE.sh
```

Edit these variables directly in the script:

```bash
# Line ~20-30: Set your configuration
DATASET_NAMES="001 002 003"
ENABLE_ANNOTATION=true
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4
```

### Step 3: Run on Server

Still on the **remote server**:

```bash
# Make script executable
chmod +x scripts/06_run_overnight_REMOTE.sh

# Run directly
./scripts/06_run_overnight_REMOTE.sh

# Or run in background (recommended for long jobs)
nohup ./scripts/06_run_overnight_REMOTE.sh > nohup.out 2>&1 &
echo $! > overnight.pid  # Save process ID

# Check status later
tail -f overnight_output/logs/overnight_run_*.log
```

### Step 4: Monitor from Server

On the **remote server**:

```bash
# Watch progress
tail -f overnight_output/logs/overnight_run_*.log | grep -E "(episode|COMPLETED|ERROR)"

# Or use monitor script
chmod +x scripts/06_monitor_overnight.sh
./scripts/06_monitor_overnight.sh overnight_output/
```

### Step 5: Check Results

On the **remote server**:

```bash
# Results are in
ls -lh overnight_output/
ls -lh lerobot_datasets/

# Or retrieve locally
# On your local machine:
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
    your_username@server.example.com:~/SmolVLA-Testing/lerobot_datasets/ \
    ./remote_datasets/
```

---

## Comparing Workflows

### Workflow 1: SSH from Local (REMOTE_EXECUTION=true)

**Pros:**
- Control everything from your laptop
- Real-time output streaming
- Automatic log retrieval
- Easy to monitor and adjust
- Can run script in background: `nohup ./script.sh &`

**Cons:**
- Network connection must stay open
- If you close terminal, connection drops (but can use `nohup` or `tmux`)

**Best for:** Interactive development, debugging, small runs

### Workflow 2: Direct Server Execution

**Pros:**
- Runs on server even if you close laptop
- Can disconnect and check later
- Better for long overnight runs
- Less network overhead

**Cons:**
- Need to SSH to edit scripts
- Harder to debug issues
- Must manually retrieve results

**Best for:** Fire-and-forget overnight jobs, production runs

---

## Configuration Differences

### For LOCAL MACHINE EXECUTION

Edit `scripts/06_overnight_params.sh`:

```bash
REMOTE_EXECUTION=false

# These are relative to your LOCAL project root
RAW_DATASETS_ROOT="raw_datasets"
CLEANED_DATASETS_ROOT="cleaned_datasets"
LEROBOT_DATASETS_ROOT="lerobot_datasets"
```

### For REMOTE SSH EXECUTION

Edit `scripts/06_overnight_params.sh`:

```bash
REMOTE_EXECUTION=true

REMOTE_USER="your_username"
GPU_NODE="server.example.com"
REMOTE_PROJECT_ROOT="/home/your_username/SmolVLA-Testing"
SSH_KEY_FILE="${HOME}/.ssh/id_ed25519"

# Paths are relative to REMOTE_PROJECT_ROOT on the server
RAW_DATASETS_ROOT="raw_datasets"
CLEANED_DATASETS_ROOT="cleaned_datasets"
LEROBOT_DATASETS_ROOT="lerobot_datasets"
```

### For DIRECT SERVER EXECUTION

Edit `scripts/06_run_overnight_REMOTE.sh` on the server:

```bash
DATASET_NAMES="001 002 003"
ENABLE_ANNOTATION=true
NUM_GPUS=1

# Paths are relative to the script's location on the server
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
```

---

## SSH Setup (One-time)

If you haven't set up SSH keys:

### Generate SSH Key on Local Machine

```bash
# Create key (one-time)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Display public key
cat ~/.ssh/id_ed25519.pub
```

### Install Public Key on Server

On the **remote server**:

```bash
# Create .ssh directory if needed
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public key
echo "your_public_key_content_here" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Test Connection

Back on your **local machine**:

```bash
ssh -i ~/.ssh/id_ed25519 your_username@server.example.com "echo 'Success!'"
```

---

## Troubleshooting

### "SSH connection refused"

```bash
# Check if SSH server is running on remote
ssh your_username@server.example.com "ps aux | grep sshd"

# Verify SSH key permissions
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### "Permission denied (publickey)"

```bash
# Ensure public key is in ~/.ssh/authorized_keys on server
ssh-copy-id -i ~/.ssh/id_ed25519 your_username@server.example.com

# Or manually:
ssh your_username@server.example.com "cat >> ~/.ssh/authorized_keys" < ~/.ssh/id_ed25519.pub
```

### "vllm not found" on Remote

SSH to server and install:

```bash
ssh your_username@server.example.com

# Activate environment
source ../lerobot/.venv/bin/activate

# Install Qwen dependencies
pip install vllm>=0.7 qwen-vl-utils

# Test
python3 -c "from vllm import LLM; print('OK')"
```

### Network Connection Drops

Use `tmux` or `screen` on the remote server:

```bash
# SSH to server
ssh your_username@server.example.com

# Start tmux session
tmux new-session -d -s overnight

# Send command to session
tmux send-keys -t overnight "cd ~/SmolVLA-Testing && ./scripts/06_run_overnight_REMOTE.sh" Enter

# Disconnect (session continues running)
tmux detach

# Check on job later
tmux attach -t overnight
```

### Storage Space Issues

Check before running:

```bash
# Local check
du -sh raw_datasets/* cleaned_datasets/* lerobot_datasets/*

# Remote check
ssh your_username@server.example.com "du -sh ~/SmolVLA-Testing/{raw_datasets,cleaned_datasets,lerobot_datasets}"
```

---

## Recommended Setup

For most users, we recommend:

```bash
# 1. Setup SSH (one-time)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -i ~/.ssh/id_ed25519 your_username@server.example.com

# 2. Edit local config
vi scripts/06_overnight_params.sh
# Set: REMOTE_EXECUTION=true and other params

# 3. Run from local machine
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh

# 4. Monitor in another terminal (local)
./scripts/06_monitor_overnight.sh overnight_output/

# 5. Results automatically available locally
```

This gives you:
- Full control from your laptop
- Real-time monitoring
- Automatic result retrieval
- Easy debugging

---

## Next Steps

1. Choose your workflow (SSH or direct server)
2. Set up SSH keys if needed
3. Edit configuration file
4. Test with 1 dataset: `--max-episodes 5 --skip-annotation`
5. Run full overnight batch

For issues, check:
- `overnight_output/logs/overnight_run_*.log`
- SSH connectivity with `ssh -v`
- GPU availability with `nvidia-smi` on remote
- Disk space with `df -h` on remote
