# Server Setup: Clone & Test

After you clone the repo on the server, follow these exact steps to test Qwen works:

## Step 1: SSH to Server

```bash
ssh your_user@server.example.com
```

## Step 2: Clone Repo

```bash
# If not already cloned:
git clone <repo_url> ~/SmolVLA-Testing
cd ~/SmolVLA-Testing
```

## Step 3: Activate LeRobot Environment

```bash
# Activate the lerobot venv if it exists
source ../lerobot/.venv/bin/activate

# Or create new venv if needed
python3 -m venv venv
source venv/bin/activate
```

## Step 4: Install Qwen Dependencies

```bash
# Inside the activated venv:
pip install vllm>=0.7 qwen-vl-utils
```

## Step 5: Run Quick Test

```bash
# Make test script executable
chmod +x scripts/06_test_qwen.sh

# Run test
./scripts/06_test_qwen.sh
```

Expected output:
```
✓ Python OK
✓ vLLM installed
✓ qwen-vl-utils installed
✓ Found 1 GPU(s)
Loading Qwen3-VL model (this may take 1-2 minutes)...
  Loading: Qwen/Qwen3-VL-30B-A3B-Instruct
  ✓ Model loaded successfully!
```

If all checks pass ✓, your server is ready for the full overnight job.

## Step 6: Run Full Test (Single Dataset, 5 Episodes)

If test passes, try a minimal overnight run:

```bash
python3 run_overnight_pipeline.py \
    --raw-datasets raw_datasets \
    --dataset-names 001 \
    --enable-annotation \
    --num-gpus 1 \
    --max-episodes 5 \
    --output-dir overnight_output
```

This should:
- Clean dataset 001
- Annotate 5 episodes with Qwen
- Convert to lerobot format
- Complete in 10-15 minutes

Then you're ready to run the full batch.

## Troubleshooting

### "vllm not found"
```bash
pip install vllm>=0.7 qwen-vl-utils
python3 -c "from vllm import LLM; print('OK')"
```

### "GPU not found"
```bash
nvidia-smi
```
If no output, contact your server admin—GPU may not be available or drivers not installed.

### "Out of memory"
If you get CUDA OOM:
```bash
# Use quantized model instead (40% less VRAM)
python3 run_overnight_pipeline.py \
    --raw-datasets raw_datasets \
    --dataset-names 001 \
    --qwen-model "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ" \
    --enable-annotation \
    --num-gpus 1 \
    --max-episodes 5
```

### "Model download fails"
If Hugging Face download times out, you may need to download the model manually first:
```bash
python3 << 'EOF'
from vllm import LLM
LLM(model="Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True)
print("Model downloaded and cached")
EOF
```

## Next: Run Full Overnight Batch

Once the test passes, edit this on your **local machine**:

```bash
vi scripts/06_overnight_params.sh
```

Set:
```bash
DATASET_NAMES="001 002 003"  # Your actual datasets
ENABLE_ANNOTATION=true
NUM_GPUS=1
```

Then either:

**Option A: SSH and run directly on server**
```bash
ssh your_user@server.com
cd ~/SmolVLA-Testing
./scripts/06_run_overnight_REMOTE.sh
```

**Option B: Control from local machine via SSH**
```bash
./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
```

Done! 🎉
