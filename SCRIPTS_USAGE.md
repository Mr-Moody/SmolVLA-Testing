# Training Scripts Guide

Two convenient scripts for training ACT models without remembering all the command-line flags.

## Windows Users

### Using `train_act.bat`

**Make it executable** (Windows automatically recognizes `.bat` files):
```bash
# Just run it directly
train_act.bat --help
```

**Quick training:**
```bash
train_act.bat --dataset-root lerobot_datasets\001 --steps 20000
```

**Using presets:**
```bash
# Quick (5 min)
train_act.bat --preset quick --dataset-root lerobot_datasets\001

# Balanced (30 min)
train_act.bat --preset balanced --dataset-root lerobot_datasets\001

# Production (90 min)
train_act.bat --preset production --dataset-root lerobot_datasets\001

# Limited VRAM
train_act.bat --preset limited-vram --dataset-root lerobot_datasets\001
```

**Custom parameters:**
```bash
train_act.bat --dataset-root lerobot_datasets\001 ^
    --batch-size 4 --steps 50000 --vision-backbone resnet50
```

## Linux/Mac Users

### Using `train_act.sh`

**Make it executable:**
```bash
chmod +x train_act.sh
```

**Quick training:**
```bash
./train_act.sh --dataset-root lerobot_datasets/001 --steps 20000
```

**Using presets:**
```bash
# Quick (5 min)
./train_act.sh --preset quick --dataset-root lerobot_datasets/001

# Balanced (30 min)
./train_act.sh --preset balanced --dataset-root lerobot_datasets/001

# Production (90 min)
./train_act.sh --preset production --dataset-root lerobot_datasets/001

# Limited VRAM
./train_act.sh --preset limited-vram --dataset-root lerobot_datasets/001
```

**Custom parameters:**
```bash
./train_act.sh --dataset-root lerobot_datasets/001 \
    --batch-size 4 --steps 50000 --vision-backbone resnet50
```

## Common Use Cases

### Scenario 1: Quick Experiment
**Windows:**
```bash
train_act.bat --preset quick --dataset-root lerobot_datasets\001
```

**Linux/Mac:**
```bash
./train_act.sh --preset quick --dataset-root lerobot_datasets/001
```

Expected: Trains in ~5 minutes with quick feedback on whether model/data pipeline works.

### Scenario 2: Good Quality (Default)
**Windows:**
```bash
train_act.bat --dataset-root lerobot_datasets\001
```

**Linux/Mac:**
```bash
./train_act.sh --dataset-root lerobot_datasets/001
```

Expected: 20,000 steps in ~30 minutes, reasonable accuracy.

### Scenario 3: Production Quality
**Windows:**
```bash
train_act.bat --preset production --dataset-root lerobot_datasets\001
```

**Linux/Mac:**
```bash
./train_act.sh --preset production --dataset-root lerobot_datasets/001
```

Expected: 50,000 steps with ResNet50 in ~90 minutes, high accuracy.

### Scenario 4: GPU with Limited VRAM
**Windows:**
```bash
train_act.bat --preset limited-vram --dataset-root lerobot_datasets\001
```

**Linux/Mac:**
```bash
./train_act.sh --preset limited-vram --dataset-root lerobot_datasets/001
```

Expected: Smaller batch size and AMP for GPUs with <8GB VRAM.

### Scenario 5: Custom Configuration
**Windows:**
```bash
train_act.bat --dataset-root lerobot_datasets\001 ^
    --batch-size 4 ^
    --steps 50000 ^
    --vision-backbone resnet50 ^
    --learning-rate 5e-5
```

**Linux/Mac:**
```bash
./train_act.sh --dataset-root lerobot_datasets/001 \
    --batch-size 4 \
    --steps 50000 \
    --vision-backbone resnet50 \
    --learning-rate 5e-5
```

## Presets Explained

| Preset | Batch | Steps | Backbone | Time | Best For |
|--------|-------|-------|----------|------|----------|
| `quick` | 16 | 5,000 | ResNet18 | 5 min | Testing |
| `balanced` | 8 | 20,000 | ResNet18 | 30 min | Good quality |
| `production` | 8 | 50,000 | ResNet50 | 90 min | Best quality |
| `limited-vram` | 2 | 10,000 | ResNet18+AMP | 15 min | <8GB GPU |

## Available Arguments

### Required
- `--dataset-root PATH` - Path to LeRobotDataset v3 directory

### Presets (Alternative to manual config)
- `--preset quick` - Fast experiment
- `--preset balanced` - Balanced quality/speed
- `--preset production` - Best quality
- `--preset limited-vram` - For limited GPU memory

### Training Hyperparameters
- `--steps N` - Training steps (default: 20000)
- `--batch-size N` - Batch size (default: 8)
- `--learning-rate LR` - Learning rate (default: 1e-5)
- `--seed N` - Random seed (default: 1000)
- `--device DEVICE` - cuda|cpu|mps (default: cuda)

### Model Architecture
- `--chunk-size N` - Action chunk size (default: 100)
- `--vision-backbone BACKBONE` - resnet18|resnet34|resnet50 (default: resnet18)
- `--no-vae` - Disable VAE (default: enabled)

### Performance
- `--use-amp` - Enable Automatic Mixed Precision
- `--standalone` - Use standalone script

### Output
- `--output-dir PATH` - Custom output directory
- `--lerobot-root PATH` - Path to lerobot repo

## Full Examples

### Full Quick Test (Windows)
```bash
REM Prepare data
python main.py clean 001 --force
python main.py convert 001 --primary-camera ee_zed_m

REM Train with quick preset
train_act.bat --preset quick --dataset-root lerobot_datasets\001
```

### Full Quick Test (Linux/Mac)
```bash
# Prepare data
python main.py clean 001 --force
python main.py convert 001 --primary-camera ee_zed_m

# Train with quick preset
./train_act.sh --preset quick --dataset-root lerobot_datasets/001
```

### Full Production Workflow (Windows)
```bash
REM 1. Clean dataset
python main.py clean 001 --force

REM 2. Convert to LeRobot format
python main.py convert 001 --primary-camera ee_zed_m

REM 3. Train ACT model
train_act.bat --preset production --dataset-root lerobot_datasets\001

REM 4. Check results
dir outputs\001_act
```

### Full Production Workflow (Linux/Mac)
```bash
# 1. Clean dataset
python main.py clean 001 --force

# 2. Convert to LeRobot format
python main.py convert 001 --primary-camera ee_zed_m

# 3. Train ACT model
./train_act.sh --preset production --dataset-root lerobot_datasets/001

# 4. Check results
ls outputs/001_act/
```

## Troubleshooting

### Script doesn't run (Windows)
- Make sure you're in the SmolVLA-Testing directory
- Try: `python train_act_standalone.py --help` (direct Python method)

### Script doesn't run (Linux/Mac)
```bash
# Make it executable
chmod +x train_act.sh

# Then run
./train_act.sh --help
```

### CUDA Out of Memory
```bash
# Windows
train_act.bat --preset limited-vram --dataset-root lerobot_datasets\001

# Linux/Mac
./train_act.sh --preset limited-vram --dataset-root lerobot_datasets/001
```

### Need more customization
Use the standalone script directly:
```bash
# Windows
python train_act_standalone.py --dataset-root lerobot_datasets\001 ^
    --batch-size 2 --learning-rate 5e-5

# Linux/Mac
python train_act_standalone.py --dataset-root lerobot_datasets/001 \
    --batch-size 2 --learning-rate 5e-5
```

## Help

**Windows:**
```bash
train_act.bat --help
```

**Linux/Mac:**
```bash
./train_act.sh --help
```

## Creating Your Own Preset

Edit the script and add a new case in the preset section:

**Windows (train_act.bat):**
```batch
if "%~1"=="custom" (
    set "BATCH_SIZE=4"
    set "STEPS=30000"
    set "CHUNK_SIZE=80"
    set "VISION_BACKBONE=resnet34"
    set "USE_AMP=true"
    echo [INFO] Applied preset: custom
)
```

**Linux/Mac (train_act.sh):**
```bash
custom)
    BATCH_SIZE=4
    STEPS=30000
    CHUNK_SIZE=80
    VISION_BACKBONE="resnet34"
    USE_AMP=true
    print_info "Applied preset: custom"
    ;;
```

Then use it:
```bash
# Windows
train_act.bat --preset custom --dataset-root lerobot_datasets\001

# Linux/Mac
./train_act.sh --preset custom --dataset-root lerobot_datasets/001
```

## Next Steps

1. **Prepare your dataset** using `main.py clean` and `main.py convert`
2. **Run a quick preset** to test the pipeline
3. **Use production preset** for your final model
4. **Check outputs** in the `outputs/` directory

Good luck! 🚀
