#!/bin/bash
# train_act.sh - Bash script for training ACT models
#
# Usage:
#   ./train_act.sh --dataset-root lerobot_datasets/001 --steps 20000
#   ./train_act.sh --preset quick --dataset-root lerobot_datasets/001
#   ./train_act.sh --preset production --dataset-root lerobot_datasets/001
#
# For help:
#   ./train_act.sh --help

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET_ROOT=""
OUTPUT_DIR=""
STEPS=20000
BATCH_SIZE=8
DEVICE="cuda"
LEARNING_RATE="1e-5"
CHUNK_SIZE=100
VISION_BACKBONE="resnet18"
USE_VAE=true
SEED=1000
USE_AMP=false
USE_STANDALONE=false
LEROBOT_ROOT=""
PRESET=""

# Help message
show_help() {
    cat << EOF
${BLUE}ACT Model Training Script${NC}

${GREEN}Usage:${NC}
  ./train_act.sh --dataset-root PATH [options]
  ./train_act.sh --preset PRESET_NAME --dataset-root PATH

${GREEN}Required Arguments:${NC}
  --dataset-root PATH           Path to LeRobotDataset v3 directory

${GREEN}Preset Options:${NC}
  --preset quick                Quick experiment (5-10 min, low quality)
  --preset balanced             Balanced (15-30 min, good quality)
  --preset production           Production (60+ min, high quality)
  --preset limited-vram         For GPUs with <8GB VRAM

${GREEN}Common Arguments:${NC}
  --steps N                     Training steps (default: 20000)
  --batch-size N               Batch size (default: 8)
  --learning-rate LR           Learning rate (default: 1e-5)
  --seed N                     Random seed (default: 1000)
  --device DEVICE              cuda|cpu|mps (default: cuda)
  --output-dir PATH            Output directory (auto-generated if omitted)
  --lerobot-root PATH          Path to lerobot repo (auto-detected if omitted)

${GREEN}ACT Architecture Arguments:${NC}
  --chunk-size N               Chunk size (default: 100)
  --vision-backbone BACKBONE   resnet18|resnet34|resnet50 (default: resnet18)
  --no-vae                     Disable VAE
  --no-amp                     Disable Automatic Mixed Precision

${GREEN}Script Options:${NC}
  --standalone                 Use standalone script instead of main.py
  --use-amp                    Enable Automatic Mixed Precision
  --help                       Show this help message

${GREEN}Examples:${NC}
  # Quick test
  ./train_act.sh --dataset-root lerobot_datasets/001 --steps 5000

  # Quick preset (fastest)
  ./train_act.sh --preset quick --dataset-root lerobot_datasets/001

  # Production quality
  ./train_act.sh --preset production --dataset-root lerobot_datasets/001

  # Large model on good GPU
  ./train_act.sh --preset production --dataset-root lerobot_datasets/001 \\
    --vision-backbone resnet50 --batch-size 4

  # Limited VRAM
  ./train_act.sh --preset limited-vram --dataset-root lerobot_datasets/001

  # Standalone script
  ./train_act.sh --standalone --dataset-root lerobot_datasets/001

${GREEN}Preset Details:${NC}
  quick:
    - Batch: 16, Steps: 5,000, Backbone: ResNet18
    - Best for: Testing pipeline
    - Time: ~5 min on V100

  balanced:
    - Batch: 8, Steps: 20,000, Backbone: ResNet18
    - Best for: Good quality, reasonable time
    - Time: ~30 min on V100

  production:
    - Batch: 8, Steps: 50,000, Backbone: ResNet50
    - Best for: High quality model
    - Time: ~90 min on V100

  limited-vram:
    - Batch: 2, Steps: 10,000, Backbone: ResNet18, AMP: enabled
    - Best for: GPUs with <8GB VRAM
    - Time: ~15 min on RTX 2080

EOF
}

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Apply preset configuration
apply_preset() {
    local preset=$1
    case "$preset" in
        quick)
            BATCH_SIZE=16
            STEPS=5000
            CHUNK_SIZE=50
            VISION_BACKBONE="resnet18"
            USE_AMP=true
            print_info "Applied preset: quick (batch=16, steps=5000, backbone=resnet18)"
            ;;
        balanced)
            BATCH_SIZE=8
            STEPS=20000
            CHUNK_SIZE=75
            VISION_BACKBONE="resnet18"
            print_info "Applied preset: balanced (batch=8, steps=20000, backbone=resnet18)"
            ;;
        production)
            BATCH_SIZE=8
            STEPS=50000
            CHUNK_SIZE=100
            VISION_BACKBONE="resnet50"
            SEED=42
            print_info "Applied preset: production (batch=8, steps=50000, backbone=resnet50)"
            ;;
        limited-vram)
            BATCH_SIZE=2
            STEPS=10000
            CHUNK_SIZE=50
            VISION_BACKBONE="resnet18"
            USE_AMP=true
            print_info "Applied preset: limited-vram (batch=2, steps=10000, amp=enabled)"
            ;;
        *)
            print_error "Unknown preset: $preset"
            echo "Available presets: quick, balanced, production, limited-vram"
            exit 1
            ;;
    esac
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset-root)
                DATASET_ROOT="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --steps)
                STEPS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --chunk-size)
                CHUNK_SIZE="$2"
                shift 2
                ;;
            --vision-backbone)
                VISION_BACKBONE="$2"
                shift 2
                ;;
            --lerobot-root)
                LEROBOT_ROOT="$2"
                shift 2
                ;;
            --preset)
                PRESET="$2"
                shift 2
                ;;
            --use-amp)
                USE_AMP=true
                shift
                ;;
            --no-amp)
                USE_AMP=false
                shift
                ;;
            --no-vae)
                USE_VAE=false
                shift
                ;;
            --standalone)
                USE_STANDALONE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    if [[ -z "$DATASET_ROOT" ]]; then
        print_error "Missing required argument: --dataset-root"
        echo "Use --help for usage information"
        exit 1
    fi

    if [[ ! -d "$DATASET_ROOT" ]]; then
        print_error "Dataset root not found: $DATASET_ROOT"
        exit 1
    fi

    if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
        print_error "Dataset metadata not found: $DATASET_ROOT/meta/info.json"
        echo "Make sure to run: python main.py convert dataset_name"
        exit 1
    fi

    # Validate device
    case "$DEVICE" in
        cuda|cpu|mps) ;;
        *)
            print_error "Invalid device: $DEVICE (must be cuda, cpu, or mps)"
            exit 1
            ;;
    esac

    # Validate backbone
    case "$VISION_BACKBONE" in
        resnet18|resnet34|resnet50) ;;
        *)
            print_error "Invalid vision backbone: $VISION_BACKBONE (must be resnet18, resnet34, or resnet50)"
            exit 1
            ;;
    esac
}

# Build command
build_command() {
    local cmd=""

    if [[ "$USE_STANDALONE" == true ]]; then
        cmd="python train_act_standalone.py"
    else
        cmd="python main.py train --model-type act"
    fi

    cmd="$cmd --dataset-root $DATASET_ROOT"
    cmd="$cmd --steps $STEPS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --seed $SEED"
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --chunk-size $CHUNK_SIZE"
    cmd="$cmd --vision-backbone $VISION_BACKBONE"

    if [[ -n "$OUTPUT_DIR" ]]; then
        cmd="$cmd --output-dir $OUTPUT_DIR"
    fi

    if [[ "$USE_AMP" == true ]]; then
        cmd="$cmd --use-amp"
    fi

    if [[ "$USE_VAE" == false ]]; then
        cmd="$cmd --no-vae"
    fi

    if [[ -n "$LEROBOT_ROOT" ]]; then
        cmd="$cmd --lerobot-root $LEROBOT_ROOT"
    fi

    echo "$cmd"
}

# Print configuration
print_config() {
    echo ""
    print_info "Training Configuration:"
    echo "  Dataset:        $DATASET_ROOT"
    if [[ -n "$OUTPUT_DIR" ]]; then
        echo "  Output dir:     $OUTPUT_DIR"
    fi
    echo "  Steps:          $STEPS"
    echo "  Batch size:     $BATCH_SIZE"
    echo "  Device:         $DEVICE"
    echo "  Learning rate:  $LEARNING_RATE"
    echo "  Seed:           $SEED"
    echo ""
    print_info "ACT Configuration:"
    echo "  Chunk size:     $CHUNK_SIZE"
    echo "  Vision backbone: $VISION_BACKBONE"
    echo "  Use VAE:        $USE_VAE"
    echo "  Use AMP:        $USE_AMP"
    if [[ -n "$LEROBOT_ROOT" ]]; then
        echo "  LeRobot root:   $LEROBOT_ROOT"
    fi
    echo ""
    print_info "Script Settings:"
    echo "  Use standalone: $USE_STANDALONE"
    if [[ -n "$PRESET" ]]; then
        echo "  Preset:         $PRESET"
    fi
    echo ""
}

# Main function
main() {
    # Parse arguments
    parse_args "$@"

    # Apply preset if specified
    if [[ -n "$PRESET" ]]; then
        apply_preset "$PRESET"
    fi

    # Validate arguments
    validate_args

    # Print configuration
    print_config

    # Build and execute command
    local cmd=$(build_command)
    
    print_success "Starting ACT training..."
    echo ""
    
    eval "$cmd"
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo ""
        print_success "Training completed successfully!"
        if [[ -n "$OUTPUT_DIR" ]]; then
            echo "Output directory: $OUTPUT_DIR"
        else
            local dataset_name=$(basename "$DATASET_ROOT")
            echo "Output directory: outputs/${dataset_name}_act"
        fi
    else
        echo ""
        print_error "Training failed with exit code $exit_code"
        exit $exit_code
    fi
}

# Run main
main "$@"
