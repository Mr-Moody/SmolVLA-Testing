#!/usr/bin/env bash
# Example: Multi-GPU annotation with quantized Qwen model (2 GPUs, AWQ quantization)
# Use for faster annotation across multiple GPUs

DATASET_NAMES="001 002 003 004 005 006 007 008 009 010"
RAW_DATASETS_ROOT="raw_datasets"
CLEANED_DATASETS_ROOT="cleaned_datasets"
LEROBOT_DATASETS_ROOT="lerobot_datasets"
OVERNIGHT_OUTPUT_DIR="overnight_output_multi_gpu_quantized"

# Enable annotation with quantized model
ENABLE_ANNOTATION=true
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"  # Quantized = lower VRAM
NUM_GPUS=2                                        # Tensor parallelism across 2 GPUs
BATCH_SIZE_ANNOTATION=8                          # Larger batch on multiple GPUs

MAX_EPISODES_PER_DATASET=""

CAMERA_TOLERANCE_MS=150.0
PRIMARY_CAMERA="ee_zed_m"

# Settings for multi-GPU
CONTINUE_ON_ERROR=false
ENABLE_CHECKPOINT=true
LOG_LEVEL="INFO"
MONITOR_INTERVAL=30  # Update monitor every 30s
