#!/usr/bin/env bash
# Example: Large batch without annotation (20+ datasets, clean and convert only)
# Use when you want to skip Qwen annotation for speed

DATASET_NAMES="001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020"
RAW_DATASETS_ROOT="raw_datasets"
CLEANED_DATASETS_ROOT="cleaned_datasets"
LEROBOT_DATASETS_ROOT="lerobot_datasets"
OVERNIGHT_OUTPUT_DIR="overnight_output_large_no_annotation"

# Skip annotation for speed
ENABLE_ANNOTATION=false
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4

# Limit per-dataset processing
MAX_EPISODES_PER_DATASET=""

CAMERA_TOLERANCE_MS=150.0
PRIMARY_CAMERA="ee_zed_m"

# Robust settings for large batch
CONTINUE_ON_ERROR=true
ENABLE_CHECKPOINT=true
LOG_LEVEL="INFO"
