@echo off
REM train_act.bat - Batch script for training ACT models on Windows
REM
REM Usage:
REM   train_act.bat --dataset-root lerobot_datasets\001 --steps 20000
REM   train_act.bat --preset quick --dataset-root lerobot_datasets\001
REM   train_act.bat --help

setlocal enabledelayedexpansion

REM Default values
set "DATASET_ROOT="
set "OUTPUT_DIR="
set "STEPS=20000"
set "BATCH_SIZE=8"
set "DEVICE=cuda"
set "LEARNING_RATE=1e-5"
set "CHUNK_SIZE=100"
set "VISION_BACKBONE=resnet18"
set "USE_VAE=true"
set "SEED=1000"
set "USE_AMP=false"
set "USE_STANDALONE=false"
set "LEROBOT_ROOT="
set "PRESET="

REM Parse arguments
:parse_args
if "%~1"=="" goto check_args
if "%~1"=="--dataset-root" (
    set "DATASET_ROOT=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--output-dir" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--steps" (
    set "STEPS=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--batch-size" (
    set "BATCH_SIZE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--learning-rate" (
    set "LEARNING_RATE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--seed" (
    set "SEED=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--device" (
    set "DEVICE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--chunk-size" (
    set "CHUNK_SIZE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--vision-backbone" (
    set "VISION_BACKBONE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--lerobot-root" (
    set "LEROBOT_ROOT=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--preset" (
    set "PRESET=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--use-amp" (
    set "USE_AMP=true"
    shift
    goto parse_args
)
if "%~1"=="--no-amp" (
    set "USE_AMP=false"
    shift
    goto parse_args
)
if "%~1"=="--no-vae" (
    set "USE_VAE=false"
    shift
    goto parse_args
)
if "%~1"=="--standalone" (
    set "USE_STANDALONE=true"
    shift
    goto parse_args
)
if "%~1"=="--help" (
    goto show_help
)
if "%~1"=="-h" (
    goto show_help
)
echo Error: Unknown option %~1
echo Use --help for usage information
exit /b 1

:check_args
if "!DATASET_ROOT!"=="" (
    echo Error: Missing required argument: --dataset-root
    echo Use --help for usage information
    exit /b 1
)

if not exist "!DATASET_ROOT!" (
    echo Error: Dataset root not found: !DATASET_ROOT!
    exit /b 1
)

if not exist "!DATASET_ROOT!\meta\info.json" (
    echo Error: Dataset metadata not found: !DATASET_ROOT!\meta\info.json
    echo Make sure to run: python main.py convert dataset_name
    exit /b 1
)

REM Apply preset if specified
if not "!PRESET!"=="" (
    call :apply_preset !PRESET!
)

REM Print configuration
echo.
echo [INFO] Training Configuration:
echo   Dataset:        !DATASET_ROOT!
if not "!OUTPUT_DIR!"=="" (
    echo   Output dir:     !OUTPUT_DIR!
)
echo   Steps:          !STEPS!
echo   Batch size:     !BATCH_SIZE!
echo   Device:         !DEVICE!
echo   Learning rate:  !LEARNING_RATE!
echo   Seed:           !SEED!
echo.
echo [INFO] ACT Configuration:
echo   Chunk size:     !CHUNK_SIZE!
echo   Vision backbone: !VISION_BACKBONE!
echo   Use VAE:        !USE_VAE!
echo   Use AMP:        !USE_AMP!
if not "!LEROBOT_ROOT!"=="" (
    echo   LeRobot root:   !LEROBOT_ROOT!
)
echo.
echo [INFO] Script Settings:
echo   Use standalone: !USE_STANDALONE!
if not "!PRESET!"=="" (
    echo   Preset:         !PRESET!
)
echo.

REM Build command
set "CMD="
if "!USE_STANDALONE!"=="true" (
    set "CMD=python train_act_standalone.py"
) else (
    set "CMD=python main.py train --model-type act"
)

set "CMD=!CMD! --dataset-root !DATASET_ROOT!"
set "CMD=!CMD! --steps !STEPS!"
set "CMD=!CMD! --batch-size !BATCH_SIZE!"
set "CMD=!CMD! --learning-rate !LEARNING_RATE!"
set "CMD=!CMD! --seed !SEED!"
set "CMD=!CMD! --device !DEVICE!"
set "CMD=!CMD! --chunk-size !CHUNK_SIZE!"
set "CMD=!CMD! --vision-backbone !VISION_BACKBONE!"

if not "!OUTPUT_DIR!"=="" (
    set "CMD=!CMD! --output-dir !OUTPUT_DIR!"
)

if "!USE_AMP!"=="true" (
    set "CMD=!CMD! --use-amp"
)

if "!USE_VAE!"=="false" (
    set "CMD=!CMD! --no-vae"
)

if not "!LEROBOT_ROOT!"=="" (
    set "CMD=!CMD! --lerobot-root !LEROBOT_ROOT!"
)

echo [SUCCESS] Starting ACT training...
echo.

REM Execute command
%CMD%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Training completed successfully!
    if not "!OUTPUT_DIR!"=="" (
        echo Output directory: !OUTPUT_DIR!
    ) else (
        for %%F in (!DATASET_ROOT!) do (
            echo Output directory: outputs\%%~nxF_act
        )
    )
) else (
    echo.
    echo [ERROR] Training failed with exit code %errorlevel%
    exit /b %errorlevel%
)

goto :eof

REM Apply preset configuration
:apply_preset
if "%~1"=="quick" (
    set "BATCH_SIZE=16"
    set "STEPS=5000"
    set "CHUNK_SIZE=50"
    set "VISION_BACKBONE=resnet18"
    set "USE_AMP=true"
    echo [INFO] Applied preset: quick (batch=16, steps=5000, backbone=resnet18)
) else if "%~1"=="balanced" (
    set "BATCH_SIZE=8"
    set "STEPS=20000"
    set "CHUNK_SIZE=75"
    set "VISION_BACKBONE=resnet18"
    echo [INFO] Applied preset: balanced (batch=8, steps=20000, backbone=resnet18)
) else if "%~1"=="production" (
    set "BATCH_SIZE=8"
    set "STEPS=50000"
    set "CHUNK_SIZE=100"
    set "VISION_BACKBONE=resnet50"
    set "SEED=42"
    echo [INFO] Applied preset: production (batch=8, steps=50000, backbone=resnet50)
) else if "%~1"=="limited-vram" (
    set "BATCH_SIZE=2"
    set "STEPS=10000"
    set "CHUNK_SIZE=50"
    set "VISION_BACKBONE=resnet18"
    set "USE_AMP=true"
    echo [INFO] Applied preset: limited-vram (batch=2, steps=10000, amp=enabled)
) else (
    echo Error: Unknown preset: %~1
    echo Available presets: quick, balanced, production, limited-vram
    exit /b 1
)
goto :eof

:show_help
echo.
echo ACT Model Training Script for Windows
echo.
echo Usage:
echo   train_act.bat --dataset-root PATH [options]
echo   train_act.bat --preset PRESET_NAME --dataset-root PATH
echo.
echo Required Arguments:
echo   --dataset-root PATH           Path to LeRobotDataset v3 directory
echo.
echo Preset Options:
echo   --preset quick                Quick experiment (5-10 min, low quality)
echo   --preset balanced             Balanced (15-30 min, good quality)
echo   --preset production           Production (60+ min, high quality)
echo   --preset limited-vram         For GPUs with less than 8GB VRAM
echo.
echo Common Arguments:
echo   --steps N                     Training steps (default: 20000)
echo   --batch-size N               Batch size (default: 8)
echo   --learning-rate LR           Learning rate (default: 1e-5)
echo   --seed N                     Random seed (default: 1000)
echo   --device DEVICE              cuda^|cpu^|mps (default: cuda)
echo   --output-dir PATH            Output directory (auto-generated if omitted)
echo   --lerobot-root PATH          Path to lerobot repo (auto-detected if omitted)
echo.
echo ACT Architecture Arguments:
echo   --chunk-size N               Chunk size (default: 100)
echo   --vision-backbone BACKBONE   resnet18^|resnet34^|resnet50 (default: resnet18)
echo   --no-vae                     Disable VAE
echo   --use-amp                    Enable Automatic Mixed Precision
echo.
echo Script Options:
echo   --standalone                 Use standalone script instead of main.py
echo   --help                       Show this help message
echo.
echo Examples:
echo.
echo   REM Quick test
echo   train_act.bat --dataset-root lerobot_datasets\001 --steps 5000
echo.
echo   REM Quick preset
echo   train_act.bat --preset quick --dataset-root lerobot_datasets\001
echo.
echo   REM Production quality
echo   train_act.bat --preset production --dataset-root lerobot_datasets\001
echo.
echo   REM Large model
echo   train_act.bat --preset production --dataset-root lerobot_datasets\001 ^
echo     --vision-backbone resnet50 --batch-size 4
echo.
echo   REM Limited VRAM
echo   train_act.bat --preset limited-vram --dataset-root lerobot_datasets\001
echo.
echo   REM Standalone script
echo   train_act.bat --standalone --dataset-root lerobot_datasets\001
echo.
exit /b 0
