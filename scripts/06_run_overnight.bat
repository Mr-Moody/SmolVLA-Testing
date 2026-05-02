@echo off
REM
REM RUN ON: LOCAL MACHINE
REM
REM This script runs on your local machine and can execute the pipeline either:
REM   1. Locally on your Windows machine, OR
REM   2. Via SSH to a remote Linux GPU server
REM
REM For remote execution, set REMOTE_EXECUTION=true in 06_overnight_params.sh
REM
REM Usage:
REM   06_run_overnight.bat
REM   06_run_overnight.bat (uses settings from 06_overnight_params.sh)
REM
REM Overnight annotation and conversion pipeline (Windows)
REM Runs: Clean → Annotate with Qwen → Convert to lerobot format
REM
REM Usage:
REM   06_run_overnight.bat
REM   06_run_overnight.bat dataset_names="001 002 003"

setlocal enabledelayedexpansion

REM Note: Remote execution (SSH) not directly supported on Windows batch scripts
REM For remote execution on Windows, use Windows Subsystem for Linux (WSL):
REM   wsl ./scripts/06_run_overnight_LOCAL.sh scripts/06_overnight_params.sh
REM
REM Or use 06_run_overnight_LOCAL.sh which works on Windows if bash/git-bash is installed
REM
REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\.."
set PROJECT_ROOT=%CD%

REM Load parameters
set CONFIG_FILE=%SCRIPT_DIR%06_overnight_params.sh
if not exist "%CONFIG_FILE%" (
    echo ERROR: Config file not found: %CONFIG_FILE%
    exit /b 1
)

REM Parse .sh config file (basic approach - extract key variables)
for /f "tokens=1,2 delims==" %%A in ('findstr "^DATASET_NAMES\|^RAW_DATASETS_ROOT\|^CLEANED_DATASETS_ROOT\|^LEROBOT_DATASETS_ROOT\|^OVERNIGHT_OUTPUT_DIR\|^ENABLE_ANNOTATION\|^NUM_GPUS\|^BATCH_SIZE_ANNOTATION" "%CONFIG_FILE%"') do (
    set "%%A=%%B"
)

if "%DATASET_NAMES%"=="" set DATASET_NAMES=001 002 003
if "%RAW_DATASETS_ROOT%"=="" set RAW_DATASETS_ROOT=raw_datasets
if "%CLEANED_DATASETS_ROOT%"=="" set CLEANED_DATASETS_ROOT=cleaned_datasets
if "%LEROBOT_DATASETS_ROOT%"=="" set LEROBOT_DATASETS_ROOT=lerobot_datasets
if "%OVERNIGHT_OUTPUT_DIR%"=="" set OVERNIGHT_OUTPUT_DIR=overnight_output
if "%ENABLE_ANNOTATION%"=="" set ENABLE_ANNOTATION=true
if "%NUM_GPUS%"=="" set NUM_GPUS=1
if "%BATCH_SIZE_ANNOTATION%"=="" set BATCH_SIZE_ANNOTATION=4

REM Setup logging
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set TIMESTAMP=%mydate%_%mytime%
set LOG_DIR=%OVERNIGHT_OUTPUT_DIR%\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set MAIN_LOG=%LOG_DIR%\overnight_run_%TIMESTAMP%.log

echo. > "%MAIN_LOG%"
echo ================================================================================ >> "%MAIN_LOG%"
echo OVERNIGHT ANNOTATION PIPELINE >> "%MAIN_LOG%"
echo ================================================================================ >> "%MAIN_LOG%"
echo Start time: %date% %time% >> "%MAIN_LOG%"
echo Datasets: %DATASET_NAMES% >> "%MAIN_LOG%"
echo Output: %OVERNIGHT_OUTPUT_DIR% >> "%MAIN_LOG%"
echo ================================================================================ >> "%MAIN_LOG%"
echo. >> "%MAIN_LOG%"

echo.
echo ================================================================================
echo OVERNIGHT ANNOTATION PIPELINE
echo ================================================================================
echo Start time: %date% %time%
echo Datasets: %DATASET_NAMES%
echo Output: %OVERNIGHT_OUTPUT_DIR%
echo Log file: %MAIN_LOG%
echo ================================================================================
echo.

REM Check dependencies
python3 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python3 not found
    exit /b 1
)

REM Create output directories
if not exist "%OVERNIGHT_OUTPUT_DIR%" mkdir "%OVERNIGHT_OUTPUT_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Build and run pipeline
echo [%date% %time%] Running overnight pipeline...
echo [%date% %time%] Running overnight pipeline... >> "%MAIN_LOG%"

set CMD=python3 run_overnight_pipeline.py --raw-datasets "%RAW_DATASETS_ROOT%" --cleaned-datasets "%CLEANED_DATASETS_ROOT%" --lerobot-datasets "%LEROBOT_DATASETS_ROOT%" --dataset-names %DATASET_NAMES% --output-dir "%OVERNIGHT_OUTPUT_DIR%" --num-gpus %NUM_GPUS% --batch-size-annotation %BATCH_SIZE_ANNOTATION%

if "%ENABLE_ANNOTATION%"=="false" (
    set CMD=!CMD! --skip-annotation
)

echo Command: %CMD%
echo Command: %CMD% >> "%MAIN_LOG%"
echo. >> "%MAIN_LOG%"

%CMD% >> "%MAIN_LOG%" 2>&1
set EXIT_CODE=%errorlevel%

echo. >> "%MAIN_LOG%"
echo ================================================================================ >> "%MAIN_LOG%"
echo Pipeline completed with exit code: %EXIT_CODE% >> "%MAIN_LOG%"
echo End time: %date% %time% >> "%MAIN_LOG%"
echo ================================================================================ >> "%MAIN_LOG%"

echo.
echo ================================================================================
echo Pipeline completed with exit code: %EXIT_CODE%
echo Log file: %MAIN_LOG%
echo Output directory: %OVERNIGHT_OUTPUT_DIR%
echo ================================================================================
echo.

exit /b %EXIT_CODE%
