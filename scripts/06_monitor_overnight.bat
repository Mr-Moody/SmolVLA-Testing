@echo off
REM Monitor overnight pipeline progress (Windows)
REM Usage: 06_monitor_overnight.bat overnight_output/

setlocal enabledelayedexpansion

set OUTPUT_DIR=%1
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=overnight_output

if not exist "%OUTPUT_DIR%" (
    echo ERROR: Output directory not found: %OUTPUT_DIR%
    exit /b 1
)

set INTERVAL=60
if not "%2"=="" set INTERVAL=%2

:monitor_loop
cls
echo.
echo ================================================================================
echo OVERNIGHT PIPELINE PROGRESS MONITOR
echo ================================================================================
echo Output Directory: %OUTPUT_DIR%
echo Last Updated: %date% %time%
echo.

REM Show checkpoint status if it exists
if exist "%OUTPUT_DIR%\checkpoint.json" (
    echo ---- CHECKPOINT STATUS ----
    python3 << PYEOF
import json
from pathlib import Path

cp_file = Path(r"%OUTPUT_DIR%") / "checkpoint.json"
if cp_file.exists():
    data = json.loads(cp_file.read_text())
    print(f"Last checkpoint: {data.get('timestamp', 'unknown')}")
    print()
    print("Dataset Status:")
    for ds in data.get('datasets', []):
        status = ds.get('status', 'unknown')
        name = ds.get('name', '?')
        error = ds.get('error', '')
        status_icon = "OK" if status == "done" else "XX" if status == "failed" else ">>"
        if error:
            error_short = (error[:40] + "...") if len(error) > 40 else error
            print(f"  [{status_icon}] {name:20s} {status:20s}")
        else:
            print(f"  [{status_icon}] {name:20s} {status:20s}")
PYEOF
    echo.
)

REM Show latest log entries
if exist "%OUTPUT_DIR%\logs" (
    echo ---- LATEST LOG ENTRIES ----
    for /f "delims=" %%a in ('dir /b /od "%OUTPUT_DIR%\logs\*.log" 2^>nul ^| findstr /v "^$"') do (
        set latest_log=%%a
    )
    if defined latest_log (
        echo Log: !latest_log!
        echo.
        REM Show last 15 lines (Windows tail equivalent)
        python3 << PYEOF
from pathlib import Path

log_file = Path(r"%OUTPUT_DIR%\logs\!latest_log!")
if log_file.exists():
    lines = log_file.read_text().splitlines()
    for line in lines[-15:]:
        print("  " + line)
PYEOF
    )
    echo.
)

echo ================================================================================
echo Monitoring interval: %INTERVAL%s ^| Press Ctrl+C to exit
echo Next update: (in %INTERVAL%s)
echo ================================================================================

timeout /t %INTERVAL% /nobreak
goto monitor_loop
