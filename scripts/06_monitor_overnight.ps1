# RUN ON: LOCAL MACHINE (Windows)
# Monitor overnight pipeline progress

param(
    [string]$OutputDir = ".overnight_output",
    [int]$MonitorInterval = 60
)

if (-not (Test-Path $OutputDir -PathType Container)) {
    Write-Host "ERROR: Output directory not found: $OutputDir" -ForegroundColor Red
    exit 1
}

function ShowStatus {
    Clear-Host
    
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║        OVERNIGHT PIPELINE PROGRESS MONITOR                    ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Output Directory: $OutputDir"
    Write-Host "Last Updated: $(Get-Date)"
    Write-Host ""
    
    # Check for checkpoint
    $checkpoint_file = Join-Path $OutputDir "checkpoint.json"
    if (Test-Path $checkpoint_file) {
        Write-Host "━━ CHECKPOINT STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
        
        try {
            $checkpoint = Get-Content $checkpoint_file | ConvertFrom-Json
            Write-Host "Last checkpoint: $($checkpoint.timestamp)"
            Write-Host ""
            Write-Host "Dataset Status:"
            
            foreach ($ds in $checkpoint.datasets) {
                $status = $ds.status
                $name = $ds.name
                $error = $ds.error
                
                $status_icon = switch ($status) {
                    "done" { "✓" }
                    "failed" { "✗" }
                    default { "▸" }
                }
                
                $error_str = if ($error) { "  ERROR: $($error.Substring(0, [Math]::Min(50, $error.Length)))" } else { "" }
                
                Write-Host "  $status_icon $($name.PadRight(20)) $($status.PadRight(20)) $error_str"
            }
        }
        catch {
            Write-Host "  (Could not parse checkpoint)"
        }
        
        Write-Host ""
    }
    
    # Show log summary
    $log_dir = Join-Path $OutputDir "logs"
    if (Test-Path $log_dir -PathType Container) {
        Write-Host "━━ LATEST LOG ENTRIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
        
        $latest_log = Get-ChildItem "$log_dir/*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($latest_log) {
            Write-Host "Log: $($latest_log.Name)"
            Write-Host ""
            
            $log_content = Get-Content $latest_log.FullName -Tail 20
            foreach ($line in $log_content) {
                Write-Host "  $line"
            }
        }
    }
    
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "Monitoring interval: ${MonitorInterval}s | Press Ctrl+C to exit" -ForegroundColor Gray
    Write-Host "Next update: $(Get-Date).AddSeconds($MonitorInterval)" -ForegroundColor Gray
}

# Initial display
ShowStatus

# Monitor loop
while ($true) {
    Start-Sleep -Seconds $MonitorInterval
    ShowStatus
}
