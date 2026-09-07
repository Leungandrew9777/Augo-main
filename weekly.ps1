# weekly.ps1 — one-command weekly refresh: sync data -> predictions -> launch UI.
#
# Usage:
#   .\weekly.ps1                 # sync fixtures+results, run pipeline, launch Reflex
#   .\weekly.ps1 -NoSync         # skip data sync, just re-predict + launch
#   .\weekly.ps1 -Gw 33          # force a specific gameweek for the pipeline
#
# Note: if your PowerShell execution policy blocks scripts, run:
#   powershell -ExecutionPolicy Bypass -File .\weekly.ps1

param(
    [switch]$NoSync,
    [int]$Gw = 0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "== Augo weekly refresh ==" -ForegroundColor Cyan

if (-not $NoSync) {
    Write-Host "[1/3] Syncing fixtures + results from The Odds API ..." -ForegroundColor Cyan
    python sync.py all
    if ($LASTEXITCODE -ne 0) { throw "sync.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "[1/3] Skipping data sync (-NoSync)." -ForegroundColor DarkGray
}

Write-Host "[2/3] Running prediction pipeline ..." -ForegroundColor Cyan
if ($Gw -gt 0) {
    python run_pipeline.py --launch --gw $Gw
} else {
    python run_pipeline.py --launch
}
if ($LASTEXITCODE -ne 0) { throw "run_pipeline.py failed (exit $LASTEXITCODE)" }

Write-Host "[3/3] Done. Reflex UI is starting (close the window to stop)." -ForegroundColor Green
