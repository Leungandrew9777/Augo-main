# weekly.ps1 — one-command weekly refresh: sync data -> rebuild features -> predictions -> launch UI.
#
# Usage:
#   .\weekly.ps1                 # sync, rebuild features, run pipeline, launch Reflex
#   .\weekly.ps1 -NoSync         # skip data sync + feature rebuild, just re-predict + launch
#   .\weekly.ps1 -Gw 33          # force a specific gameweek for the pipeline
#
# Retraining (train_ensemble.py) is deliberately not here — run it manually
# (roughly monthly) as described in README.md.
#
# Note: if your PowerShell execution policy blocks scripts, run:
#   powershell -ExecutionPolicy Bypass -File .\weekly.ps1

param(
    [switch]$NoSync,
    [int]$Gw = 0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:AUGO_LOCAL_DATA = "1"        # use freshly written local files, never the stale published snapshot

Write-Host "== Augo weekly refresh ==" -ForegroundColor Cyan

if (-not $NoSync) {
    Write-Host "[1/4] Syncing fixtures + results from The Odds API ..." -ForegroundColor Cyan
    python sync.py all
    if ($LASTEXITCODE -ne 0) { throw "sync.py failed (exit $LASTEXITCODE)" }

    Write-Host "[2/4] Rebuilding ELO / rolling form / H2H features ..." -ForegroundColor Cyan
    python data_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw "data_pipeline.py failed (exit $LASTEXITCODE)" }
    python feature_engineering.py
    if ($LASTEXITCODE -ne 0) { throw "feature_engineering.py failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "[1/4] Skipping data sync (-NoSync)." -ForegroundColor DarkGray
    Write-Host "[2/4] Skipping feature rebuild (-NoSync)." -ForegroundColor DarkGray
}

Write-Host "[3/4] Running prediction pipeline ..." -ForegroundColor Cyan
if ($Gw -gt 0) {
    python run_pipeline.py --launch --gw $Gw
} else {
    python run_pipeline.py --launch
}
if ($LASTEXITCODE -ne 0) { throw "run_pipeline.py failed (exit $LASTEXITCODE)" }

Write-Host "[4/4] Done. Reflex UI is starting (close the window to stop)." -ForegroundColor Green
