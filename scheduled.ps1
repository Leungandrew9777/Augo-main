# scheduled.ps1 — non-interactive runner invoked by Windows Task Scheduler.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scheduled.ps1 -Mode weekly
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scheduled.ps1 -Mode daily
#
# weekly: sync data -> refresh history + features -> predict -> evaluate -> notify -> publish (push)
# daily : sync fixtures+results -> evaluate -> notify (grading summary)
#
# Model retraining (train_ensemble.py) is intentionally NOT part of this script —
# it stays a manual, roughly monthly step run at the developer's discretion.

param(
    [ValidateSet("weekly", "daily")]
    [string]$Mode = "weekly"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:AUGO_NON_INTERACTIVE = "1"   # never prompt for GW in headless runs
$env:AUGO_LOCAL_DATA = "1"        # grade/publish from local files, never stale remote data
$env:PYTHONUTF8 = "1"             # emoji/Unicode prints crash under cp1252 consoles

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir "scheduled_$Mode.log"
Start-Transcript -Path $log -Force | Out-Null

# Native commands do not throw on non-zero exit codes in PowerShell, so an
# unchecked `python ...` failure would silently run the remaining steps and
# report success. Throw explicitly instead so the catch block fires.
function Invoke-Py {
    param([string[]]$PyArgs)
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "python $($PyArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-Host "== Augo scheduled run ($Mode) ==" -ForegroundColor Cyan
    if ($Mode -eq "daily") {
        # sync.py all (not just results) so a missed weekly run self-heals the
        # next morning instead of leaving the upcoming GW out of fixtures.csv.
        Invoke-Py @("sync.py", "all")
        Invoke-Py @("data_quality.py", "--data")   # hard gate: abort on bad fixtures/results
        Invoke-Py @("evaluate.py", "--json")
        Invoke-Py @("notify.py", "--results")
        Invoke-Py @("publish.py", "--push")   # push results/eval to GitHub raw so the deployed app updates
    } else {
        Invoke-Py @("sync.py", "all")
        Invoke-Py @("data_quality.py", "--data")   # hard gate before touching history/features
        # Re-download the season history so ELO/rolling stats fold in the
        # finished GW, then rebuild the features served at inference time.
        Invoke-Py @("data_pipeline.py")
        Invoke-Py @("feature_engineering.py")
        Invoke-Py @("data_quality.py", "--history", "--artifacts")   # gate + prediction smoke test
        Invoke-Py @("run_pipeline.py")
        Invoke-Py @("evaluate.py", "--json")
        Invoke-Py @("notify.py")
        Invoke-Py @("publish.py", "--push")
    }
    Write-Host "== done ==" -ForegroundColor Green
} catch {
    Write-Host "== FAILED: $_ ==" -ForegroundColor Red
    & python notify.py --error "$($_.Exception.Message)"
    throw
} finally {
    Stop-Transcript | Out-Null
}
