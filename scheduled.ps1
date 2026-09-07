# scheduled.ps1 — non-interactive runner invoked by Windows Task Scheduler.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scheduled.ps1 -Mode weekly
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scheduled.ps1 -Mode daily
#
# weekly: sync data -> predict -> evaluate -> notify -> publish (push)
# daily : sync results -> evaluate -> notify (grading summary)

param(
    [ValidateSet("weekly", "daily")]
    [string]$Mode = "weekly"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir "scheduled_$Mode.log"
Start-Transcript -Path $log -Force | Out-Null

try {
    Write-Host "== Augo scheduled run ($Mode) ==" -ForegroundColor Cyan
    if ($Mode -eq "daily") {
        python sync.py results --days 2
        python evaluate.py --json
        python notify.py --results
    } else {
        python sync.py all
        python run_pipeline.py
        python evaluate.py --json
        python notify.py
        python publish.py --push
    }
    Write-Host "== done ==" -ForegroundColor Green
} catch {
    Write-Host "== FAILED: $_ ==" -ForegroundColor Red
    python notify.py --error $_.Exception.Message
    throw
} finally {
    Stop-Transcript | Out-Null
}
