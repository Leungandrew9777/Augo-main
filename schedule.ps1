# schedule.ps1 — register (or remove) the Augo scheduled tasks.
#
# Usage:
#   .\schedule.ps1                  # register AugoWeekly (Fri 17:00) + AugoDaily (09:00)
#   .\schedule.ps1 -DryRun          # show what would be registered (no changes)
#   .\schedule.ps1 -Remove          # unregister both tasks
#   .\schedule.ps1 -WeeklyAt "18:00" -DailyAt "08:30"

param(
    [switch]$DryRun,
    [switch]$Remove,
    [string]$WeeklyAt = "17:00",
    [string]$DailyAt = "09:00"
)

$proj = $PSScriptRoot
$taskNames = @("AugoWeekly", "AugoDaily")

if ($Remove) {
    foreach ($n in $taskNames) {
        if ($DryRun) { Write-Host "Would unregister task: $n" -ForegroundColor DarkGray; continue }
        Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Unregistered: $n"
    }
    return
}

$actionArgs = @{
    Daily  = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
               "-File", "`"$proj\scheduled.ps1`"", "-Mode", "daily")
    Weekly = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
               "-File", "`"$proj\scheduled.ps1`"", "-Mode", "weekly")
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew

if ($DryRun) {
    Write-Host "Would register:" -ForegroundColor Cyan
    Write-Host "  AugoDaily  @ $DailyAt  -> powershell $($actionArgs.Daily -join ' ')"
    Write-Host "  AugoWeekly @ $WeeklyAt  -> powershell $($actionArgs.Weekly -join ' ')"
    Write-Host "  (logs -> $proj\logs\scheduled_*.log)"
    return
}

Register-ScheduledTask -TaskName "AugoDaily" -Force -TaskPath "\Augo" `
    -Action (New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($actionArgs.Daily -join " ")) `
    -Trigger (New-ScheduledTaskTrigger -Daily -At $DailyAt) `
    -Settings $settings -Description "Augo: sync results + grading summary"
Write-Host "Registered AugoDaily @ $DailyAt"

Register-ScheduledTask -TaskName "AugoWeekly" -Force -TaskPath "\Augo" `
    -Action (New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($actionArgs.Weekly -join " ")) `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At $WeeklyAt) `
    -Settings $settings -Description "Augo: sync fixtures, run predictions, notify, publish"
Write-Host "Registered AugoWeekly @ $WeeklyAt (Friday)"

Write-Host "Verify with:  Get-ScheduledTask -TaskPath '\Augo'"
