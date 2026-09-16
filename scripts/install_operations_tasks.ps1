param(
    [ValidateSet("testnet", "demo", "live")]
    [string]$Environment = "testnet"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$RecoveryTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)

function Register-AIQuantTask {
    param([string]$Name, [string]$Arguments, $Trigger)
    $Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $ProjectRoot
    Register-ScheduledTask -TaskName $Name -Action $Action -Trigger $Trigger `
        -Principal $Principal -Settings $Settings -Force | Out-Null
}

$BackupArgs = 'scripts\database_admin.py --env-file .env backup'
Register-AIQuantTask "AIQuant-DailyBackup" $BackupArgs (New-ScheduledTaskTrigger -Daily -At "03:30")

$Monthly = New-ScheduledTaskTrigger -Daily -At "04:30"
# The trigger runs daily, while Python performs the drill only on day one.
$RestoreWrapper = "-c `"import datetime,runpy,sys; sys.argv=['database_admin.py','--env-file','.env','restore-latest']; datetime.date.today().day==1 and runpy.run_path('scripts/database_admin.py',run_name='__main__')`""
Register-AIQuantTask "AIQuant-MonthlyRestoreDrill" $RestoreWrapper $Monthly

$SupervisorArgs = "-m ai_quant_trading.operations.supervisor --project-root `"$ProjectRoot`" --interval 30"
Register-AIQuantTask "AIQuant-Testnet-Supervisor" $SupervisorArgs @(
    (New-ScheduledTaskTrigger -AtLogOn -User $UserId),
    $RecoveryTrigger
)

$WatchdogArgs = "-m ai_quant_trading.operations.watchdog --project-root `"$ProjectRoot`" --environment $Environment --host 127.0.0.1 --only-when-armed --startup-grace-seconds 180"
Register-AIQuantTask "AIQuant-Watchdog" $WatchdogArgs @(
    (New-ScheduledTaskTrigger -AtLogOn -User $UserId),
    $RecoveryTrigger
)

Start-ScheduledTask -TaskName "AIQuant-Testnet-Supervisor"
Start-ScheduledTask -TaskName "AIQuant-Watchdog"

Write-Host "Installed AI Quant backup, restore drill, Testnet supervisor, and watchdog tasks."
