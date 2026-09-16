param(
    [switch]$Uninstall,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$RuntimeRoot = $PSScriptRoot
$Executable = Join-Path $RuntimeRoot "AIQuantTradingSystem.exe"
$SupervisorTask = "AIQuant-Testnet-Supervisor"
$WatchdogTask = "AIQuant-Watchdog"

if ($Uninstall) {
    foreach ($Name in @($SupervisorTask, $WatchdogTask)) {
        Stop-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
    }
    Write-Host "Removed Testnet Supervisor and independent Watchdog tasks."
    exit 0
}

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Desktop executable not found: $Executable"
}

$UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal `
    -UserId $UserId `
    -LogonType Interactive `
    -RunLevel Limited
$LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
# 每分鐘重新觸發；IgnoreNew 讓健康程序不重複，程序死亡時則能在一分鐘內恢復。
$RecoveryTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
$RuntimeTriggers = @($LogonTrigger, $RecoveryTrigger)
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

function Register-AIQuantRuntimeTask {
    param(
        [string]$Name,
        [string]$Arguments
    )
    $Action = New-ScheduledTaskAction `
        -Execute $Executable `
        -Argument $Arguments `
        -WorkingDirectory $RuntimeRoot
    Register-ScheduledTask `
        -TaskName $Name `
        -Action $Action `
        -Trigger $RuntimeTriggers `
        -Principal $Principal `
        -Settings $Settings `
        -Force | Out-Null
}

Register-AIQuantRuntimeTask `
    -Name $SupervisorTask `
    -Arguments "--testnet-supervisor --interval 30"
Register-AIQuantRuntimeTask `
    -Name $WatchdogTask `
    -Arguments "--watchdog-worker --environment testnet --interval 30 --startup-grace-seconds 180 --port 9108"

if (-not $NoStart) {
    Start-ScheduledTask -TaskName $SupervisorTask
    Start-ScheduledTask -TaskName $WatchdogTask
}

Write-Host "Installed Testnet Supervisor and independent Watchdog tasks."
Write-Host "The Supervisor stays idle until Testnet auto-resume is enabled in the App."
