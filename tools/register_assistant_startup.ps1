param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$taskName = "NEAT Shared Assistant Server"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed the NEAT assistant startup task."
    exit 0
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$launcher = Join-Path $PSScriptRoot "start_assistant_server.ps1"

if (-not (Test-Path -LiteralPath $python)) {
    throw "The NEAT virtual environment was not found at $python"
}
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "The NEAT assistant launcher was not found at $launcher"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`"" `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT30S"
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Starts the locally hosted NEAT AI assistant after Windows sign-in." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $taskName
Write-Output "Registered and started the NEAT assistant startup task."
Write-Output "Health check: http://127.0.0.1:8765/health"
Write-Output "Startup log: $env:LOCALAPPDATA\NEAT\logs\assistant_shared_server.log"
