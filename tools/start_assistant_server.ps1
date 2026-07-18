param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $env:LOCALAPPDATA "NEAT\logs"
$logPath = Join-Path $logDirectory "assistant_shared_server.log"
$stdoutPath = Join-Path $logDirectory "assistant_shared_server.stdout.log"
$stderrPath = Join-Path $logDirectory "assistant_shared_server.stderr.log"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Write-ServerLog([string]$Message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $logPath -Value "[$timestamp] $Message" -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $python)) {
    Write-ServerLog "ERROR: The NEAT virtual environment was not found at $python"
    exit 2
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = "1"

function Test-ServerHealth {
    try {
        $response = Invoke-WebRequest `
            -Uri "http://127.0.0.1:8765/health" `
            -UseBasicParsing `
            -TimeoutSec 4
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-ServerProcess {
    Write-ServerLog "Starting NEAT shared assistant server."
    try {
        # Uvicorn writes normal INFO startup messages to stderr. Redirect the
        # native streams directly so PowerShell does not turn them into error
        # records. Do not wait on the Windows Store Python launcher: it can exit
        # after handing execution to the real interpreter while the server is
        # still healthy.
        $process = Start-Process `
            -FilePath $python `
            -ArgumentList "-m", "tools.assistant_shared_server" `
            -WorkingDirectory $projectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
        Write-ServerLog "Python launcher started with PID $($process.Id)."
    } catch {
        Write-ServerLog "ERROR: $($_.Exception.Message)"
    }
}

Write-ServerLog "Health supervisor started."
$lastStart = [DateTime]::MinValue

while ($true) {
    if (-not (Test-ServerHealth)) {
        $secondsSinceStart = ((Get-Date) - $lastStart).TotalSeconds
        if ($secondsSinceStart -ge 45) {
            Start-ServerProcess
            $lastStart = Get-Date
        }
    }
    Start-Sleep -Seconds 10
}
