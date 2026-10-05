param(
    [Parameter(Mandatory = $true)][string]$MsiPath,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [Parameter(Mandatory = $true)][string]$LogDirectory,
    [string]$InstallRoot
)

$ErrorActionPreference = "Stop"
$msi = (Resolve-Path -LiteralPath $MsiPath).Path
$manifest = (Resolve-Path -LiteralPath $ManifestPath).Path
$logs = [System.IO.Path]::GetFullPath($LogDirectory)
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$tempRoot = if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    [System.IO.Path]::GetTempPath()
} else {
    $env:RUNNER_TEMP
}
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Join-Path $tempRoot "NEAT-installed"
}
$installLog = Join-Path $logs "install.log"
$uninstallLog = Join-Path $logs "uninstall.log"
$smokeResult = Join-Path $logs "installed-release-smoke.txt"
$installRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$installedExe = Join-Path $installRoot "NEAT.exe"
$installed = $false
$uninstalled = $false

function Invoke-Msi([string[]]$Arguments, [string]$Operation) {
    $process = Start-Process -FilePath "msiexec.exe" -ArgumentList $Arguments `
        -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "$Operation returned Windows Installer exit code $($process.ExitCode); expected 0. See $installLog and $uninstallLog."
    }
}

function Get-NeatUninstallEntries {
    $uninstallRoots = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    )
    foreach ($root in $uninstallRoots) {
        if (Test-Path $root) {
            Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
                $entry = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
                if ($entry.DisplayName -eq "NEAT") {
                    [pscustomobject]@{ Hive = $_.PSDrive.Name; Key = $_.PSPath; Entry = $entry }
                }
            }
        }
    }
}

function Get-NeatShortcutTargets {
    $programs = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
    if (-not (Test-Path $programs)) { return @() }
    $shell = New-Object -ComObject WScript.Shell
    Get-ChildItem -LiteralPath $programs -Filter "*.lnk" -File -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { $shell.CreateShortcut($_.FullName).TargetPath }
}

if (Get-NeatUninstallEntries) {
    throw "A NEAT install is already registered; the MSI lifecycle test requires a clean runner."
}
if (Test-Path $installRoot) {
    throw "The deterministic test install location already exists: $installRoot"
}

$userDataRoot = Join-Path $tempRoot "issue18-user-data"
$neatUserData = Join-Path $userDataRoot "NEAT"
$settingsPath = Join-Path $neatUserData "assistant_settings.json"
$cachePath = Join-Path $neatUserData "assistant_cache\chroma\preserve-sentinel.txt"
New-Item -ItemType Directory -Path (Split-Path $settingsPath), (Split-Path $cachePath) -Force | Out-Null
[System.IO.File]::WriteAllText($settingsPath, '{"issue18":"preserve"}', [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($cachePath, "preserve user cache", [System.Text.Encoding]::UTF8)
$settingsHash = (Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash
$cacheHash = (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash

try {
    Invoke-Msi @("/i", $msi, "/qn", "/norestart", "/l*v", $installLog, "ALLUSERS=2", "MSIINSTALLPERUSER=1", "INSTALLFOLDER=$installRoot") "MSI install"
    $installed = $true
    if (-not (Test-Path -LiteralPath $installedExe -PathType Leaf)) {
        throw "MSI completed but the expected installed executable was not found: $installedExe"
    }

    python -m tools.windows_distribution verify-installed --payload $installRoot --manifest $manifest
    if ($LASTEXITCODE -ne 0) { throw "Installed payload does not match its manifest." }

    $registryEntries = @(Get-NeatUninstallEntries)
    if ($registryEntries.Count -ne 1 -or $registryEntries[0].Hive -ne "HKCU") {
        $registrations = ($registryEntries | ForEach-Object { "$($_.Hive):$($_.Key)" }) -join "; "
        throw "Expected exactly one per-user NEAT uninstall registration; found $($registryEntries.Count): $registrations"
    }
    if (-not ((Get-NeatShortcutTargets) -contains $installedExe)) {
        throw "The NEAT Start Menu launcher does not target $installedExe"
    }

    $previousLocalAppData = $env:LOCALAPPDATA
    $previousSmokeResult = $env:NEAT_RELEASE_SMOKE_RESULT
    $env:LOCALAPPDATA = $userDataRoot
    $env:NEAT_RELEASE_SMOKE_RESULT = $smokeResult
    try {
        $app = Start-Process -FilePath $installedExe -ArgumentList @("--release-smoke-test") `
            -Wait -PassThru -WindowStyle Hidden
    } finally {
        $env:LOCALAPPDATA = $previousLocalAppData
        $env:NEAT_RELEASE_SMOKE_RESULT = $previousSmokeResult
    }
    if ($app.ExitCode -ne 0) { throw "Installed NEAT smoke test returned $($app.ExitCode), expected 0." }
    if (-not (Test-Path -LiteralPath $smokeResult)) { throw "Installed release smoke did not produce $smokeResult" }
    $smokeText = Get-Content -LiteralPath $smokeResult -Raw
    foreach ($marker in @("OK provider adapters", "OK knowledge sections=", "OK BM25 fallback", "OK automatic shared access configured", "PASS")) {
        if (-not $smokeText.Contains($marker)) { throw "Installed smoke result is missing required marker: $marker" }
    }

    Invoke-Msi @("/x", $msi, "/qn", "/norestart", "/l*v", $uninstallLog, "ALLUSERS=2", "MSIINSTALLPERUSER=1") "MSI uninstall"
    $uninstalled = $true
    if (Test-Path -LiteralPath $installedExe) { throw "Uninstall left the application executable behind: $installedExe" }
    if (Get-NeatShortcutTargets | Where-Object { $_ -eq $installedExe }) {
        throw "Uninstall left the NEAT Start Menu launcher behind."
    }
    if (Get-NeatUninstallEntries) { throw "Uninstall left the NEAT installed-app registration behind." }
    if ((Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash -ne $settingsHash) {
        throw "Uninstall removed or changed the per-user assistant settings sentinel."
    }
    if ((Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash -ne $cacheHash) {
        throw "Uninstall removed or changed the per-user assistant cache sentinel."
    }
    if (Test-Path -LiteralPath $installRoot) {
        $leftover = Get-ChildItem -LiteralPath $installRoot -Force -Recurse -ErrorAction SilentlyContinue
        if ($leftover) { throw "Uninstall left application payload files under $installRoot" }
    }
    Write-Output "MSI install, manifest comparison, installed smoke, Start Menu, uninstall, and user-data preservation passed."
} finally {
    if ($installed -and -not $uninstalled) {
        $cleanupLog = Join-Path $logs "cleanup-uninstall.log"
        $cleanup = Start-Process -FilePath "msiexec.exe" `
            -ArgumentList @("/x", $msi, "/qn", "/norestart", "/l*v", $cleanupLog, "ALLUSERS=2", "MSIINSTALLPERUSER=1") `
            -Wait -PassThru -WindowStyle Hidden
        if ($cleanup.ExitCode -ne 0) {
            Write-Warning "Best-effort cleanup uninstall returned $($cleanup.ExitCode); see $cleanupLog"
        }
    }
}
