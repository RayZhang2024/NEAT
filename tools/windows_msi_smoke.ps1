param(
    [Parameter(Mandatory = $true)][string]$MsiPath,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [Parameter(Mandatory = $true)][string]$LogDirectory,
    [string]$InstallRoot
)

$ErrorActionPreference = "Stop"
$safetyModule = Join-Path $PSScriptRoot "windows_user_data_safety.psm1"
Import-Module -Name $safetyModule -Force -ErrorAction Stop
$contextModule = Join-Path $PSScriptRoot "windows_msi_context.psm1"
Import-Module -Name $contextModule -Force -ErrorAction Stop
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

function Get-NeatShortcutTargets([string]$ProgramsPath) {
    $programs = $ProgramsPath
    if (-not (Test-Path $programs)) { return @() }
    $shell = New-Object -ComObject WScript.Shell
    Get-ChildItem -LiteralPath $programs -Filter "*.lnk" -File -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object {
            $shortcut = $shell.CreateShortcut($_.FullName)
            [pscustomobject]@{ Path = $_.FullName; Name = $_.BaseName; Target = $shortcut.TargetPath }
        }
}

$identity = Get-NeatMsiIdentity -MsiPath $msi
Write-Output "MSI ProductCode: $($identity.ProductCode)"
Write-Output "MSI UpgradeCode: $($identity.UpgradeCode)"
$initialInstances = @(Get-NeatMsiProductInstances -ProductCode $identity.ProductCode)
if ($initialInstances.Count -ne 0) {
    $existing = ($initialInstances | ForEach-Object { "$($_.Context):$($_.Sid)" }) -join "; "
    throw "The MSI ProductCode is already registered before installation: $existing"
}
Write-Output "Initial MSI ProductCode context: not registered"
$initialArpEntries = @(Get-NeatUninstallEntries)
Write-Output "Initial ARP registrations (diagnostic only): $($initialArpEntries.Count)"
foreach ($entry in $initialArpEntries) { Write-Output "ARP diagnostic: $($entry.Hive):$($entry.Key)" }
if (Test-Path $installRoot) {
    throw "The deterministic test install location already exists: $installRoot"
}

$userDataScope = New-NeatUserDataTestScope -LocalAppDataRoot $env:LOCALAPPDATA
$neatUserData = $userDataScope.DataRoot

try {
    $expectedUserDataRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "NEAT"))
    if ($neatUserData -ine $expectedUserDataRoot) {
        throw "MSI preservation test must use the current user's real NEAT data root: $expectedUserDataRoot"
    }
    Write-Output "Real per-user NEAT data root under preservation test: $neatUserData"
    $settingsPath = Join-Path $neatUserData "assistant_settings.json"
    $cachePath = Join-Path $neatUserData "assistant_cache\chroma\preserve-sentinel.txt"
    New-Item -ItemType Directory -Path (Split-Path $settingsPath), (Split-Path $cachePath) -Force | Out-Null
    [System.IO.File]::WriteAllText($settingsPath, '{"issue18":"preserve"}', [System.Text.Encoding]::UTF8)
    [System.IO.File]::WriteAllText($cachePath, "preserve user cache", [System.Text.Encoding]::UTF8)
    $settingsHash = (Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash
    $cacheHash = (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash

    Write-Output "Requested MSI properties: ALLUSERS=2; MSIINSTALLPERUSER=1; INSTALLFOLDER=$installRoot"
    Invoke-Msi @("/i", $msi, "/qn", "/norestart", "/l*v", $installLog, "ALLUSERS=2", "MSIINSTALLPERUSER=1", "INSTALLFOLDER=$installRoot") "MSI install"
    $installed = $true
    if (-not (Test-Path -LiteralPath $installedExe -PathType Leaf)) {
        throw "MSI completed but the expected installed executable was not found: $installedExe"
    }

    python -m tools.windows_distribution verify-installed --payload $installRoot --manifest $manifest
    if ($LASTEXITCODE -ne 0) { throw "Installed payload does not match its manifest." }

    $instances = @(Get-NeatMsiProductInstances -ProductCode $identity.ProductCode)
    $installedInstance = Resolve-NeatMsiProductContext -ProductCode $identity.ProductCode -Instances $instances
    Write-Output "Installed MSI context: $($installedInstance.Context); SID=$($installedInstance.Sid)"
    $registryEntries = @(Get-NeatUninstallEntries)
    Write-Output "Installed ARP registrations (diagnostic only): $($registryEntries.Count)"
    foreach ($entry in $registryEntries) { Write-Output "ARP diagnostic: $($entry.Hive):$($entry.Key)" }

    $userPrograms = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
    $allUsersPrograms = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs"
    $userShortcuts = @(Get-NeatShortcutTargets -ProgramsPath $userPrograms)
    $allUsersShortcuts = @(Get-NeatShortcutTargets -ProgramsPath $allUsersPrograms)
    if (-not (Test-NeatInstalledUserShortcut -Shortcuts $userShortcuts -InstalledExe $installedExe)) {
        throw "The NEAT Start Menu launcher does not target $installedExe"
    }
    $equivalentAllUsers = @($allUsersShortcuts | Where-Object {
        Test-NeatEquivalentStartMenuShortcut -Shortcut $_ -InstalledExe $installedExe
    })
    if ($equivalentAllUsers.Count -gt 0) {
        $paths = ($equivalentAllUsers | ForEach-Object { $_.Path }) -join "; "
        throw "An equivalent all-users NEAT Start Menu launcher exists: $paths"
    }
    Write-Output "Start Menu verification: per-user launcher resolves to installed NEAT.exe; no all-users NEAT launcher."

    $previousSmokeResult = $env:NEAT_RELEASE_SMOKE_RESULT
    $env:NEAT_RELEASE_SMOKE_RESULT = $smokeResult
    try {
        $app = Start-Process -FilePath $installedExe -ArgumentList @("--release-smoke-test") `
            -Wait -PassThru -WindowStyle Hidden
    } finally {
        $env:NEAT_RELEASE_SMOKE_RESULT = $previousSmokeResult
    }
    if ($app.ExitCode -ne 0) { throw "Installed NEAT smoke test returned $($app.ExitCode), expected 0." }
    if (-not (Test-Path -LiteralPath $smokeResult)) { throw "Installed release smoke did not produce $smokeResult" }
    $smokeText = Get-Content -LiteralPath $smokeResult -Raw
    foreach ($marker in @("OK provider adapters", "OK knowledge sections=", "OK BM25 fallback", "OK automatic shared access configured", "PASS")) {
        if (-not $smokeText.Contains($marker)) { throw "Installed smoke result is missing required marker: $marker" }
    }

    Write-Output "Requested MSI uninstall properties: ALLUSERS=2; MSIINSTALLPERUSER=1"
    Invoke-Msi @("/x", $msi, "/qn", "/norestart", "/l*v", $uninstallLog, "ALLUSERS=2", "MSIINSTALLPERUSER=1") "MSI uninstall"
    $uninstalled = $true
    if (Test-Path -LiteralPath $installedExe) { throw "Uninstall left the application executable behind: $installedExe" }
    if (Test-NeatInstalledUserShortcut -Shortcuts @(Get-NeatShortcutTargets -ProgramsPath $userPrograms) -InstalledExe $installedExe) {
        throw "Uninstall left the NEAT Start Menu launcher behind."
    }
    $remainingAllUsersShortcuts = @(Get-NeatShortcutTargets -ProgramsPath $allUsersPrograms |
        Where-Object { Test-NeatEquivalentStartMenuShortcut -Shortcut $_ -InstalledExe $installedExe })
    if ($remainingAllUsersShortcuts.Count -gt 0) {
        $paths = ($remainingAllUsersShortcuts | ForEach-Object { $_.Path }) -join "; "
        throw "An all-users NEAT Start Menu launcher remains after uninstall: $paths"
    }
    $remainingInstances = @(Get-NeatMsiProductInstances -ProductCode $identity.ProductCode)
    if ($remainingInstances.Count -ne 0) {
        $remaining = ($remainingInstances | ForEach-Object { "$($_.Context):$($_.Sid)" }) -join "; "
        throw "Uninstall left ProductCode $($identity.ProductCode) registered: $remaining"
    }
    Write-Output "Post-uninstall MSI ProductCode context: not registered"
    $remainingArp = @(Get-NeatUninstallEntries)
    Write-Output "Post-uninstall ARP registrations (diagnostic only): $($remainingArp.Count)"
    foreach ($entry in $remainingArp) { Write-Output "ARP diagnostic: $($entry.Hive):$($entry.Key)" }
    if (-not (Test-Path -LiteralPath $settingsPath -PathType Leaf) -or
        (Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash -ne $settingsHash) {
        throw "Uninstall removed or changed the per-user assistant settings sentinel."
    }
    if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf) -or
        (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash -ne $cacheHash) {
        throw "Uninstall removed or changed the per-user assistant cache sentinel."
    }
    if (Test-Path -LiteralPath $installRoot) {
        $leftover = Get-ChildItem -LiteralPath $installRoot -Force -Recurse -ErrorAction SilentlyContinue
        if ($leftover) { throw "Uninstall left application payload files under $installRoot" }
    }
    Write-Output "Verified real user-data root remained in place through uninstall: $neatUserData"
    Write-Output "Preserved assistant settings SHA-256: $settingsHash"
    Write-Output "Preserved assistant cache sentinel SHA-256: $cacheHash"
    Write-Output "MSI install, manifest comparison, installed smoke, Start Menu, uninstall, and user-data preservation passed."
} finally {
    try {
        if ($installed -and -not $uninstalled) {
            $cleanupLog = Join-Path $logs "cleanup-uninstall.log"
            $cleanup = Start-Process -FilePath "msiexec.exe" `
                -ArgumentList @("/x", $msi, "/qn", "/norestart", "/l*v", $cleanupLog, "ALLUSERS=2", "MSIINSTALLPERUSER=1") `
                -Wait -PassThru -WindowStyle Hidden
            if ($cleanup.ExitCode -ne 0) {
                Write-Warning "Best-effort cleanup uninstall returned $($cleanup.ExitCode); see $cleanupLog"
            }
        }
    } finally {
        Restore-NeatUserDataTestScope -Scope $userDataScope
        Write-Output "Restored original per-user NEAT data state after the MSI preservation test."
    }
}
