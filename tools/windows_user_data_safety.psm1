Set-StrictMode -Version Latest

function Get-NeatUserDataTreeSnapshot {
    param([Parameter(Mandatory = $true)][string]$Path)

    $root = [System.IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\', '/'))
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "NEAT user-data directory is missing: $root"
    }

    $records = @(
        foreach ($item in Get-ChildItem -LiteralPath $root -Force -Recurse -ErrorAction Stop) {
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing to snapshot a reparse point in NEAT user data: $($item.FullName)"
            }
            $relative = $item.FullName.Substring($root.Length).TrimStart([char[]]@('\', '/'))
            if ($item.PSIsContainer) {
                [pscustomobject]@{ Path = $relative; Type = 'Directory'; Length = 0; SHA256 = '' }
            } else {
                $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256 -ErrorAction Stop).Hash
                [pscustomobject]@{
                    Path = $relative
                    Type = 'File'
                    Length = $item.Length
                    SHA256 = $hash
                }
            }
        }
    )
    $records = @($records | Sort-Object -Property Path)
    return ConvertTo-Json -InputObject $records -Compress -Depth 4
}

function Remove-NeatUserDataTestTree {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$OwnerToken
    )

    if (-not (Test-Path -LiteralPath $Path)) { return }
    $directory = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $directory.PSIsContainer -or
        (($directory.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "Refusing to remove an unexpected NEAT user-data path: $Path"
    }
    $markerPath = Join-Path $Path '.issue18-test-owner'
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf) -or
        (Get-Content -LiteralPath $markerPath -Raw -ErrorAction Stop) -ne $OwnerToken) {
        throw "Refusing to remove NEAT user data without this test's ownership marker: $Path"
    }

    # Snapshot traversal rejects links/junctions before recursive removal.
    $null = Get-NeatUserDataTreeSnapshot -Path $Path
    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $Path) {
        throw "Could not remove the test-owned NEAT user-data tree: $Path"
    }
}

function New-NeatUserDataTestScope {
    param([Parameter(Mandatory = $true)][string]$LocalAppDataRoot)

    if ([string]::IsNullOrWhiteSpace($LocalAppDataRoot)) {
        throw 'LOCALAPPDATA must identify the current user data directory.'
    }
    $localRoot = [System.IO.Path]::GetFullPath($LocalAppDataRoot)
    if (-not (Test-Path -LiteralPath $localRoot -PathType Container)) {
        throw "Current user's LOCALAPPDATA directory does not exist: $localRoot"
    }

    $dataRoot = Join-Path $localRoot 'NEAT'
    $backupPath = $null
    $originalSnapshot = $null
    $originalExisted = Test-Path -LiteralPath $dataRoot
    if ($originalExisted) {
        $existing = Get-Item -LiteralPath $dataRoot -Force -ErrorAction Stop
        if (-not $existing.PSIsContainer -or
            (($existing.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "Refusing to move an unexpected or reparse-point NEAT user-data path: $dataRoot"
        }
        $originalSnapshot = Get-NeatUserDataTreeSnapshot -Path $dataRoot
        $backupPath = Join-Path $localRoot "NEAT.issue18-backup-$([guid]::NewGuid().ToString('N'))"
        if (Test-Path -LiteralPath $backupPath) {
            throw "Unique NEAT user-data backup path already exists: $backupPath"
        }

        try {
            Move-Item -LiteralPath $dataRoot -Destination $backupPath -ErrorAction Stop
            if ((Test-Path -LiteralPath $dataRoot) -or
                -not (Test-Path -LiteralPath $backupPath -PathType Container)) {
                throw "Could not safely move existing NEAT user data to backup: $backupPath"
            }
            if ((Get-NeatUserDataTreeSnapshot -Path $backupPath) -ne $originalSnapshot) {
                throw "NEAT user-data backup verification failed: $backupPath"
            }
        } catch {
            $backupFailure = $_
            if ((Test-Path -LiteralPath $backupPath) -and (Test-Path -LiteralPath $dataRoot)) {
                throw "Backup preparation failed and both the original and backup paths exist; preserve both for recovery: '$dataRoot', '$backupPath'."
            }
            if (Test-Path -LiteralPath $backupPath) {
                Move-Item -LiteralPath $backupPath -Destination $dataRoot -ErrorAction Stop
                if ((Get-NeatUserDataTreeSnapshot -Path $dataRoot) -ne $originalSnapshot) {
                    throw "Backup verification failed and original NEAT user data could not be verified after restoring it: $dataRoot"
                }
            } elseif (-not (Test-Path -LiteralPath $dataRoot)) {
                throw "Backup preparation failed and the original NEAT user data path is missing; inspect '$backupPath'."
            }
            throw $backupFailure
        }
    }

    $ownerToken = [guid]::NewGuid().ToString('N')
    $testTreeCreated = $false
    try {
        New-Item -ItemType Directory -Path $dataRoot -ErrorAction Stop | Out-Null
        $testTreeCreated = $true
        $markerPath = Join-Path $dataRoot '.issue18-test-owner'
        [System.IO.File]::WriteAllText(
            $markerPath,
            $ownerToken,
            [System.Text.UTF8Encoding]::new($false)
        )
        return [pscustomobject]@{
            LocalAppDataRoot = $localRoot
            DataRoot = $dataRoot
            BackupPath = $backupPath
            OriginalSnapshot = $originalSnapshot
            OriginalExisted = $originalExisted
            OwnerToken = $ownerToken
        }
    } catch {
        $setupFailure = $_
        if ($testTreeCreated -and (Test-Path -LiteralPath $dataRoot)) {
            $marker = Join-Path $dataRoot '.issue18-test-owner'
            if ((Test-Path -LiteralPath $marker -PathType Leaf) -and
                (Get-Content -LiteralPath $marker -Raw) -eq $ownerToken) {
                Remove-NeatUserDataTestTree -Path $dataRoot -OwnerToken $ownerToken
            } elseif (-not (Get-ChildItem -LiteralPath $dataRoot -Force -ErrorAction Stop)) {
                Remove-Item -LiteralPath $dataRoot -Force -ErrorAction Stop
            } else {
                throw "Could not initialize the test data tree safely; any original data remains backed up at '$backupPath'."
            }
        }
        if ($backupPath -and (Test-Path -LiteralPath $backupPath -PathType Container)) {
            if (Test-Path -LiteralPath $dataRoot) {
                throw "Could not restore original NEAT user data because the test path remains occupied; backup: $backupPath"
            }
            Move-Item -LiteralPath $backupPath -Destination $dataRoot -ErrorAction Stop
            if ((Get-NeatUserDataTreeSnapshot -Path $dataRoot) -ne $originalSnapshot) {
                throw "Could not verify restored NEAT user data after setup failure: $dataRoot"
            }
        }
        throw $setupFailure
    }
}

function Restore-NeatUserDataTestScope {
    param([Parameter(Mandatory = $true)][psobject]$Scope)

    Remove-NeatUserDataTestTree -Path $Scope.DataRoot -OwnerToken $Scope.OwnerToken
    if ($Scope.OriginalExisted) {
        if (-not (Test-Path -LiteralPath $Scope.BackupPath -PathType Container)) {
            throw "Pre-existing NEAT user-data backup is missing; recover it from '$($Scope.BackupPath)'."
        }
        if (Test-Path -LiteralPath $Scope.DataRoot) {
            throw "Refusing to overwrite a path that appeared during the MSI smoke; original data remains at '$($Scope.BackupPath)'."
        }
        Move-Item -LiteralPath $Scope.BackupPath -Destination $Scope.DataRoot -ErrorAction Stop
        if ((Get-NeatUserDataTreeSnapshot -Path $Scope.DataRoot) -ne $Scope.OriginalSnapshot) {
            throw "Pre-existing NEAT user data was restored but failed snapshot verification: $($Scope.DataRoot)"
        }
    } elseif (Test-Path -LiteralPath $Scope.DataRoot) {
        throw "Test-owned NEAT user data remains after cleanup: $($Scope.DataRoot)"
    }
}

Export-ModuleMember -Function New-NeatUserDataTestScope, Restore-NeatUserDataTestScope
