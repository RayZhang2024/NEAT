$ErrorActionPreference = "Stop"

if (-not ("NeatIssue18.WindowsInstallerNative" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NeatIssue18 {
    public static class WindowsInstallerNative {
        [DllImport("msi.dll", EntryPoint = "MsiEnumProductsExW", CharSet = CharSet.Unicode)]
        public static extern uint MsiEnumProductsEx(
            string productCode,
            IntPtr userSid,
            uint context,
            uint index,
            StringBuilder installedProductCode,
            out uint installedContext,
            StringBuilder sid,
            ref uint sidLength);
    }
}
'@ | Out-Null
}

function Get-NeatMsiIdentity {
    param([Parameter(Mandatory = $true)][string]$MsiPath)

    $resolved = (Resolve-Path -LiteralPath $MsiPath).Path
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $null
    try {
        $database = $installer.OpenDatabase($resolved, 0)
        $values = @{}
        foreach ($property in @("ProductCode", "UpgradeCode")) {
            $view = $database.OpenView("SELECT `Value` FROM `Property` WHERE `Property`='$property'")
            try {
                $view.Execute()
                $record = $view.Fetch()
                if ($null -eq $record) { throw "MSI has no $property property: $resolved" }
                $values[$property] = [string]$record.StringData(1)
            } finally {
                if ($null -ne $view) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($view) }
                if ($null -ne $record) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($record) }
            }
        }
        foreach ($property in @("ProductCode", "UpgradeCode")) {
            $parsed = [guid]::Empty
            if (-not [guid]::TryParse($values[$property], [ref]$parsed)) {
                throw "MSI $property is missing or is not a GUID: '$($values[$property])'"
            }
            $values[$property] = $parsed.ToString("B").ToUpperInvariant()
        }
        return [pscustomobject]$values
    } finally {
        if ($null -ne $database) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($database) }
        if ($null -ne $installer) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($installer) }
    }
}

function Get-NeatMsiProductInstances {
    param([Parameter(Mandatory = $true)][string]$ProductCode)

    $parsed = [guid]::Empty
    if (-not [guid]::TryParse($ProductCode, [ref]$parsed)) {
        throw "ProductCode is not a valid GUID: '$ProductCode'"
    }
    $canonicalCode = $parsed.ToString("B").ToUpperInvariant()
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $contexts = @(
        [pscustomobject]@{ Code = 1; Name = "USERMANAGED"; Sid = $currentSid },
        [pscustomobject]@{ Code = 2; Name = "USERUNMANAGED"; Sid = $currentSid },
        [pscustomobject]@{ Code = 4; Name = "MACHINE"; Sid = $null }
    )
    $instances = [System.Collections.Generic.List[object]]::new()
    foreach ($query in $contexts) {
        for ($index = 0; ; $index++) {
            $productBuffer = [System.Text.StringBuilder]::new(39)
            $sidBuffer = [System.Text.StringBuilder]::new(186)
            [uint32]$installedContext = 0
            [uint32]$sidLength = 186
            $sidPointer = [IntPtr]::Zero
            if ($null -ne $query.Sid) {
                $sidPointer = [Runtime.InteropServices.Marshal]::StringToHGlobalUni($query.Sid)
            }
            try {
                $result = [NeatIssue18.WindowsInstallerNative]::MsiEnumProductsEx(
                    $canonicalCode, $sidPointer, [uint32]$query.Code, [uint32]$index,
                    $productBuffer, [ref]$installedContext, $sidBuffer, [ref]$sidLength)
            } finally {
                if ($sidPointer -ne [IntPtr]::Zero) {
                    [Runtime.InteropServices.Marshal]::FreeHGlobal($sidPointer)
                }
            }
            if ($result -eq 259 -or $result -eq 1605) { break }
            if ($result -ne 0) {
                throw "MsiEnumProductsEx failed for $canonicalCode ($($query.Name)) with Windows Installer error $result."
            }
            $instanceCode = [guid]::Parse($productBuffer.ToString()).ToString("B").ToUpperInvariant()
            $contextName = switch ($installedContext) {
                1 { "USERMANAGED" }
                2 { "USERUNMANAGED" }
                4 { "MACHINE" }
                default { "UNKNOWN($installedContext)" }
            }
            $instances.Add([pscustomobject]@{
                ProductCode = $instanceCode
                Context = $contextName
                Sid = $sidBuffer.ToString()
            })
        }
    }
    return $instances.ToArray()
}

function Resolve-NeatMsiProductContext {
    param(
        [Parameter(Mandatory = $true)][string]$ProductCode,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Instances
    )

    $parsed = [guid]::Empty
    if (-not [guid]::TryParse($ProductCode, [ref]$parsed)) {
        throw "ProductCode is not a valid GUID: '$ProductCode'"
    }
    $canonicalCode = $parsed.ToString("B").ToUpperInvariant()
    $matches = @($Instances | Where-Object {
        $instanceCode = [guid]::Empty
        [guid]::TryParse([string]$_.ProductCode, [ref]$instanceCode) -and
            $instanceCode.ToString("B").ToUpperInvariant() -eq $canonicalCode
    })
    if ($matches.Count -eq 0) { throw "Windows Installer has no registered instance of ProductCode $canonicalCode." }
    if ($matches.Count -ne 1) { throw "ProductCode $canonicalCode has $($matches.Count) registered instances; context is ambiguous." }
    if ($matches[0].Context -notin @("USERUNMANAGED", "USERMANAGED")) {
        throw "ProductCode $canonicalCode is registered in $($matches[0].Context) context, not a per-user context."
    }
    return $matches[0]
}

function Test-NeatInstalledUserShortcut {
    param([object[]]$Shortcuts, [Parameter(Mandatory = $true)][string]$InstalledExe)
    return [bool]@($Shortcuts | Where-Object { $_.Target -ieq $InstalledExe }).Count
}

function Test-NeatEquivalentStartMenuShortcut {
    param([Parameter(Mandatory = $true)]$Shortcut, [Parameter(Mandatory = $true)][string]$InstalledExe)
    return ($Shortcut.Target -ieq $InstalledExe -or
        $Shortcut.Name -match "(?i)\bNEAT\b" -or
        $Shortcut.Path -match "(?i)[\\/]NEAT([\\/]|\.lnk$)")
}

Export-ModuleMember -Function Get-NeatMsiIdentity, Get-NeatMsiProductInstances, Resolve-NeatMsiProductContext, Test-NeatInstalledUserShortcut, Test-NeatEquivalentStartMenuShortcut
