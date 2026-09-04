#requires -Version 5.1
<#
.SYNOPSIS
  Idempotently installs the bounded SZL GPU bridge watchdog and guardian.

.DESCRIPTION
  Preserves the existing SZL-GPU-Bridge task principal when present, copies
  reviewed repository-local files to C:\szl-bridge, and registers two redundant
  liveness tasks. It reads no Hugging Face or GitHub credential value.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Root = 'C:\szl-bridge',
    [string]$SourceRoot = $PSScriptRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'install_watchdog.ps1 must run from an elevated identity.'
    }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Value)
    [IO.File]::WriteAllText($Path, $Value, [Text.UTF8Encoding]::new($false))
}

function Resolve-TaskUser {
    $existing = Get-ScheduledTask -TaskName 'SZL-GPU-Bridge' -ErrorAction SilentlyContinue
    if ($null -ne $existing -and -not [string]::IsNullOrWhiteSpace([string]$existing.Principal.UserId)) {
        $candidate = [string]$existing.Principal.UserId
    }
    else {
        $candidate = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    }
    if ($candidate -in @('SYSTEM', 'NT AUTHORITY\SYSTEM', 'LOCAL SERVICE', 'NETWORK SERVICE')) {
        throw 'A user-scoped bridge principal is required so existing Hugging Face authentication remains available.'
    }
    return $candidate
}

function New-FixedTaskDefinition {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)][string]$UserId,
        [Parameter(Mandatory = $true)][string]$RootPath
    )
    $powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $daemon = Join-Path $RootPath 'daemon.ps1'
    $watchdog = Join-Path $RootPath 'watchdog.ps1'
    $principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType S4U -RunLevel Highest

    switch ($TaskName) {
        'SZL-GPU-Bridge' {
            $action = New-ScheduledTaskAction -Execute $powerShell -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}"' -f $daemon)
            $triggers = @(
                (New-ScheduledTaskTrigger -AtStartup),
                (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15))
            )
            $settings = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries `
                -StartWhenAvailable `
                -MultipleInstances IgnoreNew `
                -RestartCount 3 `
                -RestartInterval (New-TimeSpan -Minutes 2) `
                -ExecutionTimeLimit (New-TimeSpan -Hours 26)
        }
        'SZL-GPU-Bridge-Watchdog' {
            $action = New-ScheduledTaskAction -Execute $powerShell -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Mode Watchdog -Root "{1}"' -f $watchdog, $RootPath)
            $triggers = @(
                (New-ScheduledTaskTrigger -AtStartup),
                (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 5))
            )
            $settings = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries `
                -StartWhenAvailable `
                -MultipleInstances IgnoreNew `
                -RestartCount 3 `
                -RestartInterval (New-TimeSpan -Minutes 1) `
                -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
        }
        'SZL-GPU-Bridge-Guardian' {
            $action = New-ScheduledTaskAction -Execute $powerShell -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Mode Guardian -Root "{1}"' -f $watchdog, $RootPath)
            $triggers = @(
                (New-ScheduledTaskTrigger -AtStartup),
                (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(3) -RepetitionInterval (New-TimeSpan -Minutes 15))
            )
            $settings = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries `
                -StartWhenAvailable `
                -MultipleInstances IgnoreNew `
                -RestartCount 3 `
                -RestartInterval (New-TimeSpan -Minutes 1) `
                -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
        }
        default { throw ('Task is outside the fixed allowlist: {0}' -f $TaskName) }
    }
    return [pscustomobject]@{
        action = $action
        triggers = $triggers
        settings = $settings
        principal = $principal
    }
}

function Register-FixedTask {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)][string]$UserId,
        [Parameter(Mandatory = $true)][string]$RootPath
    )
    $definition = New-FixedTaskDefinition -TaskName $TaskName -UserId $UserId -RootPath $RootPath
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $definition.action `
        -Trigger $definition.triggers `
        -Settings $definition.settings `
        -Principal $definition.principal `
        -Description 'SZL GPU bridge bounded liveness control plane.' `
        -Force | Out-Null
    Enable-ScheduledTask -TaskName $TaskName | Out-Null
}

Assert-Administrator
$UserId = Resolve-TaskUser
$SourceWatchdog = Join-Path $SourceRoot 'watchdog.ps1'
$SourceDaemon = Join-Path $SourceRoot 'daemon.ps1'
if (-not (Test-Path -LiteralPath $SourceWatchdog -PathType Leaf)) {
    throw ('Missing repository watchdog source: {0}' -f $SourceWatchdog)
}
if (-not (Test-Path -LiteralPath $SourceDaemon -PathType Leaf)) {
    throw ('Missing repository daemon source: {0}' -f $SourceDaemon)
}

New-Item -ItemType Directory -Path $Root -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root 'logs') -Force | Out-Null

$InstalledWatchdog = Join-Path $Root 'watchdog.ps1'
$InstalledDaemon = Join-Path $Root 'daemon.ps1'
Copy-Item -LiteralPath $SourceWatchdog -Destination $InstalledWatchdog -Force
if (-not (Test-Path -LiteralPath $InstalledDaemon -PathType Leaf)) {
    Copy-Item -LiteralPath $SourceDaemon -Destination $InstalledDaemon -Force
}

$Config = [ordered]@{
    schema = 'szl.gpu-bridge-watchdog-config/v1'
    user_id = $UserId
    window_minutes = 30
    max_starts_per_window = 3
    daemon_stale_minutes = 35
}
$ConfigPath = Join-Path $Root 'watchdog-config.json'
$ConfigTemp = $ConfigPath + '.tmp'
Write-Utf8NoBom -Path $ConfigTemp -Value (($Config | ConvertTo-Json -Depth 8) + [Environment]::NewLine)
Move-Item -LiteralPath $ConfigTemp -Destination $ConfigPath -Force

& icacls.exe $Root /inheritance:r | Out-Null
& icacls.exe $Root /grant:r 'SYSTEM:(OI)(CI)(F)' 'BUILTIN\Administrators:(OI)(CI)(F)' (('{0}:(OI)(CI)(M)' -f $UserId)) | Out-Null

foreach ($name in @('SZL-GPU-Bridge','SZL-GPU-Bridge-Watchdog','SZL-GPU-Bridge-Guardian')) {
    if ($PSCmdlet.ShouldProcess($name, 'Register exact bounded scheduled task')) {
        Register-FixedTask -TaskName $name -UserId $UserId -RootPath $Root
    }
}

Start-ScheduledTask -TaskName 'SZL-GPU-Bridge-Watchdog'
Start-ScheduledTask -TaskName 'SZL-GPU-Bridge-Guardian'
Start-Sleep -Seconds 8

$Tasks = @(
    Get-ScheduledTask -TaskName 'SZL-GPU-Bridge','SZL-GPU-Bridge-Watchdog','SZL-GPU-Bridge-Guardian' |
        ForEach-Object {
            $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
            [ordered]@{
                name = $_.TaskName
                state = [string]$_.State
                principal = [string]$_.Principal.UserId
                last_run_time = $info.LastRunTime.ToUniversalTime().ToString('o')
                last_task_result = [int64]$info.LastTaskResult
                next_run_time = $info.NextRunTime.ToUniversalTime().ToString('o')
            }
        }
)

$Receipt = [ordered]@{
    schema = 'szl.gpu-bridge-watchdog-install/v1'
    generated_at = [DateTimeOffset]::UtcNow.ToString('o')
    root = $Root
    tasks = $Tasks
    installed_files = @(
        [ordered]@{ path = 'watchdog.ps1'; sha256 = (Get-FileHash -LiteralPath $InstalledWatchdog -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ path = 'daemon.ps1'; sha256 = (Get-FileHash -LiteralPath $InstalledDaemon -Algorithm SHA256).Hash.ToLowerInvariant() },
        [ordered]@{ path = 'watchdog-config.json'; sha256 = (Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant() }
    )
    remote_commands_accepted = $false
    credential_values_read = $false
    inbound_ports_opened = $false
}
$Receipt | ConvertTo-Json -Depth 12

if ($Tasks.Count -ne 3) { throw 'Not all three exact tasks were installed.' }
if (@($Tasks | Where-Object { $_.state -eq 'Disabled' }).Count -gt 0) {
    throw 'One or more exact tasks remained disabled.'
}
