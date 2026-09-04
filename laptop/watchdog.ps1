#requires -Version 5.1
<#
.SYNOPSIS
  Bounded local watchdog for the SZL GPU bridge scheduled control plane.

.DESCRIPTION
  Reconciles three exact Task Scheduler definitions from local files only:
    - SZL-GPU-Bridge
    - SZL-GPU-Bridge-Watchdog
    - SZL-GPU-Bridge-Guardian

  It can enable, register, or start only those three tasks. It accepts no
  downloaded command, does not inspect token values, opens no port, and does
  not widen the signed pull-and-verify execution boundary of the GPU bridge.
#>

[CmdletBinding()]
param(
    [ValidateSet('Watchdog', 'Guardian', 'Status')]
    [string]$Mode = 'Watchdog',
    [string]$Root = 'C:\szl-bridge',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ConfigPath = Join-Path $Root 'watchdog-config.json'
$StatePath = Join-Path $Root 'watchdog-state.json'
$ReceiptPath = Join-Path $Root 'logs\watchdog-receipts.ndjson'
$LogPath = Join-Path $Root 'logs\watchdog.log'
$DaemonLogPath = Join-Path $Root 'logs\daemon.log'
$DaemonPath = Join-Path $Root 'daemon.ps1'
$WatchdogPath = Join-Path $Root 'watchdog.ps1'
$Now = [DateTimeOffset]::UtcNow
$Events = New-Object System.Collections.Generic.List[object]

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Write-Log {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet('INFO', 'WARN', 'ERROR')][string]$Level = 'INFO'
    )
    $line = '{0} [{1}] {2}' -f ([DateTimeOffset]::UtcNow.ToString('o')), $Level, $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    if ($Mode -eq 'Status' -or $Level -ne 'INFO') {
        Write-Output $line
    }
}

function Add-Event {
    param(
        [Parameter(Mandatory = $true)][string]$Component,
        [Parameter(Mandatory = $true)][string]$Action,
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][string]$Detail
    )
    $Events.Add([pscustomobject]@{
        component = $Component
        action = $Action
        status = $Status
        detail = $Detail
    }) | Out-Null
}

function New-DefaultState {
    return [pscustomobject]@{
        schema = 'szl.gpu-bridge-watchdog-state/v1'
        updated_at = $null
        last_healthy_at = $null
        last_status = 'UNKNOWN'
        starts = [pscustomobject]@{}
    }
}

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
        return New-DefaultState
    }
    try {
        $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if ($state.schema -ne 'szl.gpu-bridge-watchdog-state/v1') {
            throw 'unsupported state schema'
        }
        if ($null -eq $state.starts) {
            $state | Add-Member -NotePropertyName starts -NotePropertyValue ([pscustomobject]@{}) -Force
        }
        return $state
    }
    catch {
        Write-Log -Level 'WARN' -Message ('State rejected and reset: {0}' -f $_.Exception.GetType().Name)
        return New-DefaultState
    }
}

function Write-State {
    param([Parameter(Mandatory = $true)]$State)
    $temporary = $StatePath + '.tmp'
    $json = $State | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText(
        $temporary,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Get-DynamicProperty {
    param($Object, [string]$Name, $Default)
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Set-DynamicProperty {
    param($Object, [string]$Name, $Value)
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
}

function Read-Config {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw ('Missing fixed watchdog configuration: {0}' -f $ConfigPath)
    }
    $config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if ($config.schema -ne 'szl.gpu-bridge-watchdog-config/v1') {
        throw 'unsupported watchdog configuration schema'
    }
    if ([string]::IsNullOrWhiteSpace([string]$config.user_id)) {
        throw 'watchdog user_id is missing'
    }
    if ([int]$config.max_starts_per_window -lt 1 -or [int]$config.max_starts_per_window -gt 12) {
        throw 'max_starts_per_window is outside the bounded range'
    }
    if ([int]$config.window_minutes -lt 10 -or [int]$config.window_minutes -gt 240) {
        throw 'window_minutes is outside the bounded range'
    }
    if ([int]$config.daemon_stale_minutes -lt 20 -or [int]$config.daemon_stale_minutes -gt 120) {
        throw 'daemon_stale_minutes is outside the bounded range'
    }
    return $config
}

function Test-StartBudget {
    param([Parameter(Mandatory = $true)][string]$TaskName)
    $history = @(Get-DynamicProperty -Object $State.starts -Name $TaskName -Default @())
    $windowStart = $Now.AddMinutes(-[int]$Config.window_minutes)
    $recent = @($history | Where-Object {
        try { [DateTimeOffset]::Parse([string]$_) -ge $windowStart }
        catch { $false }
    })
    Set-DynamicProperty -Object $State.starts -Name $TaskName -Value $recent
    return ($recent.Count -lt [int]$Config.max_starts_per_window)
}

function Record-Start {
    param([Parameter(Mandatory = $true)][string]$TaskName)
    $history = @(Get-DynamicProperty -Object $State.starts -Name $TaskName -Default @())
    $history += [DateTimeOffset]::UtcNow.ToString('o')
    Set-DynamicProperty -Object $State.starts -Name $TaskName -Value $history
}

function Get-TaskDefinition {
    param([Parameter(Mandatory = $true)][string]$TaskName)
    $powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $principal = New-ScheduledTaskPrincipal `
        -UserId ([string]$Config.user_id) `
        -LogonType S4U `
        -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -RestartCount 6 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    switch ($TaskName) {
        'SZL-GPU-Bridge' {
            $action = New-ScheduledTaskAction `
                -Execute $powerShell `
                -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}"' -f $DaemonPath)
            $triggers = @(
                (New-ScheduledTaskTrigger -AtStartup),
                (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15))
            )
        }
        'SZL-GPU-Bridge-Watchdog' {
            $action = New-ScheduledTaskAction `
                -Execute $powerShell `
                -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Mode Watchdog -Root "{1}"' -f $WatchdogPath, $Root)
            $triggers = @(
                (New-ScheduledTaskTrigger -AtStartup),
                (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 5))
            )
        }
        'SZL-GPU-Bridge-Guardian' {
            $action = New-ScheduledTaskAction `
                -Execute $powerShell `
                -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Mode Guardian -Root "{1}"' -f $WatchdogPath, $Root)
            $triggers = @(
                (New-ScheduledTaskTrigger -AtStartup),
                (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(3) -RepetitionInterval (New-TimeSpan -Minutes 15))
            )
        }
        default { throw ('Task is outside the fixed allowlist: {0}' -f $TaskName) }
    }
    return [pscustomobject]@{
        action = $action
        triggers = $triggers
        principal = $principal
        settings = $settings
    }
}

function Ensure-Task {
    param([Parameter(Mandatory = $true)][string]$TaskName)
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $task) {
        if ($DryRun) {
            Add-Event -Component $TaskName -Action 'register' -Status 'DRY_RUN' -Detail 'fixed task would be registered'
            return $true
        }
        $definition = Get-TaskDefinition -TaskName $TaskName
        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $definition.action `
            -Trigger $definition.triggers `
            -Principal $definition.principal `
            -Settings $definition.settings `
            -Description 'SZL GPU bridge bounded liveness control plane.' `
            -Force | Out-Null
        Add-Event -Component $TaskName -Action 'register' -Status 'APPLIED' -Detail 'fixed task registered from local files'
        return $true
    }
    if ($task.State -eq 'Disabled') {
        if (-not $DryRun) {
            Enable-ScheduledTask -TaskName $TaskName | Out-Null
        }
        Add-Event -Component $TaskName -Action 'enable' -Status $(if ($DryRun) { 'DRY_RUN' } else { 'APPLIED' }) -Detail 'disabled task enabled'
    }
    return $true
}

function Start-BoundedTask {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)][string]$Reason
    )
    if (-not (Test-StartBudget -TaskName $TaskName)) {
        Add-Event -Component $TaskName -Action 'start' -Status 'COOLDOWN' -Detail 'rolling start budget exhausted'
        return $false
    }
    if ($DryRun) {
        Add-Event -Component $TaskName -Action 'start' -Status 'DRY_RUN' -Detail $Reason
        return $true
    }
    Start-ScheduledTask -TaskName $TaskName
    Record-Start -TaskName $TaskName
    Add-Event -Component $TaskName -Action 'start' -Status 'APPLIED' -Detail $Reason
    return $true
}

function Test-TaskFreshness {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)][int]$MaxAgeMinutes
    )
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $task) { return $false }
    if ($task.State -in @('Running', 'Queued')) {
        Add-Event -Component $TaskName -Action 'observe' -Status 'HEALTHY' -Detail ('state={0}' -f $task.State)
        return $true
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    $neverRan = $info.LastRunTime -le [DateTime]::MinValue.AddDays(2)
    $ageMinutes = if ($neverRan) { [double]::PositiveInfinity } else { ((Get-Date) - $info.LastRunTime).TotalMinutes }
    $lastResultHealthy = [int64]$info.LastTaskResult -in @(0, 267009)
    if ($lastResultHealthy -and $ageMinutes -le $MaxAgeMinutes) {
        Add-Event -Component $TaskName -Action 'observe' -Status 'HEALTHY' -Detail ('age_minutes={0:N1}; result={1}' -f $ageMinutes, $info.LastTaskResult)
        return $true
    }
    $reason = 'never ran'
    if (-not $neverRan) {
        $reason = 'age_minutes={0:N1}; result={1}' -f $ageMinutes, $info.LastTaskResult
    }
    return Start-BoundedTask -TaskName $TaskName -Reason $reason
}

function Test-DaemonLogFreshness {
    if (-not (Test-Path -LiteralPath $DaemonLogPath -PathType Leaf)) {
        return Start-BoundedTask -TaskName 'SZL-GPU-Bridge' -Reason 'daemon log is missing'
    }
    $age = $Now - [DateTimeOffset](Get-Item -LiteralPath $DaemonLogPath).LastWriteTimeUtc
    if ($age.TotalMinutes -le [int]$Config.daemon_stale_minutes) {
        Add-Event -Component 'daemon-log' -Action 'observe' -Status 'HEALTHY' -Detail ('age_minutes={0:N1}' -f $age.TotalMinutes)
        return $true
    }
    return Start-BoundedTask -TaskName 'SZL-GPU-Bridge' -Reason ('daemon log stale for {0:N1} minutes' -f $age.TotalMinutes)
}

function Write-Receipt {
    param([Parameter(Mandatory = $true)][string]$Status)
    $receipt = [ordered]@{
        schema = 'szl.gpu-bridge-watchdog-receipt/v1'
        generated_at = [DateTimeOffset]::UtcNow.ToString('o')
        mode = $Mode
        status = $Status
        events = @($Events)
        allowed_tasks = @(
            'SZL-GPU-Bridge',
            'SZL-GPU-Bridge-Watchdog',
            'SZL-GPU-Bridge-Guardian'
        )
        remote_commands_accepted = $false
        credential_values_read = $false
        inbound_ports_opened = $false
    }
    Add-Content -LiteralPath $ReceiptPath -Value ($receipt | ConvertTo-Json -Depth 12 -Compress) -Encoding UTF8
    return $receipt
}

Ensure-Directory -Path $Root
Ensure-Directory -Path (Join-Path $Root 'logs')
$Config = Read-Config
$State = Read-State

if ($Mode -eq 'Status') {
    [ordered]@{
        config_schema = $Config.schema
        state = $State
        tasks = @(
            Get-ScheduledTask -TaskName 'SZL-GPU-Bridge','SZL-GPU-Bridge-Watchdog','SZL-GPU-Bridge-Guardian' -ErrorAction SilentlyContinue |
                Select-Object TaskName, State
        )
        credential_values_read = $false
    } | ConvertTo-Json -Depth 12
    exit 0
}

$createdNew = $false
$mutex = New-Object System.Threading.Mutex($false, 'Global\SZLGpuBridgeWatchdog', [ref]$createdNew)
$hasLock = $false
try {
    try {
        $hasLock = $mutex.WaitOne([TimeSpan]::FromSeconds(3))
    }
    catch [System.Threading.AbandonedMutexException] {
        $hasLock = $true
        Write-Log -Level 'WARN' -Message 'Recovered an abandoned watchdog mutex.'
    }
    if (-not $hasLock) {
        Write-Log -Message 'Another watchdog cycle owns the mutex; exiting cleanly.'
        exit 0
    }

    foreach ($name in @('SZL-GPU-Bridge','SZL-GPU-Bridge-Watchdog','SZL-GPU-Bridge-Guardian')) {
        Ensure-Task -TaskName $name | Out-Null
    }

    $healthy = $true
    if (-not (Test-TaskFreshness -TaskName 'SZL-GPU-Bridge' -MaxAgeMinutes 25)) { $healthy = $false }
    if (-not (Test-DaemonLogFreshness)) { $healthy = $false }

    $counterpart = if ($Mode -eq 'Guardian') { 'SZL-GPU-Bridge-Watchdog' } else { 'SZL-GPU-Bridge-Guardian' }
    $counterpartAge = if ($counterpart -eq 'SZL-GPU-Bridge-Watchdog') { 12 } else { 35 }
    if (-not (Test-TaskFreshness -TaskName $counterpart -MaxAgeMinutes $counterpartAge)) { $healthy = $false }

    $status = if ($healthy) { 'HEALTHY' } else { 'DEGRADED' }
    $State.updated_at = [DateTimeOffset]::UtcNow.ToString('o')
    $State.last_status = $status
    if ($healthy) { $State.last_healthy_at = $State.updated_at }
    Write-State -State $State
    $receipt = Write-Receipt -Status $status
    Write-Log -Message ('Cycle complete: mode={0}; status={1}; events={2}' -f $Mode, $status, $Events.Count)
    $receipt | ConvertTo-Json -Depth 12
    if ($healthy) { exit 0 }
    exit 2
}
catch {
    Write-Log -Level 'ERROR' -Message ('Watchdog failure: {0}' -f $_.Exception.GetType().Name)
    Add-Event -Component 'watchdog' -Action 'cycle' -Status 'FAILED' -Detail $_.Exception.GetType().Name
    try { Write-Receipt -Status 'FAILED' | Out-Null } catch { }
    exit 1
}
finally {
    if ($hasLock) {
        try { $mutex.ReleaseMutex() } catch { }
    }
    $mutex.Dispose()
}
