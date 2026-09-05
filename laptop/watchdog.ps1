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
$ReceiptTipPath = Join-Path $Root 'logs\watchdog-receipt-tip.txt'
$LogPath = Join-Path $Root 'logs\watchdog.log'
$DaemonLogPath = Join-Path $Root 'logs\daemon.log'
$DaemonPath = Join-Path $Root 'daemon.ps1'
$WatchdogPath = Join-Path $Root 'watchdog.ps1'
$TaskDescription = 'SZL GPU bridge bounded liveness control plane.'
$Now = [DateTimeOffset]::UtcNow
$Events = New-Object System.Collections.Generic.List[object]

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Rotate-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Prefix
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -le 8MB) { return }
    $archive = Join-Path $item.DirectoryName ('{0}-{1}.log' -f $Prefix, $Now.ToString('yyyyMMdd-HHmmss'))
    Move-Item -LiteralPath $Path -Destination $archive -Force
    Get-ChildItem -LiteralPath $item.DirectoryName -Filter ($Prefix + '-*.log') -File |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -Skip 8 |
        Remove-Item -Force -ErrorAction SilentlyContinue
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

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
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
        Write-Log -Level 'WARN' -Message ('State rejected and reset: {0}' -f $_.Exception.GetType().Name) | Out-Null
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

    switch ($TaskName) {
        'SZL-GPU-Bridge' {
            $action = New-ScheduledTaskAction `
                -Execute $powerShell `
                -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}"' -f $DaemonPath)
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
            $action = New-ScheduledTaskAction `
                -Execute $powerShell `
                -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Mode Watchdog -Root "{1}"' -f $WatchdogPath, $Root)
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
            $action = New-ScheduledTaskAction `
                -Execute $powerShell `
                -Argument ('-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Mode Guardian -Root "{1}"' -f $WatchdogPath, $Root)
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
        principal = $principal
        settings = $settings
    }
}

function Get-TaskDefinitionFingerprint {
    param(
        [Parameter(Mandatory = $true)]$Actions,
        [Parameter(Mandatory = $true)]$Triggers,
        [Parameter(Mandatory = $true)]$Principal,
        [Parameter(Mandatory = $true)]$Settings
    )

    $actionRows = @(
        $Actions | ForEach-Object {
            [ordered]@{
                id = [string](Get-DynamicProperty -Object $_ -Name 'Id' -Default '')
                execute = [string](Get-DynamicProperty -Object $_ -Name 'Execute' -Default '')
                arguments = [string](Get-DynamicProperty -Object $_ -Name 'Arguments' -Default '')
                working_directory = [string](Get-DynamicProperty -Object $_ -Name 'WorkingDirectory' -Default '')
            }
        }
    )
    $triggerRows = @(
        $Triggers | ForEach-Object {
            $cimClass = Get-DynamicProperty -Object $_ -Name 'CimClass' -Default $null
            $repetition = Get-DynamicProperty -Object $_ -Name 'Repetition' -Default $null
            $startBoundary = [string](Get-DynamicProperty -Object $_ -Name 'StartBoundary' -Default '')
            $normalizedStartBoundary = if ([string]::IsNullOrWhiteSpace($startBoundary)) {
                ''
            }
            else {
                try {
                    $parsedStartBoundary = [DateTimeOffset]::Parse(
                        $startBoundary,
                        [Globalization.CultureInfo]::InvariantCulture
                    ).ToUniversalTime()
                    $futureTolerance = [TimeSpan]::FromMinutes(5)
                    if (
                        $null -ne $repetition -and
                        $parsedStartBoundary -le [DateTimeOffset]::UtcNow.Add($futureTolerance)
                    ) {
                        # Repeating tasks remain equivalent after their initial boundary has
                        # passed. The five-minute future window covers the fixed two/three-
                        # minute registration lead without accepting a disabled far-future
                        # schedule as healthy.
                        '<active-repeating>'
                    }
                    else {
                        $parsedStartBoundary.ToString(
                            'o',
                            [Globalization.CultureInfo]::InvariantCulture
                        )
                    }
                }
                catch {
                    '<invalid>'
                }
            }
            [ordered]@{
                type = if ($null -eq $cimClass) { '' } else { [string]$cimClass.CimClassName }
                id = [string](Get-DynamicProperty -Object $_ -Name 'Id' -Default '')
                enabled = [bool](Get-DynamicProperty -Object $_ -Name 'Enabled' -Default $true)
                start_boundary = $normalizedStartBoundary
                end_boundary = [string](Get-DynamicProperty -Object $_ -Name 'EndBoundary' -Default '')
                execution_time_limit = [string](Get-DynamicProperty -Object $_ -Name 'ExecutionTimeLimit' -Default '')
                delay = [string](Get-DynamicProperty -Object $_ -Name 'Delay' -Default '')
                random_delay = [string](Get-DynamicProperty -Object $_ -Name 'RandomDelay' -Default '')
                repetition_interval = if ($null -eq $repetition) { '' } else {
                    [string](Get-DynamicProperty -Object $repetition -Name 'Interval' -Default '')
                }
                repetition_duration = if ($null -eq $repetition) { '' } else {
                    [string](Get-DynamicProperty -Object $repetition -Name 'Duration' -Default '')
                }
                repetition_stop_at_duration_end = if ($null -eq $repetition) { $false } else {
                    [bool](Get-DynamicProperty -Object $repetition -Name 'StopAtDurationEnd' -Default $false)
                }
            }
        } | Sort-Object type, id, enabled, start_boundary, end_boundary, execution_time_limit, delay, random_delay, repetition_interval, repetition_duration, repetition_stop_at_duration_end
    )
    $requiredPrivileges = @(
        Get-DynamicProperty -Object $Principal -Name 'RequiredPrivilege' -Default @() |
            ForEach-Object { [string]$_ } |
            Sort-Object
    )
    $idleSettings = Get-DynamicProperty -Object $Settings -Name 'IdleSettings' -Default $null
    $networkSettings = Get-DynamicProperty -Object $Settings -Name 'NetworkSettings' -Default $null
    $maintenanceSettings = Get-DynamicProperty -Object $Settings -Name 'MaintenanceSettings' -Default $null
    $fingerprint = [ordered]@{
        actions = $actionRows
        triggers = $triggerRows
        principal = [ordered]@{
            user_id = [string](Get-DynamicProperty -Object $Principal -Name 'UserId' -Default '')
            group_id = [string](Get-DynamicProperty -Object $Principal -Name 'GroupId' -Default '')
            logon_type = [string](Get-DynamicProperty -Object $Principal -Name 'LogonType' -Default '')
            run_level = [string](Get-DynamicProperty -Object $Principal -Name 'RunLevel' -Default '')
            process_token_sid_type = [string](Get-DynamicProperty -Object $Principal -Name 'ProcessTokenSidType' -Default '')
            required_privileges = $requiredPrivileges
        }
        settings = [ordered]@{
            multiple_instances = [string](Get-DynamicProperty -Object $Settings -Name 'MultipleInstances' -Default '')
            compatibility = [string](Get-DynamicProperty -Object $Settings -Name 'Compatibility' -Default '')
            allow_demand_start = [bool](Get-DynamicProperty -Object $Settings -Name 'AllowDemandStart' -Default $false)
            allow_hard_terminate = [bool](Get-DynamicProperty -Object $Settings -Name 'AllowHardTerminate' -Default $false)
            delete_expired_task_after = [string](Get-DynamicProperty -Object $Settings -Name 'DeleteExpiredTaskAfter' -Default '')
            start_when_available = [bool](Get-DynamicProperty -Object $Settings -Name 'StartWhenAvailable' -Default $false)
            disallow_start_if_on_batteries = [bool](Get-DynamicProperty -Object $Settings -Name 'DisallowStartIfOnBatteries' -Default $true)
            stop_if_going_on_batteries = [bool](Get-DynamicProperty -Object $Settings -Name 'StopIfGoingOnBatteries' -Default $true)
            enabled = [bool](Get-DynamicProperty -Object $Settings -Name 'Enabled' -Default $true)
            restart_count = [int](Get-DynamicProperty -Object $Settings -Name 'RestartCount' -Default 0)
            restart_interval = [string](Get-DynamicProperty -Object $Settings -Name 'RestartInterval' -Default '')
            execution_time_limit = [string](Get-DynamicProperty -Object $Settings -Name 'ExecutionTimeLimit' -Default '')
            hidden = [bool](Get-DynamicProperty -Object $Settings -Name 'Hidden' -Default $false)
            priority = [int](Get-DynamicProperty -Object $Settings -Name 'Priority' -Default -1)
            run_only_if_idle = [bool](Get-DynamicProperty -Object $Settings -Name 'RunOnlyIfIdle' -Default $false)
            run_only_if_network_available = [bool](Get-DynamicProperty -Object $Settings -Name 'RunOnlyIfNetworkAvailable' -Default $false)
            wake_to_run = [bool](Get-DynamicProperty -Object $Settings -Name 'WakeToRun' -Default $false)
            disallow_start_on_remote_app_session = [bool](Get-DynamicProperty -Object $Settings -Name 'DisallowStartOnRemoteAppSession' -Default $false)
            use_unified_scheduling_engine = [bool](Get-DynamicProperty -Object $Settings -Name 'UseUnifiedSchedulingEngine' -Default $false)
            volatile = [bool](Get-DynamicProperty -Object $Settings -Name 'Volatile' -Default $false)
            idle = if ($null -eq $idleSettings) { $null } else {
                [ordered]@{
                    duration = [string](Get-DynamicProperty -Object $idleSettings -Name 'IdleDuration' -Default '')
                    restart_on_idle = [bool](Get-DynamicProperty -Object $idleSettings -Name 'RestartOnIdle' -Default $false)
                    stop_on_idle_end = [bool](Get-DynamicProperty -Object $idleSettings -Name 'StopOnIdleEnd' -Default $false)
                    wait_timeout = [string](Get-DynamicProperty -Object $idleSettings -Name 'WaitTimeout' -Default '')
                }
            }
            network = if ($null -eq $networkSettings) { $null } else {
                [ordered]@{
                    id = [string](Get-DynamicProperty -Object $networkSettings -Name 'Id' -Default '')
                    name = [string](Get-DynamicProperty -Object $networkSettings -Name 'Name' -Default '')
                }
            }
            maintenance = if ($null -eq $maintenanceSettings) { $null } else {
                [ordered]@{
                    deadline = [string](Get-DynamicProperty -Object $maintenanceSettings -Name 'Deadline' -Default '')
                    exclusive = [bool](Get-DynamicProperty -Object $maintenanceSettings -Name 'Exclusive' -Default $false)
                    period = [string](Get-DynamicProperty -Object $maintenanceSettings -Name 'Period' -Default '')
                }
            }
        }
    }
    return ($fingerprint | ConvertTo-Json -Depth 8 -Compress)
}

function Test-TaskDefinition {
    param(
        [Parameter(Mandatory = $true)]$Task,
        [Parameter(Mandatory = $true)]$Definition
    )
    if ([string]$Task.Description -ne $TaskDescription) { return $false }
    $actual = Get-TaskDefinitionFingerprint `
        -Actions $Task.Actions `
        -Triggers $Task.Triggers `
        -Principal $Task.Principal `
        -Settings $Task.Settings
    $expected = Get-TaskDefinitionFingerprint `
        -Actions @($Definition.action) `
        -Triggers @($Definition.triggers) `
        -Principal $Definition.principal `
        -Settings $Definition.settings
    return $actual -ceq $expected
}

function Register-FixedTask {
    param(
        [Parameter(Mandatory = $true)][string]$TaskName,
        [Parameter(Mandatory = $true)]$Definition
    )
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Definition.action `
        -Trigger $Definition.triggers `
        -Principal $Definition.principal `
        -Settings $Definition.settings `
        -Description $TaskDescription `
        -Force | Out-Null
}

function Ensure-Task {
    param([Parameter(Mandatory = $true)][string]$TaskName)
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $definition = Get-TaskDefinition -TaskName $TaskName
    if ($null -eq $task) {
        if ($DryRun) {
            Add-Event -Component $TaskName -Action 'register' -Status 'DRY_RUN' -Detail 'fixed task would be registered'
            return $false
        }
        Register-FixedTask -TaskName $TaskName -Definition $definition
        Add-Event -Component $TaskName -Action 'register' -Status 'APPLIED' -Detail 'fixed task registered from local files'
        return $false
    }
    if (-not (Test-TaskDefinition -Task $task -Definition $definition)) {
        if (-not $DryRun) {
            Register-FixedTask -TaskName $TaskName -Definition $definition
        }
        $applied = if ($DryRun) { 'DRY_RUN' } else { 'APPLIED' }
        Add-Event -Component $TaskName -Action 'reconcile' -Status $applied -Detail 'definition drift replaced from fixed local files'
        return $false
    }
    if ($task.State -eq 'Disabled') {
        if (-not $DryRun) {
            Enable-ScheduledTask -TaskName $TaskName | Out-Null
        }
        $applied = if ($DryRun) { 'DRY_RUN' } else { 'APPLIED' }
        Add-Event -Component $TaskName -Action 'enable' -Status $applied -Detail 'disabled task enabled'
        return $false
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
        return $false
    }
    try {
        Start-ScheduledTask -TaskName $TaskName
        Record-Start -TaskName $TaskName
        Add-Event -Component $TaskName -Action 'start' -Status 'APPLIED' -Detail $Reason
    }
    catch {
        Add-Event -Component $TaskName -Action 'start' -Status 'FAILED' -Detail $_.Exception.GetType().Name
    }
    return $false
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
    $neverRan = $info.LastRunTime.Year -lt 2000
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
    $task = Get-ScheduledTask -TaskName 'SZL-GPU-Bridge' -ErrorAction SilentlyContinue
    if ($null -ne $task -and $task.State -in @('Running', 'Queued')) {
        Add-Event -Component 'daemon-log' -Action 'observe' -Status 'HEALTHY' -Detail ('primary task state={0}; log freshness deferred' -f $task.State)
        return $true
    }
    if (-not (Test-Path -LiteralPath $DaemonLogPath -PathType Leaf)) {
        return Start-BoundedTask -TaskName 'SZL-GPU-Bridge' -Reason 'daemon log is missing'
    }
    $lastWrite = [DateTimeOffset]((Get-Item -LiteralPath $DaemonLogPath).LastWriteTimeUtc)
    $age = $Now - $lastWrite
    if ($age.TotalMinutes -le [int]$Config.daemon_stale_minutes) {
        Add-Event -Component 'daemon-log' -Action 'observe' -Status 'HEALTHY' -Detail ('age_minutes={0:N1}' -f $age.TotalMinutes)
        return $true
    }
    return Start-BoundedTask -TaskName 'SZL-GPU-Bridge' -Reason ('daemon log stale for {0:N1} minutes' -f $age.TotalMinutes)
}

function Write-Receipt {
    param([Parameter(Mandatory = $true)][string]$Status)
    $previous = 'GENESIS'
    if (Test-Path -LiteralPath $ReceiptTipPath -PathType Leaf) {
        $candidate = (Get-Content -LiteralPath $ReceiptTipPath -Raw).Trim()
        if ($candidate -match '^[0-9a-f]{64}$') { $previous = $candidate }
    }
    $body = [ordered]@{
        schema = 'szl.gpu-bridge-watchdog-receipt/v1'
        generated_at = [DateTimeOffset]::UtcNow.ToString('o')
        mode = $Mode
        status = $Status
        previous_sha256 = $previous
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
    $bodyJson = $body | ConvertTo-Json -Depth 12 -Compress
    $digest = Get-Sha256Hex -Bytes ([Text.Encoding]::UTF8.GetBytes($previous + "`n" + $bodyJson))
    $record = [ordered]@{ body = $body; sha256 = $digest }
    Add-Content -LiteralPath $ReceiptPath -Value ($record | ConvertTo-Json -Depth 14 -Compress) -Encoding UTF8
    [IO.File]::WriteAllText($ReceiptTipPath, $digest + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    return [pscustomobject]$record
}

Ensure-Directory -Path $Root
Ensure-Directory -Path (Join-Path $Root 'logs')
Rotate-File -Path $LogPath -Prefix 'watchdog'
Rotate-File -Path $ReceiptPath -Prefix 'watchdog-receipts'
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

    $healthy = $true
    foreach ($name in @('SZL-GPU-Bridge','SZL-GPU-Bridge-Watchdog','SZL-GPU-Bridge-Guardian')) {
        if (-not (Ensure-Task -TaskName $name)) { $healthy = $false }
    }

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
    Write-Log -Message ('Cycle complete: mode={0}; status={1}; events={2}; receipt={3}' -f $Mode, $status, $Events.Count, $receipt.sha256)
    $receipt | ConvertTo-Json -Depth 14
    exit 0
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
