param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^job-[A-Za-z0-9][A-Za-z0-9._-]*$')]
  [string]$JobId,

  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[0-9a-f]{40}$')]
  [string]$BridgeRevision,

  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[^@\s]+@sha256:[0-9a-f]{64}$')]
  [string]$Image,

  [Parameter(Mandatory = $true)]
  [string]$BridgeSource,

  [string]$BridgeRoot = "C:\szl-bridge",
  [string]$HfCache = "C:\szl-bridge-cache\huggingface\hub",
  [string]$InputCache = "C:\szl-bridge-cache\inputs"
)

$ErrorActionPreference = "Stop"

if ($env:HF_TOKEN -or $env:HUGGING_FACE_HUB_TOKEN -or $env:GH_TOKEN) {
  throw "isolated execution refuses a host process containing HF or GitHub tokens"
}

$Docker = (Get-Command docker.exe -ErrorAction Stop).Source
$Git = (Get-Command git.exe -ErrorAction Stop).Source
$ObservedRevision = (& $Git -C $BridgeSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ObservedRevision -ne $BridgeRevision) {
  throw "bridge source is not the exact approved revision: $ObservedRevision"
}
if (& $Git -C $BridgeSource status --porcelain --untracked-files=no) {
  throw "bridge source has tracked modifications"
}

$JobSpec = Join-Path $BridgeSource "queue\pending\$JobId.json"
$EngineKey = Join-Path $BridgeSource "keys\engine_pubkey.json"
$PrefetchReceipt = Join-Path $InputCache "$JobId-prefetch.json"
foreach ($Required in @($JobSpec, $EngineKey, $PrefetchReceipt)) {
  if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
    throw "required isolated-execution input is missing: $Required"
  }
}
foreach ($RequiredDirectory in @($HfCache, $InputCache)) {
  if (-not (Test-Path -LiteralPath $RequiredDirectory -PathType Container)) {
    throw "required isolated-execution cache is missing: $RequiredDirectory"
  }
}

$Prefetch = Get-Content -LiteralPath $PrefetchReceipt -Raw | ConvertFrom-Json
if (
  $Prefetch.jobId -ne $JobId -or
  $Prefetch.remoteCodeExecuted -ne $false -or
  $Prefetch.credentialPersisted -ne $false
) {
  throw "prefetch receipt does not preserve the no-code/no-credential boundary"
}

$SandboxSource = Join-Path $BridgeRoot "sandbox-source-$JobId"
$SandboxWork = Join-Path $BridgeRoot "sandbox-work-$JobId"
$Jobs = Join-Path $BridgeRoot "jobs"
$JobRoot = Join-Path $Jobs $JobId
$Outbox = Join-Path $JobRoot "receipt-outbox"
$Control = Join-Path $BridgeRoot "control"
foreach ($NewDirectory in @($SandboxSource, $SandboxWork)) {
  if (Test-Path -LiteralPath $NewDirectory) {
    throw "one-attempt sandbox path already exists: $NewDirectory"
  }
  New-Item -ItemType Directory -Path $NewDirectory | Out-Null
}
foreach ($Directory in @($Jobs, $Control)) {
  if (-not (Test-Path -LiteralPath $Directory)) {
    New-Item -ItemType Directory -Path $Directory | Out-Null
  }
}
if (
  (Test-Path -LiteralPath $Outbox) -and
  @(Get-ChildItem -LiteralPath $Outbox -Filter "*.intent.json").Count -gt 0
) {
  throw "one-attempt receipt outbox is not empty"
}

$SourceFiles = @(
  "dispatcher.py",
  "frontier_contract.py",
  "frontier_job.py",
  "frontier_runtime.py",
  "nemo_v3_contract.py",
  "runjob_nemo_v3.py"
)
foreach ($Name in $SourceFiles) {
  $Source = Join-Path $BridgeSource "laptop\$Name"
  if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "approved bridge source file is missing: $Name"
  }
  Copy-Item -LiteralPath $Source -Destination (Join-Path $SandboxSource $Name)
}
New-Item -ItemType Directory -Path (Join-Path $SandboxSource "keys") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $SandboxSource "jobs") | Out-Null
Copy-Item `
  -LiteralPath $EngineKey `
  -Destination (Join-Path $SandboxSource "keys\engine_pubkey.json")

$NotBefore = [DateTimeOffset]::UtcNow.ToString("o")
$NotBeforePath = Join-Path $Control "$JobId-not-before.txt"
Set-Content -LiteralPath $NotBeforePath -Value $NotBefore -Encoding ascii

$Arguments = @(
  "run",
  "--rm",
  "--gpus", "all",
  "--network", "none",
  "--read-only",
  "--cap-drop", "ALL",
  "--security-opt", "no-new-privileges:true",
  "--pids-limit", "2048",
  "--shm-size", "8g",
  "--mount", "type=bind,src=$SandboxSource,dst=/bridge,readonly",
  "--mount", "type=bind,src=$Jobs,dst=/bridge/jobs",
  "--mount", "type=bind,src=$JobSpec,dst=/job/spec.json,readonly",
  "--mount", "type=bind,src=$InputCache,dst=/inputs,readonly",
  "--mount", "type=bind,src=$HfCache,dst=/root/.cache/huggingface/hub,readonly",
  "--mount", "type=bind,src=$SandboxWork,dst=/workspace",
  "--tmpfs", "/tmp:rw,noexec,nosuid,size=4294967296",
  "--env", "SZL_INPUT_CACHE=/inputs",
  "--env", "SZL_RECEIPT_TRANSPORT=local-unsigned-outbox",
  "--env", "SZL_EXECUTION_ISOLATION=credentialless-networkless-container",
  "--env", "HF_HUB_OFFLINE=1",
  "--env", "TRANSFORMERS_OFFLINE=1",
  "--env", "HF_DATASETS_OFFLINE=1",
  "--env", "HF_HOME=/root/.cache/huggingface",
  "--env", "HF_HUB_CACHE=/root/.cache/huggingface/hub",
  "--env", "XDG_CACHE_HOME=/workspace/cache",
  "--entrypoint", "python",
  $Image,
  "/bridge/dispatcher.py",
  "/job/spec.json"
)

& $Docker @Arguments
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 7) {
  throw "isolated dispatcher returned $ExitCode instead of receipt-pending code 7"
}

$Intents = @(
  Get-ChildItem -LiteralPath $Outbox -Filter "*.intent.json" -File
)
if ($Intents.Count -ne 1) {
  throw "isolated executor produced $($Intents.Count) receipt intents; expected one"
}

Write-Host (
  "isolated execution complete: job=$JobId intent=$($Intents[0].Name) " +
  "not_before=$NotBefore"
)
