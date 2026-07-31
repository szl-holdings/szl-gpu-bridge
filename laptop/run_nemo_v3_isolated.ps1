param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^job-[A-Za-z0-9][A-Za-z0-9._-]*$')]
  [string]$JobId,

  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[0-9a-f]{40}$')]
  [string]$BridgeRevision,

  [Parameter(Mandatory = $true)]
  [ValidatePattern('^unsloth/unsloth@sha256:[0-9a-f]{64}$')]
  [string]$Image,

  [Parameter(Mandatory = $true)]
  [string]$BridgeSource,

  [Parameter(Mandatory = $true)]
  [string]$EnvelopePath,

  [string]$BridgeRoot = "C:\szl-bridge",
  [string]$HfCache = "C:\szl-bridge-cache\huggingface\hub",
  [string]$InputCache = "C:\szl-bridge-cache\inputs"
)

$ErrorActionPreference = "Stop"

if (
  $JobId -in @(
    "job-2026-nemo-v3-governed-attempt-2",
    "job-2026-nemo-v3-governed-successor-3",
    "job-2026-nemo-v3-governed-attempt-4"
  )
) {
  throw "job is quarantined and marked NEVER_DISPATCH"
}

if ($env:HF_TOKEN -or $env:HUGGING_FACE_HUB_TOKEN -or $env:GH_TOKEN) {
  throw "isolated execution refuses a host process containing HF or GitHub tokens"
}

$Docker = (Get-Command docker.exe -ErrorAction Stop).Source
$Git = (Get-Command git.exe -ErrorAction Stop).Source
$ImageMetadataText = & $Docker image inspect $Image
if ($LASTEXITCODE -ne 0) {
  throw "container image is not locally available by an immutable identifier: $Image"
}
try {
  $ImageMetadata = @($ImageMetadataText | ConvertFrom-Json)
} catch {
  throw "container image metadata was not valid JSON"
}
if ($ImageMetadata.Count -ne 1) {
  throw "container image metadata did not resolve exactly one image"
}
$ObservedImageId = [string]$ImageMetadata[0].Id
if ($ObservedImageId -notmatch '^sha256:[0-9a-f]{64}$') {
  throw "container image metadata has no immutable identifier"
}
$ExpectedImageId = ($Image -split "@", 2)[1]
if ($ObservedImageId -ne $ExpectedImageId) {
  throw "local image identifier drifted: $ObservedImageId != $Image"
}

$ObservedRevision = (& $Git -C $BridgeSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ObservedRevision -ne $BridgeRevision) {
  throw "bridge source is not the exact approved revision: $ObservedRevision"
}
if (& $Git -C $BridgeSource status --porcelain --untracked-files=no) {
  throw "bridge source has tracked modifications"
}

$ApprovedLauncher = Join-Path $BridgeSource "laptop\run_nemo_v3_isolated.ps1"
if (-not (Test-Path -LiteralPath $ApprovedLauncher -PathType Leaf)) {
  throw "approved isolated launcher is missing: $ApprovedLauncher"
}
if (-not $PSCommandPath) {
  throw "isolated launcher must run from a script file"
}
$InvokedLauncherSha256 = (
  Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256
).Hash.ToLowerInvariant()
$ApprovedLauncherSha256 = (
  Get-FileHash -LiteralPath $ApprovedLauncher -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($InvokedLauncherSha256 -ne $ApprovedLauncherSha256) {
  throw "running launcher does not match the exact approved bridge source"
}

$ProbeProgram = @'
import importlib.metadata
import json
import torch
import unsloth
import cryptography
import datasets
import transformers
import trl

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the immutable image")
properties = torch.cuda.get_device_properties(0)
packages = {
    name: importlib.metadata.version(name)
    for name in (
        "bitsandbytes",
        "cryptography",
        "datasets",
        "huggingface-hub",
        "peft",
        "torch",
        "trl",
        "unsloth",
        "unsloth-zoo",
        "xformers",
    )
}
receipt = {
    "cuda_available": True,
    "cuda_runtime": torch.version.cuda,
    "gpu_name": properties.name,
    "gpu_memory_bytes": properties.total_memory,
    "packages": packages,
}
print(
    "SZL_NEMO_IMAGE_PROBE_JSON="
    + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
)
'@
$ProbeArguments = @(
  "run",
  "--rm",
  "--interactive",
  "--gpus", "all",
  "--network", "none",
  "--read-only",
  "--cap-drop", "ALL",
  "--security-opt", "no-new-privileges:true",
  "--pids-limit", "256",
  "--tmpfs", "/tmp:rw,noexec,nosuid,size=1073741824",
  "--tmpfs", "/root/.cache:rw,noexec,nosuid,size=1073741824",
  "--workdir", "/tmp",
  "--env", "HF_HUB_OFFLINE=1",
  "--env", "TRANSFORMERS_OFFLINE=1",
  "--env", "XDG_CACHE_HOME=/tmp/cache",
  "--entrypoint", "python",
  $ObservedImageId,
  "-"
)
$ProbeOutput = ($ProbeProgram | & $Docker @ProbeArguments) -join "`n"
if ($LASTEXITCODE -ne 0) {
  throw "immutable training image failed its offline CUDA/package probe"
}
$ProbePrefix = "SZL_NEMO_IMAGE_PROBE_JSON="
$ProbeLines = @(
  $ProbeOutput -split "`r?`n" |
    Where-Object { $_.StartsWith($ProbePrefix) }
)
if ($ProbeLines.Count -ne 1) {
  throw "immutable training image emitted an invalid probe record"
}
$ProbeJson = $ProbeLines[0].Substring($ProbePrefix.Length)
$Probe = $ProbeJson | ConvertFrom-Json
if (
  $Probe.cuda_available -ne $true -or
  -not ($Probe.gpu_name -is [string]) -or
  -not $Probe.gpu_name.Trim() -or
  -not ($Probe.cuda_runtime -is [string]) -or
  -not $Probe.cuda_runtime.Trim()
) {
  throw "immutable training image probe has no usable CUDA identity"
}
foreach (
  $Name in @(
    "bitsandbytes",
    "cryptography",
    "datasets",
    "huggingface-hub",
    "peft",
    "torch",
    "trl",
    "unsloth",
    "unsloth-zoo",
    "xformers"
  )
) {
  $ObservedPackage = $Probe.packages.PSObject.Properties[$Name].Value
  if (-not ($ObservedPackage -is [string]) -or -not $ObservedPackage.Trim()) {
    throw "immutable training image is missing required package metadata: $Name"
  }
}
$ProbeSha = [System.Security.Cryptography.SHA256]::Create()
try {
  $EnvironmentProbeSha256 = (
    $ProbeSha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($ProbeJson)) |
      ForEach-Object { $_.ToString("x2") }
  ) -join ""
} finally {
  $ProbeSha.Dispose()
}

$JobSpec = [System.IO.Path]::GetFullPath($EnvelopePath)
$PendingRoot = Split-Path -Parent $JobSpec
$QueueRoot = Split-Path -Parent $PendingRoot
$EnvelopeSource = Split-Path -Parent $QueueRoot
$ExpectedJobSpec = [System.IO.Path]::GetFullPath(
  (Join-Path $EnvelopeSource "queue\pending\$JobId.json")
)
if ($JobSpec -cne $ExpectedJobSpec) {
  throw "verified envelope path does not select the exact governed job"
}
$EnvelopeRevision = (& $Git -C $EnvelopeSource rev-parse HEAD).Trim()
if (
  $LASTEXITCODE -ne 0 -or
  $EnvelopeRevision -notmatch '^[0-9a-f]{40}$' -or
  $EnvelopeRevision -eq $BridgeRevision
) {
  throw "envelope publication revision is not distinct immutable history"
}
$EnvelopeProtectedMain = (
  & $Git -C $EnvelopeSource rev-parse refs/remotes/origin/main
).Trim()
if (
  $LASTEXITCODE -ne 0 -or
  $EnvelopeProtectedMain -ne $EnvelopeRevision -or
  (& $Git -C $EnvelopeSource status --porcelain --untracked-files=all)
) {
  throw "envelope source is not exact clean protected Bridge main"
}
& $Git -C $EnvelopeSource merge-base --is-ancestor `
  $BridgeRevision `
  $EnvelopeRevision
if ($LASTEXITCODE -ne 0) {
  throw "signed execution revision is not protected envelope history"
}
$PrefetchReceipt = Join-Path $InputCache "$JobId-prefetch.json"
foreach ($Required in @($JobSpec, $PrefetchReceipt)) {
  if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
    throw "required isolated-execution input is missing: $Required"
  }
}
$Envelope = Get-Content -LiteralPath $JobSpec -Raw | ConvertFrom-Json
if (
  @($Envelope.signatures).Count -ne 1 -or
  [string]$Envelope.signatures[0].keyid -notmatch '^[0-9a-f]{16}$'
) {
  throw "signed job does not contain exactly one valid engine key ID"
}
$EnvelopeKeyId = [string]$Envelope.signatures[0].keyid
$SignedPayloadBytes = [Convert]::FromBase64String([string]$Envelope.payload)
$PayloadSha = [System.Security.Cryptography.SHA256]::Create()
try {
  $SignedPayloadSha256 = (
    $PayloadSha.ComputeHash($SignedPayloadBytes) |
      ForEach-Object { $_.ToString("x2") }
  ) -join ""
} finally {
  $PayloadSha.Dispose()
}
$KeysRoot = Join-Path $BridgeSource "keys"
$KeyringPath = Join-Path $KeysRoot "engine_keyring.json"
if (Test-Path -LiteralPath $KeyringPath -PathType Leaf) {
  $Keyring = Get-Content -LiteralPath $KeyringPath -Raw | ConvertFrom-Json
  if (
    $Keyring.kind -ne "szl-quant-engine-keyring" -or
    $Keyring.v -ne 1
  ) {
    throw "engine keyring contract is invalid"
  }
  $KeyEntry = $Keyring.keys.PSObject.Properties[$EnvelopeKeyId].Value
  if (
    $null -eq $KeyEntry -or
    $KeyEntry.status -ne "ACTIVE" -or
    [string]$KeyEntry.file -notmatch '^engine_pubkey(?:_[0-9a-f]{16})?\.json$'
  ) {
    throw "signed job engine key is not active for execution"
  }
  $EngineKey = Join-Path $KeysRoot ([string]$KeyEntry.file)
} else {
  $EngineKey = Join-Path $KeysRoot "engine_pubkey.json"
}
if (-not (Test-Path -LiteralPath $EngineKey -PathType Leaf)) {
  throw "enrolled engine public key is missing: $EngineKey"
}
$EnginePin = Get-Content -LiteralPath $EngineKey -Raw | ConvertFrom-Json
if ([string]$EnginePin.keyId -ne $EnvelopeKeyId) {
  throw "engine keyring entry differs from the selected public key"
}
$EngineSpki = [Convert]::FromBase64String([string]$EnginePin.publicKeySpkiBase64)
$EngineSha = [System.Security.Cryptography.SHA256]::Create()
$DerivedEngineKeyId = (
  ($EngineSha.ComputeHash($EngineSpki) | ForEach-Object { $_.ToString("x2") }) -join ""
).Substring(0, 16)
if ($DerivedEngineKeyId -ne $EnvelopeKeyId) {
  throw "selected engine public key bytes are mislabeled"
}
foreach ($RequiredDirectory in @($HfCache, $InputCache)) {
  if (-not (Test-Path -LiteralPath $RequiredDirectory -PathType Container)) {
    throw "required isolated-execution cache is missing: $RequiredDirectory"
  }
}

$Prefetch = Get-Content -LiteralPath $PrefetchReceipt -Raw | ConvertFrom-Json
if (
  $Prefetch.jobId -ne $JobId -or
  $Prefetch.signedJobPayloadSha256 -ne $SignedPayloadSha256 -or
  $Prefetch.model.repoId -ne $Prefetch.model.license.repoId -or
  $Prefetch.model.revision -ne $Prefetch.model.license.revision -or
  $Prefetch.model.license.repoType -ne "model" -or
  $Prefetch.model.license.expected -notin @($Prefetch.model.license.observed) -or
  [string]$Prefetch.model.license.readmeSha256 -notmatch '^[0-9a-f]{64}$' -or
  $Prefetch.remoteCodeExecuted -ne $false -or
  $Prefetch.credentialPersisted -ne $false
) {
  throw (
    "prefetch receipt does not preserve the exact-license/no-code/" +
    "no-credential boundary"
  )
}

$SandboxSource = Join-Path $BridgeRoot "sandbox-source-$JobId"
$SandboxWork = Join-Path $BridgeRoot "sandbox-work-$JobId"
$Jobs = Join-Path $BridgeRoot "jobs"
$JobRoot = Join-Path $Jobs $JobId
$Outbox = Join-Path $JobRoot "receipt-outbox"
$Control = Join-Path $BridgeRoot "control"
$Claims = Join-Path $Control "attempt-claims"
foreach ($NewDirectory in @($SandboxSource, $SandboxWork)) {
  if (Test-Path -LiteralPath $NewDirectory) {
    throw "one-attempt sandbox path already exists: $NewDirectory"
  }
  New-Item -ItemType Directory -Path $NewDirectory | Out-Null
}
foreach ($Directory in @($Jobs, $Control, $Claims)) {
  if (-not (Test-Path -LiteralPath $Directory)) {
    New-Item -ItemType Directory -Path $Directory | Out-Null
  }
}
if (-not (Test-Path -LiteralPath $JobRoot)) {
  New-Item -ItemType Directory -Path $JobRoot | Out-Null
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
New-Item `
  -ItemType Directory `
  -Path (Join-Path $SandboxSource "jobs\$JobId") | Out-Null
Copy-Item `
  -LiteralPath $EngineKey `
  -Destination (Join-Path $SandboxSource "keys\engine_pubkey.json")

# Prove that the exact copied source parses under the immutable training
# image interpreter before consuming the one-attempt claim. Hosted CI also
# compiles on Python 3.11, but this gate binds compatibility to the exact
# container image and exact source bytes selected for this execution.
$CompatibilityArguments = @(
  "run",
  "--rm",
  "--network", "none",
  "--read-only",
  "--cap-drop", "ALL",
  "--security-opt", "no-new-privileges:true",
  "--pids-limit", "256",
  "--mount", "type=bind,src=$SandboxSource,dst=/bridge,readonly",
  "--tmpfs", "/tmp:rw,noexec,nosuid,size=67108864",
  "--env", "PYTHONPYCACHEPREFIX=/tmp/pycache",
  "--entrypoint", "python",
  $ObservedImageId,
  "-m", "compileall", "-q", "-f", "/bridge"
)
& $Docker @CompatibilityArguments
$CompatibilityExitCode = $LASTEXITCODE
if ($CompatibilityExitCode -ne 0) {
  throw (
    "container-runtime source compatibility gate failed with exit code " +
    "$CompatibilityExitCode before the one-attempt claim"
  )
}

$Ledger = Join-Path $Jobs "seen.txt"
if (Test-Path -LiteralPath $Ledger -PathType Leaf) {
  $Seen = @(
    Get-Content -LiteralPath $Ledger |
      ForEach-Object { ($_ -split "#", 2)[0].Trim() } |
      Where-Object { $_ }
  )
  if ($JobId -in $Seen) {
    throw "one-attempt job is already present in the terminal ledger"
  }
}

# This durable CreateNew claim is the authoritative replay barrier. It is
# written only after source, prefetch, and sandbox preparation pass, but before
# Docker can start the GPU attempt. A crash after this point remains fail-closed
# and requires explicit owner recovery; an automatic retry cannot consume a
# second attempt.
$ClaimedAt = [DateTimeOffset]::UtcNow.ToString("o")
$ClaimPath = Join-Path $Claims "$JobId.json"
$Claim = [ordered]@{
  kind = "szl-nemo-v3-attempt-claim"
  v = 3
  jobId = $JobId
  jobEnvelopeSha256 = (
    (Get-FileHash -LiteralPath $JobSpec -Algorithm SHA256).Hash.ToLowerInvariant()
  )
  bridgeRevision = $BridgeRevision
  envelopeRevision = $EnvelopeRevision
  executionBridgeRevision = $BridgeRevision
  launcherSha256 = $ApprovedLauncherSha256
  trainingImage = $Image
  observedImageId = $ObservedImageId
  environmentProbeSha256 = $EnvironmentProbeSha256
  githubRunId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { $null }
  claimedAt = $ClaimedAt
}
$ClaimBytes = [System.Text.UTF8Encoding]::new($false).GetBytes(
  (($Claim | ConvertTo-Json -Depth 4) + "`n")
)
$ClaimStream = $null
try {
  $ClaimStream = [System.IO.File]::Open(
    $ClaimPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
  )
  $ClaimStream.Write($ClaimBytes, 0, $ClaimBytes.Length)
  $ClaimStream.Flush($true)
} catch [System.IO.IOException] {
  throw "one-attempt claim already exists or could not be created: $ClaimPath"
} finally {
  if ($null -ne $ClaimStream) {
    $ClaimStream.Dispose()
  }
}

$NotBefore = $ClaimedAt
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
  "--mount", "type=bind,src=$JobRoot,dst=/bridge/jobs/$JobId",
  "--mount", "type=bind,src=$JobSpec,dst=/job/spec.json,readonly",
  "--mount", "type=bind,src=$InputCache,dst=/inputs,readonly",
  "--mount", "type=bind,src=$HfCache,dst=/hf-cache,readonly",
  "--mount", "type=bind,src=$SandboxWork,dst=/workspace",
  "--workdir", "/workspace",
  "--tmpfs", "/tmp:rw,noexec,nosuid,size=4294967296",
  "--env", "SZL_INPUT_CACHE=/inputs",
  "--env", "SZL_RECEIPT_TRANSPORT=local-unsigned-outbox",
  "--env", "SZL_EXECUTION_ISOLATION=credentialless-networkless-container",
  "--env", "SZL_CONTAINER_IMAGE_REFERENCE=$Image",
  "--env", "SZL_CONTAINER_IMAGE_ID=$ObservedImageId",
  "--env", "SZL_CONTAINER_ENVIRONMENT_PROBE_SHA256=$EnvironmentProbeSha256",
  "--env", "SZL_ENVELOPE_REVISION=$EnvelopeRevision",
  "--env", "SZL_EXECUTION_BRIDGE_REVISION=$BridgeRevision",
  "--env", "SZL_LAUNCHER_SHA256=$ApprovedLauncherSha256",
  "--env", "HF_HUB_OFFLINE=1",
  "--env", "HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
  "--env", "TRANSFORMERS_OFFLINE=1",
  "--env", "HF_DATASETS_OFFLINE=1",
  "--env", "HF_HOME=/tmp/huggingface",
  "--env", "HF_HUB_CACHE=/hf-cache",
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
