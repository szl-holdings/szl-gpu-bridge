param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^job-[A-Za-z0-9][A-Za-z0-9._-]*$')]
  [string]$JobId,

  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[0-9a-f]{40}$')]
  [string]$BridgeRevision,

  [Parameter(Mandatory = $true)]
  [ValidatePattern('^sha256:[0-9a-f]{64}$')]
  [string]$Image,

  [Parameter(Mandatory = $true)]
  [string]$BridgeSource,

  [string]$BridgeRoot = "C:\szl-bridge",
  [string]$HfCache = "C:\szl-bridge-cache\huggingface\hub",
  [string]$InputCache = "C:\szl-bridge-cache\inputs",
  [string]$ImageReceiptRoot = "C:\szl-bridge\image-receipts"
)

$ErrorActionPreference = "Stop"

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
if ($ObservedImageId -ne $Image) {
  throw "local image identifier drifted: $ObservedImageId != $Image"
}
$ObservedRevisionLabel = [string](
  $ImageMetadata[0].Config.Labels.'org.opencontainers.image.revision'
)
if ($ObservedRevisionLabel -ne $BridgeRevision) {
  throw "container image revision label does not match exact bridge revision"
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

$Dockerfile = Join-Path $BridgeSource "laptop\Dockerfile.nemo-v3"
$ImageReceipt = Join-Path `
  $ImageReceiptRoot `
  "nemo-v3-image-$BridgeRevision.json"
foreach ($Required in @($Dockerfile, $ImageReceipt)) {
  if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
    throw "trusted local-image build input is missing: $Required"
  }
}
$ImageDockerfileSha256 = (
  Get-FileHash -LiteralPath $Dockerfile -Algorithm SHA256
).Hash.ToLowerInvariant()
$ExpectedBaseImage = (
  "pytorch/pytorch@sha256:" +
  "417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385"
)
$BuildReceipt = Get-Content -LiteralPath $ImageReceipt -Raw | ConvertFrom-Json
if (
  $BuildReceipt.schema -ne "szl-nemo-v3-image-build-receipt-v1" -or
  $BuildReceipt.bridgeRevision -ne $BridgeRevision -or
  $BuildReceipt.baseImage -ne $ExpectedBaseImage -or
  $BuildReceipt.imageId -ne $ObservedImageId -or
  $BuildReceipt.observedRevisionLabel -ne $ObservedRevisionLabel -or
  $BuildReceipt.dockerfileSha256 -ne $ImageDockerfileSha256 -or
  $BuildReceipt.smoke.cuda_available -ne $true -or
  -not ($BuildReceipt.smoke.gpu_name -is [string]) -or
  -not $BuildReceipt.smoke.gpu_name.Trim() -or
  -not ($BuildReceipt.smoke.cuda_runtime -is [string]) -or
  -not $BuildReceipt.smoke.cuda_runtime.Trim()
) {
  throw "local image build receipt does not bind the approved image and revision"
}
$ExpectedPackages = [ordered]@{
  "bitsandbytes" = "0.50.0"
  "datasets" = "4.3.0"
  "huggingface-hub" = "1.24.0"
  "peft" = "0.19.1"
  "pynacl" = "1.6.2"
  "trl" = "0.24.0"
  "unsloth" = "2026.7.4"
  "unsloth-zoo" = "2026.7.4"
  "xformers" = "0.0.32.post2"
}
foreach ($Name in $ExpectedPackages.Keys) {
  $ObservedPackage = $BuildReceipt.smoke.packages.PSObject.Properties[$Name].Value
  if ($ObservedPackage -ne $ExpectedPackages[$Name]) {
    throw "local image build receipt has an unapproved package version: $Name"
  }
}
$ImageBuildReceiptSha256 = (
  Get-FileHash -LiteralPath $ImageReceipt -Algorithm SHA256
).Hash.ToLowerInvariant()

$JobSpec = Join-Path $BridgeSource "queue\pending\$JobId.json"
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
  v = 2
  jobId = $JobId
  jobEnvelopeSha256 = (
    (Get-FileHash -LiteralPath $JobSpec -Algorithm SHA256).Hash.ToLowerInvariant()
  )
  bridgeRevision = $BridgeRevision
  launcherSha256 = $ApprovedLauncherSha256
  trainingImage = $Image
  observedImageId = $ObservedImageId
  observedRevisionLabel = $ObservedRevisionLabel
  imageBuildReceiptSha256 = $ImageBuildReceiptSha256
  imageDockerfileSha256 = $ImageDockerfileSha256
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

$ImageEvidenceArguments = @(
  "--env", "SZL_CONTAINER_IMAGE_REVISION=$ObservedRevisionLabel"
)
if ($ImageBuildReceiptSha256) {
  $ImageEvidenceArguments += @(
    "--env",
    "SZL_CONTAINER_IMAGE_BUILD_RECEIPT_SHA256=$ImageBuildReceiptSha256"
  )
}
if ($ImageDockerfileSha256) {
  $ImageEvidenceArguments += @(
    "--env",
    "SZL_CONTAINER_IMAGE_DOCKERFILE_SHA256=$ImageDockerfileSha256"
  )
}

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
  "--mount", "type=bind,src=$HfCache,dst=/root/.cache/huggingface/hub,readonly",
  "--mount", "type=bind,src=$SandboxWork,dst=/workspace",
  "--tmpfs", "/tmp:rw,noexec,nosuid,size=4294967296",
  "--env", "SZL_INPUT_CACHE=/inputs",
  "--env", "SZL_RECEIPT_TRANSPORT=local-unsigned-outbox",
  "--env", "SZL_EXECUTION_ISOLATION=credentialless-networkless-container",
  "--env", "SZL_CONTAINER_IMAGE_REFERENCE=$Image",
  "--env", "SZL_CONTAINER_IMAGE_ID=$ObservedImageId",
  "--env", "SZL_LAUNCHER_SHA256=$ApprovedLauncherSha256"
) + $ImageEvidenceArguments + @(
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
