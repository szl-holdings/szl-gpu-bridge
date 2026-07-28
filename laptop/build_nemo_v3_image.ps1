param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[0-9a-f]{40}$')]
  [string]$BridgeRevision,

  [Parameter(Mandatory = $true)]
  [string]$BridgeSource,

  [ValidatePattern('^[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9][A-Za-z0-9._-]*$')]
  [string]$ImageTag = "szl-nemo-v3:local",

  [string]$ReceiptRoot = "C:\szl-bridge\image-receipts"
)

$ErrorActionPreference = "Stop"

$Docker = (Get-Command docker.exe -ErrorAction Stop).Source
$Git = (Get-Command git.exe -ErrorAction Stop).Source
$Dockerfile = Join-Path $BridgeSource "laptop\Dockerfile.nemo-v3"
if (-not (Test-Path -LiteralPath $Dockerfile -PathType Leaf)) {
  throw "Nemo v3 Dockerfile is missing: $Dockerfile"
}

$ObservedRevision = (& $Git -C $BridgeSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ObservedRevision -ne $BridgeRevision) {
  throw "bridge source is not the exact requested revision: $ObservedRevision"
}
if (& $Git -C $BridgeSource status --porcelain --untracked-files=no) {
  throw "bridge source has tracked modifications"
}

$BaseImage = (
  "pytorch/pytorch@sha256:" +
  "417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385"
)
& $Docker image inspect $BaseImage *> $null
if ($LASTEXITCODE -ne 0) {
  throw "digest-pinned PyTorch base image is not present locally: $BaseImage"
}

$BuildArguments = @(
  "build",
  "--file", $Dockerfile,
  "--build-arg", "BRIDGE_REVISION=$BridgeRevision",
  "--tag", $ImageTag,
  $BridgeSource
)
& $Docker @BuildArguments
if ($LASTEXITCODE -ne 0) {
  throw "Nemo v3 image build failed"
}

$ObservedImageId = (& $Docker image inspect --format "{{.Id}}" $ImageTag).Trim()
if (
  $LASTEXITCODE -ne 0 -or
  $ObservedImageId -notmatch '^sha256:[0-9a-f]{64}$'
) {
  throw "built image did not resolve to an immutable local image ID"
}
$ImageMetadataText = (& $Docker image inspect $ObservedImageId) -join "`n"
if ($LASTEXITCODE -ne 0) {
  throw "built image metadata could not be inspected"
}
try {
  $ImageMetadata = @($ImageMetadataText | ConvertFrom-Json)
} catch {
  throw "built image metadata was not valid JSON"
}
if ($ImageMetadata.Count -ne 1) {
  throw "built image metadata did not resolve exactly one image"
}
$ObservedRevisionLabel = [string](
  $ImageMetadata[0].Config.Labels.'org.opencontainers.image.revision'
)
if ($ObservedRevisionLabel -ne $BridgeRevision) {
  throw "built image revision label does not match exact bridge revision"
}

$SmokeProgram = @'
import importlib.metadata
import json
import torch
import unsloth

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the immutable image")

properties = torch.cuda.get_device_properties(0)
receipt = json.dumps({
    "cuda_available": True,
    "cuda_runtime": torch.version.cuda,
    "gpu_name": properties.name,
    "gpu_memory_bytes": properties.total_memory,
    "packages": {
        name: importlib.metadata.version(name)
        for name in (
            "bitsandbytes",
            "datasets",
            "huggingface-hub",
            "peft",
            "pynacl",
            "torch",
            "trl",
            "unsloth",
            "unsloth-zoo",
            "xformers",
        )
    },
}, sort_keys=True)
print("SZL_NEMO_IMAGE_SMOKE_JSON=" + receipt)
'@
$SmokeArguments = @(
  "run",
  "--rm",
  "--interactive",
  "--gpus", "all",
  "--network", "none",
  "--read-only",
  "--cap-drop", "ALL",
  "--security-opt", "no-new-privileges:true",
  "--tmpfs", "/tmp:rw,noexec,nosuid,size=1073741824",
  "--tmpfs", "/root/.cache:rw,noexec,nosuid,size=1073741824",
  "--entrypoint", "python",
  $ObservedImageId,
  "-"
)
$SmokeOutput = ($SmokeProgram | & $Docker @SmokeArguments) -join "`n"
if ($LASTEXITCODE -ne 0) {
  throw "offline CUDA import smoke failed"
}
$SmokePrefix = "SZL_NEMO_IMAGE_SMOKE_JSON="
$SmokeLines = @(
  $SmokeOutput -split "`r?`n" |
    Where-Object { $_.StartsWith($SmokePrefix) }
)
if ($SmokeLines.Count -ne 1) {
  throw "offline CUDA smoke did not emit exactly one marked JSON record"
}
$Smoke = $SmokeLines[0].Substring($SmokePrefix.Length) | ConvertFrom-Json
if ($Smoke.cuda_available -ne $true) {
  throw "CUDA smoke did not report an available device"
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
  $ObservedPackage = $Smoke.packages.PSObject.Properties[$Name].Value
  if ($ObservedPackage -ne $ExpectedPackages[$Name]) {
    throw (
      "offline CUDA smoke package mismatch: " +
      "$Name=$ObservedPackage expected=$($ExpectedPackages[$Name])"
    )
  }
}

if (-not (Test-Path -LiteralPath $ReceiptRoot -PathType Container)) {
  New-Item -ItemType Directory -Path $ReceiptRoot -Force | Out-Null
}
$ReceiptPath = Join-Path $ReceiptRoot "nemo-v3-image-$BridgeRevision.json"
$Receipt = [ordered]@{
  schema = "szl-nemo-v3-image-build-receipt-v1"
  bridgeRevision = $BridgeRevision
  baseImage = $BaseImage
  dockerfileSha256 = (
    Get-FileHash -LiteralPath $Dockerfile -Algorithm SHA256
  ).Hash.ToLower()
  imageTag = $ImageTag
  imageId = $ObservedImageId
  observedRevisionLabel = $ObservedRevisionLabel
  smoke = $Smoke
  observedAt = [DateTimeOffset]::UtcNow.ToString("o")
}
$Receipt | ConvertTo-Json -Depth 8 | Set-Content `
  -LiteralPath $ReceiptPath `
  -Encoding utf8

Write-Host "immutable Nemo v3 image ready: $ObservedImageId"
Write-Host "build receipt: $ReceiptPath"
