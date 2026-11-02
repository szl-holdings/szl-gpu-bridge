param(
  [string]$BootstrapPath = (Join-Path $PSScriptRoot "..\laptop\bootstrap.ps1")
)

# Execute only the dependency block with an in-process pip fixture. No real
# installer, network request, credential, signing key, GPU or task is touched.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$source = Get-Content -LiteralPath $BootstrapPath -Raw
$start = $source.IndexOf("# Windows compatibility constraints")
$end = if ($start -ge 0) { $source.IndexOf("# hf auth:", $start) } else { -1 }
if (($start -lt 0) -or ($end -le $start)) {
  throw "bootstrap dependency block was not found"
}
$installBlock = [ScriptBlock]::Create($source.Substring($start, $end - $start))

function Invoke-BootstrapPipFixture {
  $arguments = @($args | ForEach-Object { [string]$_ })
  $callIndex = $script:nativeCalls.Count
  $script:nativeCalls += ,$arguments
  $global:LASTEXITCODE = 0
  if ($callIndex -eq $script:failureAt) { $global:LASTEXITCODE = 17 }
  if ($arguments[0] -eq "freeze") {
    if ($LASTEXITCODE -eq 0) {
      "synthetic-package==1.0"
    } elseif ($script:partialFreeze) {
      "partial-package==0.0"
    }
  }
}

function Test-InstallCase(
  [int]$FailureAt,
  [bool]$HasSnapshot = $false,
  [bool]$PartialFreeze = $false
) {
  $script:failureAt = $FailureAt
  $script:partialFreeze = $PartialFreeze
  $script:nativeCalls = @()
  $caseLabel = "dependency call $FailureAt (existing snapshot=$HasSnapshot; partial freeze=$PartialFreeze)"
  $Root = Join-Path ([IO.Path]::GetTempPath()) ("szl-bootstrap-test-" + [guid]::NewGuid())
  $Pip = "Invoke-BootstrapPipFixture"
  New-Item -ItemType Directory -Path $Root | Out-Null
  $snapshotPath = Join-Path $Root "stack-freeze.txt"
  $originalBytes = [Text.Encoding]::UTF8.GetBytes(
    "last-valid-package==9.0" + [Environment]::NewLine
  )
  if ($HasSnapshot) { [IO.File]::WriteAllBytes($snapshotPath, $originalBytes) }
  try {
    $failed = $false
    $failureMessage = ""
    $output = [Collections.Generic.List[object]]::new()
    try {
      & $installBlock 6>&1 | ForEach-Object { $output.Add($_) }
    } catch {
      $failed = $true
      $failureMessage = $_.Exception.Message
    }
    $expectedFailure = $FailureAt -ge 0
    if ($failed -ne $expectedFailure) {
      throw "$caseLabel failure was not preserved: $failureMessage"
    }
    $expectedMessages = @(
      "pip toolchain upgrade failed",
      "CUDA torch installation failed",
      "Windows GPU support installation failed",
      "training dependency installation failed",
      "training dependency snapshot failed"
    )
    if ($expectedFailure -and ($failureMessage -ne $expectedMessages[$FailureAt])) {
      throw "$caseLabel raised an unexpected failure: $failureMessage"
    }
    $expectedCalls = 5
    if ($expectedFailure) { $expectedCalls = $FailureAt + 1 }
    if ($script:nativeCalls.Count -ne $expectedCalls) {
      throw "dependency installation continued after failed call $FailureAt"
    }
    if ($expectedFailure) {
      if ($HasSnapshot) {
        $actual = [Convert]::ToBase64String([IO.File]::ReadAllBytes($snapshotPath))
        if ($actual -ne [Convert]::ToBase64String($originalBytes)) {
          throw "dependency failure changed the last valid stack snapshot"
        }
      } elseif (Test-Path -LiteralPath $snapshotPath) {
        throw "dependency failure published a stack snapshot"
      }
      if (($output | Out-String) -match "training stack ready") {
        throw "dependency failure reported the training stack ready"
      }
    } else {
      $snapshot = Get-Content -LiteralPath $snapshotPath -Raw
      if ($snapshot.Trim() -ne "synthetic-package==1.0") {
        throw "successful installation did not preserve the exact snapshot"
      }
      $expectedHash = (Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash.ToLower()
      if (($output | Out-String) -notmatch ("training stack ready; exact package snapshot sha256 " + $expectedHash)) {
        throw "successful installation did not report its exact snapshot hash"
      }
    }
    $expectedFiles = if ($HasSnapshot -or -not $expectedFailure) { 1 } else { 0 }
    if (@(Get-ChildItem -LiteralPath $Root -File).Count -ne $expectedFiles) {
      throw "dependency setup left a staged snapshot behind"
    }
  } finally {
    Remove-Item -LiteralPath $Root -Recurse -Force
  }
}

foreach ($hasSnapshot in @($false, $true)) {
  foreach ($failure in 0..3) { Test-InstallCase $failure $hasSnapshot }
  Test-InstallCase 4 $hasSnapshot $false
  Test-InstallCase 4 $hasSnapshot $true
  Test-InstallCase -1 $hasSnapshot
}
Write-Host "PASS: 14 offline dependency cases preserve prior snapshots and publish only successful freezes"
