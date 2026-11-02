param(
  [string]$BootstrapPath = (Join-Path $PSScriptRoot "..\laptop\bootstrap.ps1")
)

# Execute only the dependency block with an in-process pip fixture. No real
# installer, network request, credential, signing key, GPU or task is touched.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$source = Get-Content -LiteralPath $BootstrapPath -Raw
$start = $source.IndexOf("# Windows compatibility constraints")
$end = $source.IndexOf("# hf auth:", $start)
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
  if (($arguments[0] -eq "freeze") -and ($LASTEXITCODE -eq 0)) {
    "synthetic-package==1.0"
  }
}

function Test-InstallCase([int]$FailureAt) {
  $script:failureAt = $FailureAt
  $script:nativeCalls = @()
  $Root = Join-Path ([IO.Path]::GetTempPath()) ("szl-bootstrap-test-" + [guid]::NewGuid())
  $Pip = "Invoke-BootstrapPipFixture"
  New-Item -ItemType Directory -Path $Root | Out-Null
  try {
    $failed = $false
    $output = @()
    try { $output = @(& $installBlock 6>&1) } catch { $failed = $true }
    $expectedFailure = $FailureAt -ge 0
    if ($failed -ne $expectedFailure) {
      throw "dependency call $FailureAt failure was not preserved"
    }
    $expectedCalls = 5
    if ($expectedFailure) { $expectedCalls = $FailureAt + 1 }
    if ($script:nativeCalls.Count -ne $expectedCalls) {
      throw "dependency installation continued after failed call $FailureAt"
    }
    if (($FailureAt -ge 0) -and ($FailureAt -lt 4) -and
        (Test-Path -LiteralPath (Join-Path $Root "stack-freeze.txt"))) {
      throw "dependency failure still produced a stack snapshot"
    }
    if (-not $expectedFailure) {
      $snapshot = Get-Content -LiteralPath (Join-Path $Root "stack-freeze.txt") -Raw
      if ($snapshot.Trim() -ne "synthetic-package==1.0") {
        throw "successful installation did not preserve the exact snapshot"
      }
      if (($output | Out-String) -notmatch "training stack ready") {
        throw "successful installation did not report its snapshot"
      }
    }
  } finally {
    Remove-Item -LiteralPath $Root -Recurse -Force
  }
}

foreach ($failure in 0..4) { Test-InstallCase $failure }
Test-InstallCase -1
Write-Host "PASS: four install failures, snapshot failure, and successful installation"
