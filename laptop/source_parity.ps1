#requires -Version 5.1
<#
Compare two reviewed UTF-8 script files without writing to disk.
Only CRLF versus LF is ignored: no arbitrary whitespace, case, BOM, code,
path, signature, source revision, or line-content differences are ignored.
This is source *content* parity, not signer or authorized owner attestation.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$ExpectedSource,
  [Parameter(Mandatory = $true)][string]$InstalledSource
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-CanonicalSourceSha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw 'SOURCE_PARITY_FILE_ABSENT'
  }
  $bytes = [IO.File]::ReadAllBytes($Path)
  $strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
  try {
    $text = $strictUtf8.GetString($bytes)
  } catch {
    throw 'SOURCE_PARITY_INVALID_UTF8'
  }
  if ($text.Contains([char]0)) {
    throw 'SOURCE_PARITY_NUL'
  }
  $canonical = $text.Replace("`r`n", "`n")
  if ($canonical.Contains("`r")) {
    throw 'SOURCE_PARITY_STRAY_CR'
  }
  $digest = [Security.Cryptography.SHA256]::Create()
  try {
    $normalizedBytes = $strictUtf8.GetBytes($canonical)
    return ([BitConverter]::ToString($digest.ComputeHash($normalizedBytes))).Replace('-', '').ToLowerInvariant()
  } finally {
    $digest.Dispose()
  }
}

$expected = Get-CanonicalSourceSha256 -Path $ExpectedSource
$actual = Get-CanonicalSourceSha256 -Path $InstalledSource
if ($expected -cne $actual) {
  throw 'SOURCE_PARITY_MISMATCH'
}
Write-Output 'SOURCE_PARITY_VERIFIED'
