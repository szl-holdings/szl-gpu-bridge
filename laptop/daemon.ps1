# daemon.ps1 — poll the public queue, verify-first, dispatch, and ledger.
# Runs as a scheduled task (see bootstrap.ps1). Single-flight and fail-closed.

param(
  [ValidatePattern('^job-[A-Za-z0-9][A-Za-z0-9._-]*$')]
  [string]$OnlyJobId = ""
)

$ErrorActionPreference = "Stop"
$Root = "C:\szl-bridge"
$Log = "$Root\logs\daemon.log"
$Lock = "$Root\daemon.lock"
$Ledger = "$Root\jobs\seen.txt"
$Py = "$env:USERPROFILE\miniconda3\envs\szl-bridge\python.exe"
$Api = "https://api.github.com/repos/szl-holdings/szl-gpu-bridge/contents/queue/pending"
$Raw = "https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/queue/pending"

function Log($m) { "$(Get-Date -Format o)  $m" | Add-Content -Path $Log }

# single flight — a long training run must not be trampled by the 15-min trigger
if (Test-Path $Lock) {
  $age = (Get-Date) - (Get-Item $Lock).LastWriteTime
  if ($age.TotalHours -lt 26) { Log "lock present (age $([int]$age.TotalMinutes)m) — another run in flight, exiting"; exit 0 }
  Log "stale lock (> 26h) — clearing"
  Remove-Item $Lock -Force
}
New-Item -ItemType File -Path $Lock -Force | Out-Null

try {
  if (-not (Test-Path $Ledger)) { New-Item -ItemType File -Path $Ledger -Force | Out-Null }
  $seen = @{}
  Get-Content $Ledger | ForEach-Object { if ($_ -match '\S') { $seen[($_ -split '\s+#')[0].Trim()] = $true } }

  # List pending specs over the public GitHub API. A cachebuster avoids CDN stale
  # reads; no repository credential or inbound laptop access is required.
  $bust = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  $resp = Invoke-RestMethod -Uri "$Api`?t=$bust" -Headers @{ "User-Agent" = "szl-gpu-bridge-daemon"; "Accept" = "application/vnd.github+json" }
  $allFiles = @($resp | Where-Object { $_.name -like "job-*.json" } | Sort-Object name)
  if ([string]::IsNullOrWhiteSpace($OnlyJobId)) {
    $files = $allFiles
  } else {
    $files = @($allFiles | Where-Object { $_.name -eq "${OnlyJobId}.json" })
    if (-not $seen[$OnlyJobId] -and $files.Count -ne 1) {
      throw "requested job is not present in the public queue: $OnlyJobId"
    }
  }
  Log "poll: $($files.Count) pending spec(s)"

  foreach ($f in $files) {
    $jobId = $f.name -replace '\.json$', ''
    if ($seen[$jobId]) { continue }

    $specPath = "$Root\jobs\$($f.name)"
    Invoke-WebRequest -Uri "$Raw/$($f.name)?t=$bust" -OutFile $specPath -Headers @{ "User-Agent" = "szl-gpu-bridge-daemon" }
    Log "picked up $jobId — verify-first dispatcher will select an allowlisted local runner"

    # dispatcher.py verifies the DSSE envelope and pinned engine identity before
    # reading kind/jobId/output fields. Exit codes:
    #   0 = receipts uploaded (success or honest signed BLOCKED)
    #   3 = permanently refused (bad signature/schema/unsupported signed contract)
    # other = local infrastructure failure (retry next cycle)
    & $Py "$Root\dispatcher.py" $specPath *>> $Log
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      Add-Content -Path $Ledger -Value $jobId
      Log "$jobId complete (validated receipts pushed)"
    } elseif ($code -eq 3) {
      Add-Content -Path $Ledger -Value "$jobId  # REFUSED-UNVERIFIED-OR-UNSUPPORTED"
      Log "!! $jobId REFUSED — ledgered as consumed; see logs\refused-specs.jsonl"
    } else {
      Log "$jobId LOCAL FAILURE (exit $code) — left off the ledger for retry next cycle"
    }
  }
} catch {
  Log "daemon error (fail closed, no job ledger mutation): $($_.Exception.Message)"
} finally {
  Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
