# daemon.ps1 — poll the public queue, verify, run. Fail closed; never crash the loop host.
# Runs as a scheduled task (see bootstrap.ps1). Single-flight; seen-ledger for idempotency.

$ErrorActionPreference = "Stop"
$Root = "C:\szl-bridge"
$Log = "$Root\logs\daemon.log"
$Lock = "$Root\daemon.lock"
$Ledger = "$Root\jobs\seen.txt"
$Py = "$env:USERPROFILE\miniconda3\envs\szl-bridge\python.exe"
$Api = "https://api.github.com/repos/szl-holdings/szl-gpu-bridge/contents/queue/pending"
$Raw = "https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/queue/pending"

function Log($m) { "$(Get-Date -Format o)  $m" | Add-Content -Path $Log }

# single flight — a 20h training run must not be trampled by the 15-min trigger
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
  Get-Content $Ledger | ForEach-Object { if ($_ -match '\S') { $seen[$_.Trim()] = $true } }

  # list pending jobs (public API, no auth; cachebuster defeats CDN staleness)
  $bust = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  $resp = Invoke-RestMethod -Uri "$Api`?t=$bust" -Headers @{ "User-Agent" = "szl-gpu-bridge-daemon"; "Accept" = "application/vnd.github+json" }
  $files = @($resp | Where-Object { $_.name -like "job-*.json" } | Sort-Object name)
  Log "poll: $($files.Count) pending spec(s)"

  foreach ($f in $files) {
    $jobId = $f.name -replace '\.json$', ''
    if ($seen[$jobId]) { continue }

    $specPath = "$Root\jobs\$($f.name)"
    Invoke-WebRequest -Uri "$Raw/$($f.name)?t=$bust" -OutFile $specPath -Headers @{ "User-Agent" = "szl-gpu-bridge-daemon" }
    Log "picked up $jobId — handing to runjob.py (verify-first, fail-closed)"

    # runjob.py verifies the DSSE envelope BEFORE acting; exit codes:
    # 0 = receipts uploaded (success or honest BLOCKED)
    # 3 = spec REFUSED as unverifiable (bad sig/pin — permanently bad, never retried)
    # other = local infra failure (retried next cycle)
    & $Py "$Root\runjob.py" $specPath *>> $Log
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      Add-Content -Path $Ledger -Value $jobId
      Log "$jobId complete (receipts pushed)"
    } elseif ($code -eq 3) {
      Add-Content -Path $Ledger -Value "$jobId  # REFUSED-UNVERIFIED"
      Log "!! $jobId REFUSED — spec did not verify against the pinned engine key. Ledgered as consumed (a bad signature never heals). See logs\refused-specs.jsonl"
    } else {
      Log "$jobId LOCAL FAILURE (exit $code) — left OFF the ledger for retry next cycle"
    }
  }
} catch {
  Log "daemon error (fail closed, no state mutated): $($_.Exception.Message)"
} finally {
  Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
