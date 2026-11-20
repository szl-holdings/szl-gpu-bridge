# daemon.ps1 -- poll the public queue, verify-first, dispatch, and ledger.
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
# Fixed operator storage gate (docs/runbooks/workstation-storage-preflight.md).
# Its thresholds live here and are never read from a job spec.
$Preflight = "$Root\storage_preflight.py"
$PreflightEvidence = "$Root\logs\storage-preflight"

function Log($m) { "$(Get-Date -Format o)  $m" | Add-Content -Path $Log }

# Single-flight admission is an OS-owned exclusive handle, not a check-then-
# create sentinel. OpenOrCreate + FileShare.None is atomic on Windows; the
# handle stays live through the entire dispatch pass and is released by the
# operating system if this process exits unexpectedly. A leftover lock file
# is not proof of an active owner and is never deleted to force admission.
$LockHandle = $null
try {
  $LockHandle = [System.IO.File]::Open(
    $Lock,
    [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
  )
} catch {
  # Reporting a denied lease must not override the documented HOLD exit when
  # the log directory is absent or unwritable. Never attempt dispatch here.
  $leaseError = $_.Exception.GetType().Name
  try {
    Log "exclusive daemon lease unavailable ($leaseError) -- HOLD, no queue or job dispatch"
  } catch {
    Write-Warning "exclusive daemon lease unavailable ($leaseError), logging failed -- HOLD"
  }
  exit 78
}

$DaemonFailureExit = 0
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
    Log "picked up $jobId -- verify-first dispatcher will select an allowlisted local runner"

    # Storage admission is re-measured immediately before every dispatch, so a
    # restart or a retried job never relies on an earlier reading. A missing
    # script or any non-zero exit (78 = HOLD) refuses this dispatch without a
    # ledger mutation; the job is retried next cycle and this cycle stops here.
    if (-not (Test-Path -LiteralPath $Preflight -PathType Leaf)) {
      Log "$jobId STORAGE HOLD -- storage_preflight.py is absent; not dispatched, left off the ledger"
      break
    }
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $preflightReceipt = "$PreflightEvidence\$stamp-storage-preflight.json"
    & $Py -I -B $Preflight --root 'C:\' --samples 5 --interval-sec 2 --min-free-gib 50 --max-swing-gib 2 --receipt $preflightReceipt *>> $Log
    $preflightCode = $LASTEXITCODE
    if ($preflightCode -ne 0) {
      Log "$jobId STORAGE HOLD (preflight exit $preflightCode) -- not dispatched, left off the ledger for retry next cycle"
      break
    }
    Log "$jobId storage preflight passed (storage only; every other gate remains the dispatcher's)"

    # dispatcher.py verifies the DSSE envelope and pinned engine identity before
    # reading kind/jobId/output fields. Exit codes:
    #   0 = receipts uploaded (success or honest signed BLOCKED)
    #   3 = permanently refused (bad signature/schema/unsupported signed contract)
    #   7 = unsigned receipt intent awaits trusted signing/upload/readback
    # other = local infrastructure failure (retry next cycle)
    & $Py "$Root\dispatcher.py" $specPath *>> $Log
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      Add-Content -Path $Ledger -Value $jobId
      Log "$jobId complete (validated receipts pushed)"
    } elseif ($code -eq 3) {
      Add-Content -Path $Ledger -Value "$jobId  # REFUSED-UNVERIFIED-OR-UNSUPPORTED"
      Log "!! $jobId REFUSED -- ledgered as consumed; see logs\refused-specs.jsonl"
    } else {
      Log "$jobId LOCAL FAILURE (exit $code) -- left off the ledger for retry next cycle"
    }
  }
} catch {
  $DaemonFailureExit = 70
  Log "daemon unexpected exception; reconcile per-job ledger and receipts before retry: $($_.Exception.Message)"
} finally {
  $LockHandle.Dispose()
}
if ($DaemonFailureExit -ne 0) { exit $DaemonFailureExit }
