# bootstrap.ps1 — one-paste, idempotent setup of the SZL GPU bridge daemon.
# Elevated PowerShell:  irm https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/laptop/bootstrap.ps1 | iex
#
# What it does (and nothing else):
#   1. Creates a constrained py3.12 training environment and records the exact
#      installed package snapshot used by receipts.
#   2. Bakes the engine pubkey pin (the ONLY job-spec signer this laptop obeys).
#   3. Generates the laptop's ed25519 signing key (never leaves the machine).
#   4. Installs the verify-first dispatcher plus v1/v2 runners.
#   5. Registers a scheduled task that polls the public queue and runs jobs.
# It installs no remote-control software and opens no inbound ports.

$ErrorActionPreference = "Stop"
$Root = "C:\szl-bridge"
$Repo = "https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main"

Write-Host "== SZL GPU bridge bootstrap ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $Root, "$Root\jobs", "$Root\logs", "$Root\keys", "$Root\schema" | Out-Null

# ---- 1. pinned engine pubkey (trust root: verify BEFORE obeying anything) ----
$EnginePin = @'
{
  "kind": "szl-quant-engine-pubkey",
  "keyId": "5c6cf59741ade920",
  "publicKeySpkiBase64": "MCowBQYDK2VwAyEArBOmZZSDK+n7Qq1HJYbqNuX9YymnsRWbzSGHHnhsERM="
}
'@
# self-check BEFORE writing: keyId must equal sha256(SPKI)[:16] or this file
# could install a mislabeled trust root (fail closed, install nothing).
$PinObj = $EnginePin | ConvertFrom-Json
$SpkiBytes = [Convert]::FromBase64String($PinObj.publicKeySpkiBase64)
$Sha = [System.Security.Cryptography.SHA256]::Create()
$DerivedKeyId = (($Sha.ComputeHash($SpkiBytes) | ForEach-Object { $_.ToString("x2") }) -join "").Substring(0, 16)
if ($DerivedKeyId -ne $PinObj.keyId) {
  Write-Host "FATAL: engine pubkey self-check failed (derived $DerivedKeyId ≠ labeled $($PinObj.keyId)) — refusing to install a mislabeled trust root." -ForegroundColor Red
  exit 1
}
Set-Content -Path "$Root\keys\engine_pubkey.json" -Value $EnginePin -Encoding utf8
Write-Host "engine pubkey pinned (keyId $DerivedKeyId, self-check passed)"

# ---- 2. constrained training stack + exact installed snapshot ---------------
$CondaBase = "$env:USERPROFILE\miniconda3"
if (-not (Test-Path "$CondaBase\Scripts\conda.exe")) {
  Write-Host "installing Miniconda (user scope)..."
  $mc = "$env:TEMP\miniconda.exe"
  Invoke-WebRequest "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $mc
  Start-Process $mc -ArgumentList "/S","/InstallationType=JustMe","/AddToPath=0","/D=$CondaBase" -Wait
}
$Conda = "$CondaBase\Scripts\conda.exe"
if (-not (Test-Path "$CondaBase\envs\szl-bridge")) {
  & $Conda create -y -n szl-bridge python=3.12
}
$Pip = "$CondaBase\envs\szl-bridge\Scripts\pip.exe"
$Py  = "$CondaBase\envs\szl-bridge\python.exe"

# Windows compatibility constraints follow the upstream Unsloth lane:
# torch cu124-class; triton-windows required; two known-bad bitsandbytes
# Windows versions excluded. Exact resolved versions are frozen below and their
# sha256 is embedded in every v2 training receipt. This is honest evidence, not
# a claim that floating upstream resolution is reproducible forever.
& $Pip install --upgrade pip
& $Pip install torch --index-url https://download.pytorch.org/whl/cu124
& $Pip install "triton-windows" "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0" xformers
# Pin the fast-moving public interfaces used by the v2 contract. Lower-level GPU
# wheels remain hardware-resolved; the exact resolved environment is frozen and
# hashed immediately after installation.
& $Pip install `
  "unsloth==2026.7.4" `
  "huggingface_hub[cli]==1.24.0" `
  "PyNaCl==1.6.2" `
  "datasets==5.0.0" `
  "trl==1.8.0" `
  "peft==0.19.1"
if ($LASTEXITCODE -ne 0) { throw "training dependency installation failed" }
& $Pip freeze --all | Set-Content -Path "$Root\stack-freeze.txt" -Encoding utf8
$StackSha = (Get-FileHash "$Root\stack-freeze.txt" -Algorithm SHA256).Hash.ToLower()
Write-Host "training stack ready; exact package snapshot sha256 $StackSha"

# hf auth: reuse the laptop's existing login (the proven upload path).
& $Py -c "from huggingface_hub import HfApi; print('hf auth OK as', HfApi().whoami()['name'])"
if ($LASTEXITCODE -ne 0) {
  Write-Host "hf CLI is NOT logged in — run: $CondaBase\envs\szl-bridge\Scripts\hf.exe auth login" -ForegroundColor Yellow
  Write-Host "(bridge will poll but BLOCK uploads until login exists — fail closed)" -ForegroundColor Yellow
}

# ---- 3. laptop signing key (ed25519, never leaves this machine) -------------
if (-not (Test-Path "$Root\keys\laptop_key.pem")) {
  $KeygenPy = @'
from nacl.signing import SigningKey
import base64, hashlib, json, pathlib
sk = SigningKey.generate()
raw = bytes(sk)
pem = ("-----BEGIN SZL ED25519 SEED-----\n" + base64.b64encode(raw).decode() + "\n-----END SZL ED25519 SEED-----\n")
root = pathlib.Path(r"C:\szl-bridge\keys")
(root / "laptop_key.pem").write_text(pem)
spki = b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + bytes(sk.verify_key)
key_id = hashlib.sha256(spki).hexdigest()[:16]
(root / "laptop_pubkey.json").write_text(json.dumps({
  "kind": "szl-bridge-laptop-pubkey", "keyId": key_id,
  "publicKeySpkiBase64": base64.b64encode(spki).decode(),
}, indent=2))
print("laptop signing keyId:", key_id)
'@
  Set-Content -Path "$Root\keygen.py" -Value $KeygenPy -Encoding utf8
  & $Py "$Root\keygen.py"
  Remove-Item "$Root\keygen.py" -Force
}
$LapKeyId = (Get-Content "$Root\keys\laptop_pubkey.json" | ConvertFrom-Json).keyId
Write-Host ""
Write-Host ">>> ANNOUNCE THIS KEYID TO THE CLOUD SESSION: $LapKeyId <<<" -ForegroundColor Green
Write-Host "    (the cloud pins it before trusting any receipt from this laptop)"
Write-Host ""

# ---- 4. fetch dispatcher, both runners, helpers, and schemas ----------------
$Files = @(
  @("laptop/daemon.ps1", "$Root\daemon.ps1"),
  @("laptop/dispatcher.py", "$Root\dispatcher.py"),
  @("laptop/frontier_contract.py", "$Root\frontier_contract.py"),
  @("laptop/frontier_runtime.py", "$Root\frontier_runtime.py"),
  @("laptop/frontier_job.py", "$Root\frontier_job.py"),
  @("laptop/runjob.py", "$Root\runjob.py"),
  @("laptop/runjob_frontier.py", "$Root\runjob_frontier.py"),
  @("schema/jobspec.v1.json", "$Root\schema\jobspec.v1.json"),
  @("schema/jobspec.v2.json", "$Root\schema\jobspec.v2.json")
)
foreach ($Pair in $Files) {
  Invoke-WebRequest "$Repo/$($Pair[0])" -OutFile $Pair[1]
}
& $Py -m py_compile "$Root\dispatcher.py" "$Root\frontier_contract.py" "$Root\frontier_runtime.py" "$Root\frontier_job.py" "$Root\runjob.py" "$Root\runjob_frontier.py"
if ($LASTEXITCODE -ne 0) { throw "downloaded bridge Python failed compilation" }

# GGUF jobs are accepted only when a real llama-cli is discoverable for reload
# smoke. The frontier runner also searches existing szl-forge/llama.cpp paths.
$Llama = Get-Command llama-cli.exe -ErrorAction SilentlyContinue
if (-not $Llama) {
  Write-Host "llama-cli not on PATH — adapter/merged jobs remain available; GGUF jobs will return signed BLOCKED until llama-cli is installed." -ForegroundColor Yellow
}

# ---- 5. register scheduled task ---------------------------------------------
$TaskName = "SZL-GPU-Bridge"
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File $Root\daemon.ps1"
$Triggers = @(
  (New-ScheduledTaskTrigger -AtStartup),
  (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15))
)
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 26) -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Triggers -Settings $Settings -Principal $Principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "scheduled task '$TaskName' registered (startup + every 15 min; single-flight) and started."
Write-Host "logs: $Root\logs\daemon.log — bootstrap complete." -ForegroundColor Cyan
