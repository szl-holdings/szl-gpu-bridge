# Windows GPU bridge watchdog

The GPU bridge already uses an outbound, verify-first pull model. The Windows
watchdog adds a separate liveness plane without converting that model into a
remote shell or opening an inbound port.

## Three-task closure

| Task | Cadence | Role |
|---|---:|---|
| `SZL-GPU-Bridge` | startup + every 15 minutes | Runs the signed queue dispatcher and long-running ML jobs. |
| `SZL-GPU-Bridge-Watchdog` | startup + every 5 minutes | Reconciles the primary task, daemon-log freshness, and guardian. |
| `SZL-GPU-Bridge-Guardian` | startup + every 15 minutes | Independently reconciles the primary task and watchdog. |

Every task uses `MultipleInstances IgnoreNew`. The primary retains a 26-hour
execution limit so a legitimate training run is not killed by a short watchdog
window. The two observers have ten-minute limits and recover each other.

## Recovery discipline

The watchdog can register, enable, or start only those three exact task names.
It consumes only `C:\szl-bridge\daemon.ps1`, `watchdog.ps1`, and the locally
written fixed configuration. It cannot download a command, invoke caller-supplied
PowerShell, change a model job, read a Hub or GitHub credential value, or open a
network listener.

A task receives at most three starts in a rolling thirty-minute window. Further
attempts enter `COOLDOWN` and the next scheduled cycle reassesses the condition.
This prevents a malformed dependency or unavailable provider from becoming a
restart storm.

Expected degraded recovery cycles finish normally after recording their state.
Only an unhandled watchdog-engine failure returns an error to Task Scheduler,
which then applies its own bounded restart policy.

## User-scoped authentication

The bridge's existing Hugging Face login is user-scoped. The installer therefore
preserves the principal of the current `SZL-GPU-Bridge` task and refuses to move
the bridge to `SYSTEM`, `LOCAL SERVICE`, or `NETWORK SERVICE`. All three tasks use
the same S4U principal, so unattended execution does not require an interactive
login while retaining the existing credential boundary.

The installation workflow never prints or uploads the principal. Its artifact
contains only task names, task states, task result codes, timestamps, file
hashes, the local receipt-chain tip, the source revision, and explicit negative
authority flags.

## State and evidence

The watchdog writes state atomically to:

```text
C:\szl-bridge\watchdog-state.json
```

Each cycle appends a SHA-256-linked record to:

```text
C:\szl-bridge\logs\watchdog-receipts.ndjson
C:\szl-bridge\logs\watchdog-receipt-tip.txt
```

Logs and receipt files rotate after eight MiB and retain eight archives. Receipt
records state that remote commands were not accepted, credential values were not
read, and inbound ports were not opened.

## Installation

The protected-main workflow
`.github/workflows/windows-watchdog-install.yml` runs on an existing Windows x64
self-hosted bridge node after the package merges. It first requires either the
existing `SZL-GPU-Bridge` task or `C:\szl-bridge\daemon.ps1`, preventing the
package from silently installing on an unrelated Windows runner.

Manual local installation remains idempotent:

```powershell
Set-Location <szl-gpu-bridge-checkout>\laptop
.\install_watchdog.ps1 -Root 'C:\szl-bridge'
```

Run it from an elevated PowerShell session. It copies the reviewed watchdog,
keeps an existing installed daemon in place, writes the fixed configuration,
sets a restricted ACL, registers all three tasks, and starts both observers.

## Verification

```powershell
Get-ScheduledTask `
  'SZL-GPU-Bridge', `
  'SZL-GPU-Bridge-Watchdog', `
  'SZL-GPU-Bridge-Guardian' |
  Select-Object TaskName, State

powershell -NoProfile -ExecutionPolicy RemoteSigned `
  -File 'C:\szl-bridge\watchdog.ps1' `
  -Mode Status `
  -Root 'C:\szl-bridge'

Get-Content 'C:\szl-bridge\logs\watchdog.log' -Tail 50
Get-Content 'C:\szl-bridge\logs\watchdog-receipts.ndjson' -Tail 3
```

Repository qualification includes the complete Python/Node bridge suite, static
watchdog contracts on Linux, and a Windows Server 2022 PowerShell parser and
authority-contract job. The installation workflow then produces a separate
machine-executed proof from the real owner node.

## Availability boundary

This package removes routine human intervention for modeled Task Scheduler,
process, local-file, and bridge-daemon failures. It cannot recover a powered-off
machine whose firmware will not restart, a failed GPU or storage device, a home
network without power, or simultaneous loss of every Internet path. A physical
24/7 target still requires BIOS restore-on-AC, a UPS for the node and network,
remote power or Wake-on-LAN from an independently powered relay, and a secondary
Internet path.
