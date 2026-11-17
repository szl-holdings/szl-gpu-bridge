# Workstation disk-capacity preflight — source-only, fail-closed

**Scope:** read-only owner-machine storage admission check for a separately
authorized *future* local training or source-preparation job. This is not the
existing signed bridge dispatcher, an automatically installed watchdog, or
authorization to modify Flagship54, SaladCloud, a GPU process, any checkpoint
or the preserved training completion guard.

## Observed reason this gate exists (2026-11-17)

The authorized BETTERWITHAGE Windows node showed widely varying C: free space
during one bounded operational investigation:

- A prior observation showed approximately **0.25 GiB** free.
- A later reading showed approximately **10.59 GiB** free.
- A subsequent reading returned to **0.25 GiB**, later around 2.27 GiB.
- Windows Temp was about **1.95 GB / 69,700 files**; only about 0.93 GB was
  older than one day. These figures do *not* explain the much larger transient
  free-space swings, and no causal attribution is made.
- The exact bridge tasks `SZL-GPU-Bridge`, `SZL-GPU-Bridge-Watchdog`,
  and `SZL-GPU-Bridge-Guardian` and expected
  `C:\szl-bridge\*.ps1` owner installation were **absent on
  BETTERWITHAGE** at observation time. That does not establish whether another
  owner node runs the bridge; no second dispatcher was installed.

**Therefore no local GPU/model training launch or new large download is
admitted by this note.** Green GitHub CI and a healthy Hugging Face CPU Space
do not prove local disk capacity, live trainer health or reliable persistence.

## Actual read-only command

Run this in the exact reviewed `szl-gpu-bridge` source checkout:

```powershell
py -3 -B laptop\storage_preflight.py `
  --root C:\ `
  --samples 5 `
  --interval-sec 2 `
  --min-free-gib 50 `
  --max-swing-gib 2 `
  --receipt "C:\Users\<owner>\SZL-private-evidence\storage-preflight-<UTC>.json"
```

`50 GiB` is a conservative model-training *operator preflight threshold*,
not an immutable hardware target or an automatically derived capability
claim. A future workload with a justified lower threshold needs a separately
reviewed storage budget (weights, optimizer, datasets, temporary and
checkpoint headroom, rollback and reserved recovery space) before admission.

The standalone command takes bounded, time-separated independent
`shutil.disk_usage` observations of one volume. It fails closed when:

- Minimum sampled free space is below 50 GiB.
- Any free-space swing across the sample window exceeds 2 GiB.
- The volume is missing, inconsistent, resized during the sample, or its
  usage cannot be read.
- Arguments, sample count or observed capacity data are invalid.
- An exclusive evidence receipt already exists or its readback differs.

Exit code `0` means **capacity gate passed only**.
Exit code `78` means **HOLD / UNAVAILABLE**. All source, data rights,
signer, provider image, signed-job, watchdog ownership, compute budget,
training/resume and model-quality gates remain separately required.
A passing sample cannot prove a system will never fill up later. It is
an immediately-before-launch *necessary but not sufficient* gate.

The optional receipt uses exclusive `xb` creation, filesystem fsync and
exact-byte readback. Its SHA-256 is an **integrity checksum, not a
cryptographic signature**. No credentials, process command lines,
private addresses, file names or personal details are required.
The command never deletes files, installs tasks, downloads packages,
stops jobs or attempts administrator access.

## Source and system protection

The script is deliberately **not installed into any currently running
dispatcher**, existing Task Scheduler entry, active HF Space, or GPU
runtime by this source-only change. That integration is a separate
owner-approved change that requires an exact signed-job boundary and
tests for start/restart/resume ordering. Do not introduce a second task
on BETTERWITHAGE just because a task was not present there; establish
which machine currently owns the bridge first.

**Do not** indiscriminately delete `Temp`, `.etl` Windows traces,
`.vhdx` swap disks, audit repositories, provenance ledgers,
model checkpoints, CUDA caches or preserved private evidence to make
a single sample green. Use a separate storage-ownership inventory
and verify that any proposed deletion is not the only retained copy,
an active file or a forensic artifact. Prefer provisioning sufficient
headroom, model-artifact lifecycle controls and explicit retention.

## Reproducibility and tests

```powershell
py -3 -B -m unittest tests.test_storage_preflight -v
```

The offline tests simulate stable >50 GiB, low capacity, fluctuations,
exact threshold boundaries, bad types, filesystem identity changes,
unavailable roots, source-only receipt immutability, and no unintended
training authorization. They never allocate 50 GiB, alter a real
training controller or require a provider token.

The current bridge source of truth is protected GitHub
`szl-holdings/szl-gpu-bridge`. The implementation must pass its
normal full CI, including Windows parser/contract checks if configured,
before merge. This storage *preflight source* is not a production
claim about the bridge runtime, and intentionally does not weaken any
existing watchdog, installer, signed-job or source-protection rule.
