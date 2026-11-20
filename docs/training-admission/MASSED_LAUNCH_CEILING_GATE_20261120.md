# Massed Launch-Ceiling Admission Gate (2026-11-20)

Doctrine owner: CTO lane (Vibe/GLM), squad round 7 concord (Kimi
kimi-k3.1-pro:cloud, Mistral mistral-large-4, Astra gpt-6.2 watcher;
receipts kimi_w7_clean2.txt / mistral_w7_clean2.txt /
astra_w7_reply.txt in the estate wave records).

Governing rule: Massed Compute launches are admitted BEFORE the launch,
not receipted after. The provider exposes no billing/receipt API, so
admission is the only spend-stop layer. The gate is pure, owns no
credential, performs no I/O and never touches the provider.

## Laws encoded (tools/massed/admission.py)

* OWNER LEASE: lease_ok must be literally True. Without it every op
  denies (OWNER_LEASE_UNVERIFIED). This is the no-credential law: the
  gate takes no key at all.
* STARTED-HOUR BILLING: worst_case = started_hours x price, one second
  costs one full hour. The 3h30m canary deadline from the draft campaign
  plan is UNLAWFUL under this law (4 started hours = 2,280,000 uUSD);
  the lawful A6000 canary is 3h = 1,710,000 uUSD against the 2,000,000
  cap.
* KIMI BLACKWELL BLOCKER: a product whose single started hour exceeds
  the canary cap can never canary (measured PRO 6000 Blackwell =
  2,190,000 uUSD/hr).
* PRODUCT WHITELIST: gpu_1x_a6000_high_ram and gpu_1x_a30 only; any
  other product denies (PRODUCT_NOT_WHITELISTED, HIGH).
* PRICE GATE: products above 970,000 uUSD/hr (97c) deny outright
  (PRODUCT_PRICE_OVER_OWNER_GATE) - the owner gated any pricier SKU.
* LANE PIN MISMATCH: intent product/region/image must equal the
  owner-pinned lane exactly (no drifting launch parameters).
* PRICE RE-READ: the caller passes price_snapshot_uusd_hr measured from
  the verified inventory immediately before judgement; absent or
  drifted prices deny (PRICE_DRIFT_OR_UNMEASURED, HIGH).
* BUDGET FLOOR (RATIFY-B mirror): all-zero caps and accounts are an
  unpinned MODELED placeholder and deny (BUDGET_FLOOR_UNPINNED).
  String placeholders ("MODELED") fail registry validation.
* STORAGE PIN: storage/egress pricing is UNKNOWN on this provider; a
  lane without an owner-pinned storage ceiling denies
  (STORAGE_UNPINNED). Never assume zero.
* LAUNCH_CEILING set (all must hold at once): canary_cap, per_job_cap,
  lane_allocated, wave_cap (per-wave planned), cumulative
  projected-at-completion (measured + wave planned + worst)
  <= 900,000,000, campaign new-commitment cap 423,620,000.
* ACCOUNT SPLIT: wave_planned_uusd is per-wave; campaign_committed_uusd
  is the campaign-level commitment; the wave cap never silently
  doubles as a campaign cap.
* STATE MACHINE: PENDING_LAUNCH -> RUNNING -> TERMINATED. TERMINATED is
  terminal (LANE_TERMINAL). terminate is admitted only with the lease.

## Receipt chain

Every ADMIT and DENY is hash-chained (controller_gates.chain_record):
tamper-evidence, NOT authentication, mirroring the Salad gate.
receipt["massed_mutation_authorized"] is True only on ADMIT.

## Test suite (tests/test_massed_admission.py)

24 tests, 24 passed at land time. The suite is a mocked dry-run set:
no network, no keys, all unsafe paths deny, single lawful ADMIT = the
3h A6000 canary with every owner pin in place.

## Standing owner gates (unchanged)

No Massed launch is authorized by this module. The registry arrives
with MODELED placeholder budgets until the owner pins: the lane lease +
integer budgets, product whitelist sign-off, storage ceiling, canary
family pick, HF token rotation, and key rotation for the pasted
provider key (compromised on exposure).