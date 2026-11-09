# Non-Stop Training Runbook — the 54-hour guarantee

**Policy (owner directive):** the training legs — the 54-hr fine-tune, the brains, the
Collective — must never silently stop. This runbook makes "never stop" a checkable
procedure, not a hope.

**The rule (Law 7 + Law 9):** a leg with no receipt inside its sweep window and no
live process is **STOPPED**. Stopped legs are **restarted on A6000 nodes, not
SaladCloud**, because Salad is burst capacity with a proven silent-failure mode
(six containers died pre-Python in the 2026-11-09 incident; heredoc-in-JSON
packaging; CF 1010 API blocking).

---

## 1. Fleet truth (run from any networked machine — NOT the audit sandbox)

```bash
export SALADCLOUD_API_KEY=<rotated-key>   # never paste keys in chat
curl -s -H "Authorization: Bearer $SALADCLOUD_API_KEY" \
  "https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers" \
  | python3 -c "import json,sys; [print(c['name'], c['state']) for c in json.load(sys.stdin)]"
```

If the API returns Cloudflare 1010: retry with delays (it is intermittent from
datacenter IPs) or read the Salad dashboard directly.

## 2. Node truth (SSH — .145 first; NEVER touch .31's GPU, it is production inference)

```bash
ssh Ubuntu@216.81.248.145    # hatun-mcp-prod: GPU idle-safe for training
  screen -ls                  # the "nemo" session must be alive
  nvidia-smi                  # GPU utilization > 0 while a leg runs
  tail -n 50 ~/szl/*.log      # fresh loss lines = alive
```

Free nodes for (re)placement: `.145` (training-safe), `.118`, `.191`.
`.31` is production inference — GPU has 40 GB in use. **DO NOT TOUCH.**

## 3. Decision rule

| Observation | Verdict | Action |
|---|---|---|
| Receipt in `frontier-54-runs` within sweep window | MOVING | none |
| No receipt, live process, fresh logs | MOVING (receipt lag) | none; note lag |
| No receipt, no process | **STOPPED** | Restart per §4 |
| Container in `error`/`exited` state on Salad | **STOPPED** | Restart **on A6000** |

## 4. Restart procedure (A6000, sustained work)

```bash
ssh Ubuntu@216.81.248.145
cd ~/szl
screen -S <leg-name>              # one screen session per leg
# start the leg with a BOOT receipt as the FIRST line of Python:
python3 -c 'import json,datetime; print(json.dumps({"kind":"szl.boot-receipt/v1","leg":"<leg-name>","ts":datetime.datetime.now(datetime.timezone.utc).isoformat()}))' > boot-receipt.jsonl
python3 <leg-entrypoint> >> boot-receipt.jsonl 2>&1
# detach: Ctrl-A d
```

**Boot-receipt-first** is mandatory: the 2026-11-09 postmortem showed containers
died before Python started, so the first thing every leg does is prove it booted.
No boot receipt = the leg never started; don't wait for a training receipt.

## 5. The legs of record (as of 2026-11-09)

| Leg | Hardware (intended) | State |
|---|---|---|
| Frontier-54 v2 (54-hr fine-tune) | 5090 / Salad — MOVE TO A6000 on any restart | UNKNOWN from audit seat |
| Brain v3.5 (Prime Principle) | 5090 / Salad — same rule | UNKNOWN |
| Brain-2c (mathlib) | 4090 / Salad — same rule | UNKNOWN |
| Nano-2 | 4090 / Salad — same rule | UNKNOWN |
| The Collective (teach/dream/create) | Salad — crashed 2026-11-09, needs A6000 restart | STOPPED |
| Nemotron v3 chain | A6000 .145 via gpu-bridge | **SETTLED — attempt-17 terminal receipt** ✅ |
| A6000 bench leg | .145 | completed; next run needs the energy-meter patch (PR #134) |

## 6. What makes this "non-stop" honest

- Every leg boot emits a boot receipt; every sweep (Infra Warden, 15 min) reads it.
- A missing boot receipt within one sweep window triggers §4 automatically.
- Restarts land on A6000s (proven reliable) — Salad remains burst/eval only.
- Energy receipts are honest after PR #134 merges: UNAVAILABLE, never zero.

*Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.*
