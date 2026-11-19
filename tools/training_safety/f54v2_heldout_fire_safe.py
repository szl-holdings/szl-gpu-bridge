"""Fail-closed replacement for f54v2_heldout_fire3_20261119.py (admission only).

Differences from the original:
  * every PATCH is judged by HTTP status + body (classify_write); a curl/shell
    rc of 0 is never success. HTTP 400 pending_update_in_progress stops the run;
  * no automatic retry/variant fallback, and no variant that writes the HF token
    into the container command line;
  * the Salad key is read from its file per request and never placed in argv;
  * admission requires admission_readback(): reconciled group, requested
    replicas, no pending change and >=1 instance actually running;
  * exits non-zero on every failure and writes a receipt with the reason;
  * default is --plan (no provider writes). --apply is required to PATCH.

The container spec is read from a JSON file the owner prepares (no secrets in
it); how the evaluation container receives its HF credential is an owner
decision outside this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from szl_controller_gate import classify_write, admission_readback  # noqa: E402
from salad_io import Salad, BASE_DEFAULT  # noqa: E402

EXIT_OK, EXIT_SPEC, EXIT_CONTAINER_PATCH, EXIT_REPLICA_PATCH, EXIT_NOT_ADMITTED, EXIT_PRECHECK = 0, 2, 3, 4, 5, 6


def run(cfg, salad, *, apply=False, sleep=time.sleep, receipt=None):
    rec = {"schema": "szl.launch-admission/v1", "group": cfg["group"], "apply": apply,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def done(code, state):
        rec.update(exit=code, state=state)
        if receipt:
            receipt(rec)
        return code, rec

    spec = cfg.get("container")
    if not isinstance(spec, dict):
        return done(EXIT_SPEC, "SPEC_INVALID")
    env = spec.get("environment_variables") or {}
    if any(any(t in str(k).upper() for t in ("TOKEN", "KEY", "SECRET", "PASSWORD")) for k in env) \
            or "Bearer " in json.dumps(spec.get("command", [])):
        return done(EXIT_SPEC, "SPEC_CONTAINS_CREDENTIAL_FIELD")
    # Precheck: never write into a group that has an update in flight.
    pre = salad.get_group(cfg["group"])
    ok, code = classify_write(pre)
    rec["precheck"] = code
    if not ok:
        return done(EXIT_PRECHECK, "PRECHECK_" + code)
    try:
        g = json.loads(pre["body"].decode("utf-8"))
    except Exception:
        return done(EXIT_PRECHECK, "PRECHECK_BODY_INVALID")
    if g.get("pending_change") is True:
        return done(EXIT_PRECHECK, "PRECHECK_UPDATE_PENDING")
    if not apply:
        return done(EXIT_OK, "PLAN_ONLY_NO_WRITES")
    ok, code = classify_write(salad.patch(cfg["group"], {"container": spec}))
    rec["container_patch"] = code
    if not ok:
        return done(EXIT_CONTAINER_PATCH, "CONTAINER_PATCH_" + code)
    ok, code = classify_write(salad.patch(cfg["group"], {"replicas": int(cfg.get("replicas", 1))}))
    rec["replicas_patch"] = code
    if not ok:
        return done(EXIT_REPLICA_PATCH, "REPLICAS_PATCH_" + code)
    adm = admission_readback(cfg["group"], int(cfg.get("replicas", 1)),
                             lambda: salad.get_group(cfg["group"]),
                             lambda: salad.get_instances(cfg["group"]),
                             polls=cfg.get("admission_polls", 45),
                             interval=cfg.get("admission_interval", 60), sleep=sleep)
    rec["admission"] = {k: adm[k] for k in ("admitted", "state", "polls")}
    if not adm["admitted"]:
        # Not admitted: do not claim a running evaluation; leave parking to the
        # owner/watcher (a zero-replica decision has its own gate).
        return done(EXIT_NOT_ADMITTED, adm["state"])
    return done(EXIT_OK, "ADMITTED_RUNNING_READBACK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    salad = Salad(cfg["salad_key_file"], cfg.get("salad_base") or BASE_DEFAULT)

    def receipt(rec):
        with open(cfg["receipt"], "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    code, rec = run(cfg, salad, apply=a.apply, receipt=receipt)
    print(json.dumps({"exit": code, "state": rec["state"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
