"""Replacement watcher for watch_arm6/watch_arm7/rlvr_watch4/f54v2_watch (2026-11-19).

Default is OBSERVE-ONLY: it never mutates SaladCloud. With --apply, the only
mutation is PATCH {"replicas": 0}, and only when decide_stop() returns
STOP_ALLOWED from completion-guard evidence. Machine churn, missing/partial final
adapters, lane.log without a completion marker, unreachable provider or Hub all
end in HOLD (+ alert receipt). DONE requires confirm_zero() readback.

Usage: python watch_safe.py --config watch_config.json [--once] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from szl_controller_gate import classify_write, decide_stop, confirm_zero, _json_obj  # noqa: E402
import hub_evidence  # noqa: E402
from salad_io import Salad, BASE_DEFAULT  # noqa: E402
from szl_controller_gate import provider_call  # noqa: E402


def fetch_lane_log(url, token_file, call=provider_call):
    """Return (text, code). 404 -> (None, 'LANE_LOG_ABSENT'). Text is small and
    is only searched for the owner's completion marker."""
    with open(token_file, encoding="utf-8") as f:
        hdr = {"Authorization": "Bearer " + f.read().strip()}
    r = call("GET", url, hdr)
    if r.get("status") == 404:
        return None, "LANE_LOG_ABSENT"
    ok, code = classify_write(dict(r, body=b""))
    if not ok:
        return None, "LANE_LOG_" + code
    return r["body"].decode("utf-8", "replace"), None


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Out:
    def __init__(self, log_path, receipt_path):
        self.log_path, self.receipt_path = log_path, receipt_path

    def log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(utc() + " " + msg + "\n")

    def receipt(self, rec):
        rec = dict(rec, ts=utc(), actor="watch_safe/v1", note="no secrets recorded")
        with open(self.receipt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def observe_group(salad, name):
    g, i = salad.get_group(name), salad.get_instances(name)
    gok, gcode = classify_write(g)
    iok, icode = classify_write(i)
    if not gok or not iok:
        return None, (gcode if not gok else icode)
    gv, _ = _json_obj(g["body"])
    iv, _ = _json_obj(i["body"])
    if not isinstance(gv, dict) or not isinstance(iv, dict):
        return None, "READBACK_INVALID"
    return (gv, iv.get("instances") or []), None


def churn_seen(machines, insts, threshold):
    run = 0
    for x in insts:
        st = x.get("state") or x.get("status")
        if st == "running":
            run += 1
        mid = x.get("machine_id")
        m = x.get("machine")
        if mid is None and isinstance(m, dict):
            mid = m.get("id")
        if mid is None and isinstance(m, str):
            mid = m
        if mid:
            machines.add(str(mid))
    return len(machines) >= threshold and run == 0


def cycle(cfg, salad, out, state, *, apply=False, collect=hub_evidence.collect,
          lane_log=fetch_lane_log, sleep=time.sleep):
    results = {}
    for g in cfg["groups"]:
        name = g["name"]
        if name in state["done"]:
            results[name] = "DONE"
            continue
        obs, err = observe_group(salad, name)
        if err:
            out.log("%s READBACK_FAIL %s (HOLD, no mutation)" % (name, err))
            results[name] = "HOLD_" + err
            if err.startswith(("ACCESS_DENIED", "RATE_LIMITED")):
                state["stop_polling"] = True
            continue
        group, insts = obs
        st = (group.get("current_state") or {}).get("status")
        machines = state["machines"].setdefault(name, set())
        if churn_seen(machines, insts, cfg.get("churn_threshold", 3)):
            ev = {"reason": "churn"}
            d = decide_stop(ev)
            out.log("%s CHURN uniq=%d running=0 -> %s (no stop)" % (name, len(machines), d["code"]))
            out.receipt({"schema": "szl.controller-alert/v1", "group": name, "decision": d})
            results[name] = d["code"]
            continue
        if not g.get("repo") or st == "stopped":
            results[name] = "OBSERVED_%s" % st
            continue
        lane_text = None
        if g.get("reason") == "lane_log":
            lane_text, lerr = lane_log(g["lane_log_url"], cfg["hf_token_file"])
            if lerr:
                results[name] = "HOLD_" + lerr
                continue
        # Completion candidate signals are only triggers for evidence collection.
        ev, err = collect(dict(g, reason=g.get("reason", "completion")), cfg["hf_token_file"])
        if err:
            out.log("%s EVIDENCE_FAIL %s (HOLD)" % (name, err))
            results[name] = "HOLD_" + err
            continue
        if g.get("reason") == "lane_log":
            ev["lane_log_text"] = lane_text
        d = decide_stop(ev)
        if not d["stop_allowed"]:
            out.log("%s HOLD %s" % (name, d["code"]))
            results[name] = d["code"]
            continue
        out.receipt({"schema": "szl.scale-down-intent/v1", "group": name, "decision": d,
                     "revision": ev.get("revision"), "apply": apply})
        if not apply:
            out.log("%s STOP_ALLOWED (observe-only; no PATCH)" % name)
            results[name] = "STOP_ALLOWED_OBSERVE_ONLY"
            continue
        ok, code = classify_write(salad.patch(name, {"replicas": 0}))
        out.log("%s PATCH replicas=0 -> %s" % (name, code))
        if not ok:
            results[name] = "HOLD_PATCH_" + code
            continue
        z = {"done": False, "state": "NOT_POLLED"}
        for _ in range(cfg.get("zero_polls", 10)):
            z = confirm_zero(name, lambda: salad.get_group(name), lambda: salad.get_instances(name))
            if z["done"]:
                break
            sleep(cfg.get("zero_interval", 30))
        out.receipt({"schema": "szl.spend-stop/v2", "group": name, "patch": code,
                     "zero_readback": z["state"], "done": z["done"]})
        results[name] = "DONE" if z["done"] else "STOP_REQUESTED_NOT_CONFIRMED_" + z["state"]
        if z["done"]:
            state["done"].add(name)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--apply", action="store_true", help="allow PATCH replicas=0 on STOP_ALLOWED")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    salad = Salad(cfg["salad_key_file"], cfg.get("salad_base") or BASE_DEFAULT)
    out = Out(cfg["log"], cfg["receipts"])
    state = {"machines": {}, "done": set(), "stop_polling": False}
    out.log("watch_safe start pid=%d apply=%s groups=%s" % (os.getpid(), a.apply, [g["name"] for g in cfg["groups"]]))
    for _ in range(1 if a.once else cfg.get("cycles", 96)):
        res = cycle(cfg, salad, out, state, apply=a.apply)
        out.log("cycle " + json.dumps(res, sort_keys=True))
        if state["stop_polling"]:
            out.log("auth/rate failure: stopping polls (HOLD)")
            return 2
        if a.once:
            break
        time.sleep(cfg.get("interval", 300))
    return 0


if __name__ == "__main__":
    sys.exit(main())
