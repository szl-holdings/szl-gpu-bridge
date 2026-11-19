"""Replacement watcher for watch_arm6/watch_arm7/rlvr_watch4/f54v2_watch (2026-11-19).

Default is OBSERVE-ONLY: it never mutates SaladCloud. With --apply, the only
mutation is PATCH {"replicas": 0}, and only when decide_stop() returns
STOP_ALLOWED from completion-guard evidence, and only when the config holds no
'<OWNER: ...>' placeholder. Machine churn, missing/partial final adapters,
lane.log without a completion marker, unreachable provider or Hub all end in
HOLD (no action). A HOLD is logged every cycle and a receipt is written when a
group's HOLD code changes. DONE requires confirm_zero() readback. An error in
one group or cycle is logged and receipted as a HOLD and the watcher keeps going.

Usage: python watch_safe.py --config watch_config.json [--once] [--apply]
Exit codes: 0 finished, 2 auth/rate failure or bad config, 3 too many failed
cycles in a row, 4 --apply refused (placeholders remain).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from szl_controller_gate import (  # noqa: E402
    HF_API_HOSTS,
    UnsafeUrl,
    check_url,
    classify_write,
    confirm_zero,
    decide_stop,
    fetch_tail,
    find_placeholders,
    _json_obj,
)
import hub_evidence  # noqa: E402
from salad_io import Salad, BASE_DEFAULT  # noqa: E402

EXIT_OK, EXIT_AUTH_OR_CONFIG, EXIT_FAILING, EXIT_APPLY_REFUSED = 0, 2, 3, 4
CONNECTION_FIELDS = ("salad_key_file", "hf_token_file", "log", "receipts")
# Results that already are an action/observation, not a HOLD needing a receipt.
NO_HOLD_RECEIPT = ("DONE", "OBSERVED_", "STOP_ALLOWED", "STOP_REQUESTED", "CHURN_ALERT")


def fetch_lane_log(url, token_file, call=None):
    """Return (text, code). 404 -> (None, 'LANE_LOG_ABSENT'). The credential goes to
    huggingface.co only (never to a redirect target). Only the TAIL (last 1 MiB)
    is searched for the owner's completion marker; a log whose tail cannot be
    read is a HOLD, never 'no marker'."""
    try:
        check_url(url, HF_API_HOSTS)
    except UnsafeUrl as e:
        return None, "LANE_LOG_" + str(e)
    with open(token_file, encoding="utf-8") as f:
        hdr = {"Authorization": "Bearer " + f.read().strip()}
    data, code, _cut = (call or fetch_tail)(url, hdr)
    if code == 404:
        return None, "LANE_LOG_ABSENT"
    if code is not None:
        return None, "LANE_LOG_%s" % (
            code if isinstance(code, str) else "HTTP_%s" % code
        )
    try:
        return data.decode("utf-8"), None
    except UnicodeError:
        return data.decode("utf-8", "replace"), None


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

    def safe_log(self, msg):
        """Never raises: a failing log must not stop the watcher."""
        try:
            self.log(msg)
        except Exception as e:
            print("log write failed: %s" % type(e).__name__, file=sys.stderr)

    def safe_receipt(self, rec):
        try:
            self.receipt(rec)
        except Exception as e:
            print("receipt write failed: %s" % type(e).__name__, file=sys.stderr)


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


def _group_step(g, cfg, salad, out, state, apply, collect, lane_log, sleep):
    """One group, one cycle. Returns the result string; may raise (the caller
    turns any exception into a HOLD)."""
    name = g["name"]
    if name in state["done"]:
        return "DONE"
    obs, err = observe_group(salad, name)
    if err:
        out.safe_log("%s READBACK_FAIL %s (HOLD, no mutation)" % (name, err))
        if err.startswith(("ACCESS_DENIED", "RATE_LIMITED")):
            state["stop_polling"] = True
        return "HOLD_" + err
    group, insts = obs
    st = (group.get("current_state") or {}).get("status")
    machines = state["machines"].setdefault(name, set())
    if churn_seen(machines, insts, cfg.get("churn_threshold", 3)):
        d = decide_stop({"reason": "churn"})
        out.log(
            "%s CHURN uniq=%d running=0 -> %s (no stop)"
            % (name, len(machines), d["code"])
        )
        out.receipt({"schema": "szl.controller-alert/v1", "group": name, "decision": d})
        return d["code"]
    if not g.get("repo") or st == "stopped":
        return "OBSERVED_%s" % st
    lane_text = None
    if g.get("reason") == "lane_log":
        lane_text, lerr = lane_log(g["lane_log_url"], cfg["hf_token_file"])
        if lerr:
            return "HOLD_" + lerr
    # Completion candidate signals are only triggers for evidence collection.
    ev, err = collect(
        dict(g, reason=g.get("reason", "completion")), cfg["hf_token_file"]
    )
    if err:
        out.safe_log("%s EVIDENCE_FAIL %s (HOLD)" % (name, err))
        return "HOLD_" + err
    if g.get("reason") == "lane_log":
        ev["lane_log_text"] = lane_text
    d = decide_stop(ev)
    if not d["stop_allowed"]:
        out.safe_log("%s HOLD %s" % (name, d["code"]))
        return d["code"]
    out.receipt(
        {
            "schema": "szl.scale-down-intent/v1",
            "group": name,
            "decision": d,
            "revision": ev.get("revision"),
            "apply": apply,
        }
    )
    if not apply:
        out.log("%s STOP_ALLOWED (observe-only; no PATCH)" % name)
        return "STOP_ALLOWED_OBSERVE_ONLY"
    ok, code = classify_write(salad.patch(name, {"replicas": 0}))
    out.log("%s PATCH replicas=0 -> %s" % (name, code))
    if not ok:
        return "HOLD_PATCH_" + code
    z = {"done": False, "state": "NOT_POLLED"}
    for _ in range(cfg.get("zero_polls", 10)):
        z = confirm_zero(
            name, lambda: salad.get_group(name), lambda: salad.get_instances(name)
        )
        if z["done"]:
            break
        sleep(cfg.get("zero_interval", 30))
    out.receipt(
        {
            "schema": "szl.spend-stop/v2",
            "group": name,
            "patch": code,
            "zero_readback": z["state"],
            "done": z["done"],
        }
    )
    if z["done"]:
        state["done"].add(name)
        return "DONE"
    return "STOP_REQUESTED_NOT_CONFIRMED_" + z["state"]


def cycle(
    cfg,
    salad,
    out,
    state,
    *,
    apply=False,
    collect=hub_evidence.collect,
    lane_log=fetch_lane_log,
    sleep=time.sleep,
):
    results = {}
    if apply and find_placeholders(cfg):
        # Never act on a half-filled config, even if some group would qualify.
        out.safe_log("APPLY_REFUSED_OWNER_PLACEHOLDERS (no provider call, no action)")
        return {g["name"]: "APPLY_REFUSED_OWNER_PLACEHOLDERS" for g in cfg["groups"]}
    last = state.setdefault("last", {})
    for g in cfg["groups"]:
        name = g.get("name", "?")
        try:
            res = _group_step(
                g, cfg, salad, out, state, apply, collect, lane_log, sleep
            )
        except Exception as e:  # fail safe: no action, keep watching the others
            res = "HOLD_CONTROLLER_EXCEPTION_" + type(e).__name__
            out.safe_log("%s %s (HOLD, no action; message withheld)" % (name, res))
        results[name] = res
        if not res.startswith(NO_HOLD_RECEIPT) and last.get(name) != res:
            out.safe_receipt(
                {"schema": "szl.controller-hold/v1", "group": name, "code": res}
            )
        last[name] = res
    return results


def preflight(cfg, apply):
    """(exit_code, message) for a config that must not run, else (None, None)."""
    if apply and find_placeholders(cfg):
        return (
            EXIT_APPLY_REFUSED,
            "--apply refused: config still has <OWNER: ...> placeholders",
        )
    for k in ("salad_key_file", "log", "receipts", "groups"):
        if k not in cfg:
            return EXIT_AUTH_OR_CONFIG, "config missing %s" % k
    bad = [k for k in CONNECTION_FIELDS if find_placeholders(cfg.get(k))]
    if bad:
        return (
            EXIT_AUTH_OR_CONFIG,
            "config incomplete: %s still hold <OWNER> placeholders" % bad,
        )
    for g in cfg["groups"]:
        if g.get("reason") == "lane_log" and g.get("repo"):
            u = g.get("lane_log_url")
            if not isinstance(u, str) or find_placeholders(u):
                continue  # unset owner field: the group HOLDs
            try:
                check_url(u, HF_API_HOSTS)
            except UnsafeUrl as e:
                return EXIT_AUTH_OR_CONFIG, "lane_log_url rejected: %s" % e
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--apply", action="store_true", help="allow PATCH replicas=0 on STOP_ALLOWED"
    )
    a = ap.parse_args(argv)
    try:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
        code, msg = preflight(cfg, a.apply)
        if code is not None:
            print(msg, file=sys.stderr)
            return code
        salad = Salad(cfg["salad_key_file"], cfg.get("salad_base") or BASE_DEFAULT)
    except (OSError, ValueError, KeyError, TypeError, UnsafeUrl) as e:
        print("config error: %s" % type(e).__name__, file=sys.stderr)
        return EXIT_AUTH_OR_CONFIG
    out = Out(cfg["log"], cfg["receipts"])
    state = {"machines": {}, "done": set(), "stop_polling": False}
    out.safe_log(
        "watch_safe start pid=%d apply=%s groups=%s"
        % (os.getpid(), a.apply, [g.get("name") for g in cfg["groups"]])
    )
    max_fail = int(cfg.get("max_consecutive_failed_cycles", 12))
    failing = 0
    for _ in range(1 if a.once else cfg.get("cycles", 96)):
        try:
            res = cycle(cfg, salad, out, state, apply=a.apply)
        except Exception as e:  # defence in depth; cycle already isolates groups
            res = {"*": "HOLD_CONTROLLER_EXCEPTION_" + type(e).__name__}
        out.safe_log("cycle " + json.dumps(res, sort_keys=True))
        if state["stop_polling"]:
            out.safe_log("auth/rate failure: stopping polls (HOLD)")
            return EXIT_AUTH_OR_CONFIG
        failing = (
            failing + 1
            if any("CONTROLLER_EXCEPTION" in str(v) for v in res.values())
            else 0
        )
        if max_fail and failing >= max_fail:
            out.safe_log(
                "%d consecutive failing cycles: exiting non-zero (no action taken)"
                % failing
            )
            out.safe_receipt(
                {
                    "schema": "szl.controller-alert/v1",
                    "code": "WATCHER_EXITING_REPEATED_FAILURE",
                }
            )
            return EXIT_FAILING
        if a.once:
            break
        time.sleep(cfg.get("interval", 300))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
