#!/usr/bin/env python3
"""Reviewed replacement for fleet_doctor_v2 / hub_checkpoint_watch / bench_receipt_watch.

Read-only unless ``--apply`` is passed AND the owner lease pinned in the contract
verifies for the exact group. Every decision comes from ``controller_gates``; every
provider mutation is followed by an independent group + instance readback witness;
every published receipt is read back at its commit before it is called published.
One controller instance per (contract, command) holds an OS file lock.

    python -m tools.salad.fleet_controller --contract C doctor [--apply] [--cycles N]
    python -m tools.salad.fleet_controller --contract C checkpoints [--cycles N]
    python -m tools.salad.fleet_controller --contract C bench [--apply] [--cycles N]
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.error
import urllib.request

from tools.salad import controller_gates as gates
from tools.salad.artifact_qualification import observe_candidate, strict_object
from tools.salad.provider_transition_witness import witness
from tools.salad.training_observer import (
    ACCESS_STOP_CODES,
    NoRedirect,
    atomic_status,
    observe_training,
)

CONTRACT_SCHEMA = "szl.fleet-controller-contract/v1"
SALAD_BASE = "https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers"
MAX_BODY = 1_000_000
# Outcomes that can change without a new receipt (lease granted, provider settles).
RETRY_DECISIONS = frozenset(
    {"HOLD_PROVIDER_UNRECONCILED", "HALT_ACCESS_STOP", "ALERT_PARK_NEEDS_OWNER_LEASE"}
)


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def read_secret(path):
    value = Path(path).read_text(encoding="utf-8-sig").strip()
    if not value:
        raise ValueError("EMPTY_SECRET")
    return value


class Salad:
    """Typed provider transport. Never logs bodies, keys or exception text."""

    def __init__(self, key, opener=None):
        self._key = key
        self._open = opener or urllib.request.build_opener(NoRedirect).open

    def call(self, method, path, body=None, content_type="application/json"):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            SALAD_BASE + path,
            data=data,
            method=method,
            headers={
                "Salad-Api-Key": self._key,
                "Content-Type": content_type,
                "Accept": "application/json",
                "User-Agent": "szl-fleet-controller/1.0",
            },
        )
        try:
            with self._open(request, timeout=20) as response:
                return 0, response.status, response.read(MAX_BODY + 1)[:MAX_BODY]
        except urllib.error.HTTPError as exc:
            return 0, exc.code, exc.read(MAX_BODY)[:MAX_BODY] if exc.fp else b""
        except Exception:
            return 1, None, b""

    def get_json(self, path):
        code, status, body = self.call("GET", path)
        if code != 0 or status != 200:
            return None, status
        try:
            return strict_object(body), status
        except Exception:
            return None, status


def readback(
    salad, group, requested_replicas, *, transport_exit=0, http_status=200, body=b""
):
    """Fresh group GET + separate instance-list GET, classified by the reviewed witness."""
    group_doc, s1 = salad.get_json("/" + group)
    # A denial on the group read must stop this access lane immediately.
    # Do not issue an instance GET using the same denied credential.
    if s1 in ACCESS_STOP_CODES:
        return witness(
            transport_exit=0,
            http_status=s1,
            body=b"",
            group=None,
            instances=None,
            name=group,
            requested_replicas=requested_replicas,
        )
    inst_doc, s2 = salad.get_json("/" + group + "/instances")
    if s2 in ACCESS_STOP_CODES:
        return witness(
            transport_exit=0,
            http_status=s2,
            body=b"",
            group=None,
            instances=None,
            name=group,
            requested_replicas=requested_replicas,
        )
    return witness(
        transport_exit=transport_exit,
        http_status=http_status,
        body=body,
        group=group_doc,
        instances=inst_doc,
        name=group,
        requested_replicas=requested_replicas,
    )


class Hub:
    """Thin adapter so tests can substitute a fake Hub without network."""

    def __init__(self, token):
        from huggingface_hub import HfApi, hf_hub_download

        self.api = HfApi(token=token)
        self._dl = hf_hub_download
        self._token = token

    def fetch(
        self, *, repo_id, filename, revision, local_dir, etag_timeout, repo_type="model"
    ):
        return self._dl(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
            etag_timeout=etag_timeout,
            repo_type=repo_type,
            token=self._token,
        )

    def dataset_snapshot(self, repo_id):
        info = self.api.dataset_info(repo_id, files_metadata=True, timeout=12)
        return info.sha, {e.rfilename: e for e in (info.siblings or [])}

    def publish(self, repo_id, path_in_repo, payload):
        commit = self.api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="fleet-controller receipt",
        )
        oid = getattr(commit, "oid", None)
        listed, raw = False, None
        if isinstance(oid, str):
            listed = bool(
                self.api.get_paths_info(
                    repo_id, [path_in_repo], revision=oid, repo_type="dataset"
                )
            )
            with tempfile.TemporaryDirectory(prefix="szl-readback-") as temp:
                raw = Path(
                    self.fetch(
                        repo_id=repo_id,
                        filename=path_in_repo,
                        revision=oid,
                        local_dir=temp,
                        etag_timeout=12,
                        repo_type="dataset",
                    )
                ).read_bytes()
        return gates.publication_result(
            payload=payload, commit_oid=oid, readback=raw, path_listed=listed
        )


class Ledger:
    """Local controller state + hash-chained receipts under one state directory."""

    def __init__(self, state_dir, command):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / (command + "_state.json")
        self.receipts = self.dir / (command + "_receipts.jsonl")
        self._lock_handle = None
        self.command = command

    def lock(self):
        handle = open(self.dir / (self.command + ".lock"), "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise SystemExit("ANOTHER_CONTROLLER_HOLDS_LOCK " + self.command)
        self._lock_handle = handle

    def load(self):
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def save(self, state):
        atomic_status(self.state_path, state)

    def _last_sha(self):
        if not self.receipts.exists():
            return None
        last = None
        with self.receipts.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    last = line
        return json.loads(last)["sha256"] if last else None

    def append(self, record):
        line = gates.chain_record(record, self._last_sha())
        with self.receipts.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return line


def load_contract(path):
    contract = strict_object(Path(path).read_bytes())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SystemExit("CONTRACT_SCHEMA_INVALID")
    return contract


def lease_check(contract, group, now):
    try:
        raw = Path(contract["owner_lease_file"]).read_bytes()
    except (KeyError, OSError):
        return False, "OWNER_LEASE_UNREADABLE"
    return gates.verify_owner_lease(contract, raw, group, now)


def emit(ledger, record):
    line = ledger.append(record)
    print(
        json.dumps(
            {
                k: line.get(k)
                for k in (
                    "kind",
                    "group",
                    "lane",
                    "decision",
                    "label",
                    "verdict",
                    "reason",
                    "state",
                )
            }
        )
    )
    return line


def maybe_publish(hub, contract, kind, record):
    repo = contract.get("receipts_repo")
    if not repo or hub is None:
        return {"state": "NOT_PUBLISHED"}
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(record, indent=1, sort_keys=True).encode()
    try:
        return hub.publish(repo, "fleet-controller/%s-%s.json" % (kind, stamp), payload)
    except Exception as exc:
        return {"state": "PUBLICATION_UNVERIFIED", "error": type(exc).__name__}


# ------------------------------------------------------------------ commands


def doctor_cycle(contract, salad, hub, ledger, *, apply):
    state = ledger.load()
    lanes = state.setdefault("lanes", {})
    quals = state.setdefault("qualified_revisions", {})
    now = now_utc()
    for lane in contract.get("adopted", []):
        group = lane["group"]
        if not gates.valid_group(group) or not gates.valid_repo(lane.get("repo")):
            emit(
                ledger,
                {
                    "kind": "doctor",
                    "group": group,
                    "decision": "HOLD",
                    "reason": "CONTRACT_LANE_INVALID",
                },
            )
            continue
        replicas = lane.get("replicas", 1)
        provider = readback(salad, group, replicas)
        candidate = {
            "repo": lane["repo"],
            "target_steps": lane["target_steps"],
            "base": lane["base"],
        }
        training = observe_training(hub.api, hub.fetch, candidate)
        qualification = None
        revision = (training.get("snapshot") or {}).get("revision")
        if training.get("state") == "TERMINAL_METADATA_PRESENT" and revision:
            qualification = quals.get(revision) or observe_candidate(
                hub.api, hub.fetch, candidate
            )
            quals[revision] = qualification
        lease_ok, lease_reason = lease_check(contract, group, now)
        result = gates.decide_adopted_lane(
            contract=lane,
            lease_ok=lease_ok,
            provider=provider,
            training=training,
            qualification=qualification,
            ledger=lanes.get(group),
            now=now,
        )
        lanes[group] = result.pop("ledger")
        result.update(
            kind="doctor",
            lease=lease_reason,
            provider_state=provider.get("state"),
            training_state=training.get("state"),
            revision=revision,
        )
        if result["decision"] == "RESTART_PERMITTED":
            if apply:
                exit_code, status, body = salad.call("POST", "/" + group + "/start")
                result["action"] = {
                    "request": "POST start",
                    "post_readback": readback(
                        salad,
                        group,
                        replicas,
                        transport_exit=exit_code,
                        http_status=status,
                        body=body,
                    ),
                }
            else:
                result["action"] = {
                    "request": "NONE",
                    "reason": "DRY_RUN_WITHOUT_APPLY",
                }
                lanes[group]["attempts"] -= 1
                lanes[group]["last_restart_utc"] = None
        ledger.save(state)
        line = emit(ledger, result)
        seen = state.setdefault("last_decision", {})
        key = [result["decision"], result["reason"], revision]
        if result["decision"] != "OBSERVE" and (
            seen.get(group) != key or "action" in result
        ):
            emit(
                ledger,
                {
                    "kind": "doctor-publication",
                    "group": group,
                    **maybe_publish(hub, contract, "doctor", line),
                },
            )
        seen[group] = key
        ledger.save(state)


def checkpoint_cycle(contract, hub, ledger):
    state = ledger.load()
    last = state.setdefault("labels", {})
    for lane in contract.get("checkpoint_lanes", []):
        name, repo = lane.get("lane"), lane.get("repo")
        training = observe_training(
            hub.api, hub.fetch, {"repo": repo, "target_steps": lane.get("target_steps")}
        )
        record = gates.classify_checkpoint(training)
        key = (record["label"], record.get("revision"))
        record.update(kind="checkpoint", lane=name, repo=repo)
        streak = state.setdefault("unobservable_streak", {})
        streak[name] = (
            streak.get(name, 0) + 1 if record["label"] == "UNOBSERVABLE" else 0
        )
        if streak[name] == lane.get("alert_after_unobservable_cycles", 3):
            record["alert"] = "REPO_UNOBSERVABLE_REPEATEDLY_NOT_ABSENT"
        if list(key) != last.get(name) or "alert" in record:
            line = emit(ledger, record)
            if record["label"] == "CHECKPOINT_STATE_VERIFIED":
                line["publication"] = maybe_publish(hub, contract, "checkpoint", line)
                emit(
                    ledger,
                    {
                        "kind": "checkpoint-publication",
                        "lane": name,
                        **line["publication"],
                    },
                )
            last[name] = list(key)
        ledger.save(state)


def stop_settled_for_zero_patch(provider, expected_replicas):
    """Only an independently reconciled stopped group with no instances qualifies.

    This is a bounded provider observation, not evidence of exclusive ownership
    or training completion. The desired replica count is expected to remain
    positive until the subsequent PATCH to zero.
    """
    if type(provider) is not dict or provider.get("group_state_reconciled") is not True:
        return False
    counts = provider.get("actual_counts")
    return (
        provider.get("provider_status") == "stopped"
        and provider.get("desired_replicas") == expected_replicas
        and provider.get("instances_returned") == 0
        and type(counts) is dict
        and all(
            type(counts.get(key)) is int and counts[key] == 0 for key in gates.COUNTS
        )
    )


def bench_cycle(contract, salad, hub, ledger, *, apply):
    spec = contract["bench"]
    state = ledger.load()
    handled = state.setdefault("handled_receipts", [])
    group = spec["group"]
    cutoff = gates.parse_utc(spec["cutoff_utc"])
    if cutoff is None or not gates.valid_group(group):
        emit(
            ledger,
            {"kind": "bench", "verdict": "HOLD", "reason": "BENCH_CONTRACT_INVALID"},
        )
        return
    # An ambiguous first/second provider effect requires manual owner review.
    # Do not replay POST stop on the next cycle or for a different receipt.
    if group in state.get("stop_intents", {}):
        return
    revision, entries = hub.dataset_snapshot(spec["dataset"])
    name = gates.select_heldout_receipt(
        list(entries), prefix=spec["prefix"], cutoff_utc=cutoff
    )
    if name is None or any(h[0] == name for h in handled):
        return
    entry = entries[name]
    lfs = getattr(entry, "lfs", None)
    lfs_sha = (
        lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    )
    with tempfile.TemporaryDirectory(prefix="szl-bench-") as temp:
        raw = Path(
            hub.fetch(
                repo_id=spec["dataset"],
                filename=name,
                revision=revision,
                local_dir=temp,
                etag_timeout=12,
                repo_type="dataset",
            )
        ).read_bytes()
    verified = gates.entry_bytes_verified(
        raw,
        size=getattr(entry, "size", None),
        lfs_sha256=lfs_sha,
        blob_id=getattr(entry, "blob_id", None),
    )
    try:
        rec = strict_object(raw) if verified else None
    except Exception:
        rec = None
    verdict, gate = gates.assess_heldout_receipt(rec, spec)
    provider = readback(salad, group, spec.get("replicas", 1))
    lease_ok, lease_reason = lease_check(contract, group, now_utc())
    decision = gates.decide_bench_park(
        receipt_verified=verified and rec is not None,
        verdict=verdict,
        lease_ok=lease_ok,
        provider=provider,
    )
    record = {
        "kind": "bench",
        "group": group,
        "receipt_path": name,
        "receipt_revision": revision,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_bytes_verified": verified,
        "verdict": verdict,
        "gate": gate,
        "decision": decision,
        "lease": lease_reason,
        "provider_state": provider.get("state"),
        "note": "MEASURE, not a model verdict; owner review required; no promotion authority",
    }
    if decision == "PARK_PERMITTED" and apply:
        # Persist the group-scoped intent *before* issuing any non-idempotent
        # provider request. A crash, 429 or unknown response must never cause
        # the next cycle to replay POST stop automatically.
        intents = state.setdefault("stop_intents", {})
        intents[group] = {
            "receipt": name,
            "revision": revision,
            "phase": "POST_STOP_OUTCOME_UNRESOLVED",
        }
        ledger.save(state)
        exit_code, status, body = salad.call("POST", "/" + group + "/stop")
        action = {
            "request": "POST stop only; PATCH not authorized without settled readback",
            "stop_http_status": status if type(status) is int else None,
            "stop_transport_exit": exit_code if type(exit_code) is int else None,
        }
        if exit_code != 0 or status not in (200, 202, 204):
            decision = "HOLD_STOP_REQUEST_UNVERIFIED"
        else:
            settled = readback(
                salad,
                group,
                spec.get("replicas", 1),
                transport_exit=exit_code,
                http_status=status,
                body=body,
            )
            action["stop_readback"] = settled
            fresh_lease_ok, fresh_lease_reason = lease_check(contract, group, now_utc())
            action["second_lease"] = fresh_lease_reason
            if not stop_settled_for_zero_patch(settled, spec.get("replicas", 1)):
                decision = "HOLD_STOP_SETTLEMENT_UNVERIFIED"
            elif not fresh_lease_ok:
                decision = "HOLD_OWNER_LEASE_CHANGED"
            else:
                # A second effect is independently gated on stopped/zero
                # provider state and unchanged lease, not HTTP 202 alone.
                intents[group]["phase"] = "PATCH_ZERO_OUTCOME_UNRESOLVED"
                ledger.save(state)
                patch_exit, patch_status, patch_body = salad.call(
                    "PATCH",
                    "/" + group,
                    {"replicas": 0},
                    "application/merge-patch+json",
                )
                action["request"] = "POST stop followed by gated PATCH replicas=0"
                action["patch_http_status"] = (
                    patch_status if type(patch_status) is int else None
                )
                if patch_exit != 0 or patch_status not in (200, 202, 204):
                    decision = "HOLD_PARK_REQUEST_UNVERIFIED"
                else:
                    after_patch = readback(
                        salad,
                        group,
                        0,
                        transport_exit=patch_exit,
                        http_status=patch_status,
                        body=patch_body,
                    )
                    action["post_readback"] = after_patch
                    if (
                        after_patch.get("state")
                        == "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION"
                    ):
                        decision = "PARK_PROVIDER_ZERO_REPORTED_NOT_COMPLETION"
                        # Only independent provider zero readback clears the
                        # ambiguity barrier; this does not certify model DONE.
                        intents.pop(group, None)
                    else:
                        decision = "HOLD_PARK_READBACK_UNVERIFIED"
        record["decision"] = decision
        record["action"] = action
    elif decision == "PARK_PERMITTED":
        record["action"] = {"request": "NONE", "reason": "DRY_RUN_WITHOUT_APPLY"}
    pending = state.setdefault("pending_receipts", {})
    retry = (
        decision in RETRY_DECISIONS
        or (decision == "PARK_PERMITTED" and not apply)
        or group in state.get("stop_intents", {})
    )
    if pending.get(name) != decision or "post_readback" in record.get("action", {}):
        line = emit(ledger, record)
        emit(
            ledger,
            {
                "kind": "bench-publication",
                "group": group,
                **maybe_publish(hub, contract, "bench", line),
            },
        )
    if retry:
        pending[name] = decision
    else:
        pending.pop(name, None)
        handled.append([name, revision])
    ledger.save(state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contract", required=True)
    parser.add_argument("command", choices=("doctor", "checkpoints", "bench"))
    parser.add_argument(
        "--apply", action="store_true", help="allow lease-gated provider mutations"
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args(argv)
    if not 1 <= args.cycles <= 10_000 or not 60 <= args.interval <= 3600:
        parser.error("cycles 1..10000, interval 60..3600")
    contract = load_contract(args.contract)
    ledger = Ledger(contract["state_dir"], args.command)
    ledger.lock()
    hub = Hub(read_secret(contract["hf_token_file"]))
    salad = (
        Salad(read_secret(contract["salad_key_file"]))
        if args.command != "checkpoints"
        else None
    )
    for index in range(args.cycles):
        try:
            if args.command == "doctor":
                doctor_cycle(contract, salad, hub, ledger, apply=args.apply)
            elif args.command == "checkpoints":
                checkpoint_cycle(contract, hub, ledger)
            else:
                bench_cycle(contract, salad, hub, ledger, apply=args.apply)
        except Exception as exc:
            emit(
                ledger,
                {
                    "kind": args.command,
                    "decision": "CYCLE_ERROR",
                    "reason": type(exc).__name__,
                },
            )
        if index + 1 < args.cycles:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
