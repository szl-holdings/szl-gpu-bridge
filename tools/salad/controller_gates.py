"""Pure fail-closed decisions for the laptop fleet controllers. No I/O, no credentials.

Replaces three ad-hoc watcher rules that turned weak evidence into actions:

* fleet doctor: a ``final/adapter_model.safetensors`` filename counted as completion,
  an unknown provider status counted as ``stopped``, one quarantined group froze every
  adopted group, and the restart budget reset on any non-stopped poll instead of on
  verified trainer progress.
* checkpoint watch: any non-empty listing (even ``.gitattributes`` alone) was labelled
  ``FIRST_CHECKPOINT_LANDED``, and HTTP 404 was skipped silently forever.
* bench watch: a lexical filename cutoff admitted ``diag-*`` names, the broken-control
  arm was read from keys the receipt never writes, and the bench was parked without a
  pinned receipt, owner lease or provider readback.

Every positive result here is *eligibility*, never authority. The caller still owns
the credential, the owner lease file, the HTTP call and the post-mutation readback.
"""

import datetime as dt
import hashlib
import json
import re

SCHEMA = "szl.fleet-controller-gates/v1"
COUNTS = ("allocating_count", "creating_count", "running_count", "stopping_count")
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_GROUP = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_REPO = re.compile(r"SZLHOLDINGS/[A-Za-z0-9_.-]{1,96}\Z")
_PROGRESS_STATES = frozenset(
    {
        "CHECKPOINT_PROGRESS_OBSERVED",
        "TERMINAL_METADATA_PRESENT",
        "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE",
    }
)
_TERMINAL_OR_BROKEN = frozenset(
    {
        "TERMINAL_METADATA_PRESENT",
        "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE",
        "TERMINAL_STEP_OVERSHOOT",
        "TARGET_CONTRACT_MISMATCH",
    }
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def utc_text(moment):
    return moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value):
    if type(value) is not str or not re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z", value
    ):
        return None
    try:
        return dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


# ---------------------------------------------------------------- owner lease


def verify_owner_lease(contract, lease_bytes, group, now):
    """The controller may mutate ``group`` only under the exact lease the owner pinned.

    The contract stores sha256(lease file) and a generation. Editing the lease, bumping
    the generation, or omitting the group revokes authority without touching the code.
    """
    if type(lease_bytes) is not bytes or not 0 < len(lease_bytes) <= 65536:
        return False, "OWNER_LEASE_UNREADABLE"
    expected = contract.get("owner_lease_sha256")
    if type(expected) is not str or not _SHA64.fullmatch(expected):
        return False, "OWNER_LEASE_NOT_PINNED"
    if hashlib.sha256(lease_bytes).hexdigest() != expected:
        return False, "OWNER_LEASE_DIGEST_MISMATCH"
    try:
        lease = json.loads(lease_bytes)
    except (UnicodeError, ValueError):
        return False, "OWNER_LEASE_INVALID_JSON"
    if type(lease) is not dict:
        return False, "OWNER_LEASE_INVALID_JSON"
    generation = lease.get("generation")
    if (
        type(generation) is not int
        or generation <= 0
        or generation != contract.get("owner_generation")
    ):
        return False, "OWNER_GENERATION_MISMATCH"
    expires = parse_utc(lease.get("expires_utc"))
    if expires is None:
        return False, "OWNER_LEASE_EXPIRY_UNSET"
    if not now < expires:
        return False, "OWNER_LEASE_EXPIRED"
    groups = lease.get("groups")
    if type(groups) is not list or group not in groups:
        return False, "GROUP_NOT_LEASED"
    return True, "OWNER_LEASE_VERIFIED"


# ------------------------------------------------------------- adopted lanes


def _empty_ledger():
    return {
        "attempts": 0,
        "last_step": None,
        "last_restart_utc": None,
        "quarantined": False,
    }


def _verified_step(training):
    if type(training) is not dict or training.get("state") not in _PROGRESS_STATES:
        return None
    trainer = training.get("trainer_state")
    if type(trainer) is not dict:
        return None
    if trainer.get("integrity") != "PINNED_OBJECT_BYTES_VERIFIED":
        return None
    step = trainer.get("global_step")
    return step if type(step) is int and step >= 0 else None


def decide_adopted_lane(
    *, contract, lease_ok, provider, training, qualification, ledger, now
):
    """Classify one adopted training lane; at most RESTART_PERMITTED, never COMPLETE.

    provider: ``provider_transition_witness.witness`` over a fresh group GET plus an
    independent instance-list GET, with ``requested_replicas`` = contract replicas.
    training: ``training_observer.observe_training`` result (pinned revision).
    qualification: ``artifact_qualification.observe_candidate`` result or None.
    """
    led = dict(_empty_ledger(), **(ledger or {}))
    out = {
        "schema": SCHEMA,
        "group": contract.get("group"),
        "decision": "HOLD",
        "reason": None,
        "restart_permitted": False,
        "artifact_verified": False,
        "training_complete_verified": False,
        "promotion_authorized": False,
        "observed_utc": utc_text(now),
    }

    def done(decision, reason):
        out.update(decision=decision, reason=reason, ledger=led)
        return out

    if any(
        type(x) is dict and x.get("access_stop_required") is True
        for x in (provider, training, qualification)
    ):
        return done("HALT_ACCESS_STOP", "HTTP_401_403_OR_429_OBSERVED")
    if (
        type(qualification) is dict
        and qualification.get("status") == "ARTIFACT_VERIFIED"
        and qualification.get("ready_for_evaluation") is True
    ):
        out["artifact_verified"] = True
        return done(
            "ARTIFACT_VERIFIED_AWAITING_EVALUATION", "QUALIFIED_AT_PINNED_REVISION"
        )

    step = _verified_step(training)
    if step is not None and (led["last_step"] is None or step > led["last_step"]):
        led.update(last_step=step, attempts=0)
        out["progress_observed"] = True
    if led["quarantined"] is True:
        return done("HOLD_QUARANTINED", "OWNER_MUST_CLEAR_QUARANTINE")

    state = training.get("state") if type(training) is dict else None
    if state in _TERMINAL_OR_BROKEN:
        # Restarting a terminal or mis-targeted run would retrain over its output.
        return done("ALERT_TERMINAL_UNQUALIFIED", state)
    if type(provider) is not dict or provider.get("group_state_reconciled") is not True:
        reason = provider.get("state") if type(provider) is dict else None
        return done("HOLD_PROVIDER_UNRECONCILED", reason or "PROVIDER_UNOBSERVED")
    status = provider.get("provider_status")
    if status == "failed":
        return done("ALERT_PROVIDER_FAILED", "NO_AUTOMATIC_RESTART_FROM_FAILED")
    if status != "stopped":
        return done("OBSERVE", "PROVIDER_" + str(status).upper())
    counts = provider.get("actual_counts") or {}
    if (
        any(counts.get(k) != 0 for k in COUNTS)
        or provider.get("instances_returned") != 0
    ):
        return done("HOLD_PROVIDER_UNRECONCILED", "STOPPED_WITH_INSTANCES")
    if lease_ok is not True:
        return done("ALERT_STOPPED_NO_OWNER_LEASE", "OWNER_LEASE_UNVERIFIED")
    if state != "CHECKPOINT_PROGRESS_OBSERVED" or step is None:
        return done("ALERT_STOPPED_NO_RESUME_POINT", state or "TRAINING_UNOBSERVED")
    cooldown = contract.get("restart_cooldown_s", 1800)
    last = parse_utc(led["last_restart_utc"])
    if last is not None and (now - last).total_seconds() < cooldown:
        return done("HOLD_RESTART_COOLDOWN", "RECENT_RESTART")
    if led["attempts"] >= contract.get("max_restarts_without_progress", 3):
        led["quarantined"] = True
        return done("QUARANTINE_CRASH_LOOP", "RESTARTS_WITHOUT_VERIFIED_PROGRESS")
    led["attempts"] += 1
    led["last_restart_utc"] = utc_text(now)
    out["restart_permitted"] = True
    out["resume_step"] = step
    out["resume_revision"] = (training.get("snapshot") or {}).get("revision")
    return done("RESTART_PERMITTED", "STOPPED_WITH_VERIFIED_RESUME_POINT")


# ----------------------------------------------------------- checkpoint watch


def classify_checkpoint(training):
    """Map a pinned training observation to an honest checkpoint label.

    CHECKPOINT_STATE_VERIFIED means trainer_state.json bytes matched the Hub entry at
    one immutable revision. Adapter weights are NOT verified by this label; that needs
    ``artifact_qualification``. Repository-level HTTP 404 is UNOBSERVABLE, never absent.
    """
    if type(training) is not dict:
        return {"label": "UNOBSERVABLE", "reason": "NO_OBSERVATION"}
    state = training.get("state")
    snapshot = training.get("snapshot") or {}
    record = {
        "label": None,
        "observer_state": state,
        "revision": snapshot.get("revision"),
        "file_count": snapshot.get("file_count"),
        "adapter_weights_verified": False,
        "training_complete_verified": False,
    }
    if training.get("access_stop_required") is True:
        record.update(label="HALT_ACCESS_STOP", reason=training.get("reason"))
    elif state == "UNAVAILABLE":
        record.update(label="UNOBSERVABLE", reason=training.get("reason"))
    elif state == "NO_CHECKPOINT_OBSERVED":
        record["label"] = "NO_CHECKPOINT"
    elif _verified_step(training) is not None:
        record.update(
            label="CHECKPOINT_STATE_VERIFIED",
            step=training["trainer_state"]["global_step"],
            max_steps=training["trainer_state"].get("max_steps"),
            trainer_state_sha256=training["trainer_state"].get("trainer_state_sha256"),
        )
        if state != "CHECKPOINT_PROGRESS_OBSERVED":
            record["terminal_metadata"] = state
    else:
        record.update(label="HOLD", reason=state or "UNCLASSIFIED")
    return record


# ---------------------------------------------------------------- bench watch


def entry_bytes_verified(raw, *, size, lfs_sha256, blob_id):
    """Downloaded bytes equal the pinned Hub entry (LFS sha256, else git blob sha1)."""
    if type(raw) is not bytes or type(size) is not int or len(raw) != size:
        return False
    if lfs_sha256:
        return hashlib.sha256(raw).hexdigest() == lfs_sha256
    git = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest()
    return type(blob_id) is str and git == blob_id


def select_heldout_receipt(names, *, prefix, cutoff_utc):
    """Pick the newest strictly-named receipt at/after cutoff; diag or other names never."""
    pattern = re.compile(re.escape(prefix) + r"(\d{8}T\d{6})\.json\Z")
    best = None
    for name in names:
        match = pattern.fullmatch(name) if type(name) is str else None
        if not match:
            continue
        stamp = dt.datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=dt.timezone.utc
        )
        if stamp >= cutoff_utc and (best is None or stamp > best[0]):
            best = (stamp, name)
    return best[1] if best else None


def assess_heldout_receipt(record, contract):
    """Mechanical gate over a byte-verified receipt. MEASURE, not a model verdict."""
    gate = {"bound": False}
    if type(record) is not dict:
        return "GATE_FAIL_RECEIPT_INVALID", gate
    started = parse_utc(record.get("started_utc"))
    cutoff = parse_utc(contract.get("cutoff_utc"))
    binding = {
        "schema": record.get("schema") == "szl.salad-heldout/v2",
        "adapter_repo": record.get("adapter_repo") == contract.get("adapter_repo"),
        "base": record.get("base") == contract.get("base"),
        "base_revision": type(record.get("base_revision")) is str
        and bool(_SHA40.fullmatch(record["base_revision"]))
        and record["base_revision"]
        == contract.get("base_revision", record["base_revision"]),
        "started_after_cutoff": started is not None
        and cutoff is not None
        and started >= cutoff,
    }
    gate["binding"] = binding
    if not all(binding.values()):
        return "GATE_FAIL_RECEIPT_NOT_BOUND_TO_THIS_RUN", gate
    gate["bound"] = True
    base, adapter = record.get("base_arm") or {}, record.get("adapter_arm") or {}
    iso = record.get("base_arm_isolation") or {}
    total = iso.get("lora_layers_total")
    iso_ok = (
        type(total) is int
        and total > 0
        and iso.get("lora_layers_disabled_during_base_eval") == total
    )
    keys = ("refusal_rate", "neutral_acc", "neutral_correct")
    no_op = base.get("refusal_rate") is not None and all(
        base.get(k) == adapter.get(k) for k in keys
    )
    broken = record.get("broken_control_arm")
    failed = record.get("broken_control_failed_rules")
    gate.update(
        measured=record.get("kind") == "MEASURED",
        base_arm_isolation_ok=iso_ok,
        no_op_suspect=no_op,
        broken_control_present=type(broken) is dict,
        broken_control_failed_rules=failed if type(failed) is list else None,
        battery_admissible=record.get("battery_admissible") is True,
        base_arm={k: base.get(k) for k in keys + ("parse_fail",)},
        adapter_arm={k: adapter.get(k) for k in keys + ("parse_fail",)},
    )
    if not gate["measured"]:
        return "GATE_FAIL_NOT_MEASURED", gate
    if no_op:
        return "QUARANTINE_NOOP_SUSPECT", gate
    if not iso_ok:
        return "GATE_FAIL_BASE_ARM_NOT_ISOLATED", gate
    if type(broken) is not dict or type(failed) is not list or not failed:
        return "GATE_FAIL_BROKEN_CONTROL_NOT_DISCRIMINATING", gate
    if not gate["battery_admissible"]:
        return "GATE_FAIL_BATTERY_NOT_ADMISSIBLE", gate
    return "MEASURE_ADMISSIBLE_OWNER_REVIEW_REQUIRED", gate


def decide_bench_park(*, receipt_verified, verdict, lease_ok, provider):
    """Park only a bench whose own run produced a pinned, byte-verified receipt."""
    if type(provider) is dict and provider.get("access_stop_required") is True:
        return "HALT_ACCESS_STOP"
    if receipt_verified is not True:
        return "HOLD_RECEIPT_UNVERIFIED"
    if (
        verdict == "GATE_FAIL_RECEIPT_NOT_BOUND_TO_THIS_RUN"
        or verdict == "GATE_FAIL_RECEIPT_INVALID"
    ):
        return "HOLD_RECEIPT_NOT_BOUND"
    if type(provider) is not dict or provider.get("group_state_reconciled") is not True:
        return "HOLD_PROVIDER_UNRECONCILED"
    if provider.get("state") == "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION":
        return "ALREADY_DRAINED"
    if lease_ok is not True:
        return "ALERT_PARK_NEEDS_OWNER_LEASE"
    return "PARK_PERMITTED"


# ------------------------------------------------------ publication readback


def publication_result(*, payload, commit_oid, readback, path_listed):
    """A Hub upload counts as published only after a pinned byte-identical readback."""
    sha = hashlib.sha256(payload).hexdigest()
    ok = (
        type(commit_oid) is str
        and bool(_SHA40.fullmatch(commit_oid))
        and type(readback) is bytes
        and hashlib.sha256(readback).hexdigest() == sha
        and path_listed is True
    )
    return {
        "state": "PUBLISHED_READBACK_VERIFIED" if ok else "PUBLICATION_UNVERIFIED",
        "payload_sha256": sha,
        "commit": commit_oid if type(commit_oid) is str else None,
        "signature_status": "NOT_SIGNED",
    }


def chain_record(record, prev_sha256):
    """Hash-chain a local receipt line. A hash chain is tamper-evidence, not a signature."""
    if prev_sha256 is not None and not _SHA64.fullmatch(str(prev_sha256)):
        raise ValueError("PREVIOUS_RECEIPT_HASH_INVALID")
    body = dict(record, prev_sha256=prev_sha256, signature_status="NOT_SIGNED")
    body.pop("sha256", None)
    body["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def verify_chain(lines):
    prev = None
    for raw in lines:
        rec = json.loads(raw)
        claimed = rec.pop("sha256", None)
        if (
            rec.get("prev_sha256") != prev
            or hashlib.sha256(canonical(rec)).hexdigest() != claimed
        ):
            return False
        prev = claimed
    return True


def valid_group(name):
    return type(name) is str and bool(_GROUP.fullmatch(name))


def valid_repo(name):
    return type(name) is str and bool(_REPO.fullmatch(name))
