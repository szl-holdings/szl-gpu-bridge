"""SZL controller safety gate (stdlib only; Python 3.8+ compatible).

Fixes three defects found 2026-11-19 on BETTERWITHAGE:
  (a) provider writes judged by curl/shell rc instead of HTTP status + body;
  (b) stop decisions taken on a bare Hugging Face `final/` filename;
  (c) zero-replica decisions taken on `lane.log` presence.

Contract source: szl-gpu-bridge docs/runbooks/flagship54-completion-guard.md
@344dd604 and tools/salad/provider_transition_witness.py @817ea920 (vendored
unchanged next to this file).

Nothing here authorises training, evaluation, release or quality claims. A
STOP_ALLOWED decision only says the completion-guard evidence is present; the
caller must still be run in --apply mode by its owner. Credentials are read by
callers from files at request time and are never logged, returned or put in argv.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request

from provider_transition_witness import witness, COUNTS

OK_STATUSES = (200, 202, 204)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
FINAL_REQUIRED = ("final/adapter_model.safetensors", "final/adapter_config.json")
UNSAFE_FINAL_SUFFIXES = (".pt", ".pth", ".bin", ".pkl", ".pickle", ".ckpt", ".py")
MAX_BODY = 1 << 20


# --------------------------------------------------------------------------
# (a) provider calls: HTTP status and body decide success, never a shell rc
# --------------------------------------------------------------------------
class HttpResult(dict):
    """{'transport_ok': bool, 'status': int|None, 'body': bytes, 'error': str|None}"""


def provider_call(method, url, headers, body=None, *, timeout=60, opener=None):
    """Perform one HTTP request and return status + body. Never raises on HTTP
    errors, never logs headers. `headers` may hold a credential; it is used only
    for this request."""
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(req, timeout=timeout) as resp:
            return HttpResult(transport_ok=True, status=int(resp.status),
                              body=resp.read(MAX_BODY + 1), error=None)
    except urllib.error.HTTPError as e:  # non-2xx still carries a status/body
        try:
            raw = e.read(MAX_BODY + 1)
        except Exception:
            raw = b""
        return HttpResult(transport_ok=True, status=int(e.code), body=raw, error=None)
    except Exception as e:  # DNS, TLS, timeout: status unobserved
        return HttpResult(transport_ok=False, status=None, body=b"",
                          error=type(e).__name__)


def _json_obj(raw):
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_BODY:
        return None, "BODY_BYTE_BUDGET"
    if not raw.strip():
        return None, None
    try:
        val = json.loads(bytes(raw).decode("utf-8"))
    except (ValueError, UnicodeError):
        return None, "BODY_INVALID_JSON"
    return val, None


def classify_write(result):
    """Return (ok, code). ok is True only for transport success, a 2xx status and
    a body that is not a problem/error envelope."""
    if not isinstance(result, dict) or result.get("transport_ok") is not True:
        return False, "TRANSPORT_FAILED_OR_UNKNOWN"
    status = result.get("status")
    if type(status) is not int:
        return False, "HTTP_STATUS_UNOBSERVED"
    if status in (401, 403):
        return False, "ACCESS_DENIED_%d" % status
    if status == 429:
        return False, "RATE_LIMITED_429"
    if status not in OK_STATUSES:
        return False, "HTTP_%d" % status
    val, err = _json_obj(result.get("body", b""))
    if err:
        return False, err
    if isinstance(val, dict):
        emb = val.get("status")
        typ = val.get("type")
        if (type(emb) is int and emb >= 400) or val.get("title") == "Bad Request" \
                or (isinstance(typ, str) and ("pending_update_in_progress" in typ or "error" in typ.lower())) \
                or "error" in val or "errors" in val:
            return False, "PROVIDER_ERROR_ENVELOPE"
    return True, "HTTP_%d_ACCEPTED" % status


def classify_shell_curl(returncode, stdout, http_code_text):
    """For callers that must keep curl: run it with `-w '\\n%{http_code}'` (or a
    separate --write-out file) and pass the captured code here. rc=0 alone is
    never success."""
    if returncode != 0:
        return False, "TRANSPORT_FAILED_OR_UNKNOWN"
    try:
        status = int(str(http_code_text).strip())
    except ValueError:
        return False, "HTTP_STATUS_UNOBSERVED"
    body = stdout if isinstance(stdout, bytes) else str(stdout).encode()
    return classify_write(HttpResult(transport_ok=True, status=status, body=body, error=None))


# --------------------------------------------------------------------------
# admission: a write is only admitted after a successful state + instance readback
# --------------------------------------------------------------------------
def _readback(get_group, get_instances):
    g, i = get_group(), get_instances()
    gok, gcode = classify_write(g)
    iok, icode = classify_write(i)
    if not gok or not iok:
        return None, None, (gcode if not gok else icode)
    gv, _ = _json_obj(g["body"])
    iv, _ = _json_obj(i["body"])
    return gv, iv, None


def admission_readback(name, requested_replicas, get_group, get_instances, *,
                       polls=30, interval=60, sleep=time.sleep):
    """Poll provider readback. ADMITTED requires a reconciled group with the
    requested replicas, no pending change, running_count >= 1 and at least one
    instance in state 'running'. Anything else ends as HOLD_<code>."""
    last = "NOT_POLLED"
    for n in range(max(1, polls)):
        gv, iv, err = _readback(get_group, get_instances)
        if err:
            last = "READBACK_" + err
            if err.startswith(("ACCESS_DENIED", "RATE_LIMITED")):
                break  # never hammer an auth/rate failure
        else:
            w = witness(transport_exit=0, http_status=200, body=b"", group=gv,
                        instances=iv, name=name, requested_replicas=requested_replicas)
            last = w["state"]
            if w.get("group_state_reconciled") and w["actual_counts"]["running_count"] >= 1 \
                    and any(isinstance(x, dict) and x.get("state") == "running"
                            for x in iv.get("instances", [])):
                return {"admitted": True, "state": "ADMITTED_RUNNING_READBACK",
                        "witness": w, "polls": n + 1}
        if n + 1 < polls:
            sleep(interval)
    return {"admitted": False, "state": "HOLD_" + last, "polls": n + 1}


def confirm_zero(name, get_group, get_instances):
    """DONE after a scale-down only when desired replicas == 0 and every actual
    count and the instance list are zero (completion-guard section 'DONE')."""
    gv, iv, err = _readback(get_group, get_instances)
    if err:
        return {"done": False, "state": "READBACK_" + err}
    w = witness(transport_exit=0, http_status=200, body=b"", group=gv, instances=iv,
                name=name, requested_replicas=0)
    done = w["state"] == "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION"
    return {"done": done, "state": "ZERO_INSTANCE_READBACK" if done else w["state"], "witness": w}


# --------------------------------------------------------------------------
# (b)/(c) stop predicate: completion-guard evidence, never a filename
# --------------------------------------------------------------------------
def _hold(code, **extra):
    out = {"decision": "HOLD", "code": code, "stop_allowed": False,
           "completion_claimed": False, "quality_evaluated": False}
    out.update(extra)
    return out


GITSHA1 = re.compile(r"[0-9a-f]{40}\Z")


def _entry_ok(entry):
    """Hub metadata must give a positive size and a content id: sha256 for LFS
    files, git blob sha1 for small non-LFS files (e.g. trainer_state.json)."""
    if not isinstance(entry, dict) or type(entry.get("size")) is not int or entry["size"] <= 0:
        return False
    sha, blob = entry.get("sha256"), entry.get("git_blob_sha1")
    return bool((isinstance(sha, str) and SHA256.fullmatch(sha)) or
                (isinstance(blob, str) and GITSHA1.fullmatch(blob)))


def _match(expected, measured):
    if not isinstance(measured, dict) or measured.get("size") != expected["size"]:
        return False
    if expected.get("sha256"):
        return measured.get("sha256") == expected["sha256"]
    return measured.get("git_blob_sha1") == expected.get("git_blob_sha1")


def decide_stop(ev):
    """Pure decision over supplied evidence. Required keys:

    reason            'completion' | 'lane_log' | 'churn' | 'eval_receipt' | other
    revision          40-hex immutable Hub commit the evidence was read at
    repo_private      True (readback, not assumed)
    target_steps      frozen configured step target (int) -- not needed for eval_receipt
    latest_checkpoint int, highest checkpoint-N directory at `revision`
    trainer_state     parsed checkpoint-N/trainer_state.json {global_step, max_steps}
    hub_entries       {path: {size, sha256}} from Hub metadata at `revision`
    required_paths    optional override of artifact paths that must be verified
    local_readback    {path: {size, sha256}} measured from downloaded bytes
    backup            {location, independent: bool, files: {path: {size, sha256}}}
                      rehashed after copy to a node other than the primary
    lane_log_text     required for reason == 'lane_log'
    completion_marker regex that must match lane_log_text (owner-supplied)
    """
    reason = ev.get("reason")
    if reason == "churn":
        # Machine churn is a spend signal, not completion. Never auto-stop on it.
        backed = _backup_ok(ev, _ckpt_paths(ev)) if ev.get("backup") else False
        return _hold("CHURN_ALERT_NOT_COMPLETION", alert=True,
                     spend_park_candidate=bool(backed),
                     note="owner approval required for any spend park; "
                          "latest checkpoint backup verified" if backed else
                          "owner approval required; latest checkpoint NOT independently backed up")
    if reason not in ("completion", "lane_log", "eval_receipt"):
        return _hold("REASON_NOT_A_COMPLETION_PREDICATE")
    if not isinstance(ev.get("revision"), str) or not REVISION.fullmatch(ev["revision"]):
        return _hold("REVISION_NOT_PINNED")
    if ev.get("repo_private") is not True:
        return _hold("REPO_PRIVACY_UNVERIFIED")
    if reason == "lane_log":
        text, marker = ev.get("lane_log_text"), ev.get("completion_marker")
        if not isinstance(text, str) or not text.strip():
            return _hold("LANE_LOG_UNREAD")
        if not isinstance(marker, str) or not marker:
            return _hold("COMPLETION_MARKER_UNDEFINED")
        if not re.search(marker, text, re.M):
            return _hold("LANE_LOG_WITHOUT_COMPLETION")
    entries = ev.get("hub_entries") or {}
    if reason == "eval_receipt":
        required = list(ev.get("required_paths") or [])
        if not required:
            return _hold("EVAL_RECEIPT_PATH_UNDEFINED")
    else:
        latest, target, ts = ev.get("latest_checkpoint"), ev.get("target_steps"), ev.get("trainer_state")
        if type(target) is not int or target <= 0:
            return _hold("TARGET_STEPS_UNKNOWN")
        if type(latest) is not int or latest <= 0:
            return _hold("NO_CHECKPOINT")
        if not isinstance(ts, dict) or ts.get("global_step") != latest:
            return _hold("CHECKPOINT_STEP_MISMATCH")
        if ts.get("max_steps") != target:
            return _hold("TRAINING_TARGET_CHANGED")
        if latest < target:
            return _hold("STEP_TARGET_NOT_REACHED", latest=latest, target=target)
        final_names = [p for p in entries if p.startswith("final/")]
        if any(p.lower().endswith(UNSAFE_FINAL_SUFFIXES) for p in final_names):
            return _hold("UNSAFE_FINAL_EXPORT")
        required = list(ev.get("required_paths") or
                        list(FINAL_REQUIRED) + ["checkpoint-%d/adapter_model.safetensors" % latest,
                                                "checkpoint-%d/trainer_state.json" % latest])
    for p in required:
        if p not in entries:
            return _hold("REQUIRED_ARTIFACT_MISSING", path=p)
        if not _entry_ok(entries[p]):
            return _hold("HUB_METADATA_INCOMPLETE", path=p)
    if reason != "eval_receipt":
        ck = "checkpoint-%d/adapter_model.safetensors" % ev["latest_checkpoint"]
        if ck in entries and entries["final/adapter_model.safetensors"].get("sha256") != entries[ck].get("sha256"):
            return _hold("FINAL_DIFFERS_FROM_COMPLETED_CHECKPOINT")
    local = ev.get("local_readback") or {}
    for p in required:
        if p not in local:
            return _hold("ARTIFACT_NOT_READ_BACK", path=p)
        if not _match(entries[p], local[p]):
            return _hold("ARTIFACT_HASH_OR_SIZE_MISMATCH", path=p)
    if not _backup_ok(ev, required):
        return _hold("INDEPENDENT_BACKUP_UNVERIFIED")
    return {"decision": "STOP_ALLOWED", "code": "COMPLETION_GUARD_EVIDENCE_PRESENT",
            "stop_allowed": True, "completion_claimed": False, "quality_evaluated": False,
            "verified_paths": sorted(required),
            "note": "DONE still requires confirm_zero() readback after the PATCH"}


def _ckpt_paths(ev):
    latest = ev.get("latest_checkpoint")
    if type(latest) is not int:
        return []
    return ["checkpoint-%d/adapter_model.safetensors" % latest,
            "checkpoint-%d/trainer_state.json" % latest]


def _backup_ok(ev, paths):
    b = ev.get("backup")
    entries = ev.get("hub_entries") or {}
    if not isinstance(b, dict) or b.get("independent") is not True or not b.get("location"):
        return False
    files = b.get("files") or {}
    return bool(paths) and all(p in entries and _entry_ok(entries[p]) and _match(entries[p], files.get(p))
                               for p in paths)


def measure_file(path):
    """size, sha256 and git-blob sha1 of a local file (readback and backup rehash)."""
    import os
    size = os.path.getsize(path)
    h = hashlib.sha256()
    g = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            g.update(block)
    return {"size": size, "sha256": h.hexdigest(), "git_blob_sha1": g.hexdigest()}
