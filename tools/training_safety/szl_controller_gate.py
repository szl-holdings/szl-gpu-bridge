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
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from provider_transition_witness import witness

OK_STATUSES = (200, 202, 204)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
FINAL_REQUIRED = ("final/adapter_model.safetensors", "final/adapter_config.json")
# Allow-list, not a deny-list: anything else under final/ HOLDs the stop.
SAFE_FINAL_FILES = frozenset(
    "final/" + n
    for n in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        ".gitattributes",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
    )
)
MAX_BODY = 1 << 20
TAIL_SCAN_LIMIT = 64 << 20  # most of a log we will stream to find its tail

SALAD_HOSTS = frozenset({"api.salad.com"})
HF_API_HOSTS = frozenset({"huggingface.co"})
CREDENTIAL_HEADERS = ("salad-api-key", "authorization")
REDIRECT_CODES = (301, 302, 303, 307, 308)
OWNER_PLACEHOLDER = "<OWNER"


# --------------------------------------------------------------------------
# (a) provider calls: HTTP status and body decide success, never a shell rc
# --------------------------------------------------------------------------
class HttpResult(dict):
    """{'transport_ok': bool, 'status': int|None, 'body': bytes, 'error': str|None}"""


class UnsafeUrl(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never let urllib follow a redirect on its own: it would forward every
    request header (including credentials) to whatever host the server names."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class _AnyHost(frozenset):
    def __contains__(self, item):
        return True


_ANY_HOST = _AnyHost()


def check_url(url, allowed_hosts):
    """https only, no userinfo, default port, and the host must be allow-listed."""
    try:
        u = urllib.parse.urlsplit(url)
        host, port = u.hostname, u.port
    except ValueError:
        raise UnsafeUrl("URL_UNPARSEABLE")
    if u.scheme != "https":
        raise UnsafeUrl("URL_NOT_HTTPS")
    if u.username is not None or u.password is not None:
        raise UnsafeUrl("URL_HAS_USERINFO")
    if port not in (None, 443):
        raise UnsafeUrl("URL_NONDEFAULT_PORT")
    if not host or host.lower() not in allowed_hosts:
        raise UnsafeUrl("URL_HOST_NOT_ALLOWED")
    return url


def _has_credential(headers):
    return any(str(k).lower() in CREDENTIAL_HEADERS for k in headers)


def _reject(code):
    return HttpResult(transport_ok=False, status=None, body=b"", error=code)


def provider_call(
    method, url, headers, body=None, *, timeout=60, opener=None, allowed_hosts=None
):
    """Perform one HTTP request and return status + body. Never raises on HTTP
    errors, never logs headers, never follows a redirect (a 3xx comes back as its
    status and fails closed). `headers` may hold a credential; it is used only for
    this request, and only if `allowed_hosts` names the URL's host over https."""
    if _has_credential(headers) and allowed_hosts is None:
        return _reject("CREDENTIAL_WITHOUT_HOST_ALLOWLIST")
    try:
        check_url(url, _ANY_HOST if allowed_hosts is None else allowed_hosts)
    except UnsafeUrl as e:
        return _reject(str(e))
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
    open_fn = opener or _OPENER.open
    try:
        with open_fn(req, timeout=timeout) as resp:
            return HttpResult(
                transport_ok=True,
                status=int(resp.status),
                body=resp.read(MAX_BODY + 1),
                error=None,
            )
    except urllib.error.HTTPError as e:  # non-2xx still carries a status/body
        try:
            raw = e.read(MAX_BODY + 1)
        except Exception:
            raw = b""
        return HttpResult(transport_ok=True, status=int(e.code), body=raw, error=None)
    except Exception as e:  # DNS, TLS, timeout: status unobserved
        return HttpResult(
            transport_ok=False, status=None, body=b"", error=type(e).__name__
        )


def host_allowed_for_download(host):
    """Hosts a Hugging Face file download may be redirected to (CDN). Credentials
    are never sent to any of them; only to the exact API host."""
    h = (host or "").lower()
    return (
        h == "huggingface.co"
        or h.endswith(".huggingface.co")
        or h == "hf.co"
        or h.endswith(".hf.co")
    )


def open_following(
    url,
    headers,
    *,
    timeout=60,
    opener=None,
    max_hops=4,
    auth_hosts=HF_API_HOSTS,
    host_ok=host_allowed_for_download,
):
    """GET `url`, following redirects by hand. The first URL must be https on an
    `auth_hosts` host. Every hop must be https and satisfy `host_ok`. Credential
    headers are sent only to `auth_hosts` and are dropped on any other host.
    Returns (resp, None) with an open response (caller closes it), or
    (None, code) where code is an HTTP status int or an error string."""
    try:
        check_url(url, auth_hosts)
    except UnsafeUrl as e:
        return None, str(e)
    open_fn = opener or _OPENER.open
    cur = url
    for _ in range(max_hops + 1):
        host = (urllib.parse.urlsplit(cur).hostname or "").lower()
        hdrs = {
            k: v
            for k, v in headers.items()
            if host in auth_hosts or str(k).lower() not in CREDENTIAL_HEADERS
        }
        try:
            return open_fn(
                urllib.request.Request(cur, headers=hdrs, method="GET"), timeout=timeout
            ), None
        except urllib.error.HTTPError as e:
            if e.code not in REDIRECT_CODES:
                return None, int(e.code)
            loc = e.headers.get("Location") if e.headers else None
            if not loc:
                return None, "REDIRECT_WITHOUT_LOCATION"
            cur = urllib.parse.urljoin(cur, loc)
            try:
                u = urllib.parse.urlsplit(cur)
                check_url(cur, _ANY_HOST)
                if not host_ok(u.hostname):
                    raise UnsafeUrl("REDIRECT_HOST_NOT_ALLOWED")
            except UnsafeUrl as ue:
                return None, str(ue)
        except Exception as e:
            return None, "TRANSPORT_" + type(e).__name__
    return None, "TOO_MANY_REDIRECTS"


def fetch_tail(url, headers, *, max_bytes=MAX_BODY, opener=None, timeout=60):
    """Return (text, code): the last `max_bytes` of a (text) file, plus whether the
    front was cut. Asks the server for a suffix range; if the server ignores it,
    streams the file and keeps a rolling tail (up to TAIL_SCAN_LIMIT, else
    LANE_LOG_TOO_LARGE). Never reports a truncated read as a complete one."""
    hdrs = dict(headers)
    hdrs["Range"] = "bytes=-%d" % max_bytes
    resp, err = open_following(url, hdrs, opener=opener, timeout=timeout)
    if resp is None:
        return None, err, False
    try:
        status = int(resp.status)
        if status not in (200, 206):
            return None, status, False
        cut = False
        if status == 206:
            cr = resp.headers.get("Content-Range", "") if resp.headers else ""
            m = re.match(r"bytes\s+(\d+)-\d+/", cr)
            cut = bool(m and int(m.group(1)) > 0)
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None, "LANE_LOG_TRUNCATED", False
        else:
            data, total = b"", 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > TAIL_SCAN_LIMIT:
                    return None, "LANE_LOG_TOO_LARGE", False
                data = (data + chunk)[-max_bytes:]
            cut = total > max_bytes
    finally:
        resp.close()
    if cut:  # drop the partial first line so ^ anchors stay honest
        nl = data.find(b"\n")
        data = data[nl + 1 :] if nl >= 0 else b""
    return data, None, cut


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


def find_placeholders(obj, path="$"):
    """Paths of every string that still holds an '<OWNER: ...>' placeholder."""
    found = []
    if isinstance(obj, str):
        if OWNER_PLACEHOLDER in obj:
            found.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += find_placeholders(v, "%s.%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for n, v in enumerate(obj):
            found += find_placeholders(v, "%s[%d]" % (path, n))
    return found


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
        if (
            (type(emb) is int and emb >= 400)
            or val.get("title") == "Bad Request"
            or (
                isinstance(typ, str)
                and ("pending_update_in_progress" in typ or "error" in typ.lower())
            )
            or "error" in val
            or "errors" in val
        ):
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
    return classify_write(
        HttpResult(transport_ok=True, status=status, body=body, error=None)
    )


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


def admission_readback(
    name,
    requested_replicas,
    get_group,
    get_instances,
    *,
    polls=30,
    interval=60,
    sleep=time.sleep,
):
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
            w = witness(
                transport_exit=0,
                http_status=200,
                body=b"",
                group=gv,
                instances=iv,
                name=name,
                requested_replicas=requested_replicas,
            )
            last = w["state"]
            if (
                w.get("group_state_reconciled")
                and w["actual_counts"]["running_count"] >= 1
                and any(
                    isinstance(x, dict) and x.get("state") == "running"
                    for x in iv.get("instances", [])
                )
            ):
                return {
                    "admitted": True,
                    "state": "ADMITTED_RUNNING_READBACK",
                    "witness": w,
                    "polls": n + 1,
                }
        if n + 1 < polls:
            sleep(interval)
    return {"admitted": False, "state": "HOLD_" + last, "polls": n + 1}


def confirm_zero(name, get_group, get_instances):
    """DONE after a scale-down only when desired replicas == 0 and every actual
    count and the instance list are zero (completion-guard section 'DONE')."""
    gv, iv, err = _readback(get_group, get_instances)
    if err:
        return {"done": False, "state": "READBACK_" + err}
    w = witness(
        transport_exit=0,
        http_status=200,
        body=b"",
        group=gv,
        instances=iv,
        name=name,
        requested_replicas=0,
    )
    done = w["state"] == "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION"
    return {
        "done": done,
        "state": "ZERO_INSTANCE_READBACK" if done else w["state"],
        "witness": w,
    }


# --------------------------------------------------------------------------
# (b)/(c) stop predicate: completion-guard evidence, never a filename
# --------------------------------------------------------------------------
def _hold(code, **extra):
    out = {
        "decision": "HOLD",
        "code": code,
        "stop_allowed": False,
        "completion_claimed": False,
        "quality_evaluated": False,
    }
    out.update(extra)
    return out


GITSHA1 = re.compile(r"[0-9a-f]{40}\Z")


def _entry_ok(entry):
    """Hub metadata must give a positive size and a content id: sha256 for LFS
    files, git blob sha1 for small non-LFS files (e.g. trainer_state.json)."""
    if (
        not isinstance(entry, dict)
        or type(entry.get("size")) is not int
        or entry["size"] <= 0
    ):
        return False
    sha, blob = entry.get("sha256"), entry.get("git_blob_sha1")
    return bool(
        (isinstance(sha, str) and SHA256.fullmatch(sha))
        or (isinstance(blob, str) and GITSHA1.fullmatch(blob))
    )


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
    repo_private      True or False (readback, not assumed); None/missing HOLDs. False
                      is accepted only for reason == 'eval_receipt' with
                      allow_public_repo == True, plus the receipt content rule below
    allow_public_repo owner opt-in (bool) for a public receipt repo
    receipt_schema / receipt_required_fields
                      eval_receipt only: schema id and result fields the parsed
                      receipt must carry (owner-supplied)
    receipt_observed  {path: {schema, fields}} parsed from the downloaded receipt
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
        return _hold(
            "CHURN_ALERT_NOT_COMPLETION",
            alert=True,
            spend_park_candidate=bool(backed),
            note="owner approval required for any spend park; "
            "latest checkpoint backup verified"
            if backed
            else "owner approval required; latest checkpoint NOT independently backed up",
        )
    if reason not in ("completion", "lane_log", "eval_receipt"):
        return _hold("REASON_NOT_A_COMPLETION_PREDICATE")
    if not isinstance(ev.get("revision"), str) or not REVISION.fullmatch(
        ev["revision"]
    ):
        return _hold("REVISION_NOT_PINNED")
    for key in (
        "required_paths",
        "completion_marker",
        "receipt_schema",
        "receipt_required_fields",
        "backup",
    ):
        if find_placeholders(ev.get(key)):
            return _hold("OWNER_PLACEHOLDER_IN_EVIDENCE", field=key)
    private = ev.get("repo_private")
    if private is not True:
        # Public is acceptable only for an eval receipt, only when the owner said
        # so explicitly, and only because the receipt bytes are hash- and
        # content-verified below. Anything else (or unknown) HOLDs.
        if private is not False:
            return _hold("REPO_PRIVACY_UNVERIFIED")
        if reason != "eval_receipt" or ev.get("allow_public_repo") is not True:
            return _hold("REPO_PUBLIC_NOT_ALLOWED")
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
        latest, target, ts = (
            ev.get("latest_checkpoint"),
            ev.get("target_steps"),
            ev.get("trainer_state"),
        )
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
        if any(p.startswith("final/") and p not in SAFE_FINAL_FILES for p in entries):
            return _hold("UNSAFE_FINAL_EXPORT")
        required = list(
            ev.get("required_paths")
            or list(FINAL_REQUIRED)
            + [
                "checkpoint-%d/adapter_model.safetensors" % latest,
                "checkpoint-%d/trainer_state.json" % latest,
            ]
        )
    for p in required:
        if not _safe_rel_path(p):
            return _hold("UNSAFE_REQUIRED_PATH")
        if p not in entries:
            return _hold("REQUIRED_ARTIFACT_MISSING", path=p)
        if not _entry_ok(entries[p]):
            return _hold("HUB_METADATA_INCOMPLETE", path=p)
    if reason != "eval_receipt":
        ck = "checkpoint-%d/adapter_model.safetensors" % ev["latest_checkpoint"]
        fin = entries.get("final/adapter_model.safetensors") or {}
        if ck in entries and fin.get("sha256") != entries[ck].get("sha256"):
            return _hold("FINAL_DIFFERS_FROM_COMPLETED_CHECKPOINT")
    local = ev.get("local_readback") or {}
    for p in required:
        if p not in local:
            return _hold("ARTIFACT_NOT_READ_BACK", path=p)
        if not _match(entries[p], local[p]):
            return _hold("ARTIFACT_HASH_OR_SIZE_MISMATCH", path=p)
    if not _backup_ok(ev, required):
        return _hold("INDEPENDENT_BACKUP_UNVERIFIED")
    if reason == "eval_receipt":
        bad = _receipt_content_hold(ev, required)
        if bad:
            return bad
    return {
        "decision": "STOP_ALLOWED",
        "code": "COMPLETION_GUARD_EVIDENCE_PRESENT",
        "stop_allowed": True,
        "completion_claimed": False,
        "quality_evaluated": False,
        "verified_paths": sorted(required),
        "note": "DONE still requires confirm_zero() readback after the PATCH",
    }


def _safe_rel_path(p):
    """A relative Hub path with no '..', drive, backslash or absolute component."""
    if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p or ":" in p:
        return False
    return all(part not in ("", ".", "..") for part in p.split("/"))


def _receipt_content_hold(ev, required):
    """An eval receipt counts only if its parsed content carries the owner's
    schema id and result fields; a file merely existing at the path is not done."""
    schema, fields = ev.get("receipt_schema"), ev.get("receipt_required_fields")
    if (
        not isinstance(schema, str)
        or not schema
        or not isinstance(fields, list)
        or not fields
        or not all(isinstance(f, str) and f for f in fields)
    ):
        return _hold("RECEIPT_RULE_UNDEFINED")
    seen = ev.get("receipt_observed") or {}
    for p in required:
        o = seen.get(p)
        if not isinstance(o, dict):
            return _hold("RECEIPT_CONTENT_UNVERIFIED", path=p)
        if o.get("schema") != schema:
            return _hold("RECEIPT_SCHEMA_MISMATCH", path=p)
        if any(f not in (o.get("fields") or []) for f in fields):
            return _hold("RECEIPT_FIELDS_MISSING", path=p)
    return None


def _ckpt_paths(ev):
    latest = ev.get("latest_checkpoint")
    if type(latest) is not int:
        return []
    return [
        "checkpoint-%d/adapter_model.safetensors" % latest,
        "checkpoint-%d/trainer_state.json" % latest,
    ]


def _backup_ok(ev, paths):
    b = ev.get("backup")
    entries = ev.get("hub_entries") or {}
    if (
        not isinstance(b, dict)
        or b.get("independent") is not True
        or not b.get("location")
    ):
        return False
    files = b.get("files") or {}
    return bool(paths) and all(
        p in entries and _entry_ok(entries[p]) and _match(entries[p], files.get(p))
        for p in paths
    )


def measure_file(path):
    """size, sha256 and git-blob sha1 of a local file (readback and backup rehash)."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    g = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            g.update(block)
    return {"size": size, "sha256": h.hexdigest(), "git_blob_sha1": g.hexdigest()}
