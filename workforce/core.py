"""Shared plumbing for one-shot officer cycles (stdlib only).

Network access goes through :class:`Net` so tests inject a fake and no cycle
needs a Hugging Face Space, a persistent disk, or a long-lived process.
"""

import hashlib
import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "szl.officer/actions-v1"
DEFAULT_DATASET = "SZLHOLDINGS/frontier-54-runs"
MAX_BODY = 1_048_576

# A record must never carry credential-shaped strings. Same patterns the
# cyber-shield v3 Space scanned for, applied to every record before it is
# written, so a leak is refused structurally rather than detected later.
SECRET_PATTERN = re.compile(
    r"hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Net:
    """Real network access: bounded reads, no redirects, read-only by default.

    ``gh_read_token`` (the workflow's read-only ``github.token``) is attached
    only to requests whose host is exactly ``api.github.com``.
    """

    def __init__(self, gh_read_token="", user_agent="szl-workforce/1.0"):
        self._gh = gh_read_token
        self._ua = user_agent
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, url, data=None, headers=None, method=None):
        h = {"User-Agent": self._ua}
        if self._gh and urllib.parse.urlsplit(url).hostname == "api.github.com":
            h["Authorization"] = "Bearer " + self._gh
        h.update(headers or {})
        return urllib.request.Request(url, data=data, headers=h, method=method)

    def get_bytes(self, url, timeout=20, limit=MAX_BODY):
        try:
            with self._opener.open(self._request(url), timeout=timeout) as r:
                body = r.read(limit + 1)
        except (OSError, ValueError):
            return None
        if len(body) > limit:
            return None
        return body

    def get_json(self, url, timeout=20):
        body = self.get_bytes(url, timeout=timeout)
        if body is None:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return None

    def get_json_with_headers(self, url, headers, timeout=20, limit=MAX_BODY):
        """GET with extra headers; raises on HTTP errors (caller decides)."""
        with self._opener.open(
            self._request(url, headers=headers), timeout=timeout
        ) as r:
            body = r.read(limit + 1)
        if len(body) > limit:
            raise ValueError("response exceeds limit")
        return json.loads(body)

    def status(self, url, timeout=15):
        """HTTP status code of a GET, or None when unreachable."""
        try:
            with self._opener.open(self._request(url), timeout=timeout) as r:
                r.read(2048)
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except (OSError, ValueError):
            return None

    def resolve(self, host):
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            return False

    def post_json(self, url, payload, headers=None, timeout=60, limit=65536):
        data = json.dumps(payload).encode()
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        req = self._request(url, data=data, headers=h, method="POST")
        with self._opener.open(req, timeout=timeout) as r:
            body = r.read(limit + 1)
        if len(body) > limit:
            raise ValueError("response exceeds limit")
        return json.loads(body)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_utc(stamp):
    """Parse an ISO-8601 UTC stamp ('...Z' or '+00:00') to epoch seconds.

    The v2 auditor used time.mktime (local time) here; this is the UTC fix.
    """
    import calendar

    return calendar.timegm(time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S"))


def is_public_https_host(url):
    """True only for https URLs whose host is a DNS name, not an IP literal,
    localhost, or an internal-only suffix. Used to refuse private estate
    addresses and plain HTTP for any outbound call that carries credentials.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname.lower()
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    if host == "localhost" or "." not in host:
        return False
    for suffix in (".local", ".internal", ".lan", ".home", ".localhost", ".arpa"):
        if host.endswith(suffix):
            return False
    if parts.username or parts.password:
        return False
    return True


def run_identity(env=None):
    """Who ran this cycle. Persistent identity comes from the Sigstore
    certificate the workflow attaches (repo + workflow path + ref), not from a
    key stored on the runner. These fields only echo the run for readers."""
    env = os.environ if env is None else env
    keys = (
        "GITHUB_REPOSITORY",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_EVENT_NAME",
    )
    ident = {k.lower(): env.get(k) for k in keys if env.get(k)}
    ident["signing"] = (
        "sigstore keyless via actions/attest (GitHub artifact attestation); "
        "the record is unsigned until the workflow attests it"
    )
    return ident


def build_record(role, findings, env=None, now=None):
    body = {
        "schema": SCHEMA,
        "officer": role,
        "runner": "github-actions one-shot cycle",
        "hf_space_required": False,
        "findings": findings,
        "run": run_identity(env),
        "timestamp_utc": now or utc_now(),
        "chain": (
            "none: each record stands alone; ordering is the attested "
            "timestamp and GitHub run id (no hidden /data state)"
        ),
    }
    blob = canonical(body).encode()
    return body, blob


def refuse_secrets(blob):
    text = blob.decode("utf-8", errors="replace")
    if SECRET_PATTERN.search(text):
        raise ValueError("record contains a credential-shaped string; refusing")


def record_name(role, blob):
    return f"{role}-{sha256_hex(blob)[:12]}.json"
