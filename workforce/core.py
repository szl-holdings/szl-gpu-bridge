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
    # Cloudflare Access service token: client id is "<32 hex>.access"; the
    # secret is 64 hex, which is also the shape of a sha256 digest (records
    # carry those), so the secret is only matched next to its header/field
    # name. Cloudflare API tokens carry the scannable cfut_/cfat_/cfk_ prefix.
    r"|\b[0-9a-f]{32}\.access\b"
    r"|(?i:cf[-_]access[-_]client[-_](?:id|secret))[\"'\s:=]+[A-Za-z0-9._-]{16,}"
    r"|\bcf(?:ut|at|k)_[A-Za-z0-9]{30,}"
)

# Environment variables whose live values must never appear in a record. The
# pattern above only catches known shapes; this catches the exact configured
# secret whatever its shape. Values shorter than MIN_SECRET_LEN are ignored
# (a one-character value would match every record).
MIN_SECRET_LEN = 8
EXACT_SECRET_ENV = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
)
SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "KEY", "CLIENT_ID", "LLM_URL", "PASSWORD")


def live_secret_values(env=None):
    """{ENV_NAME: needle} for every live WORKFORCE_* secret-bearing variable.

    Non-secret WORKFORCE_* switches (ENABLED, PUBLISH, DATASET, LLM_MODEL)
    are skipped. For a URL the host is also a needle: the LLM hostname is
    configuration and must stay out of records.
    """
    env = os.environ if env is None else env
    found = {}
    for name, value in env.items():
        upper = name.upper()
        if not value or len(value) < MIN_SECRET_LEN:
            continue
        workforce_secret = upper.startswith("WORKFORCE_") and any(
            m in upper for m in SECRET_ENV_MARKERS
        )
        if not (workforce_secret or upper in EXACT_SECRET_ENV):
            continue
        found[name] = value
        if "://" in value:
            try:
                host = urllib.parse.urlsplit(value).hostname
            except ValueError:
                host = None
            if host and len(host) >= MIN_SECRET_LEN:
                found[name + "#host"] = host
    return found


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


_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",  # "this network"
        "10.0.0.0/8",  # RFC 1918
        "100.64.0.0/10",  # CGNAT (RFC 6598)
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local, incl. cloud metadata 169.254.169.254
        "172.16.0.0/12",  # RFC 1918
        "192.0.0.0/24",  # IETF protocol assignments
        "192.0.2.0/24",  # TEST-NET-1
        "192.168.0.0/16",  # RFC 1918
        "198.18.0.0/15",  # benchmarking
        "198.51.100.0/24",  # TEST-NET-2
        "203.0.113.0/24",  # TEST-NET-3
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved + broadcast
        "::/128",  # unspecified
        "::1/128",  # loopback
        "64:ff9b::/96",  # NAT64: embeds an IPv4 target
        "64:ff9b:1::/48",  # local-use NAT64
        "100::/64",  # discard-only
        "2001::/32",  # Teredo
        "2001:db8::/32",  # documentation
        "2002::/16",  # 6to4: embeds an IPv4 target
        "fc00::/7",  # unique local, incl. AWS IMDS fd00:ec2::254
        "fe80::/10",  # link-local
        "ff00::/8",  # multicast
    )
)
# Named explicitly so the intent survives any edit to the table above.
METADATA_IPS = frozenset(
    ipaddress.ip_address(a)
    for a in ("169.254.169.254", "169.254.170.2", "fd00:ec2::254")
)


def ip_is_blocked(addr):
    """True unless ``addr`` is a globally routable, non-reserved address.
    IPv4-mapped IPv6 (::ffff:a.b.c.d) is judged as its embedded IPv4."""
    try:
        ip = ipaddress.ip_address(str(addr).split("%")[0])
    except ValueError:
        return True
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip in METADATA_IPS or not ip.is_global:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS if net.version == ip.version)


def resolve_addresses(host, port=443):
    """All A/AAAA addresses for ``host`` ([] when it does not resolve)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return []
    return sorted({info[4][0] for info in infos})


def is_public_https_host(url, resolver=None):
    """True only for https URLs whose host is a DNS name that resolves, and
    resolves ONLY to globally routable addresses. Refuses IP literals,
    localhost, internal-only suffixes, userinfo, and any name with even one
    private, loopback, link-local, CGNAT or cloud-metadata address. Used for
    every outbound call that carries credentials; fails closed on a name that
    does not resolve. ``resolver(host, port)`` is injectable for tests.

    Residual: the name is resolved here and again by the HTTP client, so a
    host that changes answer between the two (DNS rebinding) is not caught.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port or 443
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
    addrs = (resolver or resolve_addresses)(host, port)
    if not addrs:
        return False
    return not any(ip_is_blocked(a) for a in addrs)


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


def refuse_secrets(blob, env=None):
    """Refuse a record that carries a credential-shaped string or the exact
    value of any live WORKFORCE_* secret in ``env`` (default: os.environ).
    The error names the variable, never the value."""
    text = blob.decode("utf-8", errors="replace")
    if SECRET_PATTERN.search(text):
        raise ValueError("record contains a credential-shaped string; refusing")
    for name, needle in sorted(live_secret_values(env).items()):
        if needle in text:
            raise ValueError(
                f"record contains the live value of {name.split('#')[0]}; refusing"
            )


def record_name(role, blob):
    return f"{role}-{sha256_hex(blob)[:12]}.json"
