"""Offline tests for the workforce officers on GitHub Actions.

Covers: missing secret (skip, no upload), no upload when unsigned, when the
attestation does not bind the record, or when cryptographic verification
refuses it (a structurally valid fake bundle must be REFUSED), duty fixtures
for all seven officers, that a cycle needs no Hugging Face Space or HF client
library, the dream-architect LLM backend refusing private/plain-HTTP/
unauthenticated routes, DNS-resolving host checks, record secret refusal
(shapes and live WORKFORCE_* values), the space-census deadline, record intake
at the privilege boundary, and the workflow hardening contract (SHA pins,
enable gate on EVERY job incl. dispatch, privilege split, pinned identity).
"""

import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workforce import core, cycle, dream_backend, duties, intake, publish  # noqa: E402

ATOM = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><id>http://arxiv.org/abs/2611.00001v1</id><title>Abstention A</title></entry>"
    "<entry><id>http://arxiv.org/abs/2611.00002v1</id><title>Abstention B</title></entry>"
    "</feed>"
).encode()


class FakeNet:
    """Routes URLs to canned responses; records every call; no sockets."""

    def __init__(self, routes=None, statuses=None, dns=None, offline=False):
        self.routes = routes or {}
        self.statuses = statuses or {}
        self.dns = dns or {}
        self.offline = offline
        self.calls = []
        self.posts = []

    def _match(self, url):
        if self.offline:
            return None
        for prefix in sorted(self.routes, key=len, reverse=True):
            if url.startswith(prefix):
                return self.routes[prefix]
        return None

    def get_bytes(self, url, timeout=20, limit=core.MAX_BODY):
        self.calls.append(url)
        v = self._match(url)
        if v is None:
            return None
        return v if isinstance(v, bytes) else json.dumps(v).encode()

    def get_json(self, url, timeout=20):
        b = self.get_bytes(url, timeout)
        return None if b is None else json.loads(b)

    def get_json_with_headers(self, url, headers, timeout=20):
        self.calls.append(url)
        self.posts.append(("GET", url, dict(headers)))
        v = self._match(url)
        if v is None:
            raise OSError("unreachable")
        return v

    def post_json(self, url, payload, headers=None, timeout=60):
        self.calls.append(url)
        self.posts.append(("POST", url, dict(headers or {})))
        v = self._match(url)
        if v is None:
            raise OSError("unreachable")
        return v

    def status(self, url, timeout=15):
        self.calls.append(url)
        return None if self.offline else self.statuses.get(url)

    def resolve(self, host):
        return False if self.offline else self.dns.get(host, False)


def spaces_routes():
    names = ["frontier-showcase", "eternal-auditor", "khipu-playground", "truth-router"]
    stages = ["RUNNING", "PAUSED", "SLEEPING", "BUILD_ERROR"]
    r = {
        "https://huggingface.co/api/spaces?author=SZLHOLDINGS": [
            {"id": f"SZLHOLDINGS/{n}"} for n in names
        ]
    }
    for n, st in zip(names, stages):
        r[f"https://huggingface.co/api/spaces/SZLHOLDINGS/{n}"] = {
            "runtime": {"stage": st}
        }
    return r


ENV = {
    "GITHUB_REPOSITORY": "szl-holdings/szl-gpu-bridge",
    "GITHUB_WORKFLOW_REF": "szl-holdings/szl-gpu-bridge/.github/workflows/workforce.yml@refs/heads/main",
    "GITHUB_SHA": "0" * 40,
    "GITHUB_RUN_ID": "1",
}


PUBLIC_IP = "93.184.216.34"


def fake_resolver(host, port=443):
    """Every test hostname resolves to one public address; nothing real."""
    return [PUBLIC_IP] if host.endswith((".example.com", ".a-11-oy.com")) else []


def setUpModule():
    # No test touches real DNS: the module-level resolver is replaced.
    global _patch
    _patch = mock.patch.object(core, "resolve_addresses", fake_resolver)
    _patch.start()


def tearDownModule():
    _patch.stop()


def ok_verifier(record_path, bundle_path, env=None):
    return None


def fake_bundle(record_bytes, signed=True, cert=True, digest=None):
    stmt = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": "r.json",
                "digest": {
                    "sha256": digest or hashlib.sha256(record_bytes).hexdigest()
                },
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
    }
    return json.dumps(
        {
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": {"certificate": {"rawBytes": "MII="}}
            if cert
            else {},
            "dsseEnvelope": {
                "payload": base64.b64encode(json.dumps(stmt).encode()).decode(),
                "payloadType": "application/vnd.in-toto+json",
                "signatures": [{"sig": "MEUC"}] if signed else [],
            },
        }
    ).encode()


class PublishStopRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.record = self.dir / "eternal-auditor-abc.json"
        self.record.write_bytes(b'{"officer":"eternal-auditor"}')
        self.bundle = self.dir / "eternal-auditor-abc.sigstore.json"
        self.commits = []

    def tearDown(self):
        self.tmp.cleanup()

    def committer(self, repo, token, files, summary):
        self.commits.append((repo, [p for p, _ in files]))

    def run_publish(self, env, verifier=ok_verifier):
        return publish.publish(
            self.record,
            self.bundle,
            env=env,
            committer=self.committer,
            verifier=verifier,
        )

    def test_missing_secret_skips_without_upload(self):
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes()))
        msg, code = self.run_publish({})
        self.assertEqual((msg, code), ("SKIPPED_NO_SECRET", 0))
        self.assertEqual(self.commits, [])

    def test_no_bundle_means_no_upload(self):
        msg, code = self.run_publish({"WORKFORCE_HF_TOKEN": "x"})
        self.assertEqual(code, 2)
        self.assertIn("unsigned", msg)
        self.assertEqual(self.commits, [])

    def test_unsigned_envelope_refused(self):
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes(), signed=False))
        msg, code = self.run_publish({"WORKFORCE_HF_TOKEN": "x"})
        self.assertEqual(code, 2)
        self.assertEqual(self.commits, [])

    def test_missing_certificate_refused(self):
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes(), cert=False))
        _, code = self.run_publish({"WORKFORCE_HF_TOKEN": "x"})
        self.assertEqual(code, 2)
        self.assertEqual(self.commits, [])

    def test_digest_mismatch_refused(self):
        self.bundle.write_bytes(fake_bundle(b"other", digest="ab" * 32))
        msg, code = self.run_publish({"WORKFORCE_HF_TOKEN": "x"})
        self.assertEqual(code, 2)
        self.assertIn("does not match", msg)
        self.assertEqual(self.commits, [])

    def test_hmac_style_signature_is_not_accepted(self):
        self.bundle.write_bytes(json.dumps({"signature": "a" * 64}).encode())
        _, code = self.run_publish({"WORKFORCE_HF_TOKEN": "x"})
        self.assertEqual(code, 2)
        self.assertEqual(self.commits, [])

    def test_secret_in_record_refused(self):
        self.record.write_bytes(b'{"leak":"hf_' + b"A" * 30 + b'"}')
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes()))
        _, code = self.run_publish({"WORKFORCE_HF_TOKEN": "x"})
        self.assertEqual(code, 2)
        self.assertEqual(self.commits, [])

    def test_attested_record_publishes_record_and_bundle(self):
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes()))
        msg, code = self.run_publish(
            {"WORKFORCE_HF_TOKEN": "x", "WORKFORCE_DATASET": "SZLHOLDINGS/t"}
        )
        self.assertEqual(code, 0, msg)
        self.assertEqual(
            self.commits,
            [
                (
                    "SZLHOLDINGS/t",
                    [
                        "officers/eternal-auditor-abc.json",
                        "officers/eternal-auditor-abc.sigstore.json",
                    ],
                )
            ],
        )


class DutyFixtures(unittest.TestCase):
    def test_auditor_uses_utc_and_census(self):
        routes = spaces_routes()
        routes[
            "https://huggingface.co/api/datasets/SZLHOLDINGS/frontier-54-runs/commits/main"
        ] = [{"date": "2026-11-19T10:00:00.000Z", "title": "officer record"}]
        now = core.parse_utc("2026-11-19T13:00:00Z")
        f = duties.eternal_auditor(FakeNet(routes), now=now)
        self.assertEqual(f[0]["value"], 3.0)
        self.assertEqual(f[0]["verdict"], "PASS")
        self.assertEqual(f[1]["running"], 1)
        self.assertEqual(f[1]["sampled"], 4)

    def test_infra_warden_separates_stages_and_expected(self):
        net = FakeNet(
            spaces_routes(),
            statuses={"https://anatomy.a-11-oy.com/": 200},
            dns={"anatomy.a-11-oy.com": True},
        )
        f = duties.infra_warden(net)
        self.assertEqual(
            f[0]["stages"], {"RUNNING": 1, "PAUSED": 1, "SLEEPING": 1, "BUILD_ERROR": 1}
        )
        paused = [x for x in f[0]["not_running"] if x["space"] == "eternal-auditor"]
        self.assertTrue(paused[0]["expected"])
        self.assertEqual(f[0]["unexpected_not_running"], 2)
        hosts = {h["host"]: h for h in f[1]["hosts"]}
        self.assertEqual(hosts["anatomy.a-11-oy.com"]["https_status"], 200)
        self.assertFalse(hosts["router.a-11-oy.com"]["dns"])
        self.assertIsNone(hosts["router.a-11-oy.com"]["https_status"])

    def test_gardener_proposes_never_writes(self):
        net = FakeNet(
            {
                "https://huggingface.co/api/datasets?author=SZLHOLDINGS": [
                    {"id": "SZLHOLDINGS/a", "tags": ["license:mit"]},
                    {"id": "SZLHOLDINGS/b", "tags": []},
                ],
                "https://huggingface.co/datasets/SZLHOLDINGS/b/raw/main/README.md": b"# B\n",
            }
        )
        f = duties.estate_gardener(net)
        props = f[0]["proposals"]
        self.assertEqual([p["dataset"] for p in props], ["SZLHOLDINGS/b"])
        self.assertTrue(props[0]["preserved_prefix"])
        self.assertEqual(net.posts, [])

    def test_chief_engineer_default_branch_and_strict_ok(self):
        routes = {}
        concl = {"a11oy": "success", "szl-forge": "failure", ".github": None}
        for rn, c in concl.items():
            routes[f"https://api.github.com/repos/szl-holdings/{rn}"] = {
                "default_branch": "main"
            }
            routes[
                f"https://api.github.com/repos/szl-holdings/{rn}/actions/runs?branch=main"
            ] = {"workflow_runs": [{"name": "ci", "conclusion": c, "status": "x"}]}
        f = duties.chief_engineer(FakeNet(routes))
        states = {x["repo"]: x["state"] for x in f if x["surface"] == "ci"}
        self.assertEqual(
            states,
            {
                "a11oy": "OK",
                "szl-forge": "FAILED",
                ".github": "NOT_CONCLUSIVE",
                "szl-frontier": "UNAVAILABLE",
            },
        )

    def test_cyber_shield_probes_and_scan(self):
        gw = duties.DEFAULT_GATEWAY
        net = FakeNet(
            {
                "https://huggingface.co/api/datasets/SZLHOLDINGS/frontier-54-runs/tree/main/officers": [
                    {"path": "officers/szl-cyber-shield-1.json"},
                    {"path": "officers/eternal-auditor-1.json"},
                ],
                "https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/raw/main/officers/szl-cyber-shield-1.json": b'{"ok":1}',
            },
            statuses={
                "https://a-11-oy.com": 200,
                gw + "/v1/completions": 200,
                "https://gpu2.a-11-oy.com": 403,
            },
        )
        f = {x["probe"]: x for x in duties.cyber_shield(net)}
        self.assertEqual(f["front_door_tls"]["state"], "PASS")
        self.assertEqual(f["paywall_stands"]["state"], "FAIL")
        self.assertEqual(f["gateway_health"]["state"], "UNAVAILABLE")
        self.assertEqual(f["records_secrets_scan"]["scanned"], 1)
        self.assertEqual(f["records_secrets_scan"]["state"], "PASS")

    def test_research_council_fetches_ids_not_constants(self):
        net = FakeNet({"https://export.arxiv.org/api/query": ATOM})
        f = duties.research_council(net)
        self.assertEqual(f[0]["state"], "MEASURED")
        self.assertEqual(f[0]["items"][0]["id"], "2611.00001v1")
        f2 = duties.research_council(FakeNet(offline=True))
        self.assertEqual(f2[0]["state"], "UNAVAILABLE")

    def test_arxiv_dtd_refused(self):
        with self.assertRaises(ValueError):
            duties.parse_arxiv_feed(b'<!DOCTYPE x [<!ENTITY a "b">]><feed/>')

    def test_dream_architect_template_when_unconfigured(self):
        net = FakeNet(
            {
                "https://api.github.com/repos/szl-holdings/szl-gpu-bridge/issues": [
                    {"number": 135, "title": "lean checker"},
                    {"number": 999, "title": "other"},
                ],
                "https://export.arxiv.org/api/query": ATOM,
            }
        )
        dreamed, draft = duties.dream_architect(net, backend=dream_backend.Backend())
        self.assertEqual(dreamed["questions_read"], 1)
        self.assertEqual(dreamed["hypotheses"][0]["drafted_by"], "template")
        self.assertEqual(draft["state"], dream_backend.NOT_CONFIGURED)
        self.assertEqual(net.posts, [])


class DreamBackend(unittest.TestCase):
    dreamed = {
        "state": "MEASURED",
        "hypotheses": [{"question": {"n": 135, "title": "lean checker"}}],
    }

    def test_plain_http_and_private_ip_refused(self):
        for url in (
            "http://llm.example.com",
            "https://10.0.0.5",
            "https://209.137.198.194",
            "https://localhost",
            "https://gpu.internal",
            "https://user:pw@llm.example.com",
        ):
            b = dream_backend.Backend(url, "id", "secret")
            net = FakeNet()
            out = b.draft(net, self.dreamed)
            self.assertEqual(out["state"], dream_backend.REFUSED, url)
            self.assertEqual(net.calls, [], url)

    def test_missing_access_token_refused(self):
        b = dream_backend.Backend("https://llm.example.com", "id", "")
        net = FakeNet()
        self.assertEqual(b.draft(net, self.dreamed)["state"], dream_backend.REFUSED)
        self.assertEqual(net.calls, [])

    def test_configured_sends_access_headers_and_labels_hypothesis(self):
        url = "https://llm.example.com"
        net = FakeNet(
            {
                url + "/v1/models": {"data": [{"id": "m1"}]},
                url + "/v1/chat/completions": {
                    "choices": [
                        {
                            "message": {
                                "content": "HYPOTHESIS: x helps | PRIOR_ART: y | TEST: z"
                            }
                        }
                    ]
                },
            }
        )
        out = dream_backend.Backend(url, "cid", "csec").draft(net, self.dreamed)
        self.assertEqual(out["state"], "MEASURED")
        self.assertEqual(out["drafts"][0]["hypothesis"], "x helps")
        for _, _, headers in net.posts:
            self.assertEqual(headers["CF-Access-Client-Id"], "cid")
        self.assertNotIn("llm.example.com", json.dumps(out))

    def test_unreachable_vm_is_unavailable_not_measured(self):
        out = dream_backend.Backend("https://llm.example.com", "a", "b").draft(
            FakeNet(), self.dreamed
        )
        self.assertEqual(out["state"], "UNAVAILABLE")


class CycleNeedsNoSpace(unittest.TestCase):
    def test_every_role_completes_offline_and_writes_one_record(self):
        with tempfile.TemporaryDirectory() as d:
            for role in duties.ROLES:
                path, digest = cycle.run(role, d, FakeNet(offline=True), env=ENV)
                blob = path.read_bytes()
                self.assertEqual(hashlib.sha256(blob).hexdigest(), digest)
                rec = json.loads(blob)
                self.assertEqual(rec["officer"], role)
                self.assertFalse(rec["hf_space_required"])
                # Offline, nothing may be reported as passing or healthy.
                text = json.dumps(rec["findings"])
                self.assertNotIn('"PASS"', text, role)
                self.assertNotIn('"OK"', text, role)
                self.assertNotIn('"dns": true', text, role)
            self.assertEqual(len(list(Path(d).glob("*.json"))), len(duties.ROLES))

    def test_no_space_runtime_or_hf_client_dependency(self):
        imported, strings = set(), []
        for p in (ROOT / "workforce").glob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(), filename=str(p))):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    strings.append(node.value)
        for mod in ("huggingface_hub", "fastapi", "hmac", "cryptography", "gradio"):
            self.assertNotIn(mod, imported, mod)
        joined = "\n".join(strings)
        self.assertNotIn(".hf.space", joined)
        self.assertNotIn("/data/", joined)
        self.assertNotIn("szl-auditor-v2-default", joined)

    def test_records_never_carry_tokens(self):
        net = FakeNet(
            {
                "https://export.arxiv.org/api/query": ATOM.replace(
                    b"Abstention A", b"hf_" + b"B" * 30
                )
            }
        )
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                cycle.run("szl-research-council", d, net, env=ENV)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_gateway_must_be_https_hostname(self):
        with self.assertRaises(SystemExit):
            cycle.gateway_from_env({"COMPUTE_GATEWAY": "http://203.0.113.9:8000"})


class PublishCryptoVerification(unittest.TestCase):
    """check_bundle is structure only. A fake bundle with a dummy certificate
    (MII=) and signature (MEUC) passes it, so publish MUST refuse on the
    cryptographic verification result."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.record = self.dir / "eternal-auditor-abc.json"
        self.record.write_bytes(b'{"officer":"eternal-auditor"}')
        self.bundle = self.dir / "eternal-auditor-abc.sigstore.json"
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes()))
        self.commits = []

    def tearDown(self):
        self.tmp.cleanup()

    def committer(self, repo, token, files, summary):
        self.commits.append(repo)

    def publish_with(self, run, env=None):
        env = {"WORKFORCE_HF_TOKEN": "hf-token-value", **(env or {})}
        verifier = lambda r, b, env=None: publish.verify_attestation(  # noqa: E731
            r, b, env=env, run=run
        )
        return publish.publish(
            self.record,
            self.bundle,
            env=env,
            committer=self.committer,
            verifier=verifier,
        )

    @staticmethod
    def proc(rc):
        return subprocess.CompletedProcess([], rc, b"", b"")

    def test_fake_bundle_passes_structure_check_which_is_why_it_is_not_trust(self):
        self.assertIsNone(
            publish.check_bundle(self.record.read_bytes(), self.bundle.read_bytes())
        )

    def test_fake_bundle_is_refused_when_verification_fails(self):
        msg, code = self.publish_with(lambda *a, **k: self.proc(1))
        self.assertEqual(code, 2)
        self.assertIn("cryptographic", msg)
        self.assertEqual(self.commits, [])

    def test_gh_missing_or_hung_is_a_refusal(self):
        def missing(*a, **k):
            raise FileNotFoundError("gh")

        def hung(*a, **k):
            raise subprocess.TimeoutExpired("gh", 1)

        for run in (missing, hung):
            msg, code = self.publish_with(run)
            self.assertEqual(code, 2, msg)
        self.assertEqual(self.commits, [])

    def test_only_a_clean_verification_publishes(self):
        msg, code = self.publish_with(lambda *a, **k: self.proc(0))
        self.assertEqual(code, 0, msg)
        self.assertEqual(len(self.commits), 1)

    def test_verification_is_pinned_and_the_hf_token_never_reaches_gh(self):
        seen = {}

        def run(cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw["env"]
            return self.proc(0)

        sha = "a" * 40
        self.publish_with(
            run,
            env={
                "GITHUB_REPOSITORY": "attacker/fork",  # must not steer the pins
                "GITHUB_REF": "refs/heads/evil",
                "GITHUB_SHA": sha,
                "GH_TOKEN": "gh-read-token",
                "PATH": "/usr/bin",
            },
        )
        cmd = seen["cmd"]
        self.assertEqual(cmd[:3], ["gh", "attestation", "verify"])

        def after(flag):
            return cmd[cmd.index(flag) + 1]

        self.assertEqual(after("--repo"), "szl-holdings/szl-gpu-bridge")
        self.assertEqual(
            after("--signer-workflow"),
            "szl-holdings/szl-gpu-bridge/.github/workflows/workforce-cycle.yml",
        )
        self.assertEqual(after("--source-ref"), "refs/heads/main")
        self.assertEqual(after("--source-digest"), sha)
        self.assertIn("--deny-self-hosted-runners", cmd)
        self.assertEqual(after("--bundle"), str(self.bundle))
        self.assertNotIn("WORKFORCE_HF_TOKEN", seen["env"])
        self.assertNotIn("attacker", json.dumps(cmd))

    @unittest.skipUnless(shutil.which("gh"), "gh CLI not installed")
    def test_fake_bundle_is_refused_by_the_real_gh_verifier(self):
        env = {
            "WORKFORCE_HF_TOKEN": "hf-token-value",
            "PATH": os.environ.get("PATH", ""),
            "HOME": self.dir.as_posix(),  # empty config: no ambient login
        }
        with mock.patch.object(publish, "GH_VERIFY_TIMEOUT_S", 60):
            msg, code = publish.publish(
                self.record, self.bundle, env=env, committer=self.committer
            )
        self.assertEqual(code, 2, msg)
        self.assertEqual(self.commits, [])

    def test_live_hf_token_in_record_refused(self):
        self.record.write_bytes(b'{"x":"hf-token-value"}')
        self.bundle.write_bytes(fake_bundle(self.record.read_bytes()))
        msg, code = self.publish_with(lambda *a, **k: self.proc(0))
        self.assertEqual(code, 2)
        self.assertIn("WORKFORCE_HF_TOKEN", msg)
        self.assertNotIn("hf-token-value", msg)
        self.assertEqual(self.commits, [])


class PublicHostResolution(unittest.TestCase):
    def check(self, url, addrs):
        return core.is_public_https_host(url, resolver=lambda h, p: addrs)

    def test_public_address_accepted(self):
        self.assertTrue(self.check("https://llm.example.com", ["93.184.216.34"]))
        self.assertTrue(
            self.check(
                "https://llm.example.com", ["2606:2800:220:1:248:1893:25c8:1946"]
            )
        )

    def test_unresolvable_fails_closed(self):
        self.assertFalse(self.check("https://llm.example.com", []))

    def test_private_loopback_linklocal_cgnat_metadata_refused(self):
        blocked = [
            "10.1.2.3",
            "172.16.0.9",
            "192.168.1.1",
            "127.0.0.1",
            "0.0.0.0",
            "169.254.1.1",
            "169.254.169.254",  # cloud metadata
            "100.64.0.1",  # CGNAT
            "100.127.255.254",
            "198.18.0.1",
            "224.0.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
            "fd00:ec2::254",  # AWS IMDS over IPv6
            "::ffff:10.0.0.1",  # IPv4-mapped private
            "::ffff:169.254.169.254",
            "64:ff9b::a00:1",  # NAT64 -> 10.0.0.1
            "2002:a00:1::1",  # 6to4 -> 10.0.0.1
        ]
        for a in blocked:
            self.assertTrue(core.ip_is_blocked(a), a)
            self.assertFalse(self.check("https://llm.example.com", [a]), a)

    def test_one_bad_answer_among_good_ones_refuses(self):
        self.assertFalse(
            self.check("https://llm.example.com", ["93.184.216.34", "10.0.0.5"])
        )

    def test_garbage_address_refused(self):
        self.assertTrue(core.ip_is_blocked("not-an-ip"))

    def test_literals_and_shapes_still_refused_without_resolving(self):
        def boom(h, p):
            raise AssertionError("must not resolve")

        for url in (
            "https://169.254.169.254",
            "https://[fd00:ec2::254]",
            "https://10.0.0.5",
            "http://llm.example.com",
            "https://localhost",
            "https://gpu.internal",
            "https://nodots",
            "https://u:p@llm.example.com",
            "https://llm.example.com:notaport",
        ):
            self.assertFalse(core.is_public_https_host(url, resolver=boom), url)

    def test_hostname_resolving_to_metadata_is_refused_end_to_end(self):
        b = dream_backend.Backend("https://llm.example.com", "id", "secret")
        with mock.patch.object(
            core, "resolve_addresses", lambda h, p=443: ["169.254.169.254"]
        ):
            net = FakeNet()
            out = b.draft(net, DreamBackend.dreamed)
        self.assertEqual(out["state"], dream_backend.REFUSED)
        self.assertEqual(net.calls, [])

    def test_gateway_resolving_private_is_refused(self):
        with mock.patch.object(
            core, "resolve_addresses", lambda h, p=443: ["10.0.0.5"]
        ):
            with self.assertRaises(SystemExit):
                cycle.gateway_from_env({"COMPUTE_GATEWAY": "https://gw.example.com"})


class RecordSecretRefusal(unittest.TestCase):
    CF_ID = "0123456789abcdef0123456789abcdef.access"
    CF_SECRET = "f" * 64

    def refused(self, text, env=None):
        with self.assertRaises(ValueError) as cm:
            core.refuse_secrets(text.encode(), env=env if env is not None else {})
        return str(cm.exception)

    def test_cloudflare_access_shapes_refused(self):
        self.refused(f'{{"id":"{self.CF_ID}"}}')
        self.refused(f'{{"CF-Access-Client-Secret":"{self.CF_SECRET}"}}')
        self.refused(f"cf_access_client_secret={self.CF_SECRET}")
        self.refused("token cfut_" + "A" * 40)

    def test_sha256_digest_alone_is_not_a_secret(self):
        core.refuse_secrets(('{"sha256":"%s"}' % self.CF_SECRET).encode(), env={})

    def test_live_workforce_values_refused_by_name_never_by_value(self):
        env = {
            "WORKFORCE_HF_TOKEN": "tokenvalue-1234",
            "WORKFORCE_CF_ACCESS_CLIENT_ID": "clientid-value-1",
            "WORKFORCE_CF_ACCESS_CLIENT_SECRET": "clientsecret-value-1",
            "WORKFORCE_LLM_URL": "https://secret-llm-host.example.com/v1",
        }
        for name, value in env.items():
            msg = self.refused(f'{{"note":"saw {value} here"}}', env)
            self.assertIn(name, msg)
            self.assertNotIn(value, msg)
        msg = self.refused('{"note":"secret-llm-host.example.com"}', env)
        self.assertIn("WORKFORCE_LLM_URL", msg)

    def test_non_secret_switches_and_short_values_ignored(self):
        env = {
            "WORKFORCE_ENABLED": "true",
            "WORKFORCE_PUBLISH": "true",
            "WORKFORCE_DATASET": "SZLHOLDINGS/frontier-54-runs",
            "WORKFORCE_LLM_MODEL": "some-model-name",
            "WORKFORCE_HF_TOKEN": "x",
            "OTHER_TOKEN": "unrelated-value-1",
        }
        core.refuse_secrets(
            b'{"a":"true","b":"SZLHOLDINGS/frontier-54-runs","c":"some-model-name",'
            b'"d":"x","e":"unrelated-value-1"}',
            env=env,
        )

    def test_cycle_refuses_a_record_carrying_a_live_secret(self):
        env = dict(ENV, WORKFORCE_CF_ACCESS_CLIENT_SECRET="live-cf-secret-value")
        net = FakeNet(
            {
                "https://export.arxiv.org/api/query": ATOM.replace(
                    b"Abstention A", b"live-cf-secret-value"
                )
            }
        )
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as cm:
                cycle.run("szl-research-council", d, net, env=env)
            self.assertNotIn("live-cf-secret-value", str(cm.exception))
            self.assertEqual(list(Path(d).iterdir()), [])


class SpaceCensusDeadline(unittest.TestCase):
    class Clock:
        def __init__(self, step):
            self.t, self.step = 0.0, step

        def __call__(self):
            self.t += self.step
            return self.t

    class TimedNet(FakeNet):
        def __init__(self, routes):
            super().__init__(routes)
            self.timeouts = []

        def get_json(self, url, timeout=20):
            self.timeouts.append(timeout)
            return super().get_json(url, timeout)

    def routes(self, n):
        r = {
            "https://huggingface.co/api/spaces?author=SZLHOLDINGS": [
                {"id": f"SZLHOLDINGS/s{i}"} for i in range(n)
            ]
        }
        for i in range(n):
            r[f"https://huggingface.co/api/spaces/SZLHOLDINGS/s{i}"] = {
                "runtime": {"stage": "RUNNING"}
            }
        return r

    def test_defaults_cannot_overrun_the_job_timeout(self):
        self.assertLessEqual(duties.SPACE_CENSUS_CAP, 50)
        self.assertLess(duties.SPACE_CENSUS_BUDGET_S, 600 / 2)
        self.assertLessEqual(duties.SPACE_CALL_TIMEOUT_S, 20)

    def test_slow_hub_stops_at_the_deadline_and_says_so(self):
        net = self.TimedNet(self.routes(30))
        census = duties._space_census(net, cap=30, budget_s=100, clock=self.Clock(30))
        self.assertTrue(census["truncated"])
        self.assertLess(census["sampled"], 30)
        self.assertEqual(census["unsampled_deadline"], 30 - census["sampled"])
        self.assertEqual(census["stages"], {"RUNNING": census["sampled"]})
        self.assertIn("100 s", census["sample_rule"])

    def test_per_call_timeout_is_clipped_to_remaining_budget(self):
        net = self.TimedNet(self.routes(3))
        duties._space_census(net, cap=3, budget_s=100, clock=self.Clock(47))
        self.assertTrue(net.timeouts)
        self.assertTrue(all(t <= duties.SPACE_CALL_TIMEOUT_S for t in net.timeouts))
        self.assertLess(min(net.timeouts), duties.SPACE_CALL_TIMEOUT_S)

    def test_fast_hub_is_complete_and_not_truncated(self):
        census = duties._space_census(self.TimedNet(self.routes(5)), cap=5)
        self.assertFalse(census["truncated"])
        self.assertEqual(census["sampled"], 5)

    def test_cap_still_bounds_the_sample(self):
        census = duties._space_census(FakeNet(self.routes(9)), cap=4)
        self.assertEqual((census["listed_public"], census["sampled"]), (9, 4))

    def test_findings_carry_the_truncation_flag(self):
        f = duties.infra_warden(FakeNet(spaces_routes()))
        self.assertIs(f[0]["truncated"], False)


class RecordIntake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, role="eternal-auditor", mutate=None, name=None):
        _, blob = core.build_record(
            role, [{"check": "x"}], env=ENV, now="2026-11-19T00:00:00Z"
        )
        if mutate:
            blob = mutate(blob)
        path = self.dir / (name or core.record_name(role, blob))
        path.write_bytes(blob)
        return path

    def test_exactly_one_matching_record_is_admitted(self):
        p = self.write()
        self.assertEqual(intake.admit(self.dir, "eternal-auditor"), p)

    def test_every_violation_is_refused(self):
        cases = {
            "empty": lambda: None,
            "two files": lambda: (
                self.write(),
                (self.dir / "extra.txt").write_text("x"),
            ),
            "wrong role": lambda: self.write(role="szl-cyber-shield"),
            "renamed": lambda: self.write(name="eternal-auditor-000000000000.json"),
            "edited after naming": lambda: self.write(
                name="eternal-auditor-" + "0" * 12 + ".json",
                mutate=lambda b: b + b" ",
            ),
            "not json": lambda: self.write(mutate=lambda b: b"nope"),
            "wrong schema": lambda: self.write(
                mutate=lambda b: json.dumps({**json.loads(b), "schema": "x"}).encode()
            ),
            "space required": lambda: self.write(
                mutate=lambda b: json.dumps(
                    {**json.loads(b), "hf_space_required": True}
                ).encode()
            ),
            "secret": lambda: self.write(
                mutate=lambda b: b.replace(b"x", b"hf_" + b"A" * 30, 1)
            ),
        }
        for label, setup in cases.items():
            for f in self.dir.iterdir():
                f.unlink()
            setup()
            with self.assertRaises(ValueError, msg=label):
                intake.admit(self.dir, "eternal-auditor", env={})

    def test_symlink_and_unknown_role_refused(self):
        target = self.write()
        link = self.dir / "elsewhere"
        link.symlink_to(target)
        target.unlink()
        with self.assertRaises(ValueError):
            intake.admit(self.dir, "eternal-auditor")
        with self.assertRaises(ValueError):
            intake.admit(self.dir, "not-a-role")

    def test_cli_exit_codes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                intake.main(["--dir", str(self.dir), "--role", "eternal-auditor"]), 3
            )
        self.write()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                intake.main(["--dir", str(self.dir), "--role", "eternal-auditor"]), 0
            )


def job_blocks(text):
    """{job name: block text} for the top-level ``jobs:`` mapping."""
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def job_if(block):
    m = re.search(r"^    if: \$\{\{ (.+) \}\}$", block, re.M)
    return m.group(1) if m else ""


REPO_GUARD = (
    "github.repository == 'szl-holdings/szl-gpu-bridge' "
    "&& !github.event.repository.fork"
)
ENABLE = "vars.WORKFORCE_ENABLED == 'true'"


def jobs_missing_enable_gate(text):
    """Names of caller jobs whose `if:` does not require WORKFORCE_ENABLED."""
    return [n for n, b in job_blocks(text).items() if ENABLE not in job_if(b)]


class WorkflowContract(unittest.TestCase):
    wf = (ROOT / ".github/workflows/workforce.yml").read_text()
    cyc = (ROOT / ".github/workflows/workforce-cycle.yml").read_text()
    ci = (ROOT / ".github/workflows/ci.yml").read_text()

    def test_actions_pinned_by_sha(self):
        for text in (self.wf, self.cyc):
            for u in re.findall(r"uses:\s*(\S+)", text):
                if u.startswith("./"):
                    continue
                self.assertRegex(u, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")
        self.assertTrue(re.findall(r"uses:\s*(\S+)", self.cyc))

    def test_every_caller_job_including_manual_dispatch_needs_the_enable_variable(self):
        jobs = job_blocks(self.wf)
        self.assertEqual(
            sorted(jobs),
            ["daily", "daily-dream", "every-15m", "every-30m", "hourly", "manual"],
        )
        self.assertEqual(jobs_missing_enable_gate(self.wf), [])
        manual = job_if(jobs["manual"])
        self.assertIn("github.event_name == 'workflow_dispatch'", manual)
        self.assertIn("github.ref == 'refs/heads/main'", manual)

    def test_gate_check_fails_if_dispatch_bypasses_the_enable_variable(self):
        manual = job_if(job_blocks(self.wf)["manual"])
        bypass = self.wf.replace(manual, manual.replace(ENABLE + " && ", "", 1), 1)
        self.assertNotEqual(bypass, self.wf)
        self.assertEqual(jobs_missing_enable_gate(bypass), ["manual"])
        # ... and a manual job with no `if:` at all.
        no_if = self.wf.replace("    if: ${{ " + manual + " }}\n", "", 1)
        self.assertEqual(jobs_missing_enable_gate(no_if), ["manual"])

    def test_every_job_in_both_workflows_has_the_canonical_repo_guard(self):
        for text in (self.wf, self.cyc):
            for name, block in job_blocks(text).items():
                self.assertIn(REPO_GUARD, job_if(block), name)

    def test_least_privilege_timeout_concurrency(self):
        for text in (self.wf, self.cyc):
            self.assertIn("\npermissions: {}\n", text)
            self.assertNotIn("write-all", text)
            self.assertNotIn("contents: write", text)
        for name, block in job_blocks(self.cyc).items():
            self.assertIn("timeout-minutes: 10", block, name)
            self.assertIn("persist-credentials: false", block, name)
        self.assertIn("group: workforce-${{ inputs.role }}", self.cyc)

    def test_duty_job_has_no_signing_rights_no_oidc_and_no_github_token(self):
        duty = job_blocks(self.cyc)["duty"]
        for needle in (
            "id-token",
            "attestations",
            "actions/attest",
            "github.token",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "WORKFORCE_HF_TOKEN",
            "ACTIONS_ID_TOKEN",
        ):
            self.assertNotIn(needle, duty, needle)
        self.assertRegex(duty, r"permissions:\n      contents: read\n    concurrency")
        self.assertIn("workforce.cycle", duty)

    def test_only_the_attest_job_can_sign_and_publish(self):
        jobs = job_blocks(self.cyc)
        att = jobs["attest"]
        self.assertEqual(sorted(jobs), ["attest", "duty"])
        self.assertIn("needs: duty", att)
        self.assertIn("id-token: write", att)
        self.assertIn("attestations: write", att)
        self.assertIn("actions/attest@", att)
        self.assertIn("workforce.intake", att)
        for grant in ("id-token: write", "attestations: write"):
            self.assertEqual(len(re.findall(rf"^\s+{grant}$", self.cyc, re.M)), 1)
        # The attest job never runs duty code or sees the LLM / Access secrets.
        self.assertNotIn("workforce.cycle", att)
        for secret in ("WORKFORCE_LLM_URL", "WORKFORCE_CF_ACCESS"):
            self.assertNotIn(secret, att)

    def test_attest_runs_only_on_main(self):
        self.assertIn(
            "github.ref == 'refs/heads/main'", job_if(job_blocks(self.cyc)["attest"])
        )

    def test_record_passes_between_jobs_by_artifact_and_is_admitted_first(self):
        duty, att = job_blocks(self.cyc)["duty"], job_blocks(self.cyc)["attest"]
        self.assertIn("actions/upload-artifact@", duty)
        self.assertLess(
            att.index("actions/download-artifact@"), att.index("workforce.intake")
        )
        self.assertLess(att.index("workforce.intake"), att.index("actions/attest@"))
        self.assertIn("steps.admit.outputs.record", att)
        self.assertNotIn("steps.run.outputs", att)

    def test_verify_identity_is_hard_pinned_not_derived_from_the_run(self):
        att = job_blocks(self.cyc)["attest"]
        verify = att[att.index("gh attestation verify") :]
        verify = verify[: verify.index("cp ")]
        self.assertIn(f"--repo {publish.PINNED_REPO}", verify)
        self.assertIn(f"--signer-workflow {publish.PINNED_SIGNER_WORKFLOW}", verify)
        self.assertIn(f"--source-ref {publish.PINNED_SOURCE_REF}", verify)
        for derived in ("GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_WORKFLOW"):
            self.assertNotIn(derived, verify)
        # The pin names the file this very workflow is, in this repository.
        self.assertTrue(
            publish.PINNED_SIGNER_WORKFLOW.startswith(publish.PINNED_REPO + "/")
        )
        self.assertTrue(
            (
                ROOT / publish.PINNED_SIGNER_WORKFLOW[len(publish.PINNED_REPO) + 1 :]
            ).is_file()
        )

    def test_publish_is_last_gated_on_enabled_and_publish_and_starts_with_success(self):
        att = job_blocks(self.cyc)["attest"]
        a = att.index("actions/attest@")
        v = att.index("gh attestation verify")
        p = att.index("workforce.publish")
        self.assertLess(a, v)
        self.assertLess(v, p)
        cond = re.search(
            r"name: Publish.*?\n        if: \$\{\{ (.+?) \}\}\n", att, re.S
        ).group(1)
        self.assertTrue(cond.startswith("success() && "), cond)
        for need in (
            ENABLE,
            "vars.WORKFORCE_PUBLISH == 'true'",
            "github.ref == 'refs/heads/main'",
            REPO_GUARD,
        ):
            self.assertIn(need, cond)

    def test_hf_token_only_reaches_the_publish_step(self):
        att = job_blocks(self.cyc)["attest"]
        self.assertEqual(att.count("secrets.WORKFORCE_HF_TOKEN"), 1)
        self.assertGreater(
            att.index("secrets.WORKFORCE_HF_TOKEN"), att.index("name: Publish")
        )

    def test_llm_secrets_only_reach_dream_architect(self):
        for block in re.split(r"\n  (?=[\w-]+:\n)", self.wf):
            if "WORKFORCE_LLM_URL" in block and "manual:" not in block:
                self.assertIn("role: szl-dream-architect", block)

    def test_ci_compiles_formats_and_lints_workforce(self):
        for needle in (
            "py_compile laptop/*.py tests/*.py workforce/*.py",
            "ruff format --check laptop tests workforce",
            "ruff check laptop tests workforce",
        ):
            self.assertIn(needle, self.ci)


if __name__ == "__main__":
    unittest.main()
