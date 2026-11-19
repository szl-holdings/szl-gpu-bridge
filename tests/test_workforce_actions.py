"""Offline tests for the workforce officers on GitHub Actions.

Covers: missing secret (skip, no upload), no upload when unsigned or when the
attestation does not bind the record, duty fixtures for all seven officers,
that a cycle needs no Hugging Face Space or HF client library, the
dream-architect LLM backend refusing private/plain-HTTP/unauthenticated
routes, and the workflow hardening contract (SHA pins, gate, permissions).
"""

import ast
import base64
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workforce import core, cycle, dream_backend, duties, publish  # noqa: E402

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

    def run_publish(self, env):
        return publish.publish(
            self.record, self.bundle, env=env, committer=self.committer
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


class WorkflowContract(unittest.TestCase):
    wf = (ROOT / ".github/workflows/workforce.yml").read_text()
    cyc = (ROOT / ".github/workflows/workforce-cycle.yml").read_text()

    def test_actions_pinned_by_sha(self):
        uses = re.findall(r"uses:\s*(\S+)", self.cyc)
        self.assertTrue(uses)
        for u in uses:
            self.assertRegex(u, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")

    def test_schedules_gated_by_enable_variable(self):
        jobs = re.findall(r"^  ([\w-]+):\n    if: \$\{\{ (.+?) \}\}$", self.wf, re.M)
        self.assertEqual(len(jobs), 6)
        for name, cond in jobs:
            if name != "manual":
                self.assertIn("vars.WORKFORCE_ENABLED == 'true'", cond, name)

    def test_least_privilege_timeout_concurrency(self):
        for text in (self.wf, self.cyc):
            self.assertIn("\npermissions: {}\n", text)
            self.assertNotIn("write-all", text)
            self.assertNotIn("contents: write", text)
        self.assertIn("timeout-minutes: 10", self.cyc)
        self.assertIn("group: workforce-${{ inputs.role }}", self.cyc)
        self.assertIn("persist-credentials: false", self.cyc)

    def test_publish_after_attest_and_verify_only_on_main(self):
        a = self.cyc.index("actions/attest@")
        v = self.cyc.index('gh attestation verify "$RECORD"')
        p = self.cyc.index("workforce.publish")
        self.assertLess(a, v)
        self.assertLess(v, p)
        self.assertIn(
            "vars.WORKFORCE_PUBLISH == 'true' && github.ref == 'refs/heads/main'",
            self.cyc,
        )

    def test_llm_secrets_only_reach_dream_architect(self):
        for block in re.split(r"\n  (?=[\w-]+:\n)", self.wf):
            if "WORKFORCE_LLM_URL" in block and "manual:" not in block:
                self.assertIn("role: szl-dream-architect", block)


if __name__ == "__main__":
    unittest.main()
