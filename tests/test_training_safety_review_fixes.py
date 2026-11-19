"""Tests for the review fixes on the training-safety controllers (M1-M6, L1-L10, T1).
Offline: a loopback HTTP server is used once to prove redirects are not followed;
everything else is mocked. No credentials, no external network."""

import hashlib
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(os.path.dirname(HERE), "tools", "training_safety")
sys.path.insert(0, TS)
sys.path.insert(0, HERE)

import szl_controller_gate as gate  # noqa: E402
import hub_evidence  # noqa: E402
import salad_io  # noqa: E402
import watch_safe  # noqa: E402
import f54v2_heldout_fire_safe as fire  # noqa: E402

from test_training_safety_controllers import (  # noqa: E402
    REV,
    FakeSalad,
    FakeSaladLaunch,
    R,
    blob,
    complete_evidence,
    ent,
    group,
    insts,
)

CFG_DIR = os.path.join(TS, "configs")


def example(name):
    with open(os.path.join(CFG_DIR, name), encoding="utf-8") as f:
        return json.load(f)


class FakeResp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status, self._b, self._p = status, body, 0
        self.headers = headers or {}

    def read(self, n=-1):
        end = len(self._b) if n is None or n < 0 else self._p + n
        chunk = self._b[self._p : end]
        self._p += len(chunk)
        return chunk

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def redirect(code, location):
    return urllib.error.HTTPError(
        "https://huggingface.co/x", code, "r", {"Location": location}, None
    )


class ScriptedOpener:
    """Returns/raises the scripted outcomes in order; records each request."""

    def __init__(self, *outcomes):
        self.outcomes, self.seen = list(outcomes), []

    def __call__(self, req, timeout=None):
        self.seen.append((req.full_url, dict(req.header_items())))
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


# ------------------------------------------------------------------ M1
class _Hits(http.server.BaseHTTPRequestHandler):
    hits = None

    def do_GET(self):
        type(self).hits.append((self.path, self.headers.get("Salad-Api-Key")))
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", type(self).target)
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


def serve(handler_hits, target=None):
    handler = type("H", (_Hits,), {"hits": handler_hits, "target": target})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestM1Redirects(unittest.TestCase):
    def test_urllib_opener_does_not_follow_redirects_or_forward_headers(self):
        hits_a, hits_b = [], []
        b = serve(hits_b)
        a = serve(hits_a, "http://localhost:%d/second" % b.server_port)
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:%d/start" % a.server_port,
                headers={"Salad-Api-Key": "fake-not-a-key"},
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                gate._OPENER.open(req, timeout=5)
            self.assertEqual(cm.exception.code, 302)
        finally:
            a.shutdown()
            b.shutdown()
        self.assertEqual(len(hits_a), 1)
        self.assertEqual(hits_b, [])  # the second host never saw the credential

    def test_provider_call_returns_3xx_status_and_fails_closed(self):
        op = ScriptedOpener(redirect(302, "https://evil.example/x"))
        r = gate.provider_call(
            "GET",
            "https://api.salad.com/api/public/x",
            {"Salad-Api-Key": "k"},
            opener=op,
            allowed_hosts=gate.SALAD_HOSTS,
        )
        self.assertEqual(r["status"], 302)
        self.assertFalse(gate.classify_write(r)[0])
        self.assertEqual(len(op.seen), 1)

    def test_credentialed_calls_need_https_and_allowlisted_host(self):
        op = ScriptedOpener()
        h = {"Salad-Api-Key": "k"}
        for url, allowed, err in (
            ("http://api.salad.com/x", gate.SALAD_HOSTS, "URL_NOT_HTTPS"),
            ("https://evil.example/x", gate.SALAD_HOSTS, "URL_HOST_NOT_ALLOWED"),
            (
                "https://api.salad.com.evil.example/x",
                gate.SALAD_HOSTS,
                "URL_HOST_NOT_ALLOWED",
            ),
            ("https://u:p@api.salad.com/x", gate.SALAD_HOSTS, "URL_HAS_USERINFO"),
            ("https://api.salad.com:8443/x", gate.SALAD_HOSTS, "URL_NONDEFAULT_PORT"),
            ("https://api.salad.com/x", None, "CREDENTIAL_WITHOUT_HOST_ALLOWLIST"),
        ):
            r = gate.provider_call("GET", url, h, opener=op, allowed_hosts=allowed)
            self.assertFalse(r["transport_ok"], url)
            self.assertEqual(r["error"], err, url)
        self.assertEqual(op.seen, [])  # nothing was ever sent

    def test_authorization_header_also_counts_as_credential(self):
        r = gate.provider_call(
            "GET", "https://huggingface.co/x", {"Authorization": "Bearer x"}
        )
        self.assertEqual(r["error"], "CREDENTIAL_WITHOUT_HOST_ALLOWLIST")

    def test_hub_redirect_to_cdn_drops_authorization(self):
        op = ScriptedOpener(
            redirect(302, "https://cdn-lfs.hf.co/signed?x=1"),
            FakeResp(200, b"data"),
        )
        resp, err = gate.open_following(
            "https://huggingface.co/a/b/resolve/" + REV + "/f",
            {"Authorization": "Bearer fake"},
            opener=op,
        )
        self.assertIsNone(err)
        self.assertEqual(resp.read(), b"data")
        first, second = op.seen
        self.assertIn("Authorization", first[1])
        self.assertNotIn("Authorization", second[1])

    def test_hub_redirect_to_other_host_or_http_is_refused(self):
        for loc, err in (
            ("https://evil.example/x", "REDIRECT_HOST_NOT_ALLOWED"),
            ("https://huggingface.co.evil.example/x", "REDIRECT_HOST_NOT_ALLOWED"),
            ("http://cdn-lfs.hf.co/x", "URL_NOT_HTTPS"),
        ):
            op = ScriptedOpener(redirect(302, loc))
            resp, got = gate.open_following(
                "https://huggingface.co/x", {"Authorization": "Bearer fake"}, opener=op
            )
            self.assertIsNone(resp)
            self.assertEqual(got, err)
            self.assertEqual(len(op.seen), 1)  # never requested the bad target

    def test_open_following_bounded_and_first_url_checked(self):
        op = ScriptedOpener(*[redirect(302, "https://huggingface.co/y")] * 10)
        self.assertEqual(
            gate.open_following("https://huggingface.co/x", {}, opener=op)[1],
            "TOO_MANY_REDIRECTS",
        )
        self.assertEqual(
            gate.open_following("https://evil.example/x", {"Authorization": "x"})[1],
            "URL_HOST_NOT_ALLOWED",
        )

    def test_salad_base_must_be_salad_public_api(self):
        for bad in (
            "http://api.salad.com/api/public/organizations/o/projects/p/containers",
            "https://evil.example/api/public/organizations/o/projects/p/containers",
            "https://api.salad.com/other",
        ):
            with self.assertRaises(gate.UnsafeUrl):
                salad_io.Salad("keyfile", bad)
        self.assertTrue(
            salad_io.Salad("keyfile").base.startswith("https://api.salad.com/")
        )

    def test_salad_calls_carry_the_host_allowlist(self):
        d = tempfile.mkdtemp()
        key = os.path.join(d, "k")
        open(key, "w").write("fake")
        seen = []

        def call(method, url, headers, body=None, **kw):
            seen.append(kw)
            return R(200)

        s = salad_io.Salad(key, call=call)
        s.get_group("g")
        s.get_instances("g")
        s.patch("g", {"replicas": 0})
        self.assertEqual([k.get("allowed_hosts") for k in seen], [gate.SALAD_HOSTS] * 3)

    def test_lane_log_url_host_checked_before_token_is_read(self):
        def boom(*a, **k):
            raise AssertionError("must not be called")

        text, err = watch_safe.fetch_lane_log(
            "https://evil.example/lane.log", "/nonexistent-token-file", call=boom
        )
        self.assertIsNone(text)
        self.assertEqual(err, "LANE_LOG_URL_HOST_NOT_ALLOWED")

    def test_preflight_rejects_bad_lane_log_url_and_salad_base(self):
        cfg = {
            "salad_key_file": "k",
            "log": "l",
            "receipts": "r",
            "groups": [
                {
                    "name": "g",
                    "repo": "o/r",
                    "reason": "lane_log",
                    "lane_log_url": "https://evil.example/lane.log",
                }
            ],
        }
        self.assertEqual(
            watch_safe.preflight(cfg, False)[0], watch_safe.EXIT_AUTH_OR_CONFIG
        )

    def test_stream_download_caps_size_and_removes_partial(self):
        d = tempfile.mkdtemp()
        dest = os.path.join(d, "f.part")
        op = ScriptedOpener(FakeResp(200, b"x" * 5000))
        st = hub_evidence.stream_download(
            "https://huggingface.co/f", {}, dest, max_bytes=100, opener=op
        )
        self.assertEqual(st, "SIZE_EXCEEDED")
        self.assertFalse(os.path.exists(dest))

    def test_stream_download_deadline_and_non_200(self):
        d = tempfile.mkdtemp()
        dest = os.path.join(d, "f.part")
        ticks = iter([0, 0, 10, 20, 30])
        op = ScriptedOpener(FakeResp(200, b"x" * (3 << 20)))
        st = hub_evidence.stream_download(
            "https://huggingface.co/f",
            {},
            dest,
            max_seconds=5,
            opener=op,
            clock=lambda: next(ticks),
        )
        self.assertEqual(st, "DEADLINE_EXCEEDED")
        self.assertFalse(os.path.exists(dest))
        op = ScriptedOpener(FakeResp(206, b"abc"))
        self.assertEqual(
            hub_evidence.stream_download(
                "https://huggingface.co/f", {}, dest, opener=op
            ),
            206,
        )
        self.assertFalse(os.path.exists(dest))


# ------------------------------------------------------------------ M2
class TestM2Placeholders(unittest.TestCase):
    def test_find_placeholders(self):
        self.assertEqual(
            gate.find_placeholders({"a": ["<OWNER: x>"], "b": "ok"}), ["$.a[0]"]
        )
        self.assertEqual(gate.find_placeholders({"a": None, "b": 1}), [])

    def test_every_example_config_is_a_template(self):
        names = sorted(n for n in os.listdir(CFG_DIR) if n.endswith(".json"))
        self.assertTrue(names)
        for n in names:
            cfg = example(n)
            self.assertTrue(gate.find_placeholders(cfg), n)
            text = json.dumps(cfg)
            for needle in ("C:\\\\", ".sk", ".ht", "SZLHOLDINGS", "szl-"):
                self.assertNotIn(needle, text, n)

    def test_watch_apply_refused_with_example_config_and_no_provider_call(self):
        calls = []
        orig = salad_io.Salad.get_group
        salad_io.Salad.get_group = lambda self, n: calls.append(n)
        try:
            for n in ("watch_arm6", "watch_arm7", "rlvr_watch4", "f54v2_watch"):
                p = os.path.join(CFG_DIR, n + ".example.json")
                self.assertEqual(
                    watch_safe.main(["--config", p, "--once", "--apply"]),
                    watch_safe.EXIT_APPLY_REFUSED,
                )
        finally:
            salad_io.Salad.get_group = orig
        self.assertEqual(calls, [])

    def test_watch_observe_on_unfilled_connection_fields_exits_config(self):
        p = os.path.join(CFG_DIR, "watch_arm6.example.json")
        self.assertEqual(
            watch_safe.main(["--config", p, "--once"]), watch_safe.EXIT_AUTH_OR_CONFIG
        )

    def test_cycle_refuses_apply_when_any_placeholder_remains(self):
        tmp = tempfile.mkdtemp()
        out = watch_safe.Out(os.path.join(tmp, "l"), os.path.join(tmp, "r"))
        n = "g1"
        cfg = {
            "hf_token_file": "x",
            "groups": [
                {
                    "name": n,
                    "repo": "o/r",
                    "backup_dir": "<OWNER: dir>",
                    "cache_dir": tmp,
                }
            ],
        }
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        s.get_group = lambda name: (_ for _ in ()).throw(
            AssertionError("provider call")
        )
        res = watch_safe.cycle(
            cfg,
            s,
            out,
            {"machines": {}, "done": set(), "stop_polling": False},
            apply=True,
        )
        self.assertEqual(res[n], "APPLY_REFUSED_OWNER_PLACEHOLDERS")
        self.assertEqual(s.patches, [])

    def test_launcher_apply_refused_with_shipped_config_zero_provider_calls(self):
        out = subprocess.run(
            [
                sys.executable,
                os.path.join(TS, "f54v2_heldout_fire_safe.py"),
                "--config",
                os.path.join(CFG_DIR, "f54v2_fire.example.json"),
                "--apply",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(out.returncode, fire.EXIT_SPEC)
        self.assertIn("OWNER_PLACEHOLDERS_REMAIN", out.stdout)

    def test_launcher_run_apply_refused_before_any_get_or_patch(self):
        cfg = json.loads(json.dumps(TestLauncherGates.GOOD))
        cfg["container"]["command"] = ["bash", "-lc", "<OWNER: pipeline>"]
        s = FakeSaladLaunch([], R(200, group(cfg["group"], 0)), R(200, insts()))
        s.get_group = lambda n: (_ for _ in ()).throw(AssertionError("GET"))
        code, rec = fire.run(cfg, s, apply=True, sleep=lambda x: None)
        self.assertEqual(
            (code, rec["state"]), (fire.EXIT_SPEC, "OWNER_PLACEHOLDERS_REMAIN")
        )
        self.assertEqual(s.patches, [])

    def test_collect_refuses_placeholder_config_before_network(self):
        def boom(*a, **k):
            raise AssertionError("network")

        ev, err = hub_evidence.collect(
            {
                "repo": "o/r",
                "reason": "completion",
                "backup_dir": "<OWNER: d>",
                "cache_dir": "c",
            },
            "tok",
            call=boom,
        )
        self.assertEqual(err, "OWNER_PLACEHOLDER_IN_CONFIG")

    def test_decide_stop_holds_on_placeholder_evidence(self):
        ev = complete_evidence()
        ev["backup"]["location"] = "<OWNER: dir>"
        self.assertEqual(gate.decide_stop(ev)["code"], "OWNER_PLACEHOLDER_IN_EVIDENCE")


# ------------------------------------------------------------------ M3
def receipt_evidence(**over):
    data = b'{"schema": "eval/v1", "verdict": "x"}'
    e = blob(data)
    ev = {
        "reason": "eval_receipt",
        "revision": REV,
        "repo_private": False,
        "allow_public_repo": True,
        "required_paths": ["collab/r.json"],
        "hub_entries": {"collab/r.json": e},
        "local_readback": {"collab/r.json": e},
        "backup": {
            "location": "D:/b",
            "independent": True,
            "files": {"collab/r.json": e},
        },
        "receipt_schema": "eval/v1",
        "receipt_required_fields": ["verdict"],
        "receipt_observed": {
            "collab/r.json": {"schema": "eval/v1", "fields": ["schema", "verdict"]}
        },
    }
    ev.update(over)
    return ev


class TestM3PublicRepo(unittest.TestCase):
    def test_public_receipt_repo_allowed_only_with_opt_in_and_verified_content(self):
        self.assertTrue(gate.decide_stop(receipt_evidence())["stop_allowed"])

    def test_public_without_opt_in_holds(self):
        d = gate.decide_stop(receipt_evidence(allow_public_repo=False))
        self.assertEqual(d["code"], "REPO_PUBLIC_NOT_ALLOWED")

    def test_public_opt_in_never_applies_to_adapters_or_lane_log(self):
        for reason in ("completion", "lane_log"):
            ev = complete_evidence()
            ev.update(reason=reason, repo_private=False, allow_public_repo=True)
            self.assertEqual(gate.decide_stop(ev)["code"], "REPO_PUBLIC_NOT_ALLOWED")

    def test_unknown_visibility_holds(self):
        for v in (None, "yes", 0):
            self.assertEqual(
                gate.decide_stop(receipt_evidence(repo_private=v))["code"],
                "REPO_PRIVACY_UNVERIFIED",
            )

    def test_public_receipt_content_must_match_owner_rule(self):
        cases = (
            ({"receipt_schema": None}, "RECEIPT_RULE_UNDEFINED"),
            ({"receipt_required_fields": []}, "RECEIPT_RULE_UNDEFINED"),
            ({"receipt_observed": {}}, "RECEIPT_CONTENT_UNVERIFIED"),
            (
                {
                    "receipt_observed": {
                        "collab/r.json": {"schema": "other", "fields": ["verdict"]}
                    }
                },
                "RECEIPT_SCHEMA_MISMATCH",
            ),
            (
                {
                    "receipt_observed": {
                        "collab/r.json": {"schema": "eval/v1", "fields": ["schema"]}
                    }
                },
                "RECEIPT_FIELDS_MISSING",
            ),
        )
        for over, code in cases:
            self.assertEqual(gate.decide_stop(receipt_evidence(**over))["code"], code)

    def test_public_receipt_hash_mismatch_still_holds(self):
        ev = receipt_evidence()
        ev["local_readback"]["collab/r.json"] = {"size": 5, "git_blob_sha1": "0" * 40}
        self.assertEqual(gate.decide_stop(ev)["code"], "ARTIFACT_HASH_OR_SIZE_MISMATCH")

    def _collect(self, private, allow, body):
        tmp = tempfile.mkdtemp()
        tok = os.path.join(tmp, "tok")
        open(tok, "w").write("fake")
        info = {
            "sha": REV,
            "private": private,
            "siblings": [
                {
                    "rfilename": "collab/r.json",
                    "size": len(body),
                    "blobId": blob(body)["git_blob_sha1"],
                }
            ],
        }
        cfg = {
            "name": "g",
            "repo": "ORG/frontier-54-runs",
            "repo_type": "dataset",
            "reason": "eval_receipt",
            "allow_public_repo": allow,
            "required_paths": ["collab/r.json"],
            "receipt_schema": "eval/v1",
            "receipt_required_fields": ["verdict"],
            "cache_dir": os.path.join(tmp, "cache"),
            "backup_dir": os.path.join(tmp, "bak"),
            "backup_independent": True,
        }

        def dl(url, headers, dest, **kw):
            open(dest, "wb").write(body)
            return 200

        ev, err = hub_evidence.collect(
            cfg, tok, call=lambda *a, **k: R(200, info), download=dl
        )
        return ev, err

    def test_real_shape_public_dataset_receipt_end_to_end(self):
        body = b'{"schema": "eval/v1", "verdict": "x"}'
        ev, err = self._collect(False, True, body)
        self.assertIsNone(err)
        self.assertTrue(gate.decide_stop(ev)["stop_allowed"], gate.decide_stop(ev))
        ev, err = self._collect(False, False, body)
        self.assertEqual(gate.decide_stop(ev)["code"], "REPO_PUBLIC_NOT_ALLOWED")
        ev, err = self._collect(False, True, b'{"schema": "stale/v0", "verdict": "x"}')
        self.assertEqual(gate.decide_stop(ev)["code"], "RECEIPT_SCHEMA_MISMATCH")
        ev, err = self._collect(False, True, b"not json")
        self.assertEqual(gate.decide_stop(ev)["code"], "RECEIPT_CONTENT_UNVERIFIED")


# ------------------------------------------------------------------ M4
MARKER = r"^INNER_EXIT 0$"


def big_log(total, tail=b"step 99\nINNER_EXIT 0\n"):
    return (b"noise line\n" * (total // 11 + 1))[: total - len(tail)] + tail


def range_opener(body, honour_range=True):
    def op(req, timeout=None):
        rng = req.get_header("Range")
        if honour_range and rng:
            n = int(rng.split("=-")[1])
            start = max(0, len(body) - n)
            return FakeResp(
                206,
                body[start:],
                {"Content-Range": "bytes %d-%d/%d" % (start, len(body) - 1, len(body))},
            )
        return FakeResp(200, body)

    return op


class TestM4LogTail(unittest.TestCase):
    URL = "https://huggingface.co/datasets/o/r/resolve/main/lane.log"

    def test_marker_in_tail_of_2mib_log_found_with_range_support(self):
        data, err, cut = gate.fetch_tail(
            self.URL, {}, opener=range_opener(big_log(2 << 20))
        )
        self.assertIsNone(err)
        self.assertTrue(cut)
        self.assertRegex(data.decode(), r"(?m)" + MARKER)
        self.assertTrue(data.startswith(b"noise line\n"))  # partial first line dropped

    def test_marker_in_tail_found_when_server_ignores_range(self):
        data, err, cut = gate.fetch_tail(
            self.URL, {}, opener=range_opener(big_log(3 << 20), honour_range=False)
        )
        self.assertIsNone(err)
        self.assertTrue(cut)
        self.assertRegex(data.decode(), r"(?m)" + MARKER)

    def test_small_log_not_cut(self):
        data, err, cut = gate.fetch_tail(
            self.URL, {}, opener=range_opener(b"a\nINNER_EXIT 0\n")
        )
        self.assertEqual((data, err, cut), (b"a\nINNER_EXIT 0\n", None, False))

    def test_oversized_range_reply_or_huge_log_is_truncation_not_no_marker(self):
        op = lambda req, timeout=None: FakeResp(206, b"x" * ((1 << 20) + 5))  # noqa: E731
        self.assertEqual(
            gate.fetch_tail(self.URL, {}, opener=op)[1], "LANE_LOG_TRUNCATED"
        )
        old = gate.TAIL_SCAN_LIMIT
        gate.TAIL_SCAN_LIMIT = 1 << 20
        try:
            r = gate.fetch_tail(
                self.URL, {}, opener=range_opener(big_log(3 << 20), honour_range=False)
            )
        finally:
            gate.TAIL_SCAN_LIMIT = old
        self.assertEqual(r[1], "LANE_LOG_TOO_LARGE")

    def test_fetch_lane_log_maps_codes_and_returns_text(self):
        d = tempfile.mkdtemp()
        tok = os.path.join(d, "t")
        open(tok, "w").write("fake")
        f = watch_safe.fetch_lane_log
        self.assertEqual(
            f(self.URL, tok, call=lambda u, h: (None, 404, False)),
            (None, "LANE_LOG_ABSENT"),
        )
        self.assertEqual(
            f(self.URL, tok, call=lambda u, h: (None, 500, False)),
            (None, "LANE_LOG_HTTP_500"),
        )
        self.assertEqual(
            f(self.URL, tok, call=lambda u, h: (None, "LANE_LOG_TRUNCATED", False)),
            (None, "LANE_LOG_LANE_LOG_TRUNCATED"),
        )
        self.assertEqual(
            f(self.URL, tok, call=lambda u, h: (b"ok\n", None, False)), ("ok\n", None)
        )

    def test_cycle_with_big_log_sees_marker_in_tail(self):
        tmp = tempfile.mkdtemp()
        tok = os.path.join(tmp, "t")
        open(tok, "w").write("fake")
        out = watch_safe.Out(os.path.join(tmp, "l"), os.path.join(tmp, "r"))
        n = "szl-rlvr"
        cfg = {
            "hf_token_file": tok,
            "groups": [
                {
                    "name": n,
                    "repo": "o/r",
                    "reason": "lane_log",
                    "lane_log_url": self.URL,
                    "completion_marker": MARKER,
                    "target_steps": 25000,
                    "cache_dir": tmp,
                    "backup_dir": tmp,
                }
            ],
        }
        ev = complete_evidence()
        ev.update(reason="lane_log", completion_marker=MARKER)
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        op = range_opener(big_log(2 << 20))
        res = watch_safe.cycle(
            cfg,
            s,
            out,
            {"machines": {}, "done": set(), "stop_polling": False},
            apply=False,
            collect=lambda *a: (dict(ev), None),
            lane_log=lambda u, t: watch_safe.fetch_lane_log(
                u, t, call=lambda url, h: gate.fetch_tail(url, h, opener=op)
            ),
        )
        self.assertEqual(res[n], "STOP_ALLOWED_OBSERVE_ONLY")
        self.assertEqual(s.patches, [])


# ------------------------------------------------------------------ M5
class RaisingSalad(FakeSalad):
    def __init__(self, bad, *a):
        FakeSalad.__init__(self, *a)
        self.bad = bad

    def get_group(self, name):
        if name == self.bad:
            raise FileNotFoundError("secret-path-should-not-be-logged/.sk")
        return FakeSalad.get_group(self, name)


class TestM5Exceptions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log = os.path.join(self.tmp, "log")
        self.rec = os.path.join(self.tmp, "rec")
        self.out = watch_safe.Out(self.log, self.rec)

    def state(self):
        return {"machines": {}, "done": set(), "stop_polling": False}

    def cfg(self, *names):
        return {
            "hf_token_file": "x",
            "groups": [{"name": n, "repo": None} for n in names],
        }

    def test_one_group_exception_holds_and_others_continue(self):
        s = RaisingSalad(
            "bad", R(200, group("ok", 0, status="stopped")), R(200, insts())
        )
        res = watch_safe.cycle(
            self.cfg("bad", "ok"), s, self.out, self.state(), apply=True
        )
        self.assertEqual(res["bad"], "HOLD_CONTROLLER_EXCEPTION_FileNotFoundError")
        self.assertEqual(res["ok"], "OBSERVED_stopped")
        self.assertEqual(s.patches, [])
        log = open(self.log).read()
        self.assertIn("HOLD_CONTROLLER_EXCEPTION_FileNotFoundError", log)
        self.assertNotIn("secret-path", log)  # exception text is withheld
        rec = [json.loads(x) for x in open(self.rec)]
        self.assertEqual(rec[0]["schema"], "szl.controller-hold/v1")
        self.assertEqual(rec[0]["group"], "bad")

    def test_hold_receipt_written_once_per_change_not_every_cycle(self):
        s = RaisingSalad(
            "bad", R(200, group("ok", 0, status="stopped")), R(200, insts())
        )
        st = self.state()
        for _ in range(3):
            watch_safe.cycle(self.cfg("bad"), s, self.out, st)
        self.assertEqual(len(open(self.rec).read().splitlines()), 1)

    def test_unwritable_log_does_not_kill_cycle(self):
        out = watch_safe.Out(self.tmp, self.tmp)  # a directory: every write raises
        s = FakeSalad(R(400, b"{}"), R(200, insts()))
        res = watch_safe.cycle(self.cfg("g"), s, out, self.state())
        self.assertTrue(res["g"].startswith("HOLD"))

    def test_unreadable_key_file_via_real_salad_holds(self):
        s = salad_io.Salad(os.path.join(self.tmp, "missing-key"))
        res = watch_safe.cycle(self.cfg("g"), s, self.out, self.state(), apply=True)
        self.assertEqual(res["g"], "HOLD_CONTROLLER_EXCEPTION_FileNotFoundError")

    def _write_cfg(self, **over):
        cfg = {
            "salad_key_file": os.path.join(self.tmp, "missing-key"),
            "hf_token_file": os.path.join(self.tmp, "missing-tok"),
            "log": self.log,
            "receipts": self.rec,
            "interval": 0,
            "cycles": 5,
            "groups": [{"name": "g", "repo": None}],
        }
        cfg.update(over)
        p = os.path.join(self.tmp, "cfg.json")
        json.dump(cfg, open(p, "w"))
        return p

    def test_main_survives_failing_cycles_then_exits_nonzero(self):
        p = self._write_cfg(max_consecutive_failed_cycles=3)
        self.assertEqual(watch_safe.main(["--config", p]), watch_safe.EXIT_FAILING)
        log = open(self.log).read()
        self.assertEqual(
            log.count("CONTROLLER_EXCEPTION"), 6
        )  # 3 group lines + 3 cycle lines
        self.assertIn("exiting non-zero (no action taken)", log)

    def test_main_continues_through_all_cycles_when_limit_not_hit(self):
        p = self._write_cfg(max_consecutive_failed_cycles=0, cycles=4)
        self.assertEqual(watch_safe.main(["--config", p]), watch_safe.EXIT_OK)
        self.assertEqual(open(self.log).read().count("cycle {"), 4)

    def test_bad_config_exits_2_without_traceback(self):
        p = os.path.join(self.tmp, "bad.json")
        open(p, "w").write("{not json")
        self.assertEqual(
            watch_safe.main(["--config", p]), watch_safe.EXIT_AUTH_OR_CONFIG
        )
        self.assertEqual(
            watch_safe.main(["--config", os.path.join(self.tmp, "nope.json")]),
            watch_safe.EXIT_AUTH_OR_CONFIG,
        )

    def test_401_and_429_stop_polling(self):
        for code in (401, 429):
            st = self.state()
            s = FakeSalad(R(code), R(200, insts()))
            res = watch_safe.cycle(self.cfg("g"), s, self.out, st)
            self.assertTrue(st["stop_polling"], code)
            self.assertTrue(res["g"].startswith("HOLD_"))
        # and a plain 500 does not
        st = self.state()
        watch_safe.cycle(
            self.cfg("g"), FakeSalad(R(500), R(200, insts())), self.out, st
        )
        self.assertFalse(st["stop_polling"])


# ------------------------------------------------------------------ run_observe (T1)
class TestRunObserve(unittest.TestCase):
    def test_refuses_apply(self):
        p = subprocess.run(
            [
                sys.executable,
                os.path.join(TS, "run_observe.py"),
                "--config",
                "x",
                "--apply",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode, 3)
        self.assertIn("refuses --apply", p.stdout)

    def test_patch_is_disabled_and_cycle_cannot_apply_or_download(self):
        code = (
            "import sys, json, tempfile, os\n"
            "sys.argv=['run_observe.py']\n"
            "sys.path.insert(0, %r)\n"
            "import run_observe, salad_io, watch_safe\n"
            "try:\n"
            "    salad_io.Salad('k').patch('g', {})\n"
            "    print('PATCH_ALLOWED')\n"
            "except RuntimeError:\n"
            "    print('PATCH_BLOCKED')\n"
            "class S:\n"
            "    patches=[]\n"
            "    def get_group(self,n):\n"
            "        return dict(transport_ok=True,status=200,error=None,body=json.dumps({'current_state':{'status':'running'}}).encode())\n"
            "    def get_instances(self,n):\n"
            "        return dict(transport_ok=True,status=200,error=None,body=b'{\"instances\": []}')\n"
            "    def patch(self,n,b):\n"
            "        S.patches.append(b)\n"
            "d=tempfile.mkdtemp()\n"
            "out=watch_safe.Out(os.path.join(d,'l'),os.path.join(d,'r'))\n"
            "cfg={'hf_token_file':'x','groups':[{'name':'g','repo':'o/r','reason':'completion','target_steps':1,'cache_dir':d,'backup_dir':d}]}\n"
            "res=watch_safe.cycle(cfg,S(),out,{'machines':{},'done':set(),'stop_polling':False},apply=True)\n"
            "print(json.dumps([res,S.patches]))\n"
        ) % TS
        p = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        lines = p.stdout.strip().splitlines()
        self.assertEqual(lines[0], "PATCH_BLOCKED", p.stderr)
        res, patches = json.loads(lines[1])
        self.assertEqual(res["g"], "HOLD_OBSERVE_ONLY_NO_DOWNLOAD")
        self.assertEqual(patches, [])


# ------------------------------------------------------------------ launcher gates (L5, L6)
class TestLauncherGates(unittest.TestCase):
    GOOD = {
        "group": "g",
        "replicas": 1,
        "admission_polls": 1,
        "admission_interval": 0,
        "container": {
            "image": "img@sha256:" + "a" * 64,
            "command": ["bash", "-lc", "python3 eval.py"],
            "environment_variables": {"HF_HOME": "/workspace/hf"},
        },
    }

    def run_cfg(self, mutate, apply=True, patches=()):
        cfg = json.loads(json.dumps(self.GOOD))
        mutate(cfg["container"])
        s = FakeSaladLaunch(list(patches), R(200, group("g", 0)), R(200, insts()))
        code, rec = fire.run(cfg, s, apply=apply, sleep=lambda x: None)
        return code, rec, s

    def test_credential_screen_names(self):
        for name in (
            "HF_AUTH",
            "CREDENTIAL",
            "MY_BEARER",
            "DB_PASSWD",
            "HF_TOKEN",
            "API_KEY",
        ):
            code, rec, s = self.run_cfg(
                lambda c: c["environment_variables"].update({name: "v"})
            )
            self.assertEqual(rec["state"], "SPEC_CONTAINS_CREDENTIAL_FIELD", name)
            self.assertEqual(s.patches, [])

    def test_credential_screen_values_and_command(self):
        fake_hf = "hf_" + "A" * 30
        for mut in (
            lambda c: c["environment_variables"].update({"X": fake_hf}),
            lambda c: c.update(
                command=["bash", "-lc", "curl -H 'Authorization: token %s' x" % fake_hf]
            ),
            lambda c: c.update(command=["bash", "-lc", "hf login --token abc"]),
            lambda c: c.update(command=["bash", "-lc", "echo Bearer abc"]),
            lambda c: c.update(command=["bash", "-lc", "curl -H 'Salad-Api-Key: x' y"]),
        ):
            code, rec, s = self.run_cfg(mut)
            self.assertEqual(rec["state"], "SPEC_CONTAINS_CREDENTIAL_FIELD")

    def test_clean_spec_passes_screen(self):
        code, rec, s = self.run_cfg(lambda c: None, apply=False)
        self.assertEqual(rec["state"], "PLAN_ONLY_NO_WRITES")

    def test_apply_requires_digest_pinned_image(self):
        code, rec, s = self.run_cfg(lambda c: c.update(image="pytorch/pytorch:2.14.1"))
        self.assertEqual(rec["state"], "IMAGE_NOT_DIGEST_PINNED")
        self.assertEqual(s.patches, [])

    def test_apply_requires_hashes_for_pip_install(self):
        cmd = ["bash", "-lc", "pip install transformers && python3 eval.py"]
        code, rec, s = self.run_cfg(lambda c: c.update(command=cmd))
        self.assertEqual(rec["state"], "PIP_INSTALL_WITHOUT_REQUIRE_HASHES")
        cmd = [
            "bash",
            "-lc",
            "pip install --require-hashes -r r.txt && python3 eval.py",
        ]
        code, rec, s = self.run_cfg(
            lambda c: c.update(command=cmd), patches=[R(200, {}), R(200, {})]
        )
        self.assertNotEqual(rec["state"], "PIP_INSTALL_WITHOUT_REQUIRE_HASHES")

    def test_plan_mode_not_blocked_by_image_pin(self):
        code, rec, s = self.run_cfg(lambda c: c.update(image="tag-only"), apply=False)
        self.assertEqual(rec["state"], "PLAN_ONLY_NO_WRITES")


# ------------------------------------------------------------------ L-fixes and survivors
class TestGateHardening(unittest.TestCase):
    def test_final_allow_list_inverted(self):  # L8
        for extra in (
            "final/model.npz",
            "final/x.joblib",
            "final/x.h5",
            "final/run.sh",
            "final/sub/a",
        ):
            ev = complete_evidence()
            ev["hub_entries"][extra] = ent("b" * 64, 10)
            self.assertEqual(gate.decide_stop(ev)["code"], "UNSAFE_FINAL_EXPORT", extra)
        ev = complete_evidence()
        ev["hub_entries"]["final/tokenizer.json"] = ent("b" * 64, 10)
        self.assertTrue(gate.decide_stop(ev)["stop_allowed"])

    def test_required_paths_without_final_do_not_raise(self):  # L9
        ev = complete_evidence()
        ck = "checkpoint-25000/adapter_model.safetensors"
        ev["required_paths"] = [ck]
        del ev["hub_entries"]["final/adapter_model.safetensors"]
        self.assertFalse(gate.decide_stop(ev)["stop_allowed"])

    def test_unsafe_required_paths_hold(self):  # L3
        for bad in ("../x", "/etc/x", "a/../../x", "C:/x", "a\\b", "a//b", ""):
            ev = receipt_evidence(
                repo_private=True, required_paths=[bad], hub_entries={bad: blob(b"x")}
            )
            self.assertEqual(gate.decide_stop(ev)["code"], "UNSAFE_REQUIRED_PATH", bad)

    def test_collect_rejects_traversal_and_bad_revision(self):  # L3
        tmp = tempfile.mkdtemp()
        base = {
            "name": "g",
            "repo": "o/r",
            "reason": "eval_receipt",
            "cache_dir": os.path.join(tmp, "c"),
            "backup_dir": os.path.join(tmp, "b"),
        }
        ev, err = hub_evidence.collect(dict(base, required_paths=["../x"]), "tok")
        self.assertEqual(err, "UNSAFE_REQUIRED_PATH")
        tok = os.path.join(tmp, "tok")
        open(tok, "w").write("fake")
        info = {"sha": "../../evil", "private": True, "siblings": []}
        ev, err = hub_evidence.collect(
            dict(base, required_paths=["a.json"]),
            tok,
            call=lambda *a, **k: R(200, info),
        )
        self.assertEqual(err, "HUB_REVISION_INVALID")
        m, err = hub_evidence.fetch_verified("o/r", "z" * 40, "a.json", {}, "tok", tmp)
        self.assertEqual(err, "UNSAFE_PATH_OR_REVISION")
        with self.assertRaises(ValueError):
            hub_evidence.backup_copy(__file__, tmp, REV, "../escape")

    def test_backup_dir_must_be_separate(self):  # L1
        tmp = tempfile.mkdtemp()
        tok = os.path.join(tmp, "creds", "tok")
        for backup, cache in (
            (tmp, tmp),
            (os.path.join(tmp, "cache", "bak"), os.path.join(tmp, "cache")),
            (os.path.join(tmp, "creds"), os.path.join(tmp, "cache")),
        ):
            ev, err = hub_evidence.collect(
                {
                    "name": "g",
                    "repo": "o/r",
                    "reason": "completion",
                    "cache_dir": cache,
                    "backup_dir": backup,
                },
                tok,
                call=lambda *a, **k: self.fail("network"),
            )
            self.assertEqual(err, "BACKUP_DIR_NOT_SEPARATE", (backup, cache))

    def test_download_non_200_and_missing_file_hold(self):  # T1 survivors
        t = TestHubFixture()
        t.setUp()
        for status in (404, 500, None, "SIZE_EXCEEDED"):
            ev, err = hub_evidence.collect(
                t.gcfg(),
                t.tok,
                call=lambda *a, **k: R(200, t.info()),
                download=lambda url, h, dest, **kw: status,
            )
            self.assertEqual(err, "DOWNLOAD_HTTP_%s" % status)

    def test_download_receives_hub_size_cap(self):  # L4
        t = TestHubFixture()
        t.setUp()
        caps = []

        def dl(url, h, dest, **kw):
            caps.append(kw.get("max_bytes"))
            open(dest, "wb").write(t.files[url.split("/resolve/" + REV + "/")[1]])
            return 200

        hub_evidence.collect(
            t.gcfg(), t.tok, call=lambda *a, **k: R(200, t.info()), download=dl
        )
        self.assertTrue(all(isinstance(c, int) and c > 0 for c in caps), caps)

    def test_target_steps_type_check(self):  # T1 survivor
        for bad in (None, "25000", 0, -1, 25000.0, True):
            ev = complete_evidence()
            ev["target_steps"] = bad
            self.assertEqual(
                gate.decide_stop(ev)["code"], "TARGET_STEPS_UNKNOWN", repr(bad)
            )

    def test_trainer_state_must_be_json_object(self):
        t = TestHubFixture()
        t.setUp()
        t.files["checkpoint-25000/trainer_state.json"] = b"not json"
        info = t.info()
        ev, err = hub_evidence.collect(
            t.gcfg(),
            t.tok,
            call=lambda *a, **k: R(200, info),
            download=lambda url, h, dest, **kw: (
                (
                    open(dest, "wb").write(
                        t.files[url.split("/resolve/" + REV + "/")[1]]
                    )
                    and 200
                )
                or 200
            ),
        )
        self.assertEqual(err, "TRAINER_STATE_INVALID")


class TestHubFixture(unittest.TestCase):
    """Same fixture shape as TestHubEvidence, usable from other classes."""

    def runTest(self):
        pass

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tok = os.path.join(self.tmp, "tok")
        open(self.tok, "w").write("not-a-real-token")
        self.files = {
            "final/adapter_model.safetensors": os.urandom(2048),
            "final/adapter_config.json": b'{"r": 16}',
            "checkpoint-25000/adapter_model.safetensors": None,
            "checkpoint-25000/trainer_state.json": b'{"global_step": 25000, "max_steps": 25000}',
        }
        self.files["checkpoint-25000/adapter_model.safetensors"] = self.files[
            "final/adapter_model.safetensors"
        ]

    def info(self):
        sib = []
        for p, d in self.files.items():
            if p.endswith(".safetensors"):
                sib.append(
                    {
                        "rfilename": p,
                        "size": len(d),
                        "lfs": {
                            "size": len(d),
                            "sha256": hashlib.sha256(d).hexdigest(),
                        },
                    }
                )
            else:
                sib.append(
                    {"rfilename": p, "size": len(d), "blobId": blob(d)["git_blob_sha1"]}
                )
        return {"sha": REV, "private": True, "siblings": sib}

    def gcfg(self):
        return {
            "name": "g",
            "repo": "o/x",
            "reason": "completion",
            "target_steps": 25000,
            "cache_dir": os.path.join(self.tmp, "cache"),
            "backup_dir": os.path.join(self.tmp, "bak"),
            "backup_independent": True,
        }


class TestReadmeClaims(unittest.TestCase):
    """M6: keep the README from drifting back into over-claiming."""

    def test_readme_states_the_limits(self):
        text = open(os.path.join(TS, "README.md"), encoding="utf-8").read()
        for needle in (
            "Nothing notifies a person",
            "Only the\n**tests** are credential-free",
            "attested by the owner",
        ):
            self.assertIn(needle, text)
        self.assertNotIn("Credential-free replacements", text)
        self.assertNotIn("alert receipt", text.split("What a HOLD does")[0])

    def test_no_checksum_file_that_nothing_verifies(self):  # L10
        self.assertFalse(os.path.exists(os.path.join(TS, "SHA256SUMS")))


if __name__ == "__main__":
    unittest.main()
