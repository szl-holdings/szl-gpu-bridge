import os, sys, tempfile, unittest, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import receipt_upload as ru


class Op:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.src = path_in_repo, path_or_fileobj


class Err(Exception):
    def __init__(self, st, ra=None):
        class R:  # noqa
            status_code = st
            headers = {"Retry-After": str(ra)} if ra else {}
        self.response = R()


class Info:
    def __init__(self, private): self.private = private


class FakeApi:
    def __init__(self, private=True, exists=True, fail=0, status=429):
        self.private, self.exists, self.fail, self.status = private, exists, fail, status
        self.commits, self.created, self.calls = [], [], 0

    def model_info(self, repo):
        if not self.exists:
            raise Err(404)
        return Info(self.private)

    def create_repo(self, **kw):
        self.created.append(kw); self.exists = True; self.private = kw["private"]

    def create_commit(self, **kw):
        self.calls += 1
        if self.calls <= self.fail:
            raise Err(self.status)
        self.commits.append(kw)
        return type("C", (), {"oid": "abc", "commit_url": "u"})()


def fake_verify(d, **kw):
    return "FAIL", 1, "VERDICT: FAIL\nexit=1\n"


def mkrun(n_hb=50, big=False):
    d = tempfile.mkdtemp()
    os.makedirs(d + "/heartbeats")
    for i in range(n_hb):
        open(d + "/heartbeats/hb-%04d.json" % i, "w").write(json.dumps({"seq": i}))
    open(d + "/claim.json", "w").write(json.dumps({"payload": {"run_id": "r1"}}))
    if big:
        open(d + "/adapter.safetensors", "wb").write(b"x" * (2 * 1024 * 1024))
    return d


def up(d, api, **kw):
    kw.setdefault("verify_fn", fake_verify); kw.setdefault("log", lambda *_: None)
    kw.setdefault("sleep", lambda s: None)
    return ru.upload_run(d, "O/r", api=api, ops_cls=Op, **kw)


class T(unittest.TestCase):
    def test_n_heartbeats_one_commit(self):
        for n in (1, 10, 200):
            api = FakeApi(); r = up(mkrun(n), api)
            self.assertEqual(len(api.commits), 1)
            self.assertEqual(r["status"], "uploaded")
            paths = [o.path_in_repo for o in api.commits[0]["operations"]]
            self.assertEqual(sum(p.startswith("runs/r1/heartbeats/") for p in paths), n)
            self.assertTrue(all(p.startswith("runs/r1/") for p in paths))

    def test_fail_verdict_still_uploaded_and_in_message(self):
        api = FakeApi(); r = up(mkrun(2), api)
        self.assertEqual(len(api.commits), 1)
        self.assertIn("verify=FAIL exit=1", api.commits[0]["commit_message"])
        self.assertEqual(r["verdict"], "FAIL")

    def test_dry_run_no_network(self):
        api = FakeApi(); r = up(mkrun(3), api, dry_run=True)
        self.assertEqual(r["status"], "dry-run"); self.assertEqual(api.calls, 0)
        self.assertEqual(api.created, [])

    def test_large_file_referenced_not_uploaded(self):
        api = FakeApi(); d = mkrun(1, big=True); up(d, api, large_mb=1)
        ops = {o.path_in_repo: o for o in api.commits[0]["operations"]}
        self.assertNotIn("runs/r1/adapter.safetensors", ops)
        ex = json.loads(ops["runs/r1/EXCLUDED_FILES.json"].src.decode())
        self.assertEqual(ex[0]["path"], "adapter.safetensors"); self.assertEqual(len(ex[0]["sha256"]), 64)

    def test_backoff_then_success(self):
        api = FakeApi(fail=2); sl = []
        up(mkrun(1), api, sleep=sl.append, max_tries=5)
        self.assertEqual(len(api.commits), 1); self.assertEqual(len(sl), 2)

    def test_gives_up_cleanly_bounded(self):
        api = FakeApi(fail=99); sl = []
        with self.assertRaises(ru.GaveUp):
            up(mkrun(1), api, sleep=sl.append, max_tries=4)
        self.assertEqual(api.calls, 4); self.assertEqual(len(sl), 3); self.assertEqual(api.commits, [])

    def test_delay_capped_and_jittered(self):
        for a in range(0, 30):
            d = ru.backoff_delay(a, 5, 300, rng=lambda: 1.0); self.assertLessEqual(d, 300)
            d0 = ru.backoff_delay(a, 5, 300, rng=lambda: 0.0); self.assertLessEqual(d0, d)
        self.assertEqual(ru.backoff_delay(20, 5, 300, rng=lambda: 1.0), 300)

    def test_non_retry_error_raises_immediately(self):
        api = FakeApi(fail=5, status=403)
        with self.assertRaises(Err): up(mkrun(1), api)
        self.assertEqual(api.calls, 1)

    def test_public_repo_refused(self):
        api = FakeApi(private=False)
        with self.assertRaises(ru.RepoNotPrivate): up(mkrun(1), api)
        self.assertEqual(api.commits, [])

    def test_missing_repo_needs_create_and_is_private(self):
        api = FakeApi(exists=False)
        with self.assertRaises(FileNotFoundError): up(mkrun(1), api)
        api = FakeApi(exists=False); up(mkrun(1), api, create=True)
        self.assertTrue(api.created[0]["private"] is True); self.assertEqual(len(api.commits), 1)

    def test_real_proofrun_verify_on_synthetic_run(self):
        d = tempfile.mkdtemp() + "/run"
        env = dict(os.environ, PYTHONPATH=ru.DEFAULT_PROOFRUN_PATH)
        subprocess.run([sys.executable, "-m", "proofrun", "demo", d], env=env, capture_output=True, check=True)
        v, code, out = ru.run_verify(d)
        self.assertIn(v, ("PASS", "FAIL")); self.assertEqual(code, 0 if v == "PASS" else 1)
        api = FakeApi(); up(d, api, verify_fn=ru.run_verify)
        self.assertIn("verify=%s exit=%s" % (v, code), api.commits[0]["commit_message"])

    def test_token_never_in_message(self):
        os.environ["HF_TOKEN"] = "hf_SECRETSECRET"
        api = FakeApi(); up(mkrun(1), api)
        self.assertNotIn("SECRETSECRET", json.dumps(api.commits[0]["commit_message"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
