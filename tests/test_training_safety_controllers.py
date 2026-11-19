"""Credential-free tests for the 2026-11-19 controller safety fixes.
All provider and Hub traffic is mocked; no network, no secrets."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "training_safety"))

import szl_controller_gate as gate  # noqa: E402
import hub_evidence  # noqa: E402
import watch_safe  # noqa: E402
import f54v2_heldout_fire_safe as fire  # noqa: E402

REV = "776f372831574717680fcde421e9d5d7e15470d4"
PENDING_400 = json.dumps(
    {
        "type": "pending_update_in_progress",
        "title": "Bad Request",
        "status": 400,
        "instance": "15480229",
    }
).encode()


def R(status, body=b"{}", transport=True):
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    return gate.HttpResult(
        transport_ok=transport,
        status=status if transport else None,
        body=body,
        error=None,
    )


def group(name, replicas, counts=(0, 0, 0, 0), status="deploying", pending=False):
    keys = ("allocating_count", "creating_count", "running_count", "stopping_count")
    return {
        "name": name,
        "replicas": replicas,
        "pending_change": pending,
        "current_state": {
            "status": status,
            "instance_status_counts": dict(zip(keys, counts)),
        },
    }


def insts(*states):
    return {
        "instances": [
            {"state": s, "machine_id": "m%d" % i} for i, s in enumerate(states)
        ]
    }


def ent(sha, size):
    return {"size": size, "sha256": sha}


def blob(data):
    return {
        "size": len(data),
        "git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
    }


# ---------------------------------------------------------------- (a) HTTP
class TestProviderStatus(unittest.TestCase):
    def test_http_400_pending_update_is_failure(self):
        ok, code = gate.classify_write(R(400, PENDING_400))
        self.assertFalse(ok)
        self.assertEqual(code, "HTTP_400")

    def test_curl_rc0_with_400_body_is_failure(self):
        # The incident: curl -s exits 0, body is a 400 problem document.
        ok, code = gate.classify_shell_curl(0, PENDING_400, "400")
        self.assertFalse(ok)
        # Even if the status were mis-captured as 200, the error envelope fails.
        ok, code = gate.classify_shell_curl(0, PENDING_400, "200")
        self.assertFalse(ok)
        self.assertEqual(code, "PROVIDER_ERROR_ENVELOPE")
        # rc=0 without any captured status is never success.
        self.assertFalse(gate.classify_shell_curl(0, b"{}", "")[0])

    def test_transport_and_auth_failures(self):
        self.assertEqual(
            gate.classify_write(R(None, transport=False))[1],
            "TRANSPORT_FAILED_OR_UNKNOWN",
        )
        self.assertEqual(gate.classify_write(R(401))[1], "ACCESS_DENIED_401")
        self.assertEqual(gate.classify_write(R(429))[1], "RATE_LIMITED_429")
        self.assertFalse(gate.classify_write(R(500))[0])
        self.assertFalse(gate.classify_write(R(200, b"not json"))[0])
        self.assertFalse(gate.classify_write(R(200, {"error": "x"}))[0])

    def test_clean_2xx_accepted(self):
        self.assertEqual(
            gate.classify_write(R(200, {"name": "g", "replicas": 1})),
            (True, "HTTP_200_ACCEPTED"),
        )
        self.assertTrue(gate.classify_write(R(202, b""))[0])

    def test_provider_call_returns_http_error_status(self):
        def opener(req, timeout):
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {}, io.BytesIO(PENDING_400)
            )

        r = gate.provider_call(
            "PATCH",
            "https://example.invalid/x",
            {"Salad-Api-Key": "k"},
            {"replicas": 1},
            opener=opener,
        )
        self.assertEqual(r["status"], 400)
        self.assertFalse(gate.classify_write(r)[0])

    def test_provider_call_transport_error(self):
        def opener(req, timeout):
            raise TimeoutError("t")

        r = gate.provider_call("GET", "https://example.invalid/x", {}, opener=opener)
        self.assertFalse(r["transport_ok"])
        self.assertNotIn("k", r.get("error") or "")


# ---------------------------------------------------------------- admission
class TestAdmission(unittest.TestCase):
    N = "szl-bench-flagship54-v2-h"

    def adm(self, g, i, polls=3):
        return gate.admission_readback(
            self.N,
            1,
            lambda: g,
            lambda: i,
            polls=polls,
            interval=0,
            sleep=lambda s: None,
        )

    def test_zero_instances_never_admitted(self):
        a = self.adm(R(200, group(self.N, 1)), R(200, insts()))
        self.assertFalse(a["admitted"])
        self.assertEqual(a["state"], "HOLD_NO_RUNNING_INSTANCE_PROVIDER_REPORTED")

    def test_pending_change_not_admitted(self):
        a = self.adm(
            R(200, group(self.N, 1, (0, 0, 1, 0), "running", pending=True)),
            R(200, insts("running")),
        )
        self.assertFalse(a["admitted"])

    def test_replica_mismatch_not_admitted(self):
        a = self.adm(
            R(200, group(self.N, 0, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        self.assertFalse(a["admitted"])

    def test_running_readback_admitted(self):
        a = self.adm(
            R(200, group(self.N, 1, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        self.assertTrue(a["admitted"])

    def test_401_stops_polling(self):
        calls = []
        a = gate.admission_readback(
            self.N,
            1,
            lambda: calls.append(1) or R(401),
            lambda: R(401),
            polls=10,
            interval=0,
            sleep=lambda s: None,
        )
        self.assertFalse(a["admitted"])
        self.assertEqual(len(calls), 1)

    def test_confirm_zero(self):
        z = gate.confirm_zero(
            self.N,
            lambda: R(200, group(self.N, 0, status="stopped")),
            lambda: R(200, insts()),
        )
        self.assertTrue(z["done"])
        z = gate.confirm_zero(
            self.N,
            lambda: R(200, group(self.N, 0, (0, 0, 0, 1), "stopping")),
            lambda: R(200, insts("stopping")),
        )
        self.assertFalse(z["done"])
        z = gate.confirm_zero(
            self.N, lambda: R(400, PENDING_400), lambda: R(200, insts())
        )
        self.assertFalse(z["done"])


# ---------------------------------------------------------------- stop predicate
# Real Hub metadata from SZLHOLDINGS/salad-flagship54-v2@776f3728 (census 2026-11-19):
ADAPTER_SIZE = 83920528
CK11250 = "8c8acb89e523459b2d62b9a6099cb0e4fed5a41373be863a6333bbb1307a4473"
TS = b'{"global_step": 25000, "max_steps": 25000}'


def complete_evidence(final_sha="f" * 64, latest=25000, target=25000):
    cfg = b'{"r": 16}'
    ts = json.dumps({"global_step": latest, "max_steps": target}).encode()
    entries = {
        "final/adapter_model.safetensors": ent(final_sha, ADAPTER_SIZE),
        "final/adapter_config.json": blob(cfg),
        "checkpoint-%d/adapter_model.safetensors" % latest: ent(
            final_sha, ADAPTER_SIZE
        ),
        "checkpoint-%d/trainer_state.json" % latest: blob(ts),
    }
    measured = {
        p: dict(e, sha256=e.get("sha256"), git_blob_sha1=e.get("git_blob_sha1"))
        for p, e in entries.items()
    }
    return {
        "reason": "completion",
        "revision": REV,
        "repo_private": True,
        "target_steps": target,
        "latest_checkpoint": latest,
        "trainer_state": {"global_step": latest, "max_steps": target},
        "hub_entries": entries,
        "local_readback": dict(measured),
        "backup": {
            "location": "D:/backup",
            "independent": True,
            "files": dict(measured),
        },
    }


class TestStopPredicate(unittest.TestCase):
    def test_complete_evidence_allows_stop_but_claims_nothing(self):
        d = gate.decide_stop(complete_evidence())
        self.assertTrue(d["stop_allowed"], d)
        self.assertFalse(d["completion_claimed"])
        self.assertFalse(d["quality_evaluated"])

    def test_bare_final_filename_is_not_enough(self):
        ev = complete_evidence()
        ev["hub_entries"]["final/adapter_model.safetensors"] = {
            "size": None,
            "sha256": None,
        }
        self.assertEqual(gate.decide_stop(ev)["code"], "HUB_METADATA_INCOMPLETE")

    def test_missing_final_adapter(self):
        ev = complete_evidence()
        del ev["hub_entries"]["final/adapter_model.safetensors"]
        self.assertEqual(gate.decide_stop(ev)["code"], "REQUIRED_ARTIFACT_MISSING")

    def test_partial_final_adapter_size_mismatch(self):
        ev = complete_evidence()
        ev["local_readback"]["final/adapter_model.safetensors"] = {
            "size": ADAPTER_SIZE // 2,
            "sha256": "f" * 64,
        }
        self.assertEqual(gate.decide_stop(ev)["code"], "ARTIFACT_HASH_OR_SIZE_MISMATCH")

    def test_corrupt_final_adapter_hash_mismatch(self):
        ev = complete_evidence()
        ev["local_readback"]["final/adapter_model.safetensors"] = {
            "size": ADAPTER_SIZE,
            "sha256": "0" * 64,
        }
        self.assertEqual(gate.decide_stop(ev)["code"], "ARTIFACT_HASH_OR_SIZE_MISMATCH")

    def test_final_not_read_back(self):
        ev = complete_evidence()
        del ev["local_readback"]["final/adapter_model.safetensors"]
        self.assertEqual(gate.decide_stop(ev)["code"], "ARTIFACT_NOT_READ_BACK")

    def test_flagship54_v2_step_short_11250_of_25000(self):
        # Synthetic step-short case requested by YUH; uses the real checkpoint-11250 adapter hash.
        ev = complete_evidence(final_sha=CK11250, latest=11250, target=25000)
        d = gate.decide_stop(ev)
        self.assertEqual(d["code"], "STEP_TARGET_NOT_REACHED")
        self.assertFalse(d["stop_allowed"])

    def test_final_differs_from_completed_checkpoint(self):
        ev = complete_evidence()
        ev["hub_entries"]["checkpoint-25000/adapter_model.safetensors"] = ent(
            "a" * 64, ADAPTER_SIZE
        )
        self.assertEqual(
            gate.decide_stop(ev)["code"], "FINAL_DIFFERS_FROM_COMPLETED_CHECKPOINT"
        )

    def test_target_changed_or_step_mismatch(self):
        ev = complete_evidence()
        ev["trainer_state"] = {"global_step": 24750, "max_steps": 25000}
        self.assertEqual(gate.decide_stop(ev)["code"], "CHECKPOINT_STEP_MISMATCH")
        ev = complete_evidence()
        ev["trainer_state"] = {"global_step": 25000, "max_steps": 37000}
        self.assertEqual(gate.decide_stop(ev)["code"], "TRAINING_TARGET_CHANGED")

    def test_backup_required_and_independent(self):
        ev = complete_evidence()
        ev["backup"]["independent"] = False
        self.assertEqual(gate.decide_stop(ev)["code"], "INDEPENDENT_BACKUP_UNVERIFIED")
        ev = complete_evidence()
        ev["backup"]["files"]["final/adapter_model.safetensors"] = {
            "size": 1,
            "sha256": "f" * 64,
        }
        self.assertEqual(gate.decide_stop(ev)["code"], "INDEPENDENT_BACKUP_UNVERIFIED")
        ev = complete_evidence()
        del ev["backup"]
        self.assertFalse(gate.decide_stop(ev)["stop_allowed"])

    def test_unpinned_revision_or_unknown_privacy(self):
        ev = complete_evidence()
        ev["revision"] = "main"
        self.assertEqual(gate.decide_stop(ev)["code"], "REVISION_NOT_PINNED")
        ev = complete_evidence()
        ev["repo_private"] = None
        self.assertEqual(gate.decide_stop(ev)["code"], "REPO_PRIVACY_UNVERIFIED")

    def test_unsafe_final_export(self):
        ev = complete_evidence()
        ev["hub_entries"]["final/training_args.bin"] = ent("b" * 64, 10)
        self.assertEqual(gate.decide_stop(ev)["code"], "UNSAFE_FINAL_EXPORT")

    def test_lane_log_without_completion(self):
        ev = complete_evidence()
        ev["reason"] = "lane_log"
        ev["completion_marker"] = r"^INNER_EXIT 0$"
        ev["lane_log_text"] = "step 400\nINNER_EXIT 137\n"
        self.assertEqual(gate.decide_stop(ev)["code"], "LANE_LOG_WITHOUT_COMPLETION")
        ev["lane_log_text"] = ""
        self.assertEqual(gate.decide_stop(ev)["code"], "LANE_LOG_UNREAD")
        ev["lane_log_text"] = "INNER_EXIT 0\n"
        ev["completion_marker"] = None
        self.assertEqual(gate.decide_stop(ev)["code"], "COMPLETION_MARKER_UNDEFINED")

    def test_lane_log_marker_alone_not_enough(self):
        ev = complete_evidence()
        ev["reason"] = "lane_log"
        ev["completion_marker"] = r"^INNER_EXIT 0$"
        ev["lane_log_text"] = "INNER_EXIT 0\n"
        del ev["hub_entries"]["final/adapter_model.safetensors"]
        self.assertFalse(gate.decide_stop(ev)["stop_allowed"])
        ev = complete_evidence()
        ev["reason"] = "lane_log"
        ev["completion_marker"] = r"^INNER_EXIT 0$"
        ev["lane_log_text"] = "INNER_EXIT 0\n"
        self.assertTrue(gate.decide_stop(ev)["stop_allowed"])

    def test_machine_churn_never_stops(self):
        d = gate.decide_stop({"reason": "churn"})
        self.assertFalse(d["stop_allowed"])
        self.assertEqual(d["code"], "CHURN_ALERT_NOT_COMPLETION")
        self.assertFalse(d["spend_park_candidate"])
        ev = complete_evidence()
        ev["reason"] = "churn"
        d = gate.decide_stop(ev)
        self.assertFalse(d["stop_allowed"])
        self.assertTrue(d["spend_park_candidate"])

    def test_unknown_reason_holds(self):
        self.assertFalse(gate.decide_stop({"reason": "EVAL_STALLED"})["stop_allowed"])

    def test_eval_receipt(self):
        data = b'{"battery_admissible": true}'
        e = blob(data)
        ev = {
            "reason": "eval_receipt",
            "revision": REV,
            "repo_private": True,
            "required_paths": ["r.json"],
            "hub_entries": {"r.json": e},
            "local_readback": {"r.json": e},
            "backup": {"location": "D:/b", "independent": True, "files": {"r.json": e}},
        }
        self.assertTrue(gate.decide_stop(ev)["stop_allowed"])
        ev["required_paths"] = []
        self.assertFalse(gate.decide_stop(ev)["stop_allowed"])


# ---------------------------------------------------------------- hub evidence (mock I/O)
class TestHubEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tok = os.path.join(self.tmp, "tok")
        open(self.tok, "w").write("not-a-real-token")
        self.adapter = os.urandom(4096)
        self.cfgb = b'{"r": 16}'
        self.ts = b'{"global_step": 25000, "max_steps": 25000}'
        self.files = {
            "final/adapter_model.safetensors": self.adapter,
            "final/adapter_config.json": self.cfgb,
            "checkpoint-25000/adapter_model.safetensors": self.adapter,
            "checkpoint-25000/trainer_state.json": self.ts,
        }

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
            "repo": "SZLHOLDINGS/x",
            "reason": "completion",
            "target_steps": 25000,
            "cache_dir": os.path.join(self.tmp, "cache"),
            "backup_dir": os.path.join(self.tmp, "bak"),
            "backup_independent": True,
        }

    def download(self, truncate=None):
        def dl(url, headers, dest):
            self.assertIn("/resolve/" + REV + "/", url)  # pinned revision
            path = url.split("/resolve/" + REV + "/")[1]
            data = self.files[path]
            if truncate and path == truncate:
                data = data[:100]
            open(dest, "wb").write(data)
            return 200

        return dl

    def test_end_to_end_verified(self):
        ev, err = hub_evidence.collect(
            self.gcfg(),
            self.tok,
            call=lambda *a, **k: R(200, self.info()),
            download=self.download(),
        )
        self.assertIsNone(err)
        self.assertTrue(gate.decide_stop(ev)["stop_allowed"], gate.decide_stop(ev))

    def test_truncated_download_holds(self):
        ev, err = hub_evidence.collect(
            self.gcfg(),
            self.tok,
            call=lambda *a, **k: R(200, self.info()),
            download=self.download(truncate="final/adapter_model.safetensors"),
        )
        self.assertEqual(gate.decide_stop(ev)["code"], "ARTIFACT_HASH_OR_SIZE_MISMATCH")

    def test_hub_401_holds(self):
        ev, err = hub_evidence.collect(
            self.gcfg(), self.tok, call=lambda *a, **k: R(401), download=self.download()
        )
        self.assertEqual(err, "HUB_INFO_ACCESS_DENIED_401")

    def test_backup_never_overwritten(self):
        cfg = self.gcfg()
        hub_evidence.collect(
            cfg,
            self.tok,
            call=lambda *a, **k: R(200, self.info()),
            download=self.download(),
        )
        p = os.path.join(cfg["backup_dir"], REV, "final", "adapter_model.safetensors")
        before = open(p, "rb").read()
        hub_evidence.collect(
            cfg,
            self.tok,
            call=lambda *a, **k: R(200, self.info()),
            download=self.download(),
        )
        self.assertEqual(open(p, "rb").read(), before)


# ---------------------------------------------------------------- watcher
class FakeSalad:
    def __init__(self, group_resp, inst_resp, after_patch=None):
        self.g, self.i, self.after = group_resp, inst_resp, after_patch
        self.patches = []

    def get_group(self, name):
        return self.after[0] if (self.patches and self.after) else self.g

    def get_instances(self, name):
        return self.after[1] if (self.patches and self.after) else self.i

    def patch(self, name, body):
        self.patches.append(body)
        return R(200, {"name": name, "replicas": body.get("replicas")})


class TestWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = watch_safe.Out(
            os.path.join(self.tmp, "log"), os.path.join(self.tmp, "rec")
        )
        self.cfg = {
            "hf_token_file": "unused",
            "zero_polls": 2,
            "zero_interval": 0,
            "groups": [
                {
                    "name": "szl-train-distill-4b-v1-r1-r2",
                    "repo": "SZLHOLDINGS/salad-distill-4b-v1-r1",
                    "target_steps": 25000,
                    "cache_dir": self.tmp,
                    "backup_dir": self.tmp,
                }
            ],
        }

    def st(self):
        return {"machines": {}, "done": set(), "stop_polling": False}

    def test_machine_churn_no_patch_even_with_apply(self):
        n = self.cfg["groups"][0]["name"]
        churn = {
            "instances": [
                {"state": "downloading", "machine_id": "m%d" % k} for k in range(3)
            ]
        }
        s = FakeSalad(R(200, group(n, 1, (0, 1, 0, 0))), R(200, churn))
        res = watch_safe.cycle(
            self.cfg,
            s,
            self.out,
            self.st(),
            apply=True,
            collect=lambda *a: self.fail("no evidence on churn"),
        )
        self.assertEqual(res[n], "CHURN_ALERT_NOT_COMPLETION")
        self.assertEqual(s.patches, [])

    def test_final_filename_with_partial_evidence_no_patch(self):
        n = self.cfg["groups"][0]["name"]
        ev = complete_evidence()
        del ev["local_readback"]["final/adapter_model.safetensors"]
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        res = watch_safe.cycle(
            self.cfg, s, self.out, self.st(), apply=True, collect=lambda *a: (ev, None)
        )
        self.assertEqual(res[n], "ARTIFACT_NOT_READ_BACK")
        self.assertEqual(s.patches, [])

    def test_observe_only_never_patches(self):
        n = self.cfg["groups"][0]["name"]
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        res = watch_safe.cycle(
            self.cfg,
            s,
            self.out,
            self.st(),
            apply=False,
            collect=lambda *a: (complete_evidence(), None),
        )
        self.assertEqual(res[n], "STOP_ALLOWED_OBSERVE_ONLY")
        self.assertEqual(s.patches, [])

    def test_apply_requires_zero_instance_readback(self):
        n = self.cfg["groups"][0]["name"]
        still = (R(200, group(n, 0, (0, 0, 1, 0), "running")), R(200, insts("running")))
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")),
            R(200, insts("running")),
            after_patch=still,
        )
        res = watch_safe.cycle(
            self.cfg,
            s,
            self.out,
            self.st(),
            apply=True,
            collect=lambda *a: (complete_evidence(), None),
            sleep=lambda x: None,
        )
        self.assertEqual(s.patches, [{"replicas": 0}])
        self.assertTrue(res[n].startswith("STOP_REQUESTED_NOT_CONFIRMED"))
        zero = (R(200, group(n, 0, status="stopped")), R(200, insts()))
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")),
            R(200, insts("running")),
            after_patch=zero,
        )
        res = watch_safe.cycle(
            self.cfg,
            s,
            self.out,
            self.st(),
            apply=True,
            collect=lambda *a: (complete_evidence(), None),
            sleep=lambda x: None,
        )
        self.assertEqual(res[n], "DONE")

    def test_readback_400_holds(self):
        n = self.cfg["groups"][0]["name"]
        s = FakeSalad(R(400, PENDING_400), R(200, insts()))
        res = watch_safe.cycle(
            self.cfg,
            s,
            self.out,
            self.st(),
            apply=True,
            collect=lambda *a: self.fail("no evidence"),
        )
        self.assertEqual(res[n], "HOLD_HTTP_400")
        self.assertEqual(s.patches, [])

    def test_rlvr_lane_log_without_completion_no_patch(self):
        n = "szl-rlvr-attempts-r2"
        cfg = dict(
            self.cfg,
            groups=[
                {
                    "name": n,
                    "repo": "SZLHOLDINGS/rlvr-attempts-r2",
                    "reason": "lane_log",
                    "lane_log_url": "https://huggingface.co/x/lane.log",
                    "completion_marker": r"^INNER_EXIT 0$",
                    "target_steps": 25000,
                    "cache_dir": self.tmp,
                    "backup_dir": self.tmp,
                }
            ],
        )
        s = FakeSalad(
            R(200, group(n, 1, (0, 0, 1, 0), "running")), R(200, insts("running"))
        )
        ev = complete_evidence()
        ev["reason"] = "lane_log"
        ev["completion_marker"] = r"^INNER_EXIT 0$"
        res = watch_safe.cycle(
            cfg,
            s,
            self.out,
            self.st(),
            apply=True,
            collect=lambda *a: (dict(ev), None),
            lane_log=lambda u, t: ("step 10\nTRACEBACK\n", None),
        )
        self.assertEqual(res[n], "LANE_LOG_WITHOUT_COMPLETION")
        res = watch_safe.cycle(
            cfg,
            s,
            self.out,
            self.st(),
            apply=True,
            collect=lambda *a: (dict(ev), None),
            lane_log=lambda u, t: (None, "LANE_LOG_ABSENT"),
        )
        self.assertEqual(res[n], "HOLD_LANE_LOG_ABSENT")
        self.assertEqual(s.patches, [])


# ---------------------------------------------------------------- launcher
class FakeSaladLaunch:
    def __init__(self, patch_resps, group_resp, inst_resp):
        self.patch_resps, self.g, self.i, self.patches = (
            list(patch_resps),
            group_resp,
            inst_resp,
            [],
        )

    def get_group(self, n):
        return self.g

    def get_instances(self, n):
        return self.i

    def patch(self, n, body):
        self.patches.append(body)
        return self.patch_resps.pop(0)


class TestLauncher(unittest.TestCase):
    N = "szl-bench-flagship54-v2-h"
    CFG = {
        "group": N,
        "replicas": 1,
        "admission_polls": 3,
        "admission_interval": 0,
        "container": {
            "image": "img",
            "command": ["bash", "-lc", "python3 eval.py"],
            "environment_variables": {"HF_HOME": "/workspace/hf"},
        },
    }

    def go(self, s, apply=True, cfg=None):
        return fire.run(cfg or self.CFG, s, apply=apply, sleep=lambda x: None)

    def test_incident_replicas_patch_400_pending_update(self):
        s = FakeSaladLaunch(
            [R(200, {"name": self.N}), R(400, PENDING_400)],
            R(200, group(self.N, 0)),
            R(200, insts()),
        )
        code, rec = self.go(s)
        self.assertEqual(code, fire.EXIT_REPLICA_PATCH)
        self.assertEqual(rec["state"], "REPLICAS_PATCH_HTTP_400")
        self.assertNotIn("admission", rec)

    def test_container_patch_400_stops_before_replicas(self):
        s = FakeSaladLaunch(
            [R(400, PENDING_400)], R(200, group(self.N, 0)), R(200, insts())
        )
        code, rec = self.go(s)
        self.assertEqual(code, fire.EXIT_CONTAINER_PATCH)
        self.assertEqual(len(s.patches), 1)

    def test_2xx_with_error_envelope_fails(self):
        s = FakeSaladLaunch(
            [R(200, PENDING_400)], R(200, group(self.N, 0)), R(200, insts())
        )
        self.assertEqual(self.go(s)[0], fire.EXIT_CONTAINER_PATCH)

    def test_accepted_but_zero_instances_not_admitted(self):
        s = FakeSaladLaunch(
            [R(200, {}), R(200, {})], R(200, group(self.N, 1)), R(200, insts())
        )
        code, rec = self.go(s)
        self.assertEqual(code, fire.EXIT_NOT_ADMITTED)
        self.assertFalse(rec["admission"]["admitted"])

    def test_admitted_only_on_running_readback(self):
        s = FakeSaladLaunch(
            [R(200, {}), R(200, {})],
            R(200, group(self.N, 1, (0, 0, 1, 0), "running")),
            R(200, insts("running")),
        )
        self.assertEqual(self.go(s)[0], fire.EXIT_OK)

    def test_precheck_pending_update_blocks_writes(self):
        s = FakeSaladLaunch([], R(200, group(self.N, 0, pending=True)), R(200, insts()))
        self.assertEqual(self.go(s)[0], fire.EXIT_PRECHECK)
        self.assertEqual(s.patches, [])

    def test_plan_mode_no_writes(self):
        s = FakeSaladLaunch([], R(200, group(self.N, 0)), R(200, insts()))
        self.assertEqual(self.go(s, apply=False)[1]["state"], "PLAN_ONLY_NO_WRITES")
        self.assertEqual(s.patches, [])

    def test_spec_with_credential_rejected(self):
        cfg = json.loads(json.dumps(self.CFG))
        cfg["container"]["environment_variables"]["HF_TOKEN"] = "x"
        s = FakeSaladLaunch([], R(200, group(self.N, 0)), R(200, insts()))
        self.assertEqual(self.go(s, cfg=cfg)[0], fire.EXIT_SPEC)
        self.assertEqual(s.patches, [])


if __name__ == "__main__":
    unittest.main()
