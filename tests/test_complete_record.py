"""Offline terminal-record tests; no credentials, network or tensor loading."""

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("complete_record.py")
if not MODULE_PATH.exists():
    MODULE_PATH = (
        Path(__file__).resolve().parents[1] / "tools/a6000-migration/complete_record.py"
    )
MODULE_SPEC = importlib.util.spec_from_file_location("complete_record", MODULE_PATH)
publisher = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(publisher)

LAUNCH = {
    "run_id": "lean-v2-hatun-test",
    "source_commit": "a" * 40,
    "source_files": {"salad_worker.py": "d" * 64},
    "host": "hatun-v7",
    "base": {
        "repo": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "revision": "b" * 40,
    },
    "dataset": {
        "repo": "SZLHOLDINGS/lean-math-sft-v2",
        "revision": "c" * 40,
    },
    "adapter_repo": publisher.TARGET,
    "max_steps": 200,
}
SUPERVISOR = {
    "schema": "szl.shared-gpu-supervisor/v1",
    "status": "exited",
    "child_pid": 123,
    "child_returncode": 0,
    "started_at": "2026-11-16T00:00:00+00:00",
    "updated_at": "2026-11-16T00:00:03+00:00",
}


def seal(receipt):
    receipt = dict(receipt)
    receipt["receipt_sha256"] = publisher.sha(publisher.canonical(receipt))
    return receipt


def worker_log(actual=200, event="complete", **extra):
    boot = seal(
        {
            "kind": "szl.salad-worker/v1",
            "event": "boot",
            "ts": "2026-11-16T00:00:01+00:00",
            "base_revision": "b" * 40,
            "dataset_revision": "c" * 40,
        }
    )
    terminal = seal(
        {
            "kind": "szl.salad-worker/v1",
            "event": event,
            "ts": "2026-11-16T00:00:02+00:00",
            "actual_steps": actual,
            "target_steps": 200,
            "adapter": publisher.TARGET,
            "completion_qualified": event == "complete",
            **extra,
        }
    )
    return (json.dumps(boot) + "\n" + json.dumps(terminal) + "\n").encode()


class MissingRemoteFile(Exception):
    response = types.SimpleNamespace(status_code=404)


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.sha = "f" * 40
        self.body = None
        self.uploads = 0
        self.lost_response = False
        self.fail = False
        self.steps = 200
        self.downloads = []

    def repo_info(self, repo, **kwargs):
        return types.SimpleNamespace(
            private=repo == publisher.TARGET,
            sha="e" * 40 if repo == publisher.TARGET else self.sha,
        )

    def list_repo_files(self, *args, **kwargs):
        assert kwargs["revision"] == "e" * 40
        return [
            "checkpoint-200/trainer_state.json",
            "final/adapter_config.json",
            "final/adapter_model.safetensors",
        ]

    def upload_file(self, **kwargs):
        self.uploads += 1
        if self.fail:
            raise TimeoutError("sentinel-secret")
        self.body = kwargs["path_or_fileobj"]
        self.sha = "1" * 40
        if self.lost_response:
            self.lost_response = False
            raise TimeoutError("lost response")
        return types.SimpleNamespace(oid=self.sha)

    def download(self, repo, name, **kwargs):
        assert len(kwargs["revision"]) == 40
        self.downloads.append((repo, name, kwargs["revision"]))
        if repo == publisher.TARGET:
            assert kwargs["revision"] == "e" * 40
            assert name.endswith(".json"), "tensor/pickle download forbidden"
            metadata = (
                {"global_step": self.steps, "max_steps": 200}
                if "trainer_state" in name
                else {"base_model_name_or_path": LAUNCH["base"]["repo"]}
            )
            body = json.dumps(metadata).encode()
        else:
            if self.body is None:
                raise MissingRemoteFile()
            body = self.body
        path = self.root / f"download-{len(self.downloads)}.json"
        path.write_bytes(body)
        return str(path)


class CompleteRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hub = FakeHub(self.root)

    def test_self_hash_mutation_rejected(self):
        records, invalid = publisher.worker_records(worker_log())
        self.assertEqual(invalid, 0)
        self.assertTrue(publisher.verified(records[-1]))
        records[-1]["actual_steps"] = 8
        self.assertFalse(publisher.verified(records[-1]))

    def test_exact_observed_steps_qualify_terminal_metadata(self):
        _, reasons = publisher.terminal_evidence(LAUNCH, worker_log(), SUPERVISOR)
        self.assertEqual(reasons, [])

    def test_requested_steps_cannot_replace_actual_or_resume_event(self):
        for payload in [
            worker_log(8),
            worker_log(True),
            worker_log(200, "already-complete", resume_global_step=200),
        ]:
            with self.subTest(payload=payload):
                self.assertTrue(
                    publisher.terminal_evidence(LAUNCH, payload, SUPERVISOR)[1]
                )

    def test_running_or_stale_supervisor_cannot_qualify(self):
        supervisor = {**SUPERVISOR, "status": "running"}
        with self.assertRaises(ValueError):
            publisher.terminal_evidence(LAUNCH, worker_log(), supervisor)
        supervisor = {
            **SUPERVISOR,
            "started_at": "2026-11-16T00:00:02+00:00",
        }
        self.assertIn(
            "receipt_time_outside_supervisor_run",
            publisher.terminal_evidence(LAUNCH, worker_log(), supervisor)[1],
        )

    def test_checkpoint_metadata_requires_exact_step_and_no_weights(self):
        metadata = publisher.artifact_metadata(self.hub, self.hub.download, LAUNCH)
        self.assertEqual(metadata["global_step"], 200)
        self.hub.steps = 8
        with self.assertRaises(ValueError):
            publisher.artifact_metadata(self.hub, self.hub.download, LAUNCH)

    def test_exact_commit_readback_and_idempotent_rerun(self):
        output = self.root / "out"
        output.mkdir()
        body = b'{"fixture":true}\n'
        first = publisher.publish(self.hub, self.hub.download, body, output)
        second = publisher.publish(self.hub, self.hub.download, body, output)
        self.assertTrue(first["verified"])
        self.assertEqual(first, second)
        self.assertEqual(self.hub.uploads, 1)

    def test_lost_upload_response_recovers_without_second_upload(self):
        self.hub.lost_response = True
        output = self.root / "out"
        output.mkdir()
        with patch.object(publisher.time, "sleep") as sleep:
            result = publisher.publish(self.hub, self.hub.download, b"payload", output)
        self.assertTrue(result["verified"])
        self.assertEqual(self.hub.uploads, 1)
        sleep.assert_called_once_with(15)

    def invoke_main(self):
        spec = {**LAUNCH, "HF_TOKEN": "sentinel-secret"}
        for name, body in [
            ("spec", json.dumps(spec).encode()),
            ("log", worker_log()),
            ("supervisor", json.dumps(SUPERVISOR).encode()),
        ]:
            (self.root / name).write_bytes(body)
        fake = types.ModuleType("huggingface_hub")
        fake.HfApi = lambda **kwargs: self.hub
        fake.hf_hub_download = self.hub.download
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict(sys.modules, huggingface_hub=fake),
            patch.dict(os.environ, HF_TOKEN="sentinel-secret"),
            patch.object(publisher.time, "sleep"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = publisher.main(
                [
                    "--launch-spec",
                    str(self.root / "spec"),
                    "--worker-log",
                    str(self.root / "log"),
                    "--supervisor-state",
                    str(self.root / "supervisor"),
                    "--outputdir",
                    str(self.root / "result"),
                ]
            )
        self.assertNotIn("sentinel-secret", stdout.getvalue() + stderr.getvalue())
        return result

    def test_success_record_scopes_claims_and_excludes_secrets(self):
        self.assertEqual(self.invoke_main(), 0)
        raw = (self.root / "result/lane_record.json").read_bytes()
        record = json.loads(raw)
        self.assertTrue(publisher.verified(record))
        self.assertTrue(record["completion_qualified"])
        self.assertFalse(record["model_quality_qualified"])
        self.assertFalse(record["provenance_qualified"])
        self.assertFalse(record["promotion_authorized"])
        self.assertNotIn(b"sentinel-secret", raw)

    def test_upload_failure_is_bounded_and_keeps_local_record(self):
        self.hub.fail = True
        self.assertEqual(self.invoke_main(), 1)
        self.assertEqual(self.hub.uploads, 3)
        self.assertTrue((self.root / "result/lane_record.json").is_file())
        self.assertFalse((self.root / "result/publication.json").exists())


if __name__ == "__main__":
    unittest.main()
