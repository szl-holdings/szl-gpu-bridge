"""Offline, network-forbidden tests for the immutable Salad resume readback.

These fixtures represent opaque checkpoint bytes; no torch/pickle/model code runs.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

from tools.salad.resume_integrity import (
    ResumeHold,
    inspect_pinned_checkpoint,
    verified_resume_checkpoint,
)

REVISION = "a" * 40


def _git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def checkpoint_files(*, step=50, target=200):
    return {
        f"checkpoint-{step}/trainer_state.json": json.dumps(
            {"global_step": step, "max_steps": target}, sort_keys=True
        ).encode(),
        f"checkpoint-{step}/adapter_model.safetensors": b"opaque-safetensor-bytes",
        f"checkpoint-{step}/adapter_config.json": b'{"r": 4}',
        f"checkpoint-{step}/optimizer.pt": b"opaque-pickle-never-load-me",
        f"checkpoint-{step}/rng_state.pth": b"opaque-rng-never-load-me",
        f"checkpoint-{step}/scheduler.pt": b"opaque-scheduler-never-load-me",
    }


class FakeHub:
    def __init__(self, files=None):
        self.files = checkpoint_files() if files is None else files
        self.calls = []
        self.private = True
        self.revision = REVISION
        self.error = None
        self.corrupt = None
        self.extra_local = None

    def model_info(self, repo_id, *, files_metadata, timeout):
        self.calls.append(("model_info", files_metadata, timeout))
        if self.error:
            raise self.error
        siblings = [
            types.SimpleNamespace(
                rfilename=name,
                size=len(raw),
                blob_id=_git_blob(raw),
                lfs=None,
            )
            for name, raw in self.files.items()
        ]
        return types.SimpleNamespace(
            sha=self.revision, private=self.private, siblings=siblings
        )

    def snapshot_download(self, **kwargs):
        self.calls.append(("snapshot", kwargs))
        if self.error:
            raise self.error
        prefix = kwargs["allow_patterns"][0].removesuffix("*")
        root = Path(kwargs["local_dir"])
        for name, raw in self.files.items():
            if name.startswith(prefix):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    self.corrupt
                    if self.corrupt and name.endswith("optimizer.pt")
                    else raw
                )
        if self.extra_local:
            (root / prefix / self.extra_local).write_bytes(b"unaccounted")
        return str(root)


class ResumeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.hub = FakeHub()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_gate(self, **kwargs):
        return verified_resume_checkpoint(
            self.hub,
            "fixture/private",
            self.temp.name,
            expected_steps=200,
            **kwargs,
        )

    def test_bytes_verified_at_one_pinned_revision(self):
        path = self.run_gate()
        self.assertTrue(path.endswith("checkpoint-50"))
        self.assertEqual(len([x for x in self.hub.calls if x[0] == "model_info"]), 1)
        snapshots = [x[1] for x in self.hub.calls if x[0] == "snapshot"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["revision"], REVISION)
        self.assertEqual(snapshots[0]["allow_patterns"], ["checkpoint-50/*"])
        self.assertEqual(
            (Path(path) / "optimizer.pt").read_bytes(),
            self.hub.files["checkpoint-50/optimizer.pt"],
        )

    def test_no_checkpoint_is_genesis_only_with_verified_metadata(self):
        self.hub.files = {"final/adapter_model.safetensors": b"final"}
        self.assertIsNone(self.run_gate())
        self.assertEqual(len(self.hub.calls), 1)

    def test_metadata_failures_do_not_degrade_to_genesis(self):
        self.hub.error = ConnectionError("opaque transport diagnostics")
        with self.assertRaisesRegex(ResumeHold, "METADATA_UNOBSERVED"):
            self.run_gate()
        self.assertEqual(len(self.hub.calls), 1)

    def test_auth_denial_and_rate_limit_stop_with_no_retry(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                self.hub.calls.clear()
                error = RuntimeError("private-response-must-not-appear")
                error.response = types.SimpleNamespace(status_code=status)
                self.hub.error = error
                with self.assertRaisesRegex(ResumeHold, "ACCESS_STOP"):
                    self.run_gate()
                self.assertEqual(len(self.hub.calls), 1)

    def test_public_or_unknown_visibility_is_hold(self):
        for value in (False, None):
            with self.subTest(value=value):
                self.hub.private = value
                with self.assertRaises(ResumeHold):
                    self.run_gate()

    def test_incomplete_latest_checkpoint_is_not_skipped(self):
        self.hub.files.update(
            {"checkpoint-60/trainer_state.json": b'{"global_step":60,"max_steps":200}'}
        )
        with self.assertRaisesRegex(ResumeHold, "REQUIRED_STATE_MISSING"):
            self.run_gate()
        self.assertEqual(len(self.hub.calls), 1)

    def test_optimizer_and_rng_are_required_as_opaque_state(self):
        for name in ("checkpoint-50/optimizer.pt", "checkpoint-50/rng_state.pth"):
            with self.subTest(name=name):
                files = checkpoint_files()
                del files[name]
                self.hub = FakeHub(files)
                with self.assertRaises(ResumeHold):
                    self.run_gate()

    def test_tampered_local_bytes_fail_after_snapshot(self):
        self.hub.corrupt = b"modified-after-provider-manifest"
        with self.assertRaisesRegex(ResumeHold, "SIZE_MISMATCH|HASH_MISMATCH"):
            self.run_gate()

    def test_unpinned_local_file_is_hold(self):
        self.hub.extra_local = "unaccounted-private-file.txt"
        with self.assertRaisesRegex(ResumeHold, "UNPINNED_LOCAL_FILE"):
            self.run_gate()

    def test_changed_target_is_hold(self):
        self.hub.files = checkpoint_files(target=201)
        with self.assertRaisesRegex(ResumeHold, "TARGET_CHANGED"):
            self.run_gate()

    def test_trainer_boolean_step_rejected_even_when_bytes_hash_match(self):
        self.hub.files["checkpoint-50/trainer_state.json"] = (
            b'{"global_step":true,"max_steps":200}'
        )
        with self.assertRaisesRegex(ResumeHold, "TRAINER_STATE_UNVERIFIED"):
            self.run_gate()

    def test_partial_legacy_directory_never_deleted_or_reused(self):
        old = Path(self.temp.name) / "checkpoint-50"
        old.mkdir()
        marker = old / "old-unqualified-bytes"
        marker.write_bytes(b"must-not-be-removed")
        path = self.run_gate()
        self.assertTrue(marker.exists())
        self.assertNotEqual(Path(path).parent, old.parent)
        self.assertEqual(marker.read_bytes(), b"must-not-be-removed")

    def test_output_symlink_refused(self):
        target = Path(self.temp.name) / "target"
        target.mkdir()
        link = Path(self.temp.name) / "symlink"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(ResumeHold, "SYMLINK"):
            verified_resume_checkpoint(
                self.hub, "fixture/private", str(link), expected_steps=200
            )

    def test_wrong_repository_object_identity_holds(self):
        plan = inspect_pinned_checkpoint(self.hub, "fixture/private")
        self.assertEqual(plan["step"], 50)
        self.assertEqual(plan["revision"], REVISION)
        self.hub.revision = "moving-ref-not-a-commit"
        with self.assertRaisesRegex(ResumeHold, "METADATA_INVALID"):
            self.run_gate()

    def test_no_trainer_or_backup_quality_claims(self):
        path = self.run_gate()
        self.assertTrue(Path(path).is_dir())
        self.assertTrue(
            all(
                name not in path.lower()
                for name in ("qualified", "evaluated", "production", "backup-verified")
            )
        )


if __name__ == "__main__":
    unittest.main()
