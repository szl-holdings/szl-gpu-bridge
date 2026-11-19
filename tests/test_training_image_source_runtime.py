"""Offline source-to-image parity and tamper tests; no GPU or credentials."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training-image" / "source_runtime_contract.py"
spec = importlib.util.spec_from_file_location("source_runtime_contract", SOURCE)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
REV = "a" * 40


class SourceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for rel in contract.SOURCE_FILES:
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic:" + rel).encode())
        self.manifest = self.root / "source-bindings.json"

    def create(self):
        return contract.create_manifest(self.root, REV, self.manifest)

    def test_exact_manifest_and_non_authority(self):
        self.create()
        result = contract.verify_manifest(self.root, REV, self.manifest)
        self.assertEqual(result["state"], "SOURCE_BYTES_MATCH_UNATTESTED")
        self.assertFalse(result["release_authorized"])
        self.assertFalse(result["training_qualified"])

    def test_cannot_reuse_existing_manifest_writer(self):
        self.create()
        with self.assertRaises(contract.SourceHold):
            self.create()

    def test_changed_source_bytes_hold(self):
        self.create()
        (self.root / contract.SOURCE_FILES[0]).write_bytes(b"malicious data")
        with self.assertRaises(contract.SourceHold) as raised:
            contract.verify_manifest(self.root, REV, self.manifest)
        self.assertEqual(str(raised.exception), "SOURCE_BYTES_DRIFT")

    def test_missing_source_file_hold(self):
        self.create()
        (self.root / contract.SOURCE_FILES[1]).unlink()
        with self.assertRaises(contract.SourceHold):
            contract.verify_manifest(self.root, REV, self.manifest)

    def test_revision_substitution_hold(self):
        self.create()
        with self.assertRaises(contract.SourceHold):
            contract.verify_manifest(self.root, "b" * 40, self.manifest)

    def test_duplicate_member_or_missing_member_hold(self):
        self.create()
        data = json.loads(self.manifest.read_text())
        del data["files"][contract.SOURCE_FILES[0]]
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(contract.SourceHold):
            contract.verify_manifest(self.root, REV, self.manifest)

    def test_unsigned_flag_must_stay_false(self):
        self.create()
        data = json.loads(self.manifest.read_text())
        data["runtime_qualified"] = True
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(contract.SourceHold):
            contract.verify_manifest(self.root, REV, self.manifest)

    def test_invalid_commit_or_size_hold(self):
        with self.assertRaises(contract.SourceHold):
            contract.create_manifest(self.root, "moving-main", self.manifest)
        (self.root / contract.SOURCE_FILES[0]).write_bytes(b"")
        with self.assertRaises(contract.SourceHold):
            contract.create_manifest(self.root, REV, self.manifest)

    def test_symlink_path_hold(self):
        candidate = self.root / contract.SOURCE_FILES[0]
        candidate.unlink()
        try:
            candidate.symlink_to(self.root / contract.SOURCE_FILES[1])
        except (OSError, NotImplementedError):
            self.skipTest("test host cannot create symlinks")
        with self.assertRaises(contract.SourceHold):
            self.create()

    def test_manifests_reject_duplicate_json_keys(self):
        self.create()
        self.manifest.write_text(
            '{"schema":"x","schema":"y","files":{},"source_revision":"z"}'
        )
        with self.assertRaises(contract.SourceHold):
            contract.verify_manifest(self.root, REV, self.manifest)

    def test_source_manifest_has_exact_members(self):
        self.assertEqual(
            set(contract.SOURCE_FILES),
            {
                "salad_worker.py",
                "tools/salad/resume_integrity.py",
                "tools/salad/training_snapshot.py",
                "step_heartbeat.py",
                "readiness_probe.py",
                "receipts/receipt_upload.py",
            },
        )

    def test_docker_uses_canonical_worker_and_helpers(self):
        docker = (ROOT / "training-image" / "Dockerfile").read_text()
        self.assertIn("COPY tools/salad/salad_worker.py /work/salad_worker.py", docker)
        self.assertIn("tools/salad/resume_integrity.py", docker)
        self.assertIn("tools/salad/training_snapshot.py", docker)
        self.assertIn("COPY training-image/source_runtime_contract.py", docker)
        self.assertIn("source_runtime_contract.py verify", docker)
        self.assertIn("ENV PYTHONPATH=/work", docker)
        self.assertNotIn("COPY training-image/salad_worker.py", docker)
        self.assertNotIn(
            "COPY readiness_probe.py step_heartbeat.py salad_worker.py", docker
        )

    def test_retired_duplicate_worker_is_not_in_working_tree(self):
        self.assertFalse((ROOT / "training-image" / "salad_worker.py").exists())

    def test_build_context_and_provenance_source(self):
        workflow = (
            ROOT / ".github" / "workflows" / "training-image-ghcr.yml"
        ).read_text()
        self.assertIn("context: .", workflow)
        self.assertIn("file: training-image/Dockerfile", workflow)
        self.assertIn("GIT_COMMIT=${{ github.sha }}", workflow)
        for path in (
            "tools/salad/salad_worker.py",
            "tools/salad/resume_integrity.py",
            "tools/salad/training_snapshot.py",
        ):
            self.assertIn(path, workflow)

    def test_pushed_digest_smoke_checks_source_and_resume_modules(self):
        workflow = (
            ROOT / ".github" / "workflows" / "training-image-ghcr.yml"
        ).read_text()
        self.assertIn("source_runtime_contract.py verify", workflow)
        self.assertIn("verified_resume_checkpoint", workflow)
        snapshot_import = "from tools.salad.training_snapshot import model_snapshot"
        self.assertIn(snapshot_import, workflow)
        self.assertIn("org.opencontainers.image.revision", workflow)
        self.assertIn('test "${REVISION}" = "${GITHUB_SHA}"', workflow)

    def test_heartbeat_only_present_on_pinned_image_marker(self):
        worker = (ROOT / "tools" / "salad" / "salad_worker.py").read_text()
        self.assertIn('os.environ.get("SZL_HEARTBEAT_FILE")', worker)
        self.assertIn("import step_heartbeat as image_heartbeat", worker)
        self.assertIn("image_heartbeat.make_callback", worker)
        self.assertIn("verified_resume_checkpoint", worker)


if __name__ == "__main__":
    unittest.main()
