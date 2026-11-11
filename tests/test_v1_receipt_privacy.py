"""Offline tests of the real legacy uploader; no keys, GPU, or Hub access."""

import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


class V1ReceiptPrivacyTests(unittest.TestCase):
    def setUp(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "laptop" / "runjob.py"
        module_spec = importlib.util.spec_from_file_location("v1_runjob_test", path)
        self.runner = importlib.util.module_from_spec(module_spec)
        # Import needs only a public-key pin; signing is never exercised here.
        with mock.patch.object(pathlib.Path, "read_text", return_value="{}"):
            module_spec.loader.exec_module(self.runner)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runner.ROOT = pathlib.Path(self.temp.name)
        (self.runner.ROOT / "jobs").mkdir()
        self.api = mock.Mock()
        self.api.repo_info.return_value = types.SimpleNamespace(private=True)
        self.factory = mock.Mock(return_value=self.api)
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = self.factory
        patcher = mock.patch.dict(sys.modules, {"huggingface_hub": hub})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.spec = {
            "jobId": "test-job",
            "outputs": {"receiptsRepoId": "owner/receipts"},
        }
        self.signed = {"bodyBase64": "dGVzdA==", "signatureBase64": "synthetic"}
        self.name = "training_receipt.signed.json"

    def upload(self):
        self.runner.upload_receipt(self.signed, self.name, self.spec)

    def test_missing_privacy_defaults_to_private_and_reads_back(self):
        self.upload()
        self.api.create_repo.assert_called_once_with(
            "owner/receipts", repo_type="dataset", exist_ok=True, private=True
        )
        self.api.repo_info.assert_called_once_with(
            "owner/receipts", repo_type="dataset"
        )
        self.api.upload_file.assert_called_once_with(
            path_or_fileobj=str(self.runner.ROOT / "jobs" / self.name),
            path_in_repo=f"test-job/{self.name}",
            repo_id="owner/receipts",
            repo_type="dataset",
        )
        self.assertEqual(
            json.loads((self.runner.ROOT / "jobs" / self.name).read_text()), self.signed
        )
        self.assertEqual(
            [call[0] for call in self.api.mock_calls],
            ["create_repo", "repo_info", "upload_file"],
        )

    def test_explicit_public_upload_requires_public_readback(self):
        self.spec["outputs"]["private"] = False
        self.api.repo_info.return_value.private = False
        self.upload()
        self.assertIs(self.api.create_repo.call_args.kwargs["private"], False)
        self.api.upload_file.assert_called_once()

    def test_private_upload_refuses_existing_public_repository(self):
        self.api.repo_info.return_value.private = False
        with self.assertRaisesRegex(ValueError, "visibility"):
            self.upload()
        self.api.upload_file.assert_not_called()

    def test_public_spec_does_not_change_existing_private_repository(self):
        self.spec["outputs"]["private"] = False
        with self.assertRaisesRegex(ValueError, "visibility"):
            self.upload()
        self.api.upload_file.assert_not_called()
        self.api.update_repo_settings.assert_not_called()

    def test_unknown_visibility_refuses_upload(self):
        for value in (None, 0, 1, "true", "false"):
            with self.subTest(value=value):
                self.api.repo_info.return_value = types.SimpleNamespace(private=value)
                with self.assertRaisesRegex(ValueError, "visibility"):
                    self.upload()
        self.api.upload_file.assert_not_called()

    def test_missing_visibility_refuses_upload(self):
        self.api.repo_info.return_value = types.SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "visibility"):
            self.upload()
        self.api.upload_file.assert_not_called()

    def test_unavailable_visibility_refuses_upload(self):
        self.api.repo_info.side_effect = OSError("Hub read unavailable")
        with self.assertRaisesRegex(OSError, "Hub read unavailable"):
            self.upload()
        self.api.upload_file.assert_not_called()

    def test_non_boolean_spec_refuses_before_api_construction(self):
        for value in (None, 0, 1, "true", "false", [], {}):
            with self.subTest(value=value):
                self.spec["outputs"]["private"] = value
                with self.assertRaisesRegex(ValueError, "boolean"):
                    self.upload()
        self.factory.assert_not_called()
        self.api.upload_file.assert_not_called()

    def test_repository_creation_failure_refuses_upload(self):
        self.api.create_repo.side_effect = OSError("Hub create unavailable")
        with self.assertRaisesRegex(OSError, "Hub create unavailable"):
            self.upload()
        self.api.repo_info.assert_not_called()
        self.api.upload_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
