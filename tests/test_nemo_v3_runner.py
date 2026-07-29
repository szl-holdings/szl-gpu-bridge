from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import runjob_nemo_v3  # noqa: E402


class NemoV3RunnerTests(unittest.TestCase):
    def test_remote_code_requires_credentialless_offline_container(self) -> None:
        spec = {"base": {"trustRemoteCode": True}}
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "isolated container"):
                runjob_nemo_v3._require_remote_code_isolation(spec)

        isolated = {
            "SZL_EXECUTION_ISOLATION": "credentialless-networkless-container",
            "SZL_RECEIPT_TRANSPORT": "local-unsigned-outbox",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        with (
            mock.patch.dict("os.environ", isolated, clear=True),
            mock.patch.object(runjob_nemo_v3, "ROOT", pathlib.Path("missing-root")),
        ):
            runjob_nemo_v3._require_remote_code_isolation(spec)

    def test_remote_code_refuses_hf_token_even_inside_isolation(self) -> None:
        spec = {"base": {"trustRemoteCode": True}}
        isolated = {
            "SZL_EXECUTION_ISOLATION": "credentialless-networkless-container",
            "SZL_RECEIPT_TRANSPORT": "local-unsigned-outbox",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_TOKEN": "must-not-enter",
        }
        with mock.patch.dict("os.environ", isolated, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                runjob_nemo_v3._require_remote_code_isolation(spec)

    def test_uploaded_terminal_evaluation_failure_is_not_retried(self) -> None:
        spec = {
            "jobId": "job-test",
            "outputs": {"candidateId": "candidate-test"},
            "source": {"repoId": "szl-holdings/source", "revision": "a" * 40},
            "base": {
                "repoId": "SZLHOLDINGS/base",
                "revision": "b" * 40,
                "licenseId": "apache-2.0",
            },
            "dataset": {"rightsBasis": "PROJECT_AUTHORED_SCENARIOS"},
        }
        evidence = {
            "state": "FAIL",
            "rows": 1,
            "passes": 0,
            "pass_rate": 0.0,
            "degenerate": 0,
            "suites": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            job_root = pathlib.Path(temporary)
            with mock.patch.object(
                runjob_nemo_v3, "deliver_receipt", return_value=0
            ) as deliver:
                exit_code = runjob_nemo_v3._complete_terminal_evaluation_failure(
                    spec, b"signed-job-payload", job_root, evidence
                )

        self.assertEqual(exit_code, 0)
        receipt, name, delivered_spec = deliver.call_args.args
        self.assertEqual(receipt["state"], "EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED")
        self.assertEqual(receipt["decision"], "TERMINAL_FAILURE_NO_AUTOMATIC_RETRY")
        self.assertIn("stack", receipt["evaluation"])
        self.assertEqual(name, "nemo-v3-terminal.signed.json")
        self.assertIs(delivered_spec, spec)

    def test_offline_input_cache_is_verified_before_copy(self) -> None:
        content = b'{"record_id":"train:1"}\n'
        descriptor = {
            "path": "model_release/szl-nemo-v3/train.jsonl",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cache = root / "cache"
            cached = cache / descriptor["path"]
            cached.parent.mkdir(parents=True)
            cached.write_bytes(content)
            target = root / "job" / "train.jsonl"
            with mock.patch.dict(
                "os.environ", {"SZL_INPUT_CACHE": str(cache)}, clear=False
            ):
                observed = runjob_nemo_v3._download_pinned(
                    {"source": {}}, descriptor, target
                )
                observed_content = observed.read_bytes()

        self.assertEqual(observed_content, content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
