from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import runjob_nemo_v3  # noqa: E402


class NemoV3RunnerTests(unittest.TestCase):
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
        signed = {"signatureBase64": "test"}

        with tempfile.TemporaryDirectory() as temporary:
            job_root = pathlib.Path(temporary)
            with (
                mock.patch.object(
                    runjob_nemo_v3, "sign_receipt", return_value=signed
                ) as sign,
                mock.patch.object(runjob_nemo_v3, "write_json") as write,
                mock.patch.object(runjob_nemo_v3, "upload_receipt") as upload,
            ):
                exit_code = runjob_nemo_v3._complete_terminal_evaluation_failure(
                    spec, b"signed-job-payload", job_root, evidence
                )

        self.assertEqual(exit_code, 0)
        receipt = sign.call_args.args[0]
        self.assertEqual(
            receipt["state"], "EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED"
        )
        self.assertEqual(receipt["decision"], "TERMINAL_FAILURE_NO_AUTOMATIC_RETRY")
        self.assertEqual(
            write.call_args.args[0],
            job_root / "receipts" / "nemo-v3-terminal.signed.json",
        )
        upload.assert_called_once_with(
            signed, "nemo-v3-terminal.signed.json", spec
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
