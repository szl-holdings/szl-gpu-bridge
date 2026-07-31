from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import runjob_nemo_v3  # noqa: E402


class NemoV3RunnerTests(unittest.TestCase):
    def _tokenizer_fixture(
        self, root: pathlib.Path
    ) -> tuple[dict[str, object], pathlib.Path, dict[str, tuple[int, str]]]:
        repo_id = "example/pinned-model"
        revision = "a" * 40
        snapshot = root / "models--example--pinned-model" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        contents = {
            "chat_template.jinja": b"{{ messages }}",
            "special_tokens_map.json": b'{"eos_token":"</s>"}',
            "tokenizer.json": b'{"version":"1.0"}',
            "tokenizer_config.json": b'{"chat_template":"{{ messages }}"}',
        }
        manifest: dict[str, tuple[int, str]] = {}
        for name, content in contents.items():
            (snapshot / name).write_bytes(content)
            manifest[name] = (len(content), hashlib.sha256(content).hexdigest())
        spec: dict[str, object] = {
            "base": {
                "repoId": repo_id,
                "revision": revision,
                "licenseId": "test-license",
            },
            "dataset": {
                "train": {"bytes": 1, "sha256": "b" * 64},
                "preregistration": {"bytes": 1, "sha256": "c" * 64},
            },
        }
        return spec, snapshot.resolve(), manifest

    def test_production_tokenizer_manifest_binds_exact_local_snapshot(self) -> None:
        identity = (
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        )
        self.assertEqual(
            runjob_nemo_v3.PINNED_OFFLINE_TOKENIZERS[identity],
            {
                "chat_template.jinja": (
                    10504,
                    "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
                ),
                "special_tokens_map.json": (
                    420,
                    "e3a4f63da745f02317a45e00e6476c17fc66ac41faf14bb1b0be1f3211b0ca53",
                ),
                "tokenizer.json": (
                    17077484,
                    "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
                ),
                "tokenizer_config.json": (
                    188034,
                    "48de4056b0b17de26e03232fdc1f55b70595c9354ceb2ed061f724f45620aa41",
                ),
            },
        )

    def test_remote_code_requires_credentialless_offline_container(self) -> None:
        spec = {"base": {"trustRemoteCode": True}}
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "isolated container"):
                runjob_nemo_v3._require_remote_code_isolation(spec)

        isolated = {
            "SZL_EXECUTION_ISOLATION": "credentialless-networkless-container",
            "SZL_RECEIPT_TRANSPORT": "local-unsigned-outbox",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
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
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_TOKEN": "must-not-enter",
        }
        with mock.patch.dict("os.environ", isolated, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                runjob_nemo_v3._require_remote_code_isolation(spec)

    def test_offline_tokenizer_snapshot_is_exact_and_does_not_mutate_spec(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            spec, expected_snapshot, manifest = self._tokenizer_fixture(cache)
            before = repr(spec)
            identity = ("example/pinned-model", "a" * 40)
            with (
                mock.patch.dict(
                    runjob_nemo_v3.PINNED_OFFLINE_TOKENIZERS,
                    {identity: manifest},
                    clear=True,
                ),
                mock.patch.dict(
                    os.environ, {"HF_HUB_CACHE": str(cache.resolve())}, clear=False
                ),
            ):
                observed = runjob_nemo_v3._verified_offline_tokenizer_snapshot(spec)

        self.assertEqual(observed, expected_snapshot)
        self.assertEqual(repr(spec), before)

    def test_offline_tokenizer_snapshot_rejects_missing_and_tampered_artifacts(
        self,
    ) -> None:
        for failure in ("missing", "tampered"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                cache = pathlib.Path(temporary)
                spec, _, manifest = self._tokenizer_fixture(cache)
                identity = ("example/pinned-model", "a" * 40)
                tokenizer_path = (
                    cache
                    / "models--example--pinned-model"
                    / "snapshots"
                    / ("a" * 40)
                    / "tokenizer.json"
                )
                if failure == "missing":
                    tokenizer_path.unlink()
                else:
                    tokenizer_path.write_bytes(b"tampered")
                with (
                    mock.patch.dict(
                        runjob_nemo_v3.PINNED_OFFLINE_TOKENIZERS,
                        {identity: manifest},
                        clear=True,
                    ),
                    mock.patch.dict(
                        os.environ, {"HF_HUB_CACHE": str(cache.resolve())}, clear=False
                    ),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "artifact is absent|artifact mismatch"
                    ):
                        runjob_nemo_v3._verified_offline_tokenizer_snapshot(spec)

    def test_loaded_tokenizer_must_be_non_null_typed_and_snapshot_bound(self) -> None:
        class ExpectedTokenizer:
            def __init__(self, name_or_path: str, chat_template: str = "template"):
                self.name_or_path = name_or_path
                self.chat_template = chat_template

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = pathlib.Path(temporary).resolve()
            tokenizer = ExpectedTokenizer(str(snapshot))
            template = runjob_nemo_v3._require_loaded_tokenizer(
                tokenizer, ExpectedTokenizer, snapshot
            )
            self.assertEqual(template, "template")

            with self.assertRaisesRegex(RuntimeError, "returned no tokenizer"):
                runjob_nemo_v3._require_loaded_tokenizer(
                    None, ExpectedTokenizer, snapshot
                )
            with self.assertRaisesRegex(RuntimeError, "unsupported tokenizer type"):
                runjob_nemo_v3._require_loaded_tokenizer(
                    object(), ExpectedTokenizer, snapshot
                )
            other_snapshot = snapshot / "other"
            other_snapshot.mkdir()
            with self.assertRaisesRegex(RuntimeError, "not the verified snapshot"):
                runjob_nemo_v3._require_loaded_tokenizer(
                    ExpectedTokenizer(str(other_snapshot)),
                    ExpectedTokenizer,
                    snapshot,
                )

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
