"""Offline one-shot launch contracts; real subprocesses execute only tiny test doubles."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/salad/bench_launch.py"
SPEC = importlib.util.spec_from_file_location("isolated_bench_launch", SOURCE)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def fake_harness(code=0):
    return (
        "SCHEMA = 'szl.salad-bench/v2'\n"
        "FLAGS = ('--run-id', '--source-revision', '--launch-contract-sha256')\n"
        "import sys\n"
        f"sys.exit({code})\n"
    ).encode()


class BenchLaunchContract(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.harness = self.root / "harness.py"
        self.harness.write_bytes(fake_harness())
        self.work = self.root / "attempts"
        self.contract = {
            "schema": launch.SCHEMA,
            "attempt_id": "1" * 32,
            "source": {
                "repository": "szl-holdings/szl-gpu-bridge",
                "revision": "a" * 40,
                "sha256": hashlib.sha256(self.harness.read_bytes()).hexdigest(),
            },
            "inputs": {
                "base": {"repo_id": "fixture/base", "revision": "b" * 40},
                "adapter": {
                    "repo_id": "fixture/adapter",
                    "revision": "c" * 40,
                    "directory": "final",
                },
                "dataset": {
                    "repo_id": "fixture/data",
                    "revision": "d" * 40,
                    "eval_file": "data/eval.jsonl",
                    "eval_sha256": "e" * 64,
                    "train_file": "data/train.jsonl",
                    "train_sha256": "f" * 64,
                },
            },
            "destination": {
                "repo_id": "fixture/receipts",
                "repo_type": "dataset",
                "prefix": "receipts/bench",
            },
            "measurement": {"max_length": 2048, "train_ref_rows": 2000, "seed": 1234},
        }
        for owner, name in ((socket, "create_connection"), (socket.socket, "connect")):
            denied = patch.object(
                owner, name, side_effect=AssertionError("network forbidden")
            )
            denied.start()
            self.addCleanup(denied.stop)

    def plan(self):
        return launch.plan(self.contract, self.harness, self.work)

    def child(self, execute=False):
        config = self.root / "contract.json"
        config.write_text(json.dumps(self.contract), encoding="utf-8")
        args = [
            sys.executable,
            "-B",
            str(SOURCE),
            "--contract",
            str(config),
            "--harness",
            str(self.harness),
            "--work-root",
            str(self.work),
        ]
        if execute:
            args.append("--execute")
        return subprocess.run(args, capture_output=True, text=True, timeout=15)

    def replace_harness(self, raw):
        self.harness.write_bytes(raw)
        self.contract["source"]["sha256"] = hashlib.sha256(raw).hexdigest()

    def test_default_cli_only_plans_without_work_directory(self):
        result = self.child()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PLANNED_NOT_EXECUTED")
        self.assertFalse(self.work.exists())

    def test_all_input_revisions_are_explicit(self):
        record, _, _ = self.plan()
        argv = record["argv"]
        for flag, value in (
            ("--base-revision", "b" * 40),
            ("--adapter-revision", "c" * 40),
            ("--dataset-revision", "d" * 40),
            ("--receipts-repo-type", "dataset"),
        ):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn("bash", argv)
        self.assertNotIn("true", argv)

    def test_attempt_source_and_contract_are_forwarded(self):
        record, _, canonical = self.plan()
        argv = record["argv"]
        self.assertEqual(argv[argv.index("--run-id") + 1], "1" * 32)
        self.assertEqual(argv[argv.index("--source-revision") + 1], "a" * 40)
        self.assertEqual(
            argv[argv.index("--launch-contract-sha256") + 1],
            hashlib.sha256(canonical).hexdigest(),
        )
        self.assertIs(record["source_admission_verified"], False)

    def test_mutable_source_revision_is_rejected(self):
        self.contract["source"]["revision"] = "main"
        with self.assertRaises(ValueError):
            self.plan()

    def test_mutable_input_revision_is_rejected(self):
        for name in ("base", "adapter", "dataset"):
            with self.subTest(name=name):
                original = self.contract["inputs"][name]["revision"]
                self.contract["inputs"][name]["revision"] = "main"
                with self.assertRaises(ValueError):
                    self.plan()
                self.contract["inputs"][name]["revision"] = original

    def test_changed_harness_bytes_are_rejected(self):
        self.harness.write_bytes(self.harness.read_bytes() + b"# changed\n")
        with self.assertRaises(ValueError):
            self.plan()
        self.assertFalse(self.work.exists())

    def test_legacy_v1_cannot_be_silently_relabelled(self):
        self.replace_harness(
            fake_harness().replace(b"szl.salad-bench/v2", b"szl.salad-bench/v1")
        )
        with self.assertRaises(ValueError):
            self.plan()

    def test_encoded_blob_is_not_automatically_decoded(self):
        import base64

        self.replace_harness(base64.b64encode(fake_harness()))
        with self.assertRaises((ValueError, SyntaxError)):
            self.plan()

    def test_missing_launch_interface_is_rejected(self):
        self.replace_harness(b"SCHEMA='szl.salad-bench/v2'\n")
        with self.assertRaises(ValueError):
            self.plan()

    def test_unexpected_source_authority_is_rejected(self):
        self.contract["source"]["repository"] = "untrusted/other"
        with self.assertRaises(ValueError):
            self.plan()

    def test_unknown_environment_override_is_rejected(self):
        self.contract["environment"] = {"EXAMPLE": "not-a-secret"}
        with self.assertRaises(ValueError):
            self.plan()

    def test_attempt_traversal_is_rejected(self):
        self.contract["attempt_id"] = "../other"
        with self.assertRaises(ValueError):
            self.plan()

    def test_repository_path_traversal_is_rejected(self):
        self.contract["inputs"]["dataset"]["eval_file"] = "../hidden"
        with self.assertRaises(ValueError):
            self.plan()

    def test_missing_receipt_type_is_rejected(self):
        del self.contract["destination"]["repo_type"]
        with self.assertRaises(ValueError):
            self.plan()

    def test_wrong_receipt_type_is_rejected(self):
        self.contract["destination"]["repo_type"] = "bucket"
        with self.assertRaises(ValueError):
            self.plan()

    def test_explicit_model_destination_remains_typed(self):
        self.contract["destination"]["repo_type"] = "model"
        argv = self.plan()[0]["argv"]
        self.assertEqual(argv[argv.index("--receipts-repo-type") + 1], "model")

    def test_boolean_measurement_bounds_are_rejected(self):
        self.contract["measurement"]["train_ref_rows"] = True
        with self.assertRaises(ValueError):
            self.plan()

    def test_plan_json_rejects_duplicate_keys(self):
        with self.assertRaises(ValueError):
            launch.load_contract(b'{"schema":"a","schema":"b"}')

    def test_plan_json_rejects_nonfinite_numbers(self):
        with self.assertRaises(ValueError):
            launch.load_contract(b'{"seed":NaN}')

    def test_large_input_is_bounded(self):
        path = self.root / "large.json"
        path.write_bytes(b" " * (launch.MAX_BYTES + 1))
        with self.assertRaises(ValueError):
            launch.read_bytes(path)

    def test_symlink_source_is_rejected_without_read(self):
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(ValueError):
                self.plan()

    def test_stage_persists_exact_source_and_canonical_contract(self):
        record, raw, canonical = self.plan()
        directory = launch.stage(record, raw, canonical)
        self.assertEqual((directory / "bench_eval.py").read_bytes(), raw)
        self.assertEqual((directory / "launch-contract.json").read_bytes(), canonical)
        self.assertFalse((directory / "bundle").exists())

    def test_only_one_stage_can_claim_an_attempt(self):
        record, raw, canonical = self.plan()
        directory = launch.stage(record, raw, canonical)
        saved = {p.name: p.read_bytes() for p in directory.iterdir()}
        with self.assertRaises(FileExistsError):
            launch.stage(record, raw, canonical)
        self.assertEqual(saved, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_existing_attempt_is_refused_in_planning(self):
        (self.work / self.contract["attempt_id"]).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_stage_flush_failure_prevents_execution(self):
        config = self.root / "contract.json"
        config.write_text(json.dumps(self.contract), encoding="utf-8")
        with (
            patch.object(
                launch.os, "fsync", side_effect=OSError("synthetic disk failure")
            ),
            patch.object(launch.os, "execv") as execute,
        ):
            result = launch.main(
                [
                    "--contract",
                    str(config),
                    "--harness",
                    str(self.harness),
                    "--work-root",
                    str(self.work),
                    "--execute",
                ]
            )
        self.assertEqual(result, 7)
        execute.assert_not_called()

    def test_child_failure_is_not_masked_as_success(self):
        self.replace_harness(fake_harness(7))
        result = self.child(execute=True)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)

    def test_child_success_code_is_preserved(self):
        result = self.child(execute=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_uncaught_child_error_is_not_masked(self):
        self.replace_harness(
            fake_harness().replace(
                b"sys.exit(0)", b"raise RuntimeError('synthetic failure')"
            )
        )
        result = self.child(execute=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_second_execution_cannot_reuse_attempt(self):
        self.assertEqual(self.child(execute=True).returncode, 0)
        result = self.child(execute=True)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("BENCH_LAUNCH", result.stdout)

    def test_actual_harness_interface_can_be_planned_without_import(self):
        self.harness = ROOT / "tools/salad/bench_eval.py"
        self.contract["source"]["sha256"] = hashlib.sha256(
            self.harness.read_bytes()
        ).hexdigest()
        record, _, _ = self.plan()
        self.assertEqual(record["status"], "PLANNED_NOT_EXECUTED")
        self.assertFalse(self.work.exists())

    def test_contract_canonicalization_is_stable(self):
        first = self.plan()[2]
        self.contract = launch.load_contract(
            json.dumps(copy.deepcopy(self.contract), indent=4).encode()
        )
        self.assertEqual(first, self.plan()[2])

    def test_windows_path_waits_for_exact_child_status(self):
        config = self.root / "contract.json"
        config.write_text(json.dumps(self.contract), encoding="utf-8")
        with (
            patch.object(launch.sys, "platform", "win32"),
            patch.object(
                launch.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 7),
            ) as child,
            patch.object(launch.os, "execv") as replacement,
        ):
            code = launch.main(
                [
                    "--contract",
                    str(config),
                    "--harness",
                    str(self.harness),
                    "--work-root",
                    str(self.work),
                    "--execute",
                ]
            )
        self.assertEqual(code, 7)
        self.assertIs(child.call_args.kwargs["check"], False)
        self.assertNotIn("shell", child.call_args.kwargs)
        replacement.assert_not_called()

    def test_linux_path_replaces_wrapper_without_shell(self):
        config = self.root / "contract.json"
        config.write_text(json.dumps(self.contract), encoding="utf-8")
        with (
            patch.object(launch.sys, "platform", "linux"),
            patch.object(launch.os, "execv", side_effect=SystemExit(7)) as replacement,
            patch.object(launch.subprocess, "run") as child,
        ):
            with self.assertRaises(SystemExit) as error:
                launch.main(
                    [
                        "--contract",
                        str(config),
                        "--harness",
                        str(self.harness),
                        "--work-root",
                        str(self.work),
                        "--execute",
                    ]
                )
        self.assertEqual(error.exception.code, 7)
        self.assertEqual(replacement.call_args.args[0], sys.executable)
        child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
