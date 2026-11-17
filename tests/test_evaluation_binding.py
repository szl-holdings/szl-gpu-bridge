"""Offline controls for exact qualification-to-evaluation binding.

Fixtures are synthetic observations, never real model inspection or admission.
The fixture writer uses the production qualification bundle format.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.salad import artifact_qualification as qualification
from tools.salad import evaluation_binding as binding


def candidate_fixture():
    """A complete synthetic observer result, including the producer's CPU fields."""
    target = 200
    state_sha = hashlib.sha256(b"synthetic trainer JSON").hexdigest()
    names = (
        f"checkpoint-{target}/trainer_state.json",
        "final/adapter_config.json",
        "final/adapter_model.safetensors",
        "final/tokenizer.json",
        "final/tokenizer_config.json",
    )
    files = {
        name: {
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "bytes": 1024,
            "integrity": "PINNED_OBJECT_BYTES_VERIFIED",
        }
        for name in names
    }
    files[names[0]]["sha256"] = state_sha
    return {
        "repo": "SZLHOLDINGS/example",
        "target_steps": target,
        "status": "ARTIFACT_VERIFIED",
        "ready_for_evaluation": True,
        "promotion_authorized": False,
        "training_execution_provenance_verified": False,
        "model_quality_measured": False,
        "signature_status": "NOT_SIGNED",
        "release_blockers": list(qualification.RELEASE_BLOCKERS),
        "observed_at_utc": "2026-01-01T00:00:00+00:00",
        "hold_reasons": [],
        "snapshot": {
            "id": "SZLHOLDINGS/example",
            "status": "OBSERVED",
            "revision": "a" * 40,
            "private": True,
            "file_count": 6,
            "latest_checkpoint": target,
            "final_adapter": True,
            "final_config": True,
            "final_tokenizer": True,
            "root_adapter": False,
            "last_modified": "2026-01-01 00:00:00+00:00",
            "file_list_basis": "same immutable metadata response",
            "artifact_integrity": "NOT_VERIFIED_BY_METADATA",
            "training_complete": "UNVERIFIED",
            "process_liveness_verified": False,
            "evaluation": "NOT_ASSESSED",
            "latest_trainer_state_present": True,
        },
        "trainer_state": {
            "global_step": target,
            "max_steps": target,
            "epoch": 1.5,
            "num_train_epochs": 2,
            "trainer_state_sha256": state_sha,
            "integrity": "PINNED_OBJECT_BYTES_VERIFIED",
        },
        "gates": {
            "adapter_config": {
                "base": "org/base",
                "rank": 32,
                "target_modules": ["q_proj"],
            },
            "files": files,
            "final_matches_terminal_checkpoint": True,
            "terminal_step_record_verified": True,
            "tensors": {
                "all_finite": True,
                "device": "cpu",
                "dtypes": ["torch.float32"],
                "elements": 8,
                "tensor_count": 2,
            },
        },
    }


def contract_fixture():
    """Identity-only synthetic contract; a harness must still prepare its data."""
    return {
        "schema": "szl.heldout-paired/v2",
        "attempt_id": "1" * 32,
        "source": {
            "repository": "szl-holdings/szl-gpu-bridge",
            "revision": "2" * 40,
            "sha256": "3" * 64,
        },
        "base": {"repo_id": "org/base", "revision": "b" * 40},
        "adapter": {
            "repo_id": "SZLHOLDINGS/example",
            "revision": "a" * 40,
            "directory": "final",
        },
        "fixtures": {
            "repo_id": "org/fixtures",
            "revision": "c" * 40,
            "path": "heldout.jsonl",
            "sha256": "4" * 64,
        },
        "training": {
            "repo_id": "org/training",
            "revision": "d" * 40,
            "path": "training.jsonl",
            "sha256": "5" * 64,
        },
        "generation": {
            "max_input_tokens": 256,
            "max_new_tokens": 16,
            "seed": 7,
            "timeout_seconds": 30,
        },
    }


def write_fixture_bundle(directory, candidate_overrides=None):
    """Return (qualification_sha256, synthetic_contract) using the real writer."""
    directory = Path(directory)
    candidate = candidate_fixture()
    if candidate_overrides:
        candidate.update(copy.deepcopy(candidate_overrides))
    qualification.write_bundle(directory, [candidate])
    pin = hashlib.sha256((directory / "qualification.json").read_bytes()).hexdigest()
    return pin, contract_fixture()


class EvaluationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "qualification"
        self.pin, self.contract = write_fixture_bundle(self.directory)

    def expect_code(self, code, function, *args):
        with self.assertRaises(binding.BindingError) as caught:
            function(*args)
        self.assertEqual(caught.exception.reason_code, code)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def load(self):
        return binding.load_qualification_bundle(self.directory, self.pin)

    def bind(self):
        return binding.bind_qualification(self.directory, self.contract, self.pin)

    def rewrite(self, name, raw, *, repair_manifest=True):
        (self.directory / name).write_bytes(raw)
        if repair_manifest and name != "manifest.json":
            path = self.directory / "manifest.json"
            manifest = json.loads(path.read_bytes())
            manifest["sha256"][name] = hashlib.sha256(raw).hexdigest()
            path.write_bytes(qualification.canonical(manifest))
        if name == "qualification.json":
            self.pin = hashlib.sha256(raw).hexdigest()

    def rewrite_report(self, mutate):
        report = json.loads((self.directory / "qualification.json").read_bytes())
        mutate(report)
        self.rewrite("qualification.json", qualification.canonical(report))

    def test_production_bundle_binds_and_never_claims_authority(self):
        before = sorted(path.relative_to(self.root) for path in self.root.rglob("*"))
        report, files = self.load()
        result = self.bind()
        self.assertEqual(set(files), set(binding.BUNDLE_FILES))
        self.assertTrue(all(isinstance(raw, bytes) for raw in files.values()))
        self.assertEqual(result["state"], "QUALIFICATION_BOUND")
        self.assertIs(result["artifact_observation_bound"], True)
        self.assertEqual(result["qualification_sha256"], self.pin)
        self.assertEqual(
            result["manifest_sha256"],
            hashlib.sha256(files["manifest.json"]).hexdigest(),
        )
        self.assertEqual(result["source_sha256"], report["code_sha256"])
        self.assertEqual(
            result["artifact"]["revision"], self.contract["adapter"]["revision"]
        )
        self.assertEqual(result["artifact"]["directory"], "final")
        self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")
        for key in (
            "authenticity_verified",
            "compute_authorized",
            "promotion_authorized",
            "training_complete_verified",
            "training_execution_provenance_verified",
            "base_revision_verified",
            "independent_holdout_verified",
            "source_admission_verified",
            "runtime_admission_verified",
            "model_quality_measured",
            "publication_performed",
            "production_ready",
        ):
            self.assertIs(result[key], False, key)
        self.assertEqual(
            result["remaining_release_blockers"], qualification.RELEASE_BLOCKERS
        )
        self.assertEqual(result, self.bind())
        self.assertEqual(
            before, sorted(path.relative_to(self.root) for path in self.root.rglob("*"))
        )

    def test_candidate_summary_has_only_allowlisted_fields(self):
        summary = binding.validate_candidate(candidate_fixture())
        self.assertEqual(
            set(summary),
            {
                "repo",
                "revision",
                "base_repo",
                "target_steps",
                "artifact_sha256",
                "observed_at_utc",
                "release_blockers",
            },
        )

    def test_contract_digest_uses_harness_encoding_and_changes_with_input(self):
        report, files = self.load()
        manifest_sha = hashlib.sha256(files["manifest.json"]).hexdigest()
        self.contract["fixtures"]["path"] = "évaluation.jsonl"
        result = binding.bind_report(report, self.contract, self.pin, manifest_sha)
        expected = (
            json.dumps(
                self.contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(
            result["contract_sha256"], hashlib.sha256(expected).hexdigest()
        )
        self.contract["generation"]["seed"] += 1
        changed = binding.bind_report(report, self.contract, self.pin, manifest_sha)
        self.assertNotEqual(changed["contract_sha256"], result["contract_sha256"])

    def test_operator_pin_is_required_and_exact(self):
        for pin in (None, "", "main", "A" * 64, "a" * 63, True):
            with self.subTest(pin=pin):
                self.expect_code(
                    "QUALIFICATION_PIN_INVALID",
                    binding.load_qualification_bundle,
                    self.directory,
                    pin,
                )
        self.expect_code(
            "QUALIFICATION_PIN_MISMATCH",
            binding.load_qualification_bundle,
            self.directory,
            "0" * 64,
        )

    def test_missing_bundle_is_unavailable(self):
        self.expect_code(
            "QUALIFICATION_UNAVAILABLE",
            binding.load_qualification_bundle,
            self.root / "absent",
            self.pin,
        )

    def test_bounded_read_rejects_oversized_file(self):
        self.rewrite("artifact_qualification.py", b"x" * (binding.MAX_INPUT_BYTES + 1))
        self.expect_code("QUALIFICATION_BYTE_BUDGET", self.load)

    def test_symlink_file_and_directory_are_rejected(self):
        target = self.directory / "qualification.json"
        alternate = self.root / "receipt.json"
        target.rename(alternate)
        try:
            target.symlink_to(alternate)
        except OSError:
            alternate.rename(target)
            self.skipTest("Creating symlinks is not permitted on this runner")
        self.expect_code("QUALIFICATION_UNSAFE_PATH", self.load)
        target.unlink()
        alternate.rename(target)
        linked = self.root / "linked"
        linked.symlink_to(self.directory, target_is_directory=True)
        self.expect_code(
            "QUALIFICATION_UNSAFE_PATH",
            binding.load_qualification_bundle,
            linked,
            self.pin,
        )

    @unittest.skipUnless(os.name == "nt", "Windows reparse-point control")
    def test_windows_junction_is_rejected(self):
        linked = self.root / "junction"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(self.directory)],
            capture_output=True,
            timeout=10,
        )
        if result.returncode:
            self.skipTest("Creating junctions is not permitted on this runner")
        try:
            self.expect_code(
                "QUALIFICATION_UNSAFE_PATH",
                binding.load_qualification_bundle,
                linked,
                self.pin,
            )
        finally:
            os.rmdir(linked)

    def test_report_raw_bytes_must_match_pin_even_with_repaired_manifest(self):
        old_pin = self.pin
        raw = (self.directory / "qualification.json").read_bytes() + b" "
        self.rewrite("qualification.json", raw)
        self.expect_code(
            "QUALIFICATION_PIN_MISMATCH",
            binding.load_qualification_bundle,
            self.directory,
            old_pin,
        )

    def test_manifest_and_report_source_hashes_are_both_checked(self):
        source = "artifact_qualification.py"
        self.rewrite(source, b"# different verifier\n", repair_manifest=False)
        self.expect_code("QUALIFICATION_FILE_HASH_MISMATCH", self.load)
        self.rewrite(source, b"# different verifier\n")
        self.expect_code("QUALIFICATION_SOURCE_HASH_MISMATCH", self.load)

    def test_archived_source_is_never_executed(self):
        name = "artifact_qualification.py"
        marker = self.root / "executed"
        raw = f"raise RuntimeError({str(marker)!r})\n".encode()
        self.rewrite(name, raw)
        self.rewrite_report(
            lambda report: report["code_sha256"].update(
                {name: hashlib.sha256(raw).hexdigest()}
            )
        )
        self.assertEqual(self.bind()["state"], "QUALIFICATION_BOUND")
        self.assertFalse(marker.exists())

    def test_source_bytes_must_be_utf8_without_nul(self):
        for raw in (b"\xff", b"# source\x00"):
            with self.subTest(raw=raw):
                self.rewrite("artifact_qualification.py", raw)
                self.rewrite_report(
                    lambda report: report["code_sha256"].update(
                        {"artifact_qualification.py": hashlib.sha256(raw).hexdigest()}
                    )
                )
                self.expect_code("QUALIFICATION_SOURCE_BYTES_INVALID", self.load)

    def test_duplicate_nonfinite_malformed_json_fail_closed(self):
        cases = (
            (b'{"schema":1,"schema":2}', "QUALIFICATION_DUPLICATE_JSON_KEY"),
            (b'{"x":NaN}', "QUALIFICATION_NONFINITE_JSON"),
            (b'{"x":1e999}', "QUALIFICATION_NONFINITE_JSON"),
            (b"[]", "QUALIFICATION_INVALID_JSON"),
            (b"{", "QUALIFICATION_INVALID_JSON"),
            (b'{"x":"\xff"}', "QUALIFICATION_INVALID_JSON"),
        )
        for raw, code in cases:
            with self.subTest(raw=raw):
                self.rewrite("qualification.json", raw)
                self.expect_code(code, self.load)

    def test_manifest_schema_and_exact_file_list(self):
        original = (self.directory / "manifest.json").read_bytes()
        for mutation in (
            lambda m: m.update(signed=True),
            lambda m: m.update(schema="other"),
            lambda m: m["sha256"].update({"../escape": "1" * 64}),
            lambda m: m["sha256"].pop("training_snapshot.py"),
        ):
            manifest = json.loads(original)
            mutation(manifest)
            self.rewrite("manifest.json", qualification.canonical(manifest))
            self.expect_code("QUALIFICATION_SCHEMA_INVALID", self.load)

    def test_report_schema_and_authority_cannot_be_upgraded(self):
        report, _ = self.load()
        cases = (
            (lambda r: r.update(schema="other"), "QUALIFICATION_SCHEMA_INVALID"),
            (lambda r: r.update(signed=True), "QUALIFICATION_SCOPE_INVALID"),
            (
                lambda r: r.update(promotion_authorized=True),
                "QUALIFICATION_SCOPE_INVALID",
            ),
            (
                lambda r: r.update(scope="production-ready"),
                "QUALIFICATION_SCOPE_INVALID",
            ),
            (
                lambda r: r.update(observed_at_utc="today"),
                "QUALIFICATION_SCHEMA_INVALID",
            ),
            (lambda r: r.update(candidates=[]), "QUALIFICATION_CANDIDATES_INVALID"),
            (
                lambda r: r["candidates"].append(copy.deepcopy(r["candidates"][0])),
                "QUALIFICATION_CANDIDATES_INVALID",
            ),
        )
        for mutate, code in cases:
            with self.subTest(code=code):
                changed = copy.deepcopy(report)
                mutate(changed)
                self.rewrite("qualification.json", qualification.canonical(changed))
                self.expect_code(code, self.load)

    def test_exact_candidate_revision_final_directory_and_base(self):
        cases = (
            (
                "adapter",
                "revision",
                "e" * 40,
                "QUALIFICATION_ADAPTER_REVISION_MISMATCH",
            ),
            (
                "adapter",
                "repo_id",
                "SZLHOLDINGS/other",
                "QUALIFICATION_CANDIDATE_NOT_FOUND",
            ),
            (
                "adapter",
                "directory",
                "checkpoint-200",
                "QUALIFICATION_ADAPTER_DIRECTORY_MISMATCH",
            ),
            (
                "adapter",
                "directory",
                "../final",
                "QUALIFICATION_ADAPTER_DIRECTORY_MISMATCH",
            ),
            ("base", "repo_id", "org/different", "QUALIFICATION_BASE_MISMATCH"),
            ("adapter", "revision", "main", "QUALIFICATION_CONTRACT_INVALID"),
        )
        for arm, key, value, code in cases:
            with self.subTest(key=key, value=value):
                self.contract = contract_fixture()
                self.contract[arm][key] = value
                self.expect_code(code, self.bind)

    def test_base_revision_is_not_claimed_as_attested(self):
        self.contract["base"]["revision"] = "f" * 40
        self.assertIs(self.bind()["base_revision_verified"], False)

    def test_holds_and_provider_unavailability_are_distinct(self):
        for reasons, code in (
            (["TRAINING_IN_PROGRESS"], "QUALIFICATION_CANDIDATE_HOLD"),
            (
                ["OBSERVATION_UNAVAILABLE:TimeoutError"],
                "QUALIFICATION_CANDIDATE_UNAVAILABLE",
            ),
        ):
            candidate = candidate_fixture()
            candidate.update(
                status="HOLD", ready_for_evaluation=False, hold_reasons=reasons
            )
            self.expect_code(code, binding.validate_candidate, candidate)
        candidate = candidate_fixture()
        candidate["ready_for_evaluation"] = False
        self.expect_code(
            "QUALIFICATION_CANDIDATE_NOT_READY", binding.validate_candidate, candidate
        )

    def test_candidate_internal_consistency_negative_controls(self):
        cases = (
            (("target_steps",), True, "QUALIFICATION_CANDIDATE_INVALID"),
            (("promotion_authorized",), True, "QUALIFICATION_SCOPE_INVALID"),
            (
                ("training_execution_provenance_verified",),
                True,
                "QUALIFICATION_SCOPE_INVALID",
            ),
            (("model_quality_measured",), True, "QUALIFICATION_SCOPE_INVALID"),
            (("release_blockers",), [], "QUALIFICATION_RELEASE_BLOCKERS_CHANGED"),
            (("snapshot", "id"), "SZLHOLDINGS/other", "QUALIFICATION_SNAPSHOT_INVALID"),
            (("snapshot", "status"), "UNAVAILABLE", "QUALIFICATION_SNAPSHOT_INVALID"),
            (("snapshot", "revision"), "main", "QUALIFICATION_SNAPSHOT_INVALID"),
            (("snapshot", "private"), 1, "QUALIFICATION_SNAPSHOT_INVALID"),
            (("snapshot", "final_tokenizer"), False, "QUALIFICATION_SNAPSHOT_INVALID"),
            (("snapshot", "training_complete"), True, "QUALIFICATION_SNAPSHOT_INVALID"),
            (("snapshot", "latest_checkpoint"), 199, "QUALIFICATION_TERMINAL_MISMATCH"),
            (("trainer_state", "global_step"), 199, "QUALIFICATION_TERMINAL_MISMATCH"),
            (("trainer_state", "max_steps"), 201, "QUALIFICATION_TERMINAL_MISMATCH"),
            (
                ("gates", "final_matches_terminal_checkpoint"),
                False,
                "QUALIFICATION_TERMINAL_MISMATCH",
            ),
            (
                ("gates", "terminal_step_record_verified"),
                1,
                "QUALIFICATION_TERMINAL_MISMATCH",
            ),
            (
                ("gates", "files", "final/tokenizer.json", "bytes"),
                0,
                "QUALIFICATION_FILE_PROOF_INVALID",
            ),
            (
                ("gates", "files", "final/tokenizer.json", "bytes"),
                True,
                "QUALIFICATION_FILE_PROOF_INVALID",
            ),
            (
                ("gates", "files", "final/tokenizer.json", "integrity"),
                "OBSERVED",
                "QUALIFICATION_FILE_PROOF_INVALID",
            ),
            (
                ("gates", "files", "checkpoint-200/trainer_state.json", "sha256"),
                "f" * 64,
                "QUALIFICATION_FILE_PROOF_INVALID",
            ),
            (
                ("gates", "adapter_config", "rank"),
                0,
                "QUALIFICATION_ADAPTER_CONFIG_INVALID",
            ),
            (
                ("gates", "adapter_config", "target_modules"),
                [],
                "QUALIFICATION_ADAPTER_CONFIG_INVALID",
            ),
            (
                ("gates", "tensors", "all_finite"),
                False,
                "QUALIFICATION_TENSOR_PROOF_INVALID",
            ),
            (
                ("gates", "tensors", "device"),
                "cuda",
                "QUALIFICATION_TENSOR_PROOF_INVALID",
            ),
            (
                ("gates", "tensors", "tensor_count"),
                True,
                "QUALIFICATION_TENSOR_PROOF_INVALID",
            ),
            (("gates", "tensors", "elements"), 1, "QUALIFICATION_TENSOR_PROOF_INVALID"),
            (
                ("gates", "tensors", "dtypes"),
                ["torch.int64"],
                "QUALIFICATION_TENSOR_PROOF_INVALID",
            ),
        )
        for path, value, code in cases:
            with self.subTest(path=path):
                candidate = candidate_fixture()
                current = candidate
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = value
                self.expect_code(code, binding.validate_candidate, candidate)

    def test_incomplete_artifact_bundle_or_tensor_proof_cannot_bind(self):
        for remove, code in (
            (
                lambda c: c["gates"]["files"].pop("final/tokenizer_config.json"),
                "QUALIFICATION_FILE_PROOF_INVALID",
            ),
            (
                lambda c: c["gates"]["tensors"].pop("elements"),
                "QUALIFICATION_TENSOR_PROOF_INVALID",
            ),
        ):
            candidate = candidate_fixture()
            remove(candidate)
            self.expect_code(code, binding.validate_candidate, candidate)

    def test_unselected_hold_does_not_replace_selected_evidence(self):
        report, files = self.load()
        held = {
            "repo": "SZLHOLDINGS/other",
            "status": "HOLD",
            "hold_reasons": ["TRAINING_IN_PROGRESS"],
        }
        report["candidates"].append(held)
        result = binding.bind_report(
            report,
            self.contract,
            self.pin,
            hashlib.sha256(files["manifest.json"]).hexdigest(),
        )
        self.assertEqual(
            result["artifact"]["repo"], self.contract["adapter"]["repo_id"]
        )

    def test_standalone_import_needs_no_repository_or_model_packages(self):
        source = Path(binding.__file__).read_bytes()
        standalone = self.root / "evaluation_binding.py"
        standalone.write_bytes(source)
        script = (
            "import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('binding',{str(standalone)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not ({'torch','transformers','huggingface_hub','peft','requests'} "
            "& set(sys.modules));"
            "assert m.SCHEMA=='szl.evaluation-artifact-binding/v1'"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_loader_never_calls_network_or_model_libraries(self):
        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            self.assertEqual(self.bind()["state"], "QUALIFICATION_BOUND")

    def test_exception_text_is_never_arbitrary_input(self):
        error = binding.BindingError("secret provider URL")
        self.assertEqual(str(error), "QUALIFICATION_INTERNAL_ERROR")


if __name__ == "__main__":
    unittest.main()
