"""Synthetic contract/negative controls. No model weights, GPU, network or cloud writes."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.salad import artifact_qualification as qualification

SOURCE = Path(__file__).resolve().parents[1] / "tools/salad/heldout_eval_v2.py"
SPEC = importlib.util.spec_from_file_location("heldout_v2", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "heldout_binding_fixtures", Path(__file__).with_name("test_evaluation_binding.py")
)
FIXTURES = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURES)
ADAPTER_BYTES = {
    "adapter_config.json": b'{"synthetic":"adapter configuration"}',
    "adapter_model.safetensors": b"synthetic adapter bytes; no real tensors",
    "tokenizer.json": b'{"synthetic":"tokenizer"}',
    "tokenizer_config.json": b'{"synthetic":"tokenizer configuration"}',
}


def binding_source_pin():
    return M.digest(SOURCE.with_name("evaluation_binding.py").read_bytes())


def archived_binding_pin(directory):
    return M.digest((directory / "qualification-binding.json").read_bytes())


def qualification_input(root, contract):
    directory = root / "qualification-input"
    candidate = FIXTURES.candidate_fixture()
    candidate["repo"] = contract["adapter"]["repo_id"]
    candidate["snapshot"].update(
        id=candidate["repo"], revision=contract["adapter"]["revision"]
    )
    candidate["gates"]["adapter_config"]["base"] = contract["base"]["repo_id"]
    for name, value in ADAPTER_BYTES.items():
        candidate["gates"]["files"]["final/" + name].update(
            bytes=len(value), sha256=M.digest(value)
        )
    qualification.write_bundle(directory, [candidate])
    return {
        "qualification_bundle": directory,
        "qualification_sha256": M.digest(
            (directory / "qualification.json").read_bytes()
        ),
    }


def execute_qualified(c, plan, raw, root, **kwargs):
    return M.execute(c, plan, raw, root, **qualification_input(root, c), **kwargs)


def archive_qualification(directory, contract):
    args = qualification_input(directory.parent, contract)
    shutil.copytree(args["qualification_bundle"], directory / "qualification")
    api = M.binding_api()
    binding = api.bind_qualification(
        args["qualification_bundle"], contract, args["qualification_sha256"]
    )
    (directory / "qualification-binding.json").write_bytes(M.encoded(binding))
    (directory / "evaluation_binding.py").write_bytes(Path(api.__file__).read_bytes())


def examples():
    return [
        {
            "id": "n1",
            "task": "neutral_mcq",
            "messages": [{"role": "user", "content": "Synthetic choice one"}],
            "expected_choice": "B",
        },
        {
            "id": "n2",
            "task": "neutral_mcq",
            "messages": [{"role": "user", "content": "Synthetic choice two"}],
            "expected_choice": "A",
        },
        {
            "id": "r1",
            "task": "refusal",
            "messages": [{"role": "user", "content": "Synthetic unscored request"}],
            "expected_choice": None,
        },
    ]


def inputs(source=None):
    source = SOURCE.read_bytes() if source is None else source
    fixtures = b"".join(M.encoded(row) for row in examples())
    training = M.encoded(
        {
            "messages": [
                {"role": "user", "content": "Distinct training prompt"},
                {"role": "assistant", "content": "Training answer"},
            ]
        }
    )
    identity = {"repo_id": "test/base", "revision": "a" * 40}
    c = {
        "schema": M.SCHEMA,
        "attempt_id": "1" * 32,
        "source": {
            "repository": "szl-holdings/szl-gpu-bridge",
            "revision": "b" * 40,
            "sha256": M.digest(source),
        },
        "base": identity,
        "adapter": {
            "repo_id": "SZLHOLDINGS/test-adapter",
            "revision": "c" * 40,
            "directory": "final",
        },
        "fixtures": {
            "repo_id": "test/fixtures",
            "revision": "d" * 40,
            "path": "fixtures.jsonl",
            "sha256": M.digest(fixtures),
        },
        "training": {
            "repo_id": "test/training",
            "revision": "e" * 40,
            "path": "train.jsonl",
            "sha256": M.digest(training),
        },
        "generation": {
            "max_input_tokens": 128,
            "max_new_tokens": 16,
            "seed": 12,
            "timeout_seconds": 30,
        },
    }
    return c, fixtures, training, source


def prepared(values=None):
    c, fixtures, training, source = values or inputs()
    return M.prepare(M.encoded(c), fixtures, training, source)


def arm_rows(arm, outputs=("B", "A", "unscored")):
    return [
        {
            "index": i,
            "id": f["id"],
            "arm": arm,
            "fixture_sha256": M.digest(M.encoded(f)),
            "prompt_sha256": M.digest(M.encoded(M.messages(f["messages"]))),
            "output": output,
            "output_sha256": M.digest(output.encode()),
            "input_tokens": 10,
            "generated_tokens": 2,
            "budget_exhausted": False,
        }
        for i, (f, output) in enumerate(zip(examples(), outputs))
    ]


class InputContractTests(unittest.TestCase):
    def test_plan_has_no_authority_or_quality_claim(self):
        _, rows, plan = prepared()
        self.assertEqual(len(rows), 3)
        self.assertEqual(plan["task_counts"], {"neutral_mcq": 2, "refusal": 1})
        for key in (
            "source_admission_verified",
            "independent_holdout_verified",
            "compute_authorized",
            "publication_performed",
        ):
            self.assertIs(plan[key], False)
        self.assertEqual(plan["quality_qualification"], "NOT_ESTABLISHED")

    def test_all_source_and_input_hashes_are_enforced(self):
        for component in ("source", "fixtures", "training"):
            with self.subTest(component=component):
                values = inputs()
                values[0][component]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    prepared(values)

    def test_mutable_revisions_are_rejected(self):
        for component in ("base", "adapter", "fixtures", "training", "source"):
            values = inputs()
            values[0][component]["revision"] = "main"
            with self.subTest(component=component), self.assertRaises(ValueError):
                prepared(values)

    def test_unknown_fields_and_nonfinite_json_are_rejected(self):
        c, f, t, s = inputs()
        c["secret"] = "never log this"
        with self.assertRaises(ValueError):
            M.prepare(M.encoded(c), f, t, s)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                M.strict_json(raw)

    def test_generation_types_and_limits(self):
        for field, value in (
            ("seed", True),
            ("timeout_seconds", 0),
            ("timeout_seconds", 86401),
            ("max_new_tokens", -1),
            ("max_input_tokens", 1),
            ("seed", 1.2),
        ):
            values = inputs()
            values[0]["generation"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                prepared(values)

    def test_unsafe_adapter_and_dataset_paths(self):
        for component, field in (
            ("adapter", "directory"),
            ("fixtures", "path"),
            ("training", "path"),
        ):
            for value in ("../outside", "/absolute", "a//b", "C:\\outside", "a/./b"):
                values = inputs()
                values[0][component][field] = value
                with (
                    self.subTest(component=component, value=value),
                    self.assertRaises(ValueError),
                ):
                    prepared(values)

    def assert_bad_fixtures(self, rows):
        c, _, t, s = inputs()
        f = b"".join(M.encoded(x) for x in rows)
        c["fixtures"]["sha256"] = M.digest(f)
        with self.assertRaises(ValueError):
            M.prepare(M.encoded(c), f, t, s)

    def test_duplicate_ids_prompts_and_empty_fixture_are_rejected(self):
        rows = examples()
        rows[1]["id"] = rows[0]["id"]
        self.assert_bad_fixtures(rows)
        rows = examples()
        rows[1]["messages"] = rows[0]["messages"]
        self.assert_bad_fixtures(rows)
        self.assert_bad_fixtures([])

    def test_invalid_fixture_roles_and_answers(self):
        for key, value in (
            ("expected_choice", "E"),
            ("task", "guess_refusal"),
            ("id", "../x"),
        ):
            rows = examples()
            rows[0][key] = value
            self.assert_bad_fixtures(rows)
        rows = examples()
        rows[0]["messages"][0]["role"] = "assistant"
        self.assert_bad_fixtures(rows)
        rows = examples()
        rows[2]["expected_choice"] = "A"
        self.assert_bad_fixtures(rows)

    def test_training_overlap_rejects_even_with_answer_removed(self):
        c, f, _, s = inputs()
        t = M.encoded(
            {
                "messages": examples()[0]["messages"]
                + [{"role": "assistant", "content": "B"}]
            }
        )
        c["training"]["sha256"] = M.digest(t)
        with self.assertRaisesRegex(ValueError, "overlap"):
            M.prepare(M.encoded(c), f, t, s)

    def test_unicode_normalization_matches_prompts(self):
        a = [{"role": "user", "content": "caf\u00e9"}]
        b = [{"role": "user", "content": "cafe\u0301"}]
        self.assertEqual(M.messages(a), M.messages(b))

    def test_empty_training_cannot_establish_disjointness(self):
        c, f, _, s = inputs()
        c["training"]["sha256"] = M.digest(b"\n")
        with self.assertRaisesRegex(ValueError, "empty"):
            M.prepare(M.encoded(c), f, b"\n", s)

    def test_plan_cli_does_not_import_models_or_create_output(self):
        c, f, t, _ = inputs()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, raw in (("c.json", M.encoded(c)), ("f.jsonl", f), ("t.jsonl", t)):
                (root / name).write_bytes(raw)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SOURCE),
                    "--contract",
                    str(root / "c.json"),
                    "--fixtures",
                    str(root / "f.jsonl"),
                    "--training",
                    str(root / "t.jsonl"),
                    "--output-root",
                    str(root / "not-created"),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "not-created").exists())
            self.assertIs(json.loads(result.stdout)["compute_authorized"], False)


class ScoringTests(unittest.TestCase):
    def score(self, arms=None):
        c, f, _ = prepared()
        return M.score_pairs(
            c, f, arms or {a: arm_rows(a) for a in ("base", "adapter")}
        )

    def test_correct_counts_never_become_scientific_acceptance(self):
        result = self.score()
        self.assertEqual(result["arms"]["base"]["neutral_mcq"]["accuracy"], 1)
        self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")
        self.assertFalse(result["acceptance_policy_applied"])
        self.assertFalse(result["production_ready"])
        self.assertIsNone(result["energy_j"])

    def test_first_embedded_letter_is_not_an_answer(self):
        result = self.score(
            {
                a: arm_rows(a, ("A thought: B", "Answer: A", "unscored"))
                for a in ("base", "adapter")
            }
        )
        m = result["arms"]["adapter"]["neutral_mcq"]
        self.assertEqual(
            (m["accuracy"], m["rows_offered"], m["format_errors"]), (0, 2, 2)
        )

    def test_empty_group_and_unadjudicated_group_stay_unavailable(self):
        result = self.score()["arms"]["base"]
        self.assertEqual(result["invented_id"]["scoring_state"], "UNAVAILABLE_EMPTY")
        self.assertIsNone(result["invented_id"]["accuracy"])
        self.assertEqual(
            result["refusal"]["scoring_state"], "REQUIRES_INDEPENDENT_ADJUDICATION"
        )
        self.assertIsNone(result["refusal"]["correct"])

    def test_truncated_choice_is_not_silently_accepted(self):
        arms = {a: arm_rows(a) for a in ("base", "adapter")}
        arms["adapter"][0].update(budget_exhausted=True, generated_tokens=16)
        m = self.score(arms)["arms"]["adapter"]["neutral_mcq"]
        self.assertEqual((m["correct"], m["accuracy"], m["rows_offered"]), (1, 0.5, 2))

    def test_missing_extra_reordered_or_relabelled_rows_fail(self):
        for operation in ("missing", "extra", "reorder", "arm", "id", "prompt"):
            arms = {a: arm_rows(a) for a in ("base", "adapter")}
            rows = arms["adapter"]
            if operation == "missing":
                rows.pop()
            elif operation == "extra":
                rows.append(rows[0])
            elif operation == "reorder":
                rows.reverse()
            elif operation == "arm":
                rows[0]["arm"] = "base"
            elif operation == "id":
                rows[0]["id"] = "n2"
            else:
                rows[0]["prompt_sha256"] = "0" * 64
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.score(arms)

    def test_forged_hash_and_token_counts_are_rejected(self):
        for key, value in (
            ("output_sha256", "0" * 64),
            ("generated_tokens", True),
            ("generated_tokens", 17),
            ("index", False),
            ("input_tokens", 129),
            ("budget_exhausted", "false"),
        ):
            arms = {a: arm_rows(a) for a in ("base", "adapter")}
            arms["base"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.score(arms)

    def test_tokenization_mismatch_is_not_paired(self):
        arms = {a: arm_rows(a) for a in ("base", "adapter")}
        arms["adapter"][0]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "tokenization"):
            self.score(arms)


class ExecutionTests(unittest.TestCase):
    def setup_run(self, root):
        c, f, t, s = inputs()
        _, _, plan = prepared((c, f, t, s))
        raw = {
            "contract.json": M.encoded(c),
            "fixtures.jsonl": f,
            "training.jsonl": t,
            "heldout_eval_v2.py": s,
        }
        return c, plan, raw

    def fake_success(self, command, **kwargs):
        arm, folder = command[command.index("--worker") + 1], Path(command[-1])
        self.assertGreater(kwargs["timeout"], 0)
        self.assertIs(kwargs["check"], False)
        (folder / (arm + ".jsonl")).write_bytes(
            b"".join(M.encoded(x) for x in arm_rows(arm))
        )
        contract = json.loads((folder / "contract.json").read_bytes())
        runtime = {
            "schema": "szl.heldout-worker/v2",
            "arm": arm,
            "state": "COMPLETE",
            "pid": 100 if arm == "base" else 101,
            "attempt_id": contract["attempt_id"],
            "contract_sha256": M.digest(M.encoded(contract)),
            "environment": {
                "python": "synthetic",
                "torch": "synthetic",
                "transformers": "synthetic",
                "peft": "synthetic",
                "cuda": "synthetic",
                "gpu_name": "synthetic-no-gpu",
                "chat_template_sha256": "f" * 64,
            },
            "runtime_admission_verified": False,
        }
        (folder / (arm + "-runtime.json")).write_bytes(M.encoded(runtime))
        return SimpleNamespace(returncode=0)

    def test_complete_bundle_replays_without_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            calls = []

            def child(command, **kw):
                calls.append(command[command.index("--worker") + 1])
                return self.fake_success(command, **kw)

            code, terminal = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 0)
            self.assertEqual(calls, ["base", "adapter"])
            self.assertEqual(terminal["state"], "MEASUREMENTS_COMPLETE_REPLAY_VERIFIED")
            directory = root / c["attempt_id"]
            with patch.object(
                subprocess,
                "run",
                side_effect=AssertionError("replay executed a process"),
            ):
                self.assertEqual(
                    M.replay(directory)["quality_qualification"], "NOT_ESTABLISHED"
                )

    def test_nonzero_base_exit_never_runs_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            calls = []

            def child(command, **kw):
                calls.append(command)
                return SimpleNamespace(returncode=23)

            code, terminal = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 7)
            self.assertEqual(len(calls), 1)
            self.assertEqual(terminal["state"], "UNAVAILABLE")
            self.assertFalse((root / c["attempt_id"] / "bundle.json").exists())

    def test_timeout_retains_partial_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)

            def child(command, **kw):
                if "base" in command:
                    return self.fake_success(command, **kw)
                raise subprocess.TimeoutExpired(command, 0.1)

            code, terminal = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 7)
            self.assertEqual(terminal["state"], "PARTIAL")
            self.assertTrue((root / c["attempt_id"] / "base.jsonl").exists())

    def test_existing_attempt_is_preserved_not_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            directory = root / c["attempt_id"]
            directory.mkdir()
            (directory / "original").write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                execute_qualified(c, plan, raw, root, run=self.fake_success)
            self.assertEqual((directory / "original").read_bytes(), b"keep")

    def test_byte_tamper_and_rehashed_false_score_both_fail_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            self.assertEqual(
                execute_qualified(c, plan, raw, root, run=self.fake_success)[0], 0
            )
            directory = root / c["attempt_id"]
            path = directory / "measurements.json"
            original = path.read_bytes()
            changed = json.loads(original)
            changed["quality_qualification"] = "PASS"
            path.write_bytes(M.encoded(changed))
            with self.assertRaisesRegex(ValueError, "bytes_mismatch"):
                M.replay(directory)
            manifest = json.loads((directory / "bundle.json").read_bytes())
            manifest["files"]["measurements.json"] = M.digest(path.read_bytes())
            (directory / "bundle.json").write_bytes(M.encoded(manifest))
            with self.assertRaisesRegex(ValueError, "replayed_measurement"):
                M.replay(directory)

    def test_partial_zero_exit_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            code, terminal = execute_qualified(
                c, plan, raw, root, run=lambda *a, **k: SimpleNamespace(returncode=0)
            )
            self.assertEqual(code, 7)
            self.assertEqual(terminal["state"], "UNAVAILABLE")

    def test_errors_do_not_echo_input_or_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "bad.json"
            path.write_bytes(b'{"PRIVATE_SECRET":')
            result = subprocess.run(
                [
                    sys.executable,
                    str(SOURCE),
                    "--contract",
                    str(path),
                    "--fixtures",
                    str(path),
                    "--training",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 7)
            self.assertNotIn("PRIVATE_SECRET", result.stdout + result.stderr)


class QualificationExecutionTests(unittest.TestCase):
    setup_run = ExecutionTests.setup_run
    fake_success = ExecutionTests.fake_success

    def completed(self, root):
        c, plan, raw = self.setup_run(root)
        code, terminal = execute_qualified(c, plan, raw, root, run=self.fake_success)
        self.assertEqual(code, 0, terminal)
        return root / c["attempt_id"]

    def test_missing_bad_pin_and_changed_identity_have_no_execution_effects(self):
        for case in (
            "missing",
            "missing_pin",
            "bad_pin",
            "revision",
            "base",
            "directory",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                parent = Path(folder)
                output = parent / "not-created"
                c, plan, raw = self.setup_run(parent)
                arguments = qualification_input(parent, c)
                if case == "missing":
                    arguments = {}
                elif case == "missing_pin":
                    arguments.pop("qualification_sha256")
                elif case == "bad_pin":
                    arguments["qualification_sha256"] = "0" * 64
                else:
                    if case == "revision":
                        c["adapter"]["revision"] = "f" * 40
                    elif case == "base":
                        c["base"]["repo_id"] = "test/different-base"
                    else:
                        c["adapter"]["directory"] = "checkpoint-200"
                    raw["contract.json"] = M.encoded(c)
                    _, _, plan = M.prepare(
                        raw["contract.json"],
                        raw["fixtures.jsonl"],
                        raw["training.jsonl"],
                        raw["heldout_eval_v2.py"],
                    )
                runner = Mock(side_effect=AssertionError("unqualified process"))
                with self.assertRaises(ValueError):
                    M.execute(c, plan, raw, output, run=runner, **arguments)
                runner.assert_not_called()
                self.assertFalse(output.exists())

    def test_v3_bundle_preserves_original_observation_and_authority_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = self.completed(root)
            manifest = json.loads((directory / "bundle.json").read_bytes())
            binding = json.loads(
                (directory / "qualification-binding.json").read_bytes()
            )
            self.assertEqual(manifest["schema"], "szl.heldout-bundle/v3")
            self.assertEqual(len(manifest["files"]), 16)
            self.assertTrue(M.BINDING_FILES.issubset(manifest["files"]))
            for name in M.QUALIFICATION_NAMES:
                self.assertEqual(
                    (directory / "qualification" / name).read_bytes(),
                    (root / "qualification-input" / name).read_bytes(),
                )
            self.assertEqual(
                binding["qualification_sha256"],
                M.digest(
                    (root / "qualification-input/qualification.json").read_bytes()
                ),
            )
            self.assertTrue(binding["artifact_observation_bound"])
            self.assertEqual(binding["quality_qualification"], "NOT_ESTABLISHED")
            for key in (
                "compute_authorized",
                "promotion_authorized",
                "authenticity_verified",
            ):
                self.assertIs(binding[key], False)

    def test_rehashed_binding_change_fails_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.completed(Path(folder))
            path = directory / "qualification-binding.json"
            value = json.loads(path.read_bytes())
            value["artifact"]["revision"] = "f" * 40
            path.write_bytes(M.encoded(value))
            manifest = json.loads((directory / "bundle.json").read_bytes())
            manifest["files"][path.name] = M.digest(path.read_bytes())
            (directory / "bundle.json").write_bytes(M.encoded(manifest))
            with self.assertRaisesRegex(ValueError, "qualification_binding_mismatch"):
                M.replay(directory)

    def test_legacy_v2_replay_never_requires_or_imports_binding_helper(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.completed(Path(folder))
            manifest = json.loads((directory / "bundle.json").read_bytes())
            manifest["schema"] = "szl.heldout-bundle/v2"
            for name in M.BINDING_FILES:
                manifest["files"].pop(name)
            shutil.rmtree(directory / "qualification")
            (directory / "evaluation_binding.py").unlink()
            (directory / "qualification-binding.json").unlink()
            (directory / "bundle.json").write_bytes(M.encoded(manifest))
            with patch.object(
                M, "binding_api", side_effect=AssertionError("legacy import")
            ):
                result = M.replay(directory)
            self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")

    def test_replay_never_executes_archived_helper(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.completed(Path(folder))
            path = directory / "evaluation_binding.py"
            path.write_bytes(b"raise AssertionError('archived code executed')\n")
            manifest = json.loads((directory / "bundle.json").read_bytes())
            manifest["files"][path.name] = M.digest(path.read_bytes())
            (directory / "bundle.json").write_bytes(M.encoded(manifest))
            result = M.replay(directory)
            self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")

    def test_standalone_worker_rejects_changed_helper_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            calls = []

            def child(command, **kwargs):
                calls.append(command)
                directory = Path(command[-1])
                (directory / "evaluation_binding.py").write_bytes(
                    b"raise AssertionError('unverified helper executed')\n"
                )
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=10, check=False
                )
                self.assertEqual(process.returncode, 7)
                self.assertEqual(
                    json.loads(process.stderr)["reason_code"],
                    "QUALIFICATION_HELPER_HASH_MISMATCH",
                )
                self.assertFalse((directory / "base.jsonl").exists())
                return SimpleNamespace(returncode=process.returncode)

            code, _ = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 7)
            self.assertEqual(len(calls), 1)

    def replace_observation_consistently(self, directory):
        path = directory / "qualification/qualification.json"
        report = json.loads(path.read_bytes())
        report["candidates"][0]["observed_at_utc"] = "2026-01-02T00:00:00+00:00"
        path.write_bytes(M.encoded(report))
        pin = M.digest(path.read_bytes())
        manifest_path = directory / "qualification/manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["sha256"]["qualification.json"] = pin
        manifest_path.write_bytes(M.encoded(manifest))
        contract = json.loads((directory / "contract.json").read_bytes())
        binding = M.binding_api().bind_qualification(
            directory / "qualification", contract, pin
        )
        (directory / "qualification-binding.json").write_bytes(M.encoded(binding))

    def test_standalone_worker_rejects_coherent_replacement_of_selected_observation(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            calls = []

            def child(command, **kwargs):
                calls.append(command)
                directory = Path(command[-1])
                self.replace_observation_consistently(directory)
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=10, check=False
                )
                self.assertEqual(process.returncode, 7)
                self.assertEqual(
                    json.loads(process.stderr)["reason_code"],
                    "QUALIFICATION_BINDING_PIN_MISMATCH",
                )
                self.assertFalse((directory / "base.jsonl").exists())
                return SimpleNamespace(returncode=process.returncode)

            code, _ = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 7)
            self.assertEqual(len(calls), 1)

    def test_completion_rechecks_selected_observation_before_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)

            def child(command, **kwargs):
                result = self.fake_success(command, **kwargs)
                if command[command.index("--worker") + 1] == "adapter":
                    self.replace_observation_consistently(Path(command[-1]))
                return result

            code, terminal = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 7)
            self.assertEqual(
                terminal["reason_code"], "QUALIFICATION_BINDING_PIN_MISMATCH"
            )
            self.assertFalse((root / c["attempt_id"] / "bundle.json").exists())

    def test_cli_binding_holds_have_fixed_codes_and_no_private_output(self):
        for case, expected in (
            ("missing", "QUALIFICATION_ARGUMENTS_REQUIRED"),
            ("bad_pin", "QUALIFICATION_PIN_MISMATCH"),
            ("revision", "QUALIFICATION_ADAPTER_REVISION_MISMATCH"),
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                c, f, t, _ = inputs()
                arguments = qualification_input(root, c)
                if case == "revision":
                    c["adapter"]["revision"] = "f" * 40
                for name, value in (
                    ("c.json", M.encoded(c)),
                    ("f.jsonl", f),
                    ("t.jsonl", t),
                ):
                    (root / name).write_bytes(value)
                output = root / "PRIVATE_OUTPUT_NEVER_CREATED"
                command = [
                    sys.executable,
                    str(SOURCE),
                    "--contract",
                    str(root / "c.json"),
                    "--fixtures",
                    str(root / "f.jsonl"),
                    "--training",
                    str(root / "t.jsonl"),
                    "--execute",
                    "--output-root",
                    str(output),
                ]
                if case != "missing":
                    command += [
                        "--qualification-bundle",
                        str(arguments["qualification_bundle"]),
                        "--qualification-sha256",
                        "0" * 64
                        if case == "bad_pin"
                        else arguments["qualification_sha256"],
                    ]
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=10, check=False
                )
                self.assertEqual(process.returncode, 7)
                self.assertEqual(json.loads(process.stderr)["reason_code"], expected)
                self.assertNotIn("PRIVATE_OUTPUT", process.stdout + process.stderr)
                self.assertFalse(output.exists())


class RuntimeBoundaryTests(unittest.TestCase):
    setup_run = ExecutionTests.setup_run
    fake_success = ExecutionTests.fake_success

    """Real process transport plus model-stack doubles, never GPU qualification."""

    def test_runtime_drift_blocks_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)

            def child(command, **kwargs):
                result = self.fake_success(command, **kwargs)
                if command[command.index("--worker") + 1] == "adapter":
                    path = Path(command[-1]) / "adapter-runtime.json"
                    value = json.loads(path.read_bytes())
                    value["environment"]["transformers"] = "different-runtime"
                    path.write_bytes(M.encoded(value))
                return result

            code, terminal = execute_qualified(c, plan, raw, root, run=child)
            self.assertEqual(code, 7)
            self.assertEqual(terminal["state"], "PARTIAL")
            self.assertFalse((root / c["attempt_id"] / "bundle.json").exists())

    def test_worker_failure_diagnostic_survives_without_raw_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, plan, raw = self.setup_run(root)
            directory = root / c["attempt_id"]
            directory.mkdir()
            for name, value in raw.items():
                (directory / name).write_bytes(value)
            (directory / "plan.json").write_bytes(M.encoded(plan))
            archive_qualification(directory, c)
            with patch.object(
                M, "worker", side_effect=RuntimeError("PRIVATE_TOKEN PRIVATE_PROMPT")
            ):
                with self.assertRaises(RuntimeError):
                    M.run_worker(
                        directory,
                        "base",
                        binding_source_sha256=binding_source_pin(),
                        binding_sha256=archived_binding_pin(directory),
                    )
            diagnostic = (directory / "base-failure.json").read_bytes()
            self.assertNotIn(b"PRIVATE", diagnostic)
            self.assertEqual(json.loads(diagnostic)["error_type"], "RuntimeError")

    def test_real_child_processes_are_separate_and_replayable(self):
        # These are tiny synthetic workers; they test process boundaries only.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = {a: arm_rows(a) for a in ("base", "adapter")}
            script = """import argparse,hashlib,json,os,pathlib
p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--bundle');p.add_argument('--binding-source-sha256');p.add_argument('--binding-sha256');a=p.parse_args()
r=pathlib.Path(a.bundle)
def enc(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\\n').encode()
c=json.loads((r/'contract.json').read_bytes())
rows=ROWS
(r/(a.worker+'.jsonl')).write_bytes(b''.join(enc(x) for x in rows[a.worker]))
e={'python':'synthetic','torch':'synthetic','transformers':'synthetic','peft':'synthetic','cuda':'synthetic','gpu_name':'synthetic-no-gpu','chat_template_sha256':'f'*64}
w={'schema':'szl.heldout-worker/v2','arm':a.worker,'state':'COMPLETE','pid':os.getpid(),'attempt_id':c['attempt_id'],'contract_sha256':hashlib.sha256(enc(c)).hexdigest(),'environment':e,'runtime_admission_verified':False}
(r/(a.worker+'-runtime.json')).write_bytes(enc(w))
""".replace("ROWS", repr(rows)).encode()
            c, f, t, s = inputs(script)
            _, _, plan = prepared((c, f, t, s))
            raw = {
                "contract.json": M.encoded(c),
                "fixtures.jsonl": f,
                "training.jsonl": t,
                "heldout_eval_v2.py": s,
            }
            code, terminal = execute_qualified(c, plan, raw, root)
            self.assertEqual(code, 0, terminal)
            directory = root / c["attempt_id"]
            base_pid = json.loads((directory / "base-runtime.json").read_bytes())["pid"]
            adapter_pid = json.loads((directory / "adapter-runtime.json").read_bytes())[
                "pid"
            ]
            self.assertNotEqual(base_pid, adapter_pid)
            self.assertEqual(
                M.replay(directory)["quality_qualification"], "NOT_ESTABLISHED"
            )

    def test_real_timeout_kills_direct_child_and_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = b"import time\ntime.sleep(10)\n"
            c, f, t, s = inputs(source)
            c["generation"]["timeout_seconds"] = 1
            _, _, plan = prepared((c, f, t, s))
            raw = {
                "contract.json": M.encoded(c),
                "fixtures.jsonl": f,
                "training.jsonl": t,
                "heldout_eval_v2.py": s,
            }
            code, terminal = execute_qualified(c, plan, raw, root)
            self.assertEqual(code, 7)
            self.assertEqual(terminal["error_type"], "TimeoutExpired")
            self.assertFalse((root / c["attempt_id"] / "adapter.jsonl").exists())

    def invoke_worker_double(
        self, directory, arm, *, injected=False, overlong=False, tampered_adapter=None
    ):
        from contextlib import nullcontext

        calls = {"adapter": 0, "generated": 0}
        test = self

        class Tokens:
            shape = (1, 129 if overlong else 10)

            def to(self, device):
                test.assertEqual(device, "cuda:0")
                return self

        class Outputs:
            def __getitem__(self, item):
                test.assertEqual(item, (0, slice(10, None)))
                return [2, 9]

        class Tokenizer:
            eos_token_id = 9
            pad_token_id = None
            chat_template = "synthetic-template"

            def get_chat_template(self):
                return self.chat_template

            def apply_chat_template(self, prompt, **kwargs):
                test.assertIs(kwargs["return_dict"], False)
                test.assertIs(kwargs["truncation"], False)
                test.assertIs(kwargs["add_generation_prompt"], True)
                return Tokens()

            def decode(self, ids, **kwargs):
                test.assertEqual(ids, [2, 9])
                return "B"

        class Model:
            device = "cuda:0"
            _hf_peft_config_loaded = injected

            def eval(self):
                return self

            def generate(self, **kwargs):
                calls["generated"] += 1
                test.assertIs(kwargs["do_sample"], False)
                test.assertEqual(kwargs["num_beams"], 1)
                test.assertEqual(kwargs["num_return_sequences"], 1)
                test.assertEqual(kwargs["eos_token_id"], 9)
                return Outputs()

        def load_base(repo, **kwargs):
            if tampered_adapter is not None:
                self.fail("changed adapter bytes reached a model loader")
            test.assertEqual(repo, "test/base")
            test.assertEqual(kwargs["revision"], "a" * 40)
            test.assertIs(kwargs["trust_remote_code"], False)
            return Model()

        def load_adapter(model, directory, **kwargs):
            calls["adapter"] += 1
            test.assertIs(kwargs["is_trainable"], False)
            return model

        torch = SimpleNamespace(
            __version__="synthetic",
            bfloat16="bf16",
            manual_seed=lambda _: None,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                is_bf16_supported=lambda: True,
                manual_seed_all=lambda _: None,
                get_device_name=lambda _: "synthetic-no-gpu",
            ),
            version=SimpleNamespace(cuda="synthetic"),
            inference_mode=nullcontext,
            ones_like=lambda ids: ids,
        )
        transformers = SimpleNamespace(
            __version__="synthetic",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_base),
        )
        peft = (
            SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load_adapter))
            if arm == "adapter"
            else None
        )
        snapshot = directory / "snapshot"
        final = snapshot / "final"
        final.mkdir(parents=True)
        for name, value in ADAPTER_BYTES.items():
            if name == tampered_adapter:
                value = bytes([value[0] ^ 1]) + value[1:]
            (final / name).write_bytes(value)

        def download(repo, **kwargs):
            test.assertEqual(repo, "SZLHOLDINGS/test-adapter")
            test.assertEqual(kwargs["revision"], "c" * 40)
            test.assertEqual(
                kwargs["allow_patterns"], ["final/" + name for name in M.ADAPTER_NAMES]
            )
            return str(snapshot)

        hub = SimpleNamespace(snapshot_download=download)
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "transformers": transformers,
                    "peft": peft,
                    "huggingface_hub": hub,
                },
            ),
            patch.object(M, "version", return_value="synthetic"),
        ):
            result = M.worker(
                directory,
                arm,
                binding_source_sha256=binding_source_pin(),
                binding_sha256=archived_binding_pin(directory),
            )
        return result, calls

    def claimed_directory(self, root):
        c, plan, raw = self.setup_run(root)
        directory = root / c["attempt_id"]
        directory.mkdir()
        for name, value in raw.items():
            (directory / name).write_bytes(value)
        (directory / "plan.json").write_bytes(M.encoded(plan))
        archive_qualification(directory, c)
        return directory

    def test_production_worker_never_loads_adapter_in_base_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.claimed_directory(Path(folder))
            result, calls = self.invoke_worker_double(directory, "base")
            self.assertEqual(calls, {"adapter": 0, "generated": 3})
            self.assertEqual(result["state"], "COMPLETE")
            self.assertTrue((directory / "base.jsonl").is_file())

    def test_production_adapter_worker_attaches_exactly_once(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.claimed_directory(Path(folder))
            _, calls = self.invoke_worker_double(directory, "adapter")
            self.assertEqual(calls, {"adapter": 1, "generated": 3})

    def test_base_auto_adapter_injection_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.claimed_directory(Path(folder))
            with self.assertRaisesRegex(ValueError, "already_contains_adapter"):
                self.invoke_worker_double(directory, "base", injected=True)

    def test_production_worker_rejects_oversized_input_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.claimed_directory(Path(folder))
            with self.assertRaisesRegex(ValueError, "no_truncation"):
                self.invoke_worker_double(directory, "base", overlong=True)

    def test_changed_qualified_files_never_reach_model_loading(self):
        for name in M.ADAPTER_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                directory = self.claimed_directory(Path(folder))
                with self.assertRaisesRegex(ValueError, "qualified_adapter_file_hash"):
                    self.invoke_worker_double(
                        directory, "adapter", tampered_adapter=name
                    )
                self.assertFalse((directory / "adapter.jsonl").exists())

    def test_worker_entrypoints_recheck_binding_before_model_imports(self):
        for entry in (M.worker, M.run_worker):
            with (
                self.subTest(entry=entry.__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                directory = self.claimed_directory(Path(folder))
                path = directory / "qualification-binding.json"
                value = json.loads(path.read_bytes())
                value["artifact"]["sha256"] = "0" * 64
                path.write_bytes(M.encoded(value))
                with (
                    patch.dict(
                        sys.modules,
                        {"torch": None, "transformers": None, "huggingface_hub": None},
                    ),
                    self.assertRaisesRegex(
                        ValueError, "qualification_binding_mismatch"
                    ),
                ):
                    entry(
                        directory,
                        "adapter",
                        binding_source_sha256=binding_source_pin(),
                        binding_sha256=archived_binding_pin(directory),
                    )
                self.assertFalse((directory / "adapter.jsonl").exists())
                self.assertFalse((directory / "adapter-failure.json").exists())


class StreamingTrainingTests(unittest.TestCase):
    def test_bytes_and_streamed_file_have_identical_identity_and_coverage(self):
        _, _, training, _ = inputs()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "training.jsonl"
            path.write_bytes(training)
            self.assertEqual(M.inspect_training(training), M.inspect_training(path))
            self.assertEqual(M.training_digest(path), M.digest(training))

    def test_streamed_training_does_not_use_small_bundle_member_limit(self):
        _, _, training, _ = inputs()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "training.jsonl"
            path.write_bytes(training * 10)
            with patch.object(M, "MAX_BYTES", 16):
                result = M.inspect_training(path)
            self.assertEqual(result["rows"], 10)
            self.assertEqual(result["bytes"], len(training) * 10)
            self.assertEqual(result["sha256"], M.digest(training * 10))

    def test_stream_line_file_and_row_limits_fail_closed(self):
        _, _, training, _ = inputs()
        for field, value in (
            ("MAX_TRAINING_BYTES", 1),
            ("MAX_TRAINING_LINE_BYTES", 1),
            ("MAX_TRAINING_ROWS", 0),
        ):
            with (
                self.subTest(field=field),
                patch.object(M, field, value),
                self.assertRaises(ValueError),
            ):
                M.inspect_training(training)
        with self.assertRaises(ValueError):
            M.inspect_training(training + b'{"messages":')

    def test_stream_copy_preserves_every_byte_and_rejects_changed_source(self):
        _, _, training, _ = inputs()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "source.jsonl"
            original.write_bytes(training)
            M.copy_training(original, root / "copy.jsonl", M.digest(training))
            self.assertEqual((root / "copy.jsonl").read_bytes(), training)
            original.write_bytes(training + b"\n")
            with self.assertRaisesRegex(ValueError, "changed_during_snapshot"):
                M.copy_training(original, root / "partial.jsonl", M.digest(training))
            self.assertEqual((root / "partial.jsonl").read_bytes(), training + b"\n")

    def test_empty_training_tool_calls_are_explicitly_normalized(self):
        plain = [
            {"role": "user", "content": "known input"},
            {"role": "assistant", "content": "known output"},
        ]
        enriched = [{**message, "tool_calls": []} for message in plain]
        self.assertEqual(
            M.messages(plain, training=True), M.messages(enriched, training=True)
        )
        for value in (None, [{"type": "function"}], "[]", False):
            bad = [{**plain[0], "tool_calls": value}, plain[1]]
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.messages(bad, training=True)
        with self.assertRaises(ValueError):
            M.messages(enriched)


class PartitionTests(unittest.TestCase):
    def row(self, prompt, answer):
        return M.encoded(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
        )

    def test_partition_preserves_raw_rows_and_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = self.row("seen", "training answer")
            novel = self.row("new", "new answer").replace(b"\n", b"\r\n")
            evaluation = self.row("seen", "different answer") + novel
            (root / "train").write_bytes(train)
            (root / "eval").write_bytes(evaluation)
            result = M.audit_splits(root / "train", root / "eval", root / "out")
            self.assertEqual(result["shared_unique_prompt_identities"], 1)
            self.assertEqual(result["quarantined_eval_rows"], 1)
            self.assertEqual(result["candidate_eval_rows"], 1)
            self.assertEqual((root / "out/candidate-eval.jsonl").read_bytes(), novel)
            self.assertEqual((root / "train").read_bytes(), train)
            self.assertEqual((root / "eval").read_bytes(), evaluation)
            self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")
            self.assertFalse(result["independent_holdout_verified"])

    def test_all_overlapping_rows_produce_empty_unavailable_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").write_bytes(self.row("seen", "x"))
            (root / "eval").write_bytes(self.row("seen", "y") * 2)
            result = M.audit_splits(root / "train", root / "eval", root / "out")
            self.assertEqual(result["candidate_state"], "EMPTY_UNAVAILABLE")
            self.assertEqual(result["candidate_eval_rows"], 0)
            self.assertEqual(result["quarantined_eval_rows"], 2)
            self.assertEqual(result["shared_unique_prompt_identities"], 1)
            self.assertEqual((root / "out/candidate-eval.jsonl").read_bytes(), b"")

    def test_novel_duplicates_remain_visible_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").write_bytes(self.row("training", "x"))
            (root / "eval").write_bytes(self.row("novel", "y") * 2)
            result = M.audit_splits(root / "train", root / "eval", root / "out")
            self.assertEqual(result["state"], "NO_EXACT_OVERLAP_OBSERVED")
            self.assertEqual(result["candidate_eval_rows"], 2)
            self.assertEqual(result["candidate_unique_prompts"], 1)
            with self.assertRaises(FileExistsError):
                M.audit_splits(root / "train", root / "eval", root / "out")

    def test_changed_training_never_mints_successful_partition_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").write_bytes(self.row("training", "x"))
            (root / "eval").write_bytes(self.row("novel", "y"))
            with (
                patch.object(M, "training_digest", return_value="0" * 64),
                self.assertRaisesRegex(ValueError, "changed_during_partition"),
            ):
                M.audit_splits(root / "train", root / "eval", root / "out")
            self.assertFalse((root / "out/split-audit.json").exists())
            self.assertTrue((root / "out/candidate-eval.jsonl").exists())

    def test_audit_cli_reports_overlap_without_running_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").write_bytes(self.row("seen", "x"))
            (root / "eval").write_bytes(self.row("seen", "y"))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SOURCE),
                    "--audit-eval",
                    str(root / "eval"),
                    "--training",
                    str(root / "train"),
                    "--output-root",
                    str(root / "out"),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertEqual(
                json.loads(result.stdout)["state"], "EXACT_OVERLAP_OBSERVED"
            )


if __name__ == "__main__":
    unittest.main()
