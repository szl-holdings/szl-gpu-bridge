"""Offline benchmark contracts. No GPUs, model code, network, or Hub writes."""

import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("SZL_BENCH_SOURCE", ROOT / "tools/salad/bench_eval.py"))
SHA_BASE, SHA_ADAPTER, SHA_DATA, SHA_PARENT, SHA_COMMIT = [c * 40 for c in "abcde"]


class Tensor:
    def __init__(self, length, correct=None):
        self.length = length
        self.shape = (1, length)
        self.correct = length if correct is None else correct

    def __getitem__(self, index):
        return Tensor(max(0, self.length - 1), self.correct)

    def argmax(self, _):
        return self

    def __eq__(self, other):
        return types.SimpleNamespace(sum=lambda: min(self.correct, other.length))


class Encoding(dict):
    def to(self, _):
        return self


class Tokenizer:
    chat_template = "synthetic-test-only"

    def apply_chat_template(self, messages, tokenize=False):
        return int(messages[0]["content"])

    def __call__(self, text, **_):
        return Encoding(input_ids=Tensor(int(text)))


class Model:
    device = "cpu"

    def eval(self):
        return self

    def __call__(self, **kwargs):
        length = kwargs["input_ids"].shape[1]
        return types.SimpleNamespace(
            loss=types.SimpleNamespace(item=lambda: float(length)),
            logits=Tensor(length),
        )


class Fixture:
    def __init__(self, root):
        self.root = root
        self.events = []
        self.uploads = {}
        self.creates = []
        self.downloads = []
        self.fail_upload = False
        self.bad_remote = False
        self.existing = False
        self.fail_pin = False
        self.raw_counter = iter([1000, 3000, 4000, 6000, 7000, 9000])
        self.rows = [
            {"messages": [{"role": "user", "content": "2"}]},
            {"messages": [{"role": "user", "content": "4"}]},
        ]
        self.write_rows()
        (root / "adapter/final").mkdir(parents=True)
        (root / "adapter/final/adapter_model.safetensors").write_bytes(
            b"TEST-DOUBLE-NOT-A-MODEL"
        )

    def write_rows(self):
        for name in ("eval.jsonl", "train.jsonl"):
            (self.root / name).write_text(
                "".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8"
            )

    def repo_info(self, repo_id, repo_type=None, revision=None):
        self.events.append(("repo_info", repo_id, repo_type, revision))
        if self.fail_pin:
            return types.SimpleNamespace(sha=None)
        mapping = {
            "test/base": SHA_BASE,
            "test/adapter": SHA_ADAPTER,
            "test/data": SHA_DATA,
            "test/receipts": SHA_PARENT,
        }
        return types.SimpleNamespace(sha=mapping[repo_id])

    def model_info(self, repo, **_):
        self.events.append(("late_model_info", repo))
        return types.SimpleNamespace(sha=SHA_ADAPTER)

    def download(self, repo_id, filename, repo_type=None, revision=None, **_):
        self.downloads.append((repo_id, filename, repo_type, revision))
        if repo_id == "test/data":
            return str(self.root / filename)
        raw = self.uploads[filename]
        dest = self.root / ("readback_" + Path(filename).name)
        dest.write_bytes(b"corrupted" if self.bad_remote else raw)
        return str(dest)

    def snapshot(self, repo, **kwargs):
        self.events.append(("snapshot", repo, kwargs.get("revision")))
        return str(self.root / "adapter")

    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        if self.fail_upload:
            raise RuntimeError("synthetic upload failure")
        self.uploads[path_in_repo] = path_or_fileobj
        self.events.append(("legacy_upload", kwargs.get("repo_type")))

    def file_exists(self, **kwargs):
        self.events.append(("exists", kwargs["revision"]))
        return self.existing

    def create_commit(self, **kwargs):
        self.creates.append(kwargs)
        if self.fail_upload:
            raise RuntimeError("synthetic upload failure")
        for op in kwargs["operations"]:
            self.uploads[op.path_in_repo] = op.path_or_fileobj
        return types.SimpleNamespace(oid=SHA_COMMIT)

    def modules(self):
        f = self
        hf = types.ModuleType("huggingface_hub")
        hf.HfApi = lambda **_: self
        hf.hf_hub_download = self.download
        hf.snapshot_download = self.snapshot
        hf.CommitOperationAdd = types.SimpleNamespace
        torch = types.ModuleType("torch")
        torch.bfloat16 = "test-only"
        torch.manual_seed = lambda _: None
        torch.no_grad = contextlib.nullcontext
        torch.tensor = lambda x: x
        torch.exp = math.exp
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: Tokenizer()
        )
        transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: Model()
        )
        peft = types.ModuleType("peft")
        peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda base, _: base)
        nvml = types.ModuleType("pynvml")
        nvml.nvmlInit = lambda: None
        nvml.nvmlDeviceGetHandleByIndex = lambda i: i
        nvml.nvmlDeviceGetName = lambda _: "SYNTHETIC-NOT-HARDWARE"
        nvml.nvmlDeviceGetTotalEnergyConsumption = lambda _: next(f.raw_counter)
        return {
            "huggingface_hub": hf,
            "torch": torch,
            "transformers": transformers,
            "peft": peft,
            "pynvml": nvml,
        }


class BenchContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)
        self.modules = self.fixture.modules()
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.net = patch.object(
            socket, "create_connection", side_effect=AssertionError("network forbidden")
        )
        self.net.start()
        self.addCleanup(self.net.stop)
        self.connect = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        )
        self.connect.start()
        self.addCleanup(self.connect.stop)
        spec = importlib.util.spec_from_file_location("isolated_salad_bench", SOURCE)
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.old_cwd)

    def run_main(self, extra=()):
        args = [
            "bench",
            "--base",
            "test/base",
            "--adapter-repo",
            "test/adapter",
            "--dataset",
            "test/data",
            "--eval-file",
            "eval.jsonl",
            "--train-file",
            "train.jsonl",
            "--receipts-repo",
            "test/receipts",
            "--receipt-dir",
            "receipts/test",
            *extra,
        ]
        self.output = io.StringIO()
        with (
            patch.object(sys, "argv", args),
            contextlib.redirect_stdout(self.output),
            contextlib.redirect_stderr(self.output),
        ):
            return self.bench.main()

    def receipt(self):
        return json.loads(
            next(
                v for k, v in self.fixture.uploads.items() if k.endswith("result.json")
            )
        )

    def make_bundle(self, status="OK"):
        # A complete synthetic measurement, not a status-only pseudo-receipt.
        self.assertEqual(self.run_main(), 0)
        rec = self.receipt()
        rec.update(
            status=status,
            run_id="unique",
            receipts_repo="test/receipts",
            receipts_repo_type="model",
            receipt_prefix="receipts/test/unique",
        )
        self.fixture.events.clear()
        self.fixture.creates.clear()
        self.fixture.downloads.clear()
        self.fixture.uploads.clear()
        return self.bench.save_bundle(
            rec, self.root / "bundle", "test/receipts", "model", "receipts/test/unique"
        )

    def test_actual_measurement_converts_millijoules_to_joules(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.receipt()["eval_base"]["energy_j"], 2.0)

    def test_actual_loss_weights_shifted_prediction_targets(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.receipt()["eval_base"]["mean_loss"], 3.5)

    def test_zero_scorable_rows_do_not_produce_perfect_metrics(self):
        self.fixture.rows = [{"messages": []}]
        self.fixture.write_rows()
        self.assertEqual(self.run_main(), 7)
        self.assertFalse(self.fixture.uploads)

    def test_one_token_rows_are_unavailable_not_zero_loss(self):
        self.fixture.rows = [{"messages": [{"role": "user", "content": "1"}]}]
        self.fixture.write_rows()
        self.assertEqual(self.run_main(), 7)
        self.assertFalse(self.fixture.uploads)

    def test_adapter_download_uses_resolved_revision(self):
        self.assertEqual(self.run_main(), 0)
        self.assertIn(("snapshot", "test/adapter", SHA_ADAPTER), self.fixture.events)
        self.assertNotIn(("late_model_info", "test/adapter"), self.fixture.events)

    def test_dataset_downloads_use_one_resolved_revision(self):
        self.assertEqual(self.run_main(), 0)
        reads = [x for x in self.fixture.downloads if x[0] == "test/data"]
        self.assertEqual({x[3] for x in reads}, {SHA_DATA})
        self.assertEqual({x[2] for x in reads}, {"dataset"})

    def test_no_immutable_pin_refuses_before_snapshot(self):
        self.fixture.fail_pin = True
        self.assertEqual(self.run_main(), 7)
        self.assertFalse(any(e[0] == "snapshot" for e in self.fixture.events))

    def test_receipt_and_harness_publish_in_one_parent_bound_commit(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(self.fixture.creates), 1)
        commit = self.fixture.creates[0]
        self.assertEqual(commit["parent_commit"], SHA_PARENT)
        self.assertEqual(commit["repo_type"], "dataset")
        self.assertEqual(len(commit["operations"]), 2)
        self.assertEqual(
            {Path(o.path_in_repo).name for o in commit["operations"]},
            {"result.json", "bench_eval.py"},
        )

    def test_upload_failure_retains_local_results_and_returns_nonzero(self):
        self.fixture.fail_upload = True
        self.assertEqual(self.run_main(), 7)
        receipts = list(self.root.glob("bench-receipts/*/result.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_text())["status"], "OK")
        self.assertNotIn("BENCH_OK", self.output.getvalue())

    def test_bad_remote_readback_never_reports_completion(self):
        self.fixture.bad_remote = True
        self.assertEqual(self.run_main(), 7)
        self.assertNotIn("BENCH_OK", self.output.getvalue())

    def test_readback_is_at_returned_immutable_commit(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(
            {d[3] for d in self.fixture.downloads if d[0] == "test/receipts"},
            {SHA_COMMIT},
        )

    def test_holdout_quality_stays_false(self):
        self.assertEqual(self.run_main(), 0)
        self.assertIs(self.receipt()["heldout_quality_measured"], False)

    def test_non_finite_loss_refuses_publication(self):
        with patch.object(
            Model,
            "__call__",
            return_value=types.SimpleNamespace(
                loss=types.SimpleNamespace(item=lambda: float("nan"))
            ),
        ):
            self.assertEqual(self.run_main(), 7)
        self.assertFalse(self.fixture.uploads)

    def test_negative_loss_refuses_publication(self):
        with patch.object(
            Model,
            "__call__",
            return_value=types.SimpleNamespace(
                loss=types.SimpleNamespace(item=lambda: -1)
            ),
        ):
            self.assertEqual(self.run_main(), 7)
        self.assertFalse(self.fixture.uploads)

    def test_unavailable_counter_does_not_break_loss_measurement(self):
        self.modules["pynvml"].nvmlDeviceGetTotalEnergyConsumption = lambda _: (
            _ for _ in ()
        ).throw(RuntimeError("not supported"))
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.receipt()["eval_base"]["energy_label"], "NOT_MEASURED")
        self.assertIsNone(self.receipt()["eval_base"]["energy_j"])

    def test_counter_reset_is_not_measured(self):
        fields = self.bench.energy_fields(1000, 999)
        self.assertIsNone(fields["energy_j"])
        self.assertEqual(fields["energy_label"], "NOT_MEASURED")

    def test_zero_observed_delta_is_valid(self):
        self.assertEqual(self.bench.energy_fields(1000, 1000)["energy_j"], 0)
        self.assertEqual(
            self.bench.energy_fields(1000, 1000)["energy_label"], "MEASURED"
        )

    def test_absent_counter_is_not_zero(self):
        for start, end in [(None, None), (None, 3), (3, None), (True, 2), (-1, 2)]:
            with self.subTest(start=start, end=end):
                self.assertIsNone(self.bench.energy_fields(start, end)["energy_j"])

    def test_meter_scope_is_not_whole_node(self):
        self.assertIn(
            "not process-isolated or whole-node",
            self.bench.energy_fields(1, 2)["energy_scope"],
        )

    def test_input_and_prediction_token_counts_are_distinct(self):
        self.assertEqual(self.run_main(), 0)
        metrics = self.receipt()["eval_base"]
        self.assertEqual(metrics["tokens"], 6)
        self.assertEqual(metrics["predicted_tokens"], 4)

    def test_skipped_row_coverage_is_explicit(self):
        self.fixture.rows.append({"messages": []})
        self.fixture.write_rows()
        self.assertEqual(self.run_main(), 0)
        metrics = self.receipt()["eval_base"]
        self.assertEqual(
            (metrics["rows_offered"], metrics["rows"], metrics["rows_skipped"]),
            (3, 2, 1),
        )

    def test_tampered_bundle_is_rejected_without_provider_effect(self):
        bundle = self.make_bundle()
        (bundle / "result.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
        self.assertFalse(self.fixture.events)

    def test_existing_remote_conflict_is_never_overwritten(self):
        bundle = self.make_bundle()
        self.fixture.existing = True
        self.fixture.uploads = {
            "receipts/test/unique/result.json": b"different",
            "receipts/test/unique/bench_eval.py": b"different",
        }
        with self.assertRaises(ValueError):
            self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
        self.assertFalse(self.fixture.creates)

    def test_identical_existing_bundle_is_reused_without_write(self):
        bundle = self.make_bundle()
        self.fixture.existing = True
        self.fixture.uploads = {
            "receipts/test/unique/" + name: (bundle / name).read_bytes()
            for name in ("result.json", "bench_eval.py")
        }
        revision = self.bench.publish_bundle(
            bundle, self.fixture, self.fixture.download
        )
        self.assertEqual(revision, SHA_PARENT)
        self.assertFalse(self.fixture.creates)

    def test_error_bundle_is_not_published(self):
        bundle = self.make_bundle("ERROR")
        with self.assertRaises(ValueError):
            self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
        self.assertFalse(self.fixture.creates)

    def test_publication_only_never_imports_training_stack(self):
        bundle = self.make_bundle()
        # Importing the training stack is unnecessary for a receipt retry.
        with patch.dict(
            sys.modules, {"torch": None, "transformers": None, "peft": None}
        ):
            self.assertEqual(self.run_main(("--publish-only", str(bundle))), 0)
        self.assertEqual(len(self.fixture.creates), 1)

    def test_bundle_prevents_local_overwrite(self):
        self.make_bundle()
        with self.assertRaises(FileExistsError):
            self.make_bundle()

    def test_nan_cannot_be_serialized_as_qualified_evidence(self):
        with self.assertRaises(ValueError):
            self.bench.save_bundle(
                {"schema": self.bench.SCHEMA, "status": "OK", "loss": float("nan")},
                self.root / "bad",
                "test/receipts",
                "model",
                "receipts/test",
            )

    def test_repository_type_is_explicit_and_consistent(self):
        self.assertEqual(self.run_main(("--receipts-repo-type", "model")), 0)
        self.assertEqual(self.fixture.creates[0]["repo_type"], "model")
        self.assertEqual(
            {d[2] for d in self.fixture.downloads if d[0] == "test/receipts"},
            {"model"},
        )

    def test_path_traversal_in_bundle_is_rejected(self):
        bundle = self.make_bundle()
        path = bundle / "bundle.json"
        manifest = json.loads(path.read_text())
        manifest["prefix"] = "receipts/../evil"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
        self.assertFalse(self.fixture.events)

    def test_no_snapshot_download_in_publication_only(self):
        bundle = self.make_bundle()
        self.assertEqual(self.run_main(("--publish-only", str(bundle))), 0)
        self.assertFalse(any(e[0] == "snapshot" for e in self.fixture.events))

    def test_partial_remote_bundle_is_not_completed_by_overwrite(self):
        bundle = self.make_bundle()
        with patch.object(self.fixture, "file_exists", side_effect=[True, False]):
            with self.assertRaises(ValueError):
                self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
        self.assertFalse(self.fixture.creates)

    def test_raw_nvml_counter_cannot_be_coerced_into_measurement(self):
        for raw in (True, "2000", 2.5, -1, None, float("nan")):
            with self.subTest(raw=repr(raw)):
                self.modules["pynvml"].nvmlDeviceGetTotalEnergyConsumption = (
                    lambda _, value=raw: value
                )
                meter = self.bench.Energy()
                self.assertIsNone(meter.read())
                self.assertFalse(meter.ok)

    def test_failed_parent_commit_does_not_retry_or_overwrite(self):
        bundle = self.make_bundle()
        with patch.object(
            self.fixture,
            "create_commit",
            side_effect=RuntimeError("synthetic parent conflict"),
        ) as create:
            with self.assertRaises(RuntimeError):
                self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
            self.assertEqual(create.call_count, 1)
        self.assertTrue((bundle / "result.json").is_file())

    def test_wall_clock_rollback_cannot_make_elapsed_duration_negative(self):
        with patch.object(self.bench.time, "time", side_effect=[1000.0, -1000.0]):
            self.assertEqual(self.run_main(), 0)
        self.assertGreaterEqual(self.receipt()["wall_seconds"], 0)

    def test_default_receipt_target_preserves_admitted_dataset_policy(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.fixture.creates[0]["repo_type"], "dataset")
        self.assertEqual(self.receipt()["receipts_repo_type"], "dataset")

    def test_missing_typed_receipt_target_fails_before_model_hydration(self):
        original = self.fixture.repo_info

        def read(repo_id, repo_type=None, revision=None):
            if repo_id == "test/receipts":
                raise FileNotFoundError("synthetic wrong repository type")
            return original(repo_id=repo_id, repo_type=repo_type, revision=revision)

        with patch.object(self.fixture, "repo_info", side_effect=read):
            self.assertEqual(self.run_main(), 7)
        self.assertFalse(any(x[0] == "snapshot" for x in self.fixture.events))
        self.assertFalse(self.fixture.creates)

    def rewrite_result(self, bundle, change):
        import hashlib

        path = bundle / "result.json"
        result = json.loads(path.read_text())
        change(result)
        raw = json.dumps(result).encode()
        path.write_bytes(raw)
        manifest_path = bundle / "bundle.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["result.json"] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def assert_no_publication(self, bundle):
        with self.assertRaises(ValueError):
            self.bench.publish_bundle(bundle, self.fixture, self.fixture.download)
        self.assertFalse(self.fixture.events)
        self.assertFalse(self.fixture.creates)

    def test_status_only_bundle_cannot_be_published(self):
        bundle = self.make_bundle()
        self.rewrite_result(bundle, lambda r: r.pop("eval_base"))
        self.assert_no_publication(bundle)

    def test_manifest_cannot_retarget_preserved_receipt(self):
        bundle = self.make_bundle()
        path = bundle / "bundle.json"
        manifest = json.loads(path.read_text())
        manifest["repo_type"] = "dataset"
        path.write_text(json.dumps(manifest))
        self.assert_no_publication(bundle)

    def test_manifest_cannot_move_attempt_prefix(self):
        bundle = self.make_bundle()
        path = bundle / "bundle.json"
        manifest = json.loads(path.read_text())
        manifest["prefix"] = "receipts/different/unique"
        path.write_text(json.dumps(manifest))
        self.assert_no_publication(bundle)

    def test_rehashed_false_metric_is_not_qualified(self):
        bundle = self.make_bundle()
        self.rewrite_result(
            bundle, lambda r: r["eval_base"].update(predicted_tokens=999)
        )
        self.assert_no_publication(bundle)

    def test_rehashed_energy_unit_error_is_rejected(self):
        bundle = self.make_bundle()
        self.rewrite_result(bundle, lambda r: r["eval_base"].update(energy_j=2000.0))
        self.assert_no_publication(bundle)

    def test_rehashed_heldout_upgrade_is_rejected(self):
        bundle = self.make_bundle()
        self.rewrite_result(bundle, lambda r: r.update(heldout_quality_measured=True))
        self.assert_no_publication(bundle)

    def test_rehashed_nonfinite_metric_is_rejected(self):
        bundle = self.make_bundle()
        self.rewrite_result(
            bundle, lambda r: r["eval_base"].update(mean_loss=float("nan"))
        )
        self.assert_no_publication(bundle)

    def test_duplicate_json_keys_are_rejected(self):
        bundle = self.make_bundle()
        path = bundle / "bundle.json"
        raw = path.read_text()
        path.write_text('{"repo_type":"dataset",' + raw[1:])
        self.assert_no_publication(bundle)

    def test_symlink_bundle_member_is_not_read_or_published(self):
        bundle = self.make_bundle()
        original = Path.is_symlink
        with patch.object(
            Path, "is_symlink", lambda p: p.name == "result.json" or original(p)
        ):
            self.assert_no_publication(bundle)

    def test_existing_bundle_directory_cannot_be_partially_modified(self):
        target = self.root / "preserved"
        target.mkdir()
        (target / "bench_eval.py").write_bytes(b"existing evidence")
        with self.assertRaises(FileExistsError):
            self.bench.save_bundle(
                {"schema": self.bench.SCHEMA, "status": "ERROR"},
                target,
                "test/receipts",
                "dataset",
                "receipts/old",
            )
        self.assertEqual({p.name for p in target.iterdir()}, {"bench_eval.py"})

    def test_saved_bundle_flushes_before_manifest_commit_marker(self):
        events = []
        original = self.bench.os.fsync

        def flush(fd):
            events.append("fsync")
            return original(fd)

        with patch.object(self.bench.os, "fsync", side_effect=flush):
            self.make_bundle()
        self.assertGreaterEqual(len(events), 3)

    def test_bundle_input_read_is_bounded(self):
        bundle = self.make_bundle()
        path = bundle / "bundle.json"
        path.write_text(path.read_text() + " " * (2 * 1024 * 1024))
        self.assert_no_publication(bundle)

    def test_boolean_energy_is_not_a_joule_measurement(self):
        bundle = self.make_bundle()
        self.rewrite_result(
            bundle,
            lambda r: r["eval_base"].update(
                energy_counter_start=0, energy_counter_end=1000, energy_j=True
            ),
        )
        self.assert_no_publication(bundle)

    def test_boolean_offered_total_is_not_a_row_count(self):
        bundle = self.make_bundle()

        def corrupt(r):
            r["rows"]["eval"] = True
            for field in ("eval_base", "eval_adapter"):
                r[field].update(
                    rows=1,
                    rows_offered=1,
                    rows_skipped=0,
                    tokens=4,
                    predicted_tokens=3,
                    correct_tokens=3,
                    token_accuracy=1.0,
                    mean_loss=4.0,
                    perplexity=math.exp(4.0),
                )
            r["criterion_1"].update(eval_loss=4.0, **{"pass": True})
            r["improvement_vs_base"].update(
                eval_loss_base=4.0, eval_loss_adapter=4.0, delta=0.0
            )

        self.rewrite_result(bundle, corrupt)
        self.assert_no_publication(bundle)

    def test_failed_criterion_remains_a_completed_failed_gate(self):
        bundle = self.make_bundle()

        def measured_failure(r):
            r["eval_adapter"].update(mean_loss=6.0, perplexity=math.exp(6.0))
            for row in r["eval_adapter"]["row_evidence"]:
                if row["status"] == "SCORED":
                    row["mean_loss"] = 6.0
            r["criterion_1"].update(eval_loss=6.0, **{"pass": False})
            r["improvement_vs_base"].update(eval_loss_adapter=6.0, delta=-2.5)

        self.rewrite_result(bundle, measured_failure)
        self.assertEqual(
            self.bench.publish_bundle(bundle, self.fixture, self.fixture.download),
            SHA_COMMIT,
        )
        result = self.receipt()
        self.assertIs(result["criterion_1"]["pass"], False)
        self.assertIs(result["heldout_quality_measured"], False)

    def test_controller_bound_attempt_and_source_are_preserved(self):
        run_id = "a" * 32
        self.assertEqual(
            self.run_main(
                (
                    "--run-id",
                    run_id,
                    "--source-revision",
                    "b" * 40,
                    "--launch-contract-sha256",
                    "c" * 64,
                )
            ),
            0,
        )
        result = self.receipt()
        self.assertEqual(result["run_id"], run_id)
        self.assertTrue(result["receipt_prefix"].endswith("/" + run_id))
        self.assertEqual(result["source_revision"], "b" * 40)
        self.assertEqual(result["launch_contract_sha256"], "c" * 64)

    def test_invalid_attempt_is_refused_before_provider_calls(self):
        with self.assertRaises(SystemExit) as error:
            self.run_main(("--run-id", "../other"))
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.fixture.events)

    def test_partial_launch_binding_is_refused_before_provider_calls(self):
        with self.assertRaises(SystemExit) as error:
            self.run_main(("--run-id", "a" * 32, "--source-revision", "b" * 40))
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.fixture.events)

    def test_publication_retry_cannot_relabel_original_attempt(self):
        bundle = self.make_bundle()
        with self.assertRaises(SystemExit):
            self.run_main(("--publish-only", str(bundle), "--run-id", "d" * 32))
        self.assertFalse(self.fixture.events)

    def test_standalone_measurement_does_not_claim_launch_binding(self):
        self.assertEqual(self.run_main(), 0)
        self.assertNotIn("source_revision", self.receipt())
        self.assertNotIn("launch_contract_sha256", self.receipt())

    def test_partial_recovered_launch_binding_is_rejected(self):
        bundle = self.make_bundle()
        self.rewrite_result(bundle, lambda r: r.update(source_revision="b" * 40))
        self.assert_no_publication(bundle)

    def test_attempt_reuse_cannot_overwrite_original_bundle(self):
        run_id = "a" * 32
        self.assertEqual(self.run_main(("--run-id", run_id)), 0)
        original = (self.root / "bench-receipts" / run_id / "result.json").read_bytes()
        self.fixture.events.clear()
        with self.assertRaises(SystemExit) as error:
            self.run_main(("--run-id", run_id))
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.fixture.events)
        self.assertEqual(
            (self.root / "bench-receipts" / run_id / "result.json").read_bytes(),
            original,
        )


if __name__ == "__main__":
    unittest.main()
