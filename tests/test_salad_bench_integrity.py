"""CPU-only evidence tests: no GPU, provider calls, or historical-result changes."""

import hashlib
import json
import types
import unittest
from unittest.mock import patch

import test_salad_bench_contract as contract


class BenchIntegrity(unittest.TestCase):
    def setUp(self):
        self.case = contract.BenchContract()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.bench = self.case.bench
        self.fixture = self.case.fixture
        self.modules = self.case.modules

    def test_base_is_scored_before_mutating_peft_attachment(self):
        events = []

        class MutatingModel(contract.Model):
            attached = False

            def __call__(self, **kwargs):
                events.append("adapter" if self.attached else "base")
                length = kwargs["input_ids"].shape[1]
                return types.SimpleNamespace(
                    loss=types.SimpleNamespace(
                        item=lambda: 1.0 if self.attached else 2.0
                    ),
                    logits=contract.Tensor(length),
                )

        model = MutatingModel()

        def attach(base, _):
            self.assertIs(base, model)
            events.append("attach")
            base.attached = True
            return base

        self.modules["transformers"].AutoModelForCausalLM.from_pretrained = (
            lambda *args, **kwargs: model
        )
        self.modules["peft"].PeftModel.from_pretrained = attach
        self.assertEqual(self.case.run_main(), 0)
        result = self.case.receipt()
        self.assertEqual(events, ["base", "base", "attach"] + ["adapter"] * 4)
        self.assertEqual(result["eval_base"]["mean_loss"], 2.0)
        self.assertEqual(result["eval_adapter"]["mean_loss"], 1.0)
        self.assertEqual(
            result["baseline_isolation"], "BASE_MEASURED_BEFORE_ADAPTER_ATTACHMENT"
        )

    def test_per_item_evidence_binds_inputs_without_exporting_row_content(self):
        marker = "PRIVATE_TEST_FIELD_MUST_NOT_APPEAR_IN_EVIDENCE"
        self.fixture.rows[0]["private_fixture"] = marker
        self.fixture.write_rows()
        self.assertEqual(self.case.run_main(), 0)
        result = self.case.receipt()
        item = result["eval_base"]["row_evidence"][0]
        expected = hashlib.sha256(
            json.dumps(
                self.fixture.rows[0],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(item["row_sha256"], expected)
        self.assertEqual(
            (
                item["index"],
                item["status"],
                item["input_tokens"],
                item["predicted_tokens"],
            ),
            (0, "SCORED", 2, 1),
        )
        self.assertNotIn(marker, json.dumps(result))
        self.assertNotIn(marker, self.case.output.getvalue())

    def test_skip_causes_are_distinct_and_keep_ordered_coverage(self):
        self.fixture.rows += [
            {"messages": []},
            {"messages": [{"role": "user", "content": "1"}]},
        ]
        self.fixture.write_rows()
        self.assertEqual(self.case.run_main(), 0)
        rows = self.case.receipt()["eval_base"]["row_evidence"]
        self.assertEqual([r["index"] for r in rows], [0, 1, 2, 3])
        self.assertEqual(
            [r["status"] for r in rows],
            ["SCORED", "SCORED", "SKIPPED_NO_MESSAGES", "SKIPPED_TOO_SHORT"],
        )

    def test_adapter_bytes_are_bound_separately_from_repository_revision(self):
        self.assertEqual(self.case.run_main(), 0)
        result = self.case.receipt()
        self.assertEqual(
            result["adapter_file_sha256"],
            hashlib.sha256(b"TEST-DOUBLE-NOT-A-MODEL").hexdigest(),
        )
        self.assertEqual(result["adapter_repo_revision"], contract.SHA_ADAPTER)
        self.assertEqual(result["dataset_revision"], contract.SHA_DATA)

    def test_total_energy_covers_transition_gaps_and_all_three_passes(self):
        self.assertEqual(self.case.run_main(), 0)
        result = self.case.receipt()
        total = result["evaluation_energy"]
        self.assertEqual(total["energy_j"], 8.0)
        self.assertEqual(
            sum(result[name]["energy_j"] for name in self.bench.MEASUREMENT_FIELDS),
            6.0,
        )
        self.assertEqual(total["energy_counter_start"], 1000)
        self.assertEqual(total["energy_counter_end"], 9000)
        self.assertIs(total["counter_sequence_monotonic"], True)
        self.assertIs(total["includes_initial_base_hydration"], False)
        self.assertIs(total["includes_adapter_attachment"], True)
        self.assertEqual(
            total["covered_measurements"], list(self.bench.MEASUREMENT_FIELDS)
        )
        self.assertIs(result["heldout_quality_measured"], False)

    def test_reset_between_valid_component_windows_invalidates_total(self):
        self.fixture.raw_counter = iter([1000, 3000, 2000, 4000, 5000, 7000])
        self.assertEqual(self.case.run_main(), 0)
        result = self.case.receipt()
        self.assertTrue(
            all(
                result[name]["energy_label"] == "MEASURED"
                for name in self.bench.MEASUREMENT_FIELDS
            )
        )
        total = result["evaluation_energy"]
        self.assertEqual(total["energy_label"], "NOT_MEASURED")
        self.assertIsNone(total["energy_j"])
        self.assertIs(total["counter_sequence_monotonic"], False)

    def test_missing_final_counter_never_becomes_partial_total_energy(self):
        self.fixture.raw_counter = iter([1000, 3000, 4000, 6000, 7000, None])
        self.assertEqual(self.case.run_main(), 0)
        total = self.case.receipt()["evaluation_energy"]
        self.assertEqual(total["energy_label"], "NOT_MEASURED")
        self.assertIsNone(total["energy_j"])
        self.assertIsNone(total["counter_sequence_monotonic"])

    def test_rehashed_per_item_loss_tampering_refuses_provider_effects(self):
        bundle = self.case.make_bundle()
        self.case.rewrite_result(
            bundle,
            lambda r: r["eval_base"]["row_evidence"][0].update(mean_loss=9000.0),
        )
        self.case.assert_no_publication(bundle)

    def test_equal_aggregate_coverage_cannot_hide_different_adapter_rows(self):
        bundle = self.case.make_bundle()
        self.case.rewrite_result(
            bundle,
            lambda r: r["eval_adapter"]["row_evidence"][0].update(row_sha256="f" * 64),
        )
        self.case.assert_no_publication(bundle)

    def test_rehashed_missing_row_or_boolean_index_refuses_publication(self):
        bundle = self.case.make_bundle()
        original = (bundle / "result.json").read_bytes()
        manifest = (bundle / "bundle.json").read_bytes()
        for change in (
            lambda r: r["eval_base"]["row_evidence"].pop(),
            lambda r: r["eval_base"]["row_evidence"][0].update(index=False),
            lambda r: r["eval_base"]["row_evidence"][0].update(raw_prompt="forbidden"),
        ):
            with self.subTest(change=change):
                (bundle / "result.json").write_bytes(original)
                (bundle / "bundle.json").write_bytes(manifest)
                self.case.rewrite_result(bundle, change)
                self.case.assert_no_publication(bundle)

    def test_complete_profile_cannot_be_stripped_from_new_source_receipt(self):
        bundle = self.case.make_bundle()

        def strip(r):
            for name in (
                "integrity_profile",
                "baseline_isolation",
                "adapter_file_sha256",
                "evaluation_energy",
            ):
                del r[name]
            for name in self.bench.MEASUREMENT_FIELDS:
                del r[name]["row_evidence"]

        self.case.rewrite_result(bundle, strip)
        self.case.assert_no_publication(bundle)

    def test_total_window_cannot_be_replaced_by_sum_of_component_windows(self):
        bundle = self.case.make_bundle()
        self.case.rewrite_result(
            bundle, lambda r: r["evaluation_energy"].update(energy_j=6.0)
        )
        self.case.assert_no_publication(bundle)

    def test_evidence_row_bound_is_checked_before_base_hydration(self):
        with (
            patch.object(self.bench, "MAX_EVIDENCE_ROWS_PER_PASS", 1),
            patch.object(
                self.modules["transformers"].AutoModelForCausalLM,
                "from_pretrained",
                side_effect=AssertionError("base hydration must not occur"),
            ) as hydrate,
        ):
            self.assertEqual(self.case.run_main(), 7)
        hydrate.assert_not_called()
        self.assertFalse(self.fixture.creates)

    def test_result_byte_bound_refuses_storage_before_directory_creation(self):
        result = {
            "schema": self.bench.SCHEMA,
            "status": "ERROR",
            "diagnostic": "x" * 2048,
        }
        target = self.case.root / "oversized"
        with patch.object(self.bench, "MAX_RESULT_FILE_BYTES", 128):
            with self.assertRaises(ValueError):
                self.bench.save_bundle(
                    result, target, "test/receipts", "model", "receipts/oversized"
                )
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
