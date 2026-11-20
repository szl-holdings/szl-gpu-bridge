"""Zero-sentinel negative controls for nonauthorizing training-wave preflight.

Pure CPU tests only. No provider imports, credentials, model imports, or GPU jobs.
"""

from __future__ import annotations

import copy
import json
import unittest

from tools.salad import training_wave_preflight as s


def _candidate():
    return {
        "model_id": "SZLHOLDINGS/fixture-not-a-real-model",
        "lane_id": "fixture-lane",
        "intent": "train",
        "artifact_kind": "adapter",
        "base_model_id": "Qwen/fixture-base",
        "base_revision": "2" * 40,
        "dataset_id": "SZLHOLDINGS/fixture-dataset",
        "dataset_revision": "3" * 40,
        "recipe_sha256": "4" * 64,
        "trainer_commit_sha": "5" * 40,
        "eval_receipt_sha256": "6" * 64,
        "canary_receipt_sha256": "7" * 64,
        "license_rights_state": "OWNER_REPORTED_REVIEWED",
        "dependencies_confirmed": True,
        "gpu_sku": "synthetic-gpu",
        "min_vram_gib": 24,
        "replicas": 1,
        "max_attempts": 1,
        "max_runtime_minutes": 30,
        "hourly_gpu_price_micro_usd": 2_000_000,
        "checkpoint_interval_steps": 25,
        "checkpoint_target": "SZLHOLDINGS/fixture-checkpoints",
    }


def _plan():
    return {
        "schema": s.SCHEMA,
        "mode": "DRY_RUN",
        "catalog_sha256": "8" * 64,
        "auth_witness_state": "EXTERNALLY_REPORTED_VALID",
        "owner_budget_approved": True,
        "owner_budget_micro_usd": 500_000_000,
        "inflight_lane_ids": [],
        "models": [_candidate()],
    }


class SentinelEvidenceContract(unittest.TestCase):
    def test_zero_git_commit_is_not_a_valid_immutable_reference(self):
        self.assertFalse(s._git_revision("0" * 40))
        self.assertTrue(s._git_revision("2efe4294fe60ad736a90346e421c14b717c8cf2a"))

    def test_zero_sha256_is_not_a_valid_receipt_digest(self):
        self.assertFalse(s._sha("0" * 64))
        self.assertTrue(s._sha("f" * 64))

    def test_all_git_revision_fields_reject_zero_sentinel(self):
        for field in ("base_revision", "dataset_revision", "trainer_commit_sha"):
            row = _candidate()
            row[field] = "0" * 40
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(s.PreflightError, "LINEAGE_INCOMPLETE"),
            ):
                s.evaluate({**_plan(), "models": [row]})

    def test_all_content_digest_fields_reject_zero_sentinel(self):
        for field in (
            "recipe_sha256",
            "eval_receipt_sha256",
            "canary_receipt_sha256",
        ):
            row = _candidate()
            row[field] = "0" * 64
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(s.PreflightError, "LINEAGE_INCOMPLETE"),
            ):
                s.evaluate({**_plan(), "models": [row]})

    def test_catalog_digest_zero_is_invalid(self):
        altered = _plan()
        altered["catalog_sha256"] = "0" * 64
        with self.assertRaisesRegex(s.PreflightError, "INPUT_INVALID"):
            s.evaluate(altered)

    def test_width_and_case_are_still_not_interchangeable(self):
        for field, bad in (
            ("base_revision", "f" * 64),
            ("dataset_revision", "f" * 39),
            ("trainer_commit_sha", "F" * 40),
            ("recipe_sha256", "b" * 40),
            ("canary_receipt_sha256", "A" * 64),
        ):
            case = _candidate()
            case[field] = bad
            with self.subTest(field=field), self.assertRaises(s.PreflightError):
                s.evaluate({**_plan(), "models": [case]})

    def test_valid_nonzero_shapes_still_never_admit_paid_training(self):
        plan = _plan()
        expected = copy.deepcopy(plan)
        before = json.dumps(plan, sort_keys=True)
        decision = s.evaluate(plan)
        self.assertEqual(decision["decision"], "HOLD_NO_COMPUTE")
        self.assertFalse(decision["training_authorized"])
        self.assertFalse(decision["provider_writes_authorized"])
        self.assertFalse(decision["credential_checked"])
        self.assertFalse(decision["source_data_authenticated"])
        self.assertFalse(decision["receipt_signatures_checked"])
        self.assertEqual(plan, expected)
        self.assertEqual(json.dumps(plan, sort_keys=True), before)

    def test_zero_sentinel_does_not_become_a_successful_preflight(self):
        row = _candidate()
        row["base_revision"] = "0" * 40
        with self.assertRaisesRegex(s.PreflightError, "LINEAGE_INCOMPLETE"):
            s.evaluate({**_plan(), "models": [row]})

    def test_nonzero_git_oid_not_evidence_of_repo_existence(self):
        self.assertTrue(s._git_revision("a" * 40))
        decision = s.evaluate(_plan())
        self.assertEqual(decision["decision"], "HOLD_NO_COMPUTE")
        self.assertFalse(decision["model_training_verified"])


if __name__ == "__main__":
    unittest.main()
