"""Offline adversarial controls for source-bound portfolio candidate recipes."""

from __future__ import annotations

import copy
import unittest

from tools import portfolio_candidate_recipe as gate
from test_estate_54h_portfolio import NOW, snapshot


def good_candidate() -> dict:
    s = snapshot()
    return {
        "schema": gate.RECIPE_SCHEMA,
        "adapter": s["public_assets"]["models"][0]["repo"],
        "adapter_sha": s["public_assets"]["models"][0]["immutable_hub_sha"],
        "base_model": s["public_assets"]["models"][1]["repo"],
        "base_sha": s["public_assets"]["models"][1]["immutable_hub_sha"],
        "adapter_config_base": s["public_assets"]["models"][1]["repo"],
        "datasets": [
            {
                "id": s["public_assets"]["datasets"][0]["repo"],
                "sha": s["public_assets"]["datasets"][0]["immutable_hub_sha"],
                "role": "train",
            },
            {
                "id": s["public_assets"]["datasets"][1]["repo"],
                "sha": s["public_assets"]["datasets"][1]["immutable_hub_sha"],
                "role": "evaluation",
            },
        ],
        "train_example_sha256": ["a" * 64],
        "heldout_example_sha256": ["b" * 64],
        "max_steps": 120,
        "max_wall_minutes": 54 * 60,
        "evaluation_recipe": "szl-holdout-v1",
        "training_dispatch_enabled": False,
    }


class CandidateReviewTests(unittest.TestCase):
    def deny(self, doc, reason, source=None):
        with self.assertRaisesRegex(gate.CandidateHold, reason):
            gate.review_candidate(
                doc, source if source is not None else snapshot(), now=NOW
            )

    def test_matching_metadata_still_hold_not_admit(self):
        result = gate.review_candidate(good_candidate(), snapshot(), now=NOW)
        self.assertEqual(result["verdict"], "HOLD")
        self.assertFalse(result["training_dispatched"])
        self.assertFalse(result["release_approved"])
        self.assertEqual(result["reward_grants"], 0)
        self.assertIn("RIGHTS_CONSENT_AND_LICENSE_COMPATIBILITY", result["unverified"])

    def test_matching_declared_base_does_not_prove_adapter_architecture(self):
        recipe = good_candidate()
        result = gate.review_candidate(recipe, snapshot(), now=NOW)
        self.assertIn(
            "ADAPTER_BASE_ARCHITECTURE_COMPATIBILITY_UNVERIFIED",
            result["unverified"],
        )
        self.assertEqual(result["verdict"], "HOLD")
        self.assertFalse(result["release_approved"])
        self.assertEqual(result["reward_grants"], 0)

    def test_repo_sha_does_not_prove_lfs_bytes(self):
        result = gate.review_candidate(good_candidate(), snapshot(), now=NOW)
        self.assertIn("ADAPTER_AND_BASE_FILE_BYTES_UNVERIFIED", result["unverified"])
        self.assertFalse(result["training_dispatched"])

    def test_unobserved_adapter_base_rejected(self):
        x = good_candidate()
        x["base_model"] = "SZLHOLDINGS/unobserved-model"
        x["adapter_config_base"] = x["base_model"]
        self.deny(x, "BASE_ID_UNOBSERVED")

    def test_adapter_base_revision_drift_rejected(self):
        x = good_candidate()
        x["base_sha"] = "c" * 40
        self.deny(x, "BASE_REVISION_DRIFT")

    def test_adapter_config_base_mismatch_rejected(self):
        x = good_candidate()
        x["adapter_config_base"] = x["adapter"]
        self.deny(x, "ADAPTER_DECLARED_BASE_MISMATCH")

    def test_base_without_listed_weight_signal_rejected(self):
        s = snapshot()
        s["public_assets"]["models"][1]["artifact_signal"] = "NO_RECOGNIZED_WEIGHT_FILE"
        s["public_assets"]["models"][1]["weight_file_count"] = 0
        self.deny(good_candidate(), "BASE_WEIGHT_SIGNAL_UNQUALIFIED", s)

    def test_missing_train_digest_set_rejected(self):
        x = good_candidate()
        x["train_example_sha256"] = []
        self.deny(x, "TRAIN_EMPTY_OR_OVER_BUDGET")

    def test_missing_holdout_digest_set_rejected(self):
        x = good_candidate()
        x["heldout_example_sha256"] = []
        self.deny(x, "HELDOUT_EMPTY_OR_OVER_BUDGET")

    def test_identical_train_holdout_digest_rejected(self):
        x = good_candidate()
        x["heldout_example_sha256"] = copy.deepcopy(x["train_example_sha256"])
        self.deny(x, "DECLARED_TRAIN_HELDOUT_COLLISION")

    def test_malformed_split_digest_rejected(self):
        x = good_candidate()
        x["heldout_example_sha256"] = ["NOT_A_SHA"]
        self.deny(x, "HELDOUT_DIGEST_INVALID")

    def test_unknown_dataset_rejected(self):
        x = good_candidate()
        x["datasets"][0]["id"] = "SZLHOLDINGS/unknown-dataset"
        self.deny(x, "DATASET_ID_UNOBSERVED")

    def test_unpinned_dataset_revision_rejected(self):
        x = good_candidate()
        x["datasets"][0]["sha"] = "main"
        self.deny(x, "DATASET_SHA_INVALID")

    def test_duplicate_dataset_id_denied(self):
        x = good_candidate()
        x["datasets"][1]["id"] = x["datasets"][0]["id"]
        self.deny(x, "DUPLICATE_DATASET_ID")

    def test_missing_evaluation_dataset_denied(self):
        x = good_candidate()
        x["datasets"][1]["role"] = "train"
        self.deny(x, "TRAIN_AND_EVAL_DATASETS_REQUIRED")

    def test_54h_ceiling_cannot_be_extended(self):
        x = good_candidate()
        x["max_wall_minutes"] = 54 * 60 + 1
        self.deny(x, "WALLTIME_OVER_54H_OR_UNBOUNDED")

    def test_zero_or_unbounded_steps_denied(self):
        for steps in (0, -1, None, 10_000_001):
            x = good_candidate()
            x["max_steps"] = steps
            with self.subTest(steps=steps):
                self.deny(x, "STEP_BOUND_INVALID")

    def test_bool_is_not_an_integer_budget(self):
        x = good_candidate()
        x["max_wall_minutes"] = True
        self.deny(x, "WALLTIME_OVER_54H_OR_UNBOUNDED")

    def test_training_flag_true_denied(self):
        x = good_candidate()
        x["training_dispatch_enabled"] = True
        self.deny(x, "EXECUTION_FLAG_DENIED")

    def test_forger_extra_authorization_field_denied(self):
        x = good_candidate()
        x["owner_signed"] = True
        self.deny(x, "RECIPE_FIELDS")

    def test_snapshot_forged_training_state_denied(self):
        s = snapshot()
        s["public_assets"]["models"][0]["training_admitted"] = True
        self.deny(good_candidate(), "SOURCE_SNAPSHOT_INVALID", s)


if __name__ == "__main__":
    unittest.main()
