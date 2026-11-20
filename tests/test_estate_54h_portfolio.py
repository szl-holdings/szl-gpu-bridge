"""Offline denial tests for the unified multi-model training review plane."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from tools import estate_54h_portfolio as v

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)


def snapshot(number: int = 2) -> dict:
    models = []
    for n in range(number):
        signal = [
            "ADAPTER_FILES_LISTED",
            "WEIGHT_FILES_LISTED",
            "NO_RECOGNIZED_WEIGHT_FILE",
        ][n % 3]
        models.append(
            {
                "repo": f"SZLHOLDINGS/synthetic-model-{n}",
                "immutable_hub_sha": f"{n + 1:040x}",
                "artifact_signal": signal,
                "weight_file_count": 0 if signal == "NO_RECOGNIZED_WEIGHT_FILE" else 1,
                "license_tags": ["apache-2.0"],
                "rights_review": "NOT_PERFORMED",
                "training_admitted": False,
            }
        )
    return {
        "schema": v.SCHEMA,
        "observed_at_utc": (NOW - timedelta(hours=1)).isoformat(),
        "scope": "Hugging Face anonymous public author search; NOT authenticated private inventory",
        "counts": {"models": len(models), "datasets": 2},
        "public_assets": {
            "models": models,
            "datasets": [
                {
                    "repo": "SZLHOLDINGS/synthetic-data-a",
                    "immutable_hub_sha": "a" * 40,
                    "declared_license": "apache-2.0",
                    "rights_review": "NOT_PERFORMED",
                    "holdout_verified": False,
                    "training_admitted": False,
                },
                {
                    "repo": "SZLHOLDINGS/synthetic-data-b",
                    "immutable_hub_sha": "b" * 40,
                    "declared_license": "other",
                    "rights_review": "NOT_PERFORMED",
                    "holdout_verified": False,
                    "training_admitted": False,
                },
            ],
        },
        "provider_write_operations": 0,
        "training_jobs_started": 0,
        "all_training_admitted": False,
        "rights_clearance": "UNVERIFIED",
    }


class EstatePortfolioTests(unittest.TestCase):
    def deny(self, doc: dict, reason: str):
        with self.assertRaisesRegex(v.Hold, reason):
            v.review_portfolio(doc, now=NOW)

    def test_every_candidate_stays_held(self):
        report = v.review_portfolio(snapshot(), now=NOW)
        self.assertEqual(report["status"], "SOURCE_REVIEW_ONLY_ALL_HELD")
        self.assertEqual(report["portfolio_model_count"], 2)
        self.assertEqual(report["requested_hours_per_model"], 54)
        self.assertEqual(report["training_jobs_started"], 0)
        self.assertEqual(report["provider_writes"], 0)
        self.assertEqual(report["reward_grants"], 0)
        for candidate in report["review_entries"]:
            self.assertEqual(candidate["verdict"], "HOLD")
            self.assertFalse(candidate["dispatched"])
            self.assertEqual(candidate["duration_state"], "REQUESTED_NOT_AUTHORIZED")
            self.assertEqual(
                candidate["reward_state"], "UNAVAILABLE_NO_INDEPENDENT_EVAL"
            )
            self.assertIn(
                "TRAINER_OWNER_SIGNER_LEASE_UNVERIFIED", candidate["reason_codes"]
            )

    def test_large_portfolio_metadata_not_trained(self):
        report = v.review_portfolio(snapshot(401), now=NOW)
        self.assertEqual(len(report["review_entries"]), 401)
        self.assertTrue(all(not row["dispatched"] for row in report["review_entries"]))
        self.assertEqual(report["model_promotions"], 0)

    def test_adapter_needs_base_identity(self):
        report = v.review_portfolio(snapshot(1), now=NOW)
        self.assertIn(
            "ADAPTER_BASE_IDENTITY_UNVERIFIED",
            report["review_entries"][0]["reason_codes"],
        )

    def test_metadata_only_never_promoted_to_weights(self):
        report = v.review_portfolio(snapshot(3), now=NOW)
        self.assertIn(
            "NO_RECOGNIZED_WEIGHT_FILE", report["review_entries"][2]["reason_codes"]
        )

    def test_rights_cannot_be_approved_from_hf_card(self):
        p = snapshot()
        p["public_assets"]["datasets"][0]["rights_review"] = "APPROVED"
        self.deny(p, "PUBLIC_METADATA_NOT_RIGHTS_APPROVAL")

    def test_explicit_training_enabled_claim_denied(self):
        p = snapshot()
        p["public_assets"]["models"][0]["training_admitted"] = True
        self.deny(p, "PUBLIC_METADATA_NOT_RIGHTS_APPROVAL")

    def test_claimed_holdout_safe_from_public_listing_denied(self):
        p = snapshot()
        p["public_assets"]["datasets"][0]["holdout_verified"] = True
        self.deny(p, "PUBLIC_METADATA_NOT_HOLDOUT_APPROVAL")

    def test_nonpublic_repo_id_denied(self):
        p = snapshot()
        p["public_assets"]["models"][0]["repo"] = "AnotherOrg/private"
        self.deny(p, "ASSET_NAMESPACE")

    def test_unpinned_revision_denied(self):
        p = snapshot()
        p["public_assets"]["models"][0]["immutable_hub_sha"] = "main"
        self.deny(p, "ASSET_SHA_UNVERIFIED")

    def test_placeholder_revision_is_not_an_immutable_hub_commit(self):
        for kind in ("models", "datasets"):
            p = snapshot()
            p["public_assets"][kind][0]["immutable_hub_sha"] = "0" * 40
            with self.subTest(kind=kind):
                self.deny(p, "ASSET_SHA_UNVERIFIED")

    def test_duplicate_model_denied(self):
        p = snapshot()
        p["public_assets"]["models"][1]["repo"] = p["public_assets"]["models"][0][
            "repo"
        ]
        self.deny(p, "DUPLICATE_ASSET_ID")

    def test_count_drift_denied(self):
        p = snapshot()
        p["counts"]["models"] = 400
        self.deny(p, "ASSET_COUNT_CONTRADICTED")

    def test_contradictory_weight_signal_denied(self):
        p = snapshot()
        p["public_assets"]["models"][0]["weight_file_count"] = 0
        self.deny(p, "MODEL_SIGNAL_CONTRADICTED")

    def test_unsafe_extra_model_field_denied(self):
        p = snapshot()
        p["public_assets"]["models"][0]["approved_compute"] = True
        self.deny(p, "MODEL_METADATA_SCHEMA")

    def test_stale_and_future_observations_denied(self):
        for timestamp in ("2025-01-01T00:00:00+00:00", "2026-11-21T00:00:00+00:00"):
            p = snapshot()
            p["observed_at_utc"] = timestamp
            with self.subTest(timestamp=timestamp):
                self.deny(p, "SNAPSHOT_STALE_OR_FUTURE")

    def test_duplicate_json_key_denied_before_plan(self):
        with self.assertRaisesRegex(v.Hold, "DUPLICATE_JSON_KEY"):
            v.parse_public_snapshot(b'{"schema":"one","schema":"two"}')

    def test_model_labeled_other_does_not_resolve_dataset_rights(self):
        p = snapshot()
        for row in p["public_assets"]["datasets"]:
            row["declared_license"] = "apache-2.0"
        report = v.review_portfolio(p, now=NOW)
        self.assertEqual(report["datasets_without_specific_license_declaration"], 0)
        self.assertFalse(report["all_dataset_uses_approved"])
        self.assertTrue(
            all(row["verdict"] == "HOLD" for row in report["review_entries"])
        )

    def test_unrecognized_format_cannot_be_assumed_trainable(self):
        p = snapshot(1)
        p["public_assets"]["models"][0]["artifact_signal"] = "NO_RECOGNIZED_WEIGHT_FILE"
        p["public_assets"]["models"][0]["weight_file_count"] = 0
        p["public_assets"]["models"][0]["license_tags"] = []
        review = v.review_portfolio(p, now=NOW)
        self.assertEqual(review["review_entries"][0]["verdict"], "HOLD")
        self.assertIn(
            "NO_RECOGNIZED_WEIGHT_FILE", review["review_entries"][0]["reason_codes"]
        )
        self.assertEqual(review["training_jobs_started"], 0)
        self.assertEqual(review["reward_grants"], 0)

    def test_changed_well_formed_sha_is_not_provider_verification(self):
        p = snapshot()
        p["public_assets"]["models"][0]["immutable_hub_sha"] = "f" * 40
        review = v.review_portfolio(p, now=NOW)
        self.assertEqual(len(review["review_entries"]), p["counts"]["models"])
        self.assertTrue(
            all(entry["verdict"] == "HOLD" for entry in review["review_entries"])
        )
        self.assertEqual(review["reward_grants"], 0)
        self.assertEqual(review["provider_writes"], 0)

    def test_exclusive_report_write_does_not_overwrite(self):
        report = v.review_portfolio(snapshot(), now=NOW)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "receipt.json"
            digest = v.write_exclusive(p, report)
            self.assertEqual(len(digest), 64)
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                v.write_exclusive(p, report)
            self.assertEqual(original, p.read_bytes())

    def test_nominal_gpu_hour_math_does_not_admit_training(self):
        report = v.review_portfolio(snapshot(401), now=NOW)
        self.assertEqual(
            report["nominal_aggregate_single_gpu_hours_unqualified"], 21654
        )
        self.assertEqual(report["training_jobs_started"], 0)
        self.assertFalse(report["all_models_admitted"])
        scenario = v.hypothetical_single_gpu_capacity(401, 20)
        self.assertEqual(scenario["nominal_aggregate_single_gpu_hours"], 21654)
        self.assertEqual(scenario["hypothetical_capacity_gpu_hours"], 1080)
        self.assertEqual(scenario["nominal_uncovered_gpu_hours"], 20574)
        self.assertEqual(
            scenario["minimum_preemptible_aggregate_lower_bound_hours"], 1083
        )
        self.assertEqual(scenario["minimum_complete_job_wallclock_hours"], 1134)
        self.assertEqual(scenario["max_complete_indivisible_jobs_in_window"], 20)
        self.assertEqual(scenario["unserved_indivisible_job_count"], 381)
        self.assertFalse(scenario["possible_within_window_under_one_gpu_assumption"])
        self.assertFalse(scenario["provider_capacity_verified"])
        self.assertFalse(scenario["spend_limit_verified"])
        self.assertFalse(scenario["training_authorized"])
        self.assertFalse(scenario["training_quality_verified"])

    def test_hypothetical_single_gpu_math_rounding_and_zero(self):
        self.assertEqual(
            v.hypothetical_single_gpu_capacity(2, 1)[
                "minimum_complete_job_wallclock_hours"
            ],
            108,
        )
        zero = v.hypothetical_single_gpu_capacity(0, 1)
        self.assertEqual(zero["nominal_aggregate_single_gpu_hours"], 0)
        self.assertTrue(zero["possible_within_window_under_one_gpu_assumption"])
        self.assertFalse(zero["training_authorized"])

    def test_aggregate_gpu_hours_cannot_overclaim_indivisible_jobs(self):
        # 3 x 54-hour jobs demand 162 GPU-hours, exactly the aggregate
        # capacity of two slots x 81h. Yet each slot can finish only one.
        s = v.hypothetical_single_gpu_capacity(3, 2, window_hours=81)
        self.assertEqual(s["nominal_aggregate_single_gpu_hours"], 162)
        self.assertEqual(s["hypothetical_capacity_gpu_hours"], 162)
        self.assertEqual(s["minimum_preemptible_aggregate_lower_bound_hours"], 81)
        self.assertEqual(s["minimum_complete_job_wallclock_hours"], 108)
        self.assertEqual(s["max_complete_indivisible_jobs_in_window"], 2)
        self.assertEqual(s["unserved_indivisible_job_count"], 1)
        self.assertFalse(s["possible_within_window_under_one_gpu_assumption"])
        # Unlike the 81h counterexample, 108h does fit 2 then 1 full jobs.
        enough = v.hypothetical_single_gpu_capacity(3, 2, window_hours=108)
        self.assertTrue(enough["possible_within_window_under_one_gpu_assumption"])
        self.assertEqual(enough["unserved_indivisible_job_count"], 0)
        self.assertFalse(enough["provider_capacity_verified"])
        self.assertFalse(enough["training_authorized"])

    def test_hypothetical_slots_and_hours_invalid(self):
        for slots in (False, True, 0, -1, 1.5, "20", 1001):
            with self.subTest(slots=slots):
                with self.assertRaisesRegex(v.Hold, "HYPOTHETICAL_SLOTS_INVALID"):
                    v.hypothetical_single_gpu_capacity(401, slots)
        for count in (-1, True, 1001):
            with self.subTest(count=count):
                with self.assertRaisesRegex(v.Hold, "MODEL_COUNT_INVALID"):
                    v.hypothetical_single_gpu_capacity(count, 20)
        for hours in (0, True, 8761):
            with self.subTest(hours=hours):
                with self.assertRaisesRegex(v.Hold, "HYPOTHETICAL_WINDOW_INVALID"):
                    v.hypothetical_single_gpu_capacity(401, 20, window_hours=hours)

    def test_scenario_is_deterministic_and_never_signs(self):
        import json

        a = v.hypothetical_single_gpu_capacity(401, 20)
        b = v.hypothetical_single_gpu_capacity(401, 20)
        self.assertEqual(
            json.dumps(a, sort_keys=True, separators=(",", ":")),
            json.dumps(b, sort_keys=True, separators=(",", ":")),
        )
        self.assertFalse(a["training_authorized"])
        self.assertNotIn("spend_verified_usd", a)


if __name__ == "__main__":
    unittest.main()
