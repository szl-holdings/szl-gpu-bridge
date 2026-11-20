"""Offline negative controls: an ABS-RL reward-gap shadow is not training auth."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "absrl" / "verifier_gap_shadow.py"
SPEC = importlib.util.spec_from_file_location("_szl_verifier_gap_shadow", SOURCE)
assert SPEC is not None and SPEC.loader is not None
GAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GAP)


def fixture():
    plan = {
        "schema": GAP.PLAN_SCHEMA,
        "source_revision": "a" * 40,
        "dataset_revision": "b" * 40,
        "oracle_revision": "c" * 40,
        "proxy_revision": "d" * 40,
        "oracle_owner_digest": "e" * 64,
        "proxy_owner_digest": "f" * 64,
        "cases": [
            {
                "case_sha256": f"{i + 1:064x}",
                "stratum": "answerable" if i < 4 else "unanswerable",
            }
            for i in range(8)
        ],
    }
    result = ["CORRECT", "CORRECT", "INCORRECT", "INCORRECT"] * 2
    credit = [2, 2, 0, 0, 1, 1, 0, 0]
    observed = {
        "schema": GAP.OBSERVATION_SCHEMA,
        "source_revision": plan["source_revision"],
        "dataset_revision": plan["dataset_revision"],
        "oracle_revision": plan["oracle_revision"],
        "proxy_revision": plan["proxy_revision"],
        "rows": [
            {
                "case_sha256": plan["cases"][i]["case_sha256"],
                "oracle_result": result[i],
                "proxy_reward_half_units": credit[i],
            }
            for i in range(8)
        ],
    }
    return plan, observed


def assess(plan=None, observed=None, *, pinned_digest=None):
    if plan is None or observed is None:
        plan, observed = fixture()
    if pinned_digest is None:
        try:
            pinned_digest = GAP.plan_commitment(plan)
        except GAP.GapHold:
            pinned_digest = "1" * 64
    return GAP.review_reward_gap(
        plan,
        observed,
        expected_plan_sha256=pinned_digest,
        expected_source_revision="a" * 40,
    )


class VerifierGapShadowTest(unittest.TestCase):
    def hold(self, plan, observed, code, *, pinned_digest=None):
        result = assess(plan, observed, pinned_digest=pinned_digest)
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["reason"], code)
        self.assertIs(result["training_authorized"], False)
        return result

    def test_exact_populated_shadow_never_authorizes_training(self):
        result = assess()
        self.assertEqual(result["state"], "SOURCE_SHADOW_CONSISTENT_NOT_ATTESTED")
        self.assertEqual(result["frozen_cases"], 8)
        self.assertEqual(result["attempted"], 8)
        self.assertEqual(result["oracle_correct"], 4)
        self.assertEqual(result["oracle_incorrect"], 4)
        for key in (
            "trusted_oracle_proven",
            "independent_evaluator_proven",
            "pre_registration_proven",
            "dataset_rights_verified",
            "model_quality_verified",
            "training_authorized",
            "release_authorized",
            "provider_writes_performed",
        ):
            self.assertIs(result[key], False)
        self.assertEqual(result["reason"], "DECLARED_PROXY_ORACLE_AGREEMENT_ONLY")
        self.assertEqual(len(result["plan_sha256"]), 64)

    def test_optimistic_proxy_on_wrong_oracle_fails(self):
        plan, obs = fixture()
        obs["rows"][2]["proxy_reward_half_units"] = 2
        result = self.hold(plan, obs, "OPTIMISTIC_PROXY_REWARD_ON_INCORRECT_ORACLE")
        self.assertEqual(result["optimistic_proxy_credit"], 1)
        self.assertNotIn(obs["rows"][2]["case_sha256"], str(result))

    def test_half_credit_for_wrong_oracle_also_fails(self):
        plan, obs = fixture()
        obs["rows"][6]["proxy_reward_half_units"] = 1
        self.hold(plan, obs, "OPTIMISTIC_PROXY_REWARD_ON_INCORRECT_ORACLE")

    def test_missing_oracle_is_not_scored_as_false(self):
        plan, obs = fixture()
        obs["rows"][0].update(oracle_result="MISSING", proxy_reward_half_units=None)
        result = self.hold(plan, obs, "INCOMPLETE_ORACLE_COVERAGE")
        self.assertEqual(
            (result["attempted"], result["missing"], result["unknown"]), (7, 1, 0)
        )
        plan, obs = fixture()
        obs["rows"][1].update(oracle_result="UNKNOWN", proxy_reward_half_units=None)
        result = self.hold(plan, obs, "INCOMPLETE_ORACLE_COVERAGE")
        self.assertEqual((result["attempted"], result["unknown"]), (7, 1))

    def test_zero_attempted_denominator_is_hold(self):
        plan, obs = fixture()
        for row in obs["rows"]:
            row.update(oracle_result="UNKNOWN", proxy_reward_half_units=None)
        result = self.hold(plan, obs, "ZERO_ATTEMPTED_DENOMINATOR")
        self.assertEqual(result["attempted"], 0)
        self.assertEqual(result["unknown"], 8)

    def test_missing_or_unknown_cannot_get_proxy_reward(self):
        plan, obs = fixture()
        obs["rows"][0]["oracle_result"] = "UNKNOWN"
        self.hold(plan, obs, "UNKNOWN_OR_MISSING_WITH_PROXY_REWARD")

    def test_no_incorrect_oracle_negative_control_is_hold(self):
        plan, obs = fixture()
        for i, row in enumerate(obs["rows"]):
            row["oracle_result"] = "CORRECT"
            row["proxy_reward_half_units"] = 2 if i < 4 else 1
        self.hold(plan, obs, "ORACLE_NEGATIVE_CONTROL_MISSING")

    def test_stratum_negative_controls_are_required_in_each_group(self):
        for stratum, indices in (
            ("answerable", (2, 3)),
            ("unanswerable", (6, 7)),
        ):
            with self.subTest(stratum=stratum):
                plan, obs = fixture()
                for i in indices:
                    obs["rows"][i]["oracle_result"] = "CORRECT"
                    obs["rows"][i]["proxy_reward_half_units"] = 2 if i < 4 else 1
                result = self.hold(plan, obs, "STRATUM_NEGATIVE_CONTROL_MISSING")
                self.assertEqual(result[stratum + "_oracle_incorrect"], 0)
                self.assertGreater(result["oracle_incorrect"], 0)

    def test_stratum_positive_controls_are_required_in_each_group(self):
        for stratum, indices in (
            ("answerable", (0, 1)),
            ("unanswerable", (4, 5)),
        ):
            with self.subTest(stratum=stratum):
                plan, obs = fixture()
                for i in indices:
                    obs["rows"][i]["oracle_result"] = "INCORRECT"
                    obs["rows"][i]["proxy_reward_half_units"] = 0
                result = self.hold(plan, obs, "STRATUM_POSITIVE_CONTROL_MISSING")
                self.assertEqual(result[stratum + "_oracle_correct"], 0)
                self.assertGreater(result["oracle_correct"], 0)

    def test_valid_shadow_preserves_cross_stratum_contingency(self):
        result = assess()
        self.assertEqual(
            (
                result["answerable_oracle_correct"],
                result["answerable_oracle_incorrect"],
                result["unanswerable_oracle_correct"],
                result["unanswerable_oracle_incorrect"],
            ),
            (2, 2, 2, 2),
        )
        self.assertIs(result["training_authorized"], False)

    def test_wrong_stratum_credit_not_hidden_by_positive_reward(self):
        plan, obs = fixture()
        obs["rows"][0]["proxy_reward_half_units"] = 1
        result = self.hold(plan, obs, "PROXY_ORACLE_REWARD_DISAGREEMENT")
        self.assertEqual(result["stratum_reward_mismatch"], 1)
        self.assertEqual(result["pessimistic_proxy_credit"], 0)

    def test_missing_or_duplicate_observation_fails_before_metric(self):
        plan, obs = fixture()
        obs["rows"].pop()
        self.hold(plan, obs, "OBSERVATION_POPULATION_INCOMPLETE")
        plan, obs = fixture()
        obs["rows"][1]["case_sha256"] = obs["rows"][0]["case_sha256"]
        self.hold(plan, obs, "DUPLICATE_OBSERVED_CASE")

    def test_substituted_case_fails_even_if_counts_constant(self):
        plan, obs = fixture()
        obs["rows"][1]["case_sha256"] = "f" * 64
        self.hold(plan, obs, "UNKNOWN_OR_SUBSTITUTED_CASE")

    def test_missing_frozen_case_and_empty_stratum_fail(self):
        plan, obs = fixture()
        plan["cases"].pop()
        self.hold(plan, obs, "FROZEN_PLAN_COMMITMENT_MISMATCH", pinned_digest="1" * 64)
        plan, obs = fixture()
        for row in plan["cases"]:
            row["stratum"] = "answerable"
        self.hold(plan, obs, "COHORT_STRATUM_COVERAGE_INVALID")

    def test_duplicate_expected_id_fails(self):
        plan, obs = fixture()
        plan["cases"][1]["case_sha256"] = plan["cases"][0]["case_sha256"]
        self.hold(plan, obs, "DUPLICATE_FROZEN_CASE")

    def test_changed_source_or_revisions_fail(self):
        plan, obs = fixture()
        plan["source_revision"] = "f" * 40
        self.hold(plan, obs, "SOURCE_IDENTITY_NOT_EXPECTED")
        plan, obs = fixture()
        obs["oracle_revision"] = "e" * 40
        self.hold(plan, obs, "OBSERVED_REVISION_MISMATCH")
        plan, obs = fixture()
        plan["proxy_revision"] = plan["oracle_revision"]
        obs["proxy_revision"] = plan["oracle_revision"]
        self.hold(plan, obs, "ORACLE_PROXY_REVISION_NOT_DISTINCT")

    def test_identical_evaluator_owners_fail(self):
        plan, obs = fixture()
        plan["oracle_owner_digest"] = plan["proxy_owner_digest"]
        self.hold(plan, obs, "EVALUATOR_OWNER_COLLISION")

    def test_frozen_plan_commitment_mismatch_fail_closed(self):
        plan, obs = fixture()
        digest = GAP.plan_commitment(plan)
        plan["dataset_revision"] = "f" * 40
        obs["dataset_revision"] = "f" * 40
        self.hold(plan, obs, "FROZEN_PLAN_COMMITMENT_MISMATCH", pinned_digest=digest)

    def test_boolean_and_noninteger_reward_rejected(self):
        for bad in (True, 1.0, 0.5, -1, 3, None, "2"):
            plan, obs = fixture()
            obs["rows"][0]["proxy_reward_half_units"] = bad
            with self.subTest(value=str(bad)):
                self.hold(plan, obs, "PROXY_REWARD_INVALID")

    def test_schema_extension_cannot_inject_raw_prompt_or_authority(self):
        plan, obs = fixture()
        obs["rows"][0]["prompt"] = "private text must not enter this gate"
        self.hold(plan, obs, "OUTCOME_SCHEMA_INVALID")
        plan, obs = fixture()
        plan["training_authorized"] = True
        self.hold(plan, obs, "PLAN_FIELDS_INVALID")

    def test_unexpected_oracle_result_is_not_false(self):
        plan, obs = fixture()
        obs["rows"][0]["oracle_result"] = "UNCERTAIN"
        self.hold(plan, obs, "ORACLE_RESULT_INVALID")

    def test_order_independence_and_no_input_mutation(self):
        plan, obs = fixture()
        before = deepcopy((plan, obs))
        result = assess(plan, obs)
        self.assertEqual((plan, obs), before)
        plan["cases"].reverse()
        obs["rows"].reverse()
        self.assertEqual(
            assess(plan, obs)["state"], "SOURCE_SHADOW_CONSISTENT_NOT_ATTESTED"
        )
        self.assertNotEqual(GAP.plan_commitment(plan), result["plan_sha256"])

    def test_no_effectful_imports_or_raw_generation_export(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        forbidden = {
            "os",
            "socket",
            "subprocess",
            "requests",
            "urllib",
            "torch",
            "trl",
            "huggingface_hub",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(
                    forbidden.intersection(a.name.split(".")[0] for a in node.names)
                )
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split(".")[0], forbidden)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"open", "eval", "exec"})


if __name__ == "__main__":
    unittest.main()
