"""Offline adversarial tests for inactive ABS-RL reward verifier v2.

No torch, TRL, GPU, network, remote provider, private dataset or checkpoint.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "absrl" / "rewards_v2.py"
SPEC = importlib.util.spec_from_file_location("szl_absrl_rewards_v2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)


class RewardV2Tests(unittest.TestCase):
    def score(self, output, gold, label):
        return reward.abstention_reward_v2(
            [output], answer=[gold], answerable=[label]
        )[0]

    def test_exact_decisive(self):
        self.assertEqual(self.score("42", "42", True), 1.0)
        self.assertEqual(self.score("FINAL: 42", "42", True), 1.0)
        self.assertEqual(
            self.score([{"role": "assistant", "content": "FINAL: 42"}], "42", 1),
            1.0,
        )

    def test_exact_abstention(self):
        self.assertEqual(self.score("UNANSWERABLE", None, False), 0.5)
        self.assertEqual(
            self.score("FINAL: UNANSWERABLE", "", "unanswerable"), 0.5
        )

    def test_gold_substring_does_not_reward(self):
        for output in (
            "14", "not 4", "the answer is 4", "so 4",
            "FINAL: 4 because yes", "FINAL: 4\nextra",
            "FINAL: 4\nFINAL: 4",
        ):
            with self.subTest(output=output):
                self.assertEqual(self.score(output, "4", True), 0.0)

    def test_abstention_marker_embedded_does_not_reward(self):
        for output in (
            "I cannot tell: UNANSWERABLE", "not UNANSWERABLE",
            "FINAL: UNANSWERABLE trailing", "FINAL: UNANSWERABLE\nThanks",
            "unanswerable", "42",
        ):
            with self.subTest(output=output):
                self.assertEqual(self.score(output, None, False), 0.0)

    def test_answerable_overabstention_zero(self):
        self.assertEqual(self.score("UNANSWERABLE", "42", True), 0.0)
        self.assertEqual(self.score("FINAL: UNANSWERABLE", "42", True), 0.0)

    def test_missing_unknown_nonfinite_label_rejected(self):
        for value in (None, "", "unknown", "maybe", 2, -1, 0.0, float("nan")):
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError):
                    self.score("UNANSWERABLE", None, value)

    def test_contradictory_label_and_gold_rejected(self):
        for output, gold, label in (
            ("UNANSWERABLE", "42", False),
            ("42", None, True),
            ("42", "", True),
            ("42", "UNANSWERABLE", True),
            ("42", 42, True),
            ("42", "4\n2", True),
        ):
            with self.subTest(label=label, gold=repr(gold)):
                with self.assertRaises(ValueError):
                    self.score(output, gold, label)

    def test_bad_completion_shape_rejected(self):
        for output in (
            None, b"42", {"content": 42},
            {"role": "user", "content": "42"},
            [{"role": "user", "content": "42"}],
            [{"role": "assistant", "content": "42"},
             {"role": "assistant", "content": "42"}],
            [],
        ):
            with self.subTest(kind=type(output).__name__):
                with self.assertRaises(ValueError):
                    self.score(output, "42", True)

    def test_full_batch_validated_before_scoring(self):
        with self.assertRaises(ValueError):
            reward.abstention_reward_v2(
                ["42", "UNANSWERABLE"],
                answer=["42", None],
                answerable=[True, "maybe"],
            )
        with self.assertRaises(ValueError):
            reward.abstention_reward_v2(
                ["42", "UNANSWERABLE"],
                answer=["42"],
                answerable=[True, False],
            )

    def test_missing_columns_and_empty_batch_fail(self):
        with self.assertRaises(ValueError):
            reward.abstention_reward_v2(["42"], answer=["42"])
        with self.assertRaises(ValueError):
            reward.abstention_reward_v2([], answer=[], answerable=[])
        with self.assertRaises(ValueError):
            reward.abstention_reward_v2(["42"], answer="42", answerable=[True])

    def test_bounded_inputs(self):
        with self.assertRaises(ValueError):
            self.score("4" * 8193, "4", True)
        with self.assertRaises(ValueError):
            self.score("4", "4" * 513, True)
        with self.assertRaises(ValueError):
            reward.abstention_reward_v2(
                ["4"] * 2049, answer=["4"] * 2049,
                answerable=[True] * 2049,
            )

    def test_unicode_and_case_not_fuzzy(self):
        self.assertEqual(self.score("４", "4", True), 0.0)
        self.assertEqual(self.score("FOUR", "four", True), 0.0)

    def test_trl_other_fields_do_not_change_reward(self):
        self.assertEqual(
            reward.abstention_reward_v2(
                ["FINAL: 42"], answer=["42"], answerable=[True],
                prompts=["Question"], completion_ids=[[1, 2, 3]],
            ), [1.0],
        )

    def test_repeatable_half_unit_rewards(self):
        def run():
            return reward.abstention_reward_v2(
                ["FINAL: 42", "FINAL: UNANSWERABLE", "wrong"],
                answer=["42", None, "42"],
                answerable=[True, False, True],
            )
        self.assertEqual(run(), [1.0, 0.5, 0.0])
        self.assertEqual(run(), run())

    def test_shadow_diagnostics_aggregate_only(self):
        result = reward.shadow_summary_v2(
            ["FINAL: 42", "FINAL: UNANSWERABLE", "wrong", "UNANSWERABLE"],
            answer=["42", None, "42", "42"],
            answerable=[True, False, True, True],
            group_size=2,
        )
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["decisive_correct_rows"], 1)
        self.assertEqual(result["honest_abstention_rows"], 1)
        self.assertEqual(result["overabstention_rows"], 1)
        self.assertEqual(result["observed_total_half_reward_units"], 3)
        self.assertEqual(result["constant_abstain_half_reward_units"], 1)
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["model_quality_verified"])
        self.assertNotIn("42", str(result))

    def test_constant_abstention_baseline(self):
        result = reward.shadow_summary_v2(
            ["UNANSWERABLE"] * 4,
            answer=[None, None, "42", "42"],
            answerable=[False, False, True, True],
            group_size=2,
        )
        self.assertEqual(result["observed_total_half_reward_units"], 2)
        self.assertEqual(result["constant_abstain_half_reward_units"], 2)
        self.assertFalse(result["beats_constant_abstain_on_this_sample"])

    def test_all_zero_groups_and_group_size(self):
        x = reward.shadow_summary_v2(
            ["wrong", "not 4", "14", "other"],
            answer=["4"] * 4, answerable=[True] * 4, group_size=2,
        )
        self.assertEqual(x["all_zero_groups"], 2)
        self.assertFalse(x["beats_constant_abstain_on_this_sample"])
        with self.assertRaises(ValueError):
            reward.shadow_summary_v2(
                ["4"] * 3, answer=["4"] * 3,
                answerable=[True] * 3, group_size=2,
            )
        with self.assertRaises(ValueError):
            reward.shadow_summary_v2(
                ["4"], answer=["4"], answerable=[True], group_size=True,
            )

    def test_candidate_has_no_effectful_calls(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        forbidden_modules = {
            "subprocess", "requests", "torch", "trl", "huggingface_hub",
            "os", "socket", "urllib",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(forbidden_modules.intersection(
                    alias.name.split(".")[0] for alias in node.names
                ))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split(".")[0], forbidden_modules)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("open", "exec", "eval"))
        # Staged production worker remains unchanged; reviewer must separately
        # admit a compatible dataset, rendered prompt and TRL version.
        worker = ROOT / "tools" / "absrl" / "grpo_absention.py"
        self.assertNotIn("rewards_v2", worker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
