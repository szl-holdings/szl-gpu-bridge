# SPDX-License-Identifier: Apache-2.0
"""Synthetic-only regression tests: no model downloads, inference or provider writes."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools/proving-grounds/agent_proving_ground.py"
spec = importlib.util.spec_from_file_location("agent_pg", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SHA = "a" * 40


class ProvenanceAndHonestyTests(unittest.TestCase):
    def test_exact_correct_answers_only(self):
        for expected, response in (
            ("PARIS", "FINAL: Paris"),
            ("391", "FINAL: 391"),
            ("GEORGE ORWELL", "FINAL: George Orwell"),
            ("ABSTAIN", "FINAL: ABSTAIN"),
        ):
            with self.subTest(expected=expected):
                self.assertTrue(mod.score_response(expected, response)["pass"])

    def test_false_positive_non_abstaining_answer_blocked(self):
        for response in (
            "London",
            "FINAL: LONDON",
            "Paris is wrong. Final: London",
            "FINAL: 400",
            "FINAL: Paris or London",
            "FINAL: ABSTAIN",
            "FINAL: Paris\nFINAL: London",
            "Paris",
            "",
        ):
            with self.subTest(response=response):
                self.assertFalse(mod.score_response("PARIS", response)["pass"])

    def test_unknown_answer_requires_explicit_abstention(self):
        for response in (
            "I do not know",
            "FINAL: 42",
            "FINAL: BLUE",
            "FINAL: Can't know",
            "FINAL: UNKNOWN",
            "FINAL: ABSTAIN; maybe blue",
            "no data",
            "",
        ):
            with self.subTest(response=response):
                self.assertFalse(mod.score_response("ABSTAIN", response)["pass"])

    def test_missing_or_extra_probes_fail_closed(self):
        with self.assertRaisesRegex(mod.Hold, "INCOMPLETE_OR_EXTRA_RESPONSES"):
            mod.grade({"a1": "FINAL: PARIS"})
        with self.assertRaisesRegex(mod.Hold, "INCOMPLETE_OR_EXTRA_RESPONSES"):
            mod.grade(
                {
                    **{ident: "FINAL: ABSTAIN" for ident, *_ in mod.PROBES},
                    "extra": "ignored",
                }
            )

    def test_six_correct_is_still_not_model_quality_verified(self):
        data = {ident: "FINAL: " + answer for ident, _, answer in mod.PROBES}
        report = mod.grade(data)
        self.assertEqual(report["passed"], 6)
        self.assertTrue(report["all_correct"])
        self.assertFalse(report["population_quality_verified"])
        self.assertFalse(report["sampling_independence_verified"])
        self.assertLess(report["wilson_95_lower_descriptive"], 0.7)
        self.assertNotIn("FINAL:", json.dumps(report))

    def test_model_response_privacy(self):
        response = "FINAL: PARIS"
        result = mod.score_response("PARIS", response)
        self.assertTrue(result["pass"])
        self.assertNotIn("PARIS", json.dumps(result))
        self.assertEqual(len(result["answer_sha256"]), 64)

    def test_reject_wrong_repository_revision_claims(self):
        for rev in (
            None,
            "main",
            "latest",
            "f" * 39,
            "g" * 40,
            "f" * 65,
            "0" * 40,
            "0" * 64,
            "0" * 64 + "hi",
        ):
            with self.subTest(rev=rev), self.assertRaises(mod.Hold):
                mod.identity("Qwen/Qwen3-1.7B", rev, "", "")

    def test_reject_ambiguous_adapter_pin(self):
        with self.assertRaisesRegex(mod.Hold, "ADAPTER_IMMUTABLE_REVISION_REQUIRED"):
            mod.identity("Qwen/Qwen3-1.7B", SHA, "SZLHOLDINGS/khipu", None)
        with self.assertRaisesRegex(mod.Hold, "ADAPTER_REVISION_WITHOUT_ADAPTER"):
            mod.identity("Qwen/Qwen3-1.7B", SHA, "", SHA)

    def test_valid_pin_does_not_pretend_remote_source_readback(self):
        result = mod.identity("Qwen/Qwen3-1.7B", SHA, "SZLHOLDINGS/khipu", "b" * 40)
        self.assertTrue(result["revision_values_syntactically_pinned"])
        self.assertFalse(result["revision_repository_readback_verified"])
        self.assertFalse(result["weight_file_integrity_verified"])

    def test_unsigned_observation_never_assumes_signed_custody(self):
        receipt = mod.observation(
            "BATTERY", {"passed": 6, "total": 6}, candidate="test"
        )
        self.assertEqual(receipt["integrity_scope"], "UNSIGNED_HASH_ONLY")
        self.assertFalse(receipt["authenticity_verified"])
        self.assertFalse(receipt["durable_archival_verified"])
        self.assertFalse(receipt["release_authorized"])
        self.assertFalse(receipt["provider_writes"])
        self.assertRegex(receipt["sha256"], r"^[0-9a-f]{64}$")

    def test_cli_never_promotes_without_proving_or_download(self):
        buffer = io.StringIO()
        with (
            redirect_stdout(buffer),
            patch.object(
                mod, "evaluate_model", side_effect=AssertionError("MUST_NOT_LOAD")
            ),
        ):
            code = mod.main(
                [
                    "--base",
                    "Qwen/Qwen3-1.7B",
                    "--base-revision",
                    SHA,
                    "--candidate-name",
                    "not-admitted",
                ]
            )
        lines = [json.loads(x) for x in buffer.getvalue().splitlines()]
        self.assertEqual(code, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[-1]["detail"]["promotion"], "DENIED")
        self.assertEqual(
            lines[-1]["detail"]["status"], "HOLD_MODEL_BATTERY_NOT_EXECUTED"
        )

    def test_improper_input_still_fails_without_gpu(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = mod.main(
                [
                    "--base",
                    "Qwen/Qwen3-1.7B",
                    "--base-revision",
                    "main",
                    "--candidate-name",
                    "test",
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(
            json.loads(buffer.getvalue())["detail"]["reason_code"],
            "BASE_IMMUTABLE_REVISION_REQUIRED",
        )

    def test_invalid_model_output_including_binary_artifacts_rejected(self):
        for value in (None, 7, {"text": "FINAL: PARIS"}, "FINAL: PARIS" * 512):
            self.assertFalse(mod.score_response("PARIS", value)["pass"])

    def test_score_math_invalid_numbers_rejected(self):
        for args in ((-1, 6), (6, 5), (6, 0), (True, 6), (5, True)):
            with self.subTest(args=args), self.assertRaises(mod.Hold):
                mod.wilson_lower(*args)

    def test_no_automatic_model_upload_or_promotion_status(self):
        text = SRC.read_text()
        self.assertNotIn("upload_file(", text)
        self.assertNotIn("PROMOTED_TO_AGENT", text)
        self.assertNotIn("publish=True", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
