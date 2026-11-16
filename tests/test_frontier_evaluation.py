"""Synthetic CPU counterexamples for the real frontier generation evaluator."""

import hashlib
import json
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

from frontier_job import _evaluate_generations  # noqa: E402
from frontier_runtime import extract_json_object  # noqa: E402

GOOD = '{"action":"ABSTAIN","conviction":0.2}'


def config():
    return {
        "suite": "frontier-heldout-v2",
        "heldOutFraction": 0.1,
        "seed": 7,
        "maxGenerations": 20,
        "maxNewTokens": 400,
        "requiredJsonKeys": ["action", "conviction"],
        "convictionCeiling": 0.97,
        "maxDegenerateRate": 0,
        "minJsonValidRate": 0.95,
        "minRequiredKeysRate": 0.95,
        "minCeilingRespectRate": 1,
    }


def evaluate(responses, evaluation=None):
    evaluation = config() if evaluation is None else evaluation
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"synthetic control {index}"},
                {"role": "assistant", "content": GOOD},
            ]
        }
        for index in range(len(responses))
    ]
    with mock.patch("frontier_job._generate", side_effect=responses) as generate:
        result = _evaluate_generations(None, None, rows, evaluation)
    return result, generate


class GenerationCounterexampleTests(unittest.TestCase):
    def test_valid_control_passes_without_exposing_the_heldout_answer(self):
        (metrics, records), generate = evaluate([GOOD])
        for field in ("jsonValidRate", "requiredKeysRate", "ceilingRespectRate"):
            self.assertEqual(metrics[field], 1)
        self.assertEqual(metrics["degenerateRate"], 0)
        self.assertEqual(metrics["generationsChecked"], 1)
        self.assertEqual(generate.call_args.args[2][-1]["role"], "user")
        self.assertEqual(
            records[0]["responseSha256"], hashlib.sha256(GOOD.encode()).hexdigest()
        )
        self.assertEqual(records[0]["responseBytes"], len(GOOD.encode()))

    def test_every_mandatory_gate_has_an_executed_failing_control(self):
        controls = (
            ("jsonValidRate", "minJsonValidRate", "not a JSON object"),
            ("requiredKeysRate", "minRequiredKeysRate", '{"conviction":0.2}'),
            (
                "ceilingRespectRate",
                "minCeilingRespectRate",
                '{"action":"ABSTAIN","conviction":0.99}',
            ),
        )
        for metric, threshold, response in controls:
            with self.subTest(metric=metric):
                (metrics, _), _ = evaluate([response])
                self.assertEqual(metrics[metric], 0)
                self.assertLess(metrics[metric], config()[threshold])
        (metrics, _), _ = evaluate(["@" * 40])
        self.assertEqual(metrics["degenerateRate"], 1)
        self.assertGreater(metrics["degenerateRate"], config()["maxDegenerateRate"])

    def test_missing_null_and_invalid_conviction_cannot_measure_compliance(self):
        responses = [
            "not a JSON object",
            '{"action":"ABSTAIN"}',
            '{"action":"ABSTAIN","conviction":null}',
            '{"action":"ABSTAIN","conviction":true}',
            '{"action":"ABSTAIN","conviction":"0.2"}',
            '{"action":"ABSTAIN","conviction":-0.01}',
            '{"action":"ABSTAIN","conviction":-Infinity}',
            '{"action":"ABSTAIN","conviction":Infinity}',
            '{"action":"ABSTAIN","conviction":NaN}',
            '{"action":"ABSTAIN","conviction":-1e9999}',
            '{"action":"ABSTAIN","conviction":0.99,"conviction":0.2}',
        ]
        for response in responses:
            with self.subTest(response=response):
                (metrics, records), _ = evaluate([response])
                self.assertEqual(metrics["ceilingRespectRate"], 0)
                self.assertIs(records[0]["ceilingValid"], False)

    def test_optional_json_key_does_not_make_ceiling_measurement_optional(self):
        evaluation = config()
        evaluation["requiredJsonKeys"] = ["action"]
        (metrics, _), _ = evaluate(['{"action":"ABSTAIN"}'], evaluation)
        self.assertEqual(metrics["requiredKeysRate"], 1)
        self.assertEqual(metrics["ceilingRespectRate"], 0)

    def test_zero_and_exact_ceiling_are_valid_measurements(self):
        for conviction in (0, 0.97):
            with self.subTest(conviction=conviction):
                response = json.dumps({"action": "ABSTAIN", "conviction": conviction})
                (metrics, _), _ = evaluate([response])
                self.assertEqual(metrics["ceilingRespectRate"], 1)
        evaluation = config()
        evaluation["convictionCeiling"] = 0
        (metrics, _), _ = evaluate(['{"action":"ABSTAIN","conviction":0}'], evaluation)
        self.assertEqual(metrics["ceilingRespectRate"], 1)

    def test_recorded_counts_preserve_fractional_acceptance_boundary(self):
        for good_count, accepted in ((19, True), (18, False)):
            with self.subTest(good_count=good_count):
                responses = [GOOD] * good_count + ["not JSON"] * (20 - good_count)
                (metrics, records), _ = evaluate(responses)
                self.assertEqual(metrics["jsonValidRate"], good_count / 20)
                self.assertEqual(sum(r["jsonValid"] for r in records), good_count)
                self.assertIs(metrics["jsonValidRate"] >= 0.95, accepted)

    def test_empty_battery_has_no_measured_rate(self):
        (metrics, records), generate = evaluate([])
        self.assertEqual(metrics["generationsChecked"], 0)
        self.assertEqual(records, [])
        generate.assert_not_called()
        for field in (
            "jsonValidRate",
            "requiredKeysRate",
            "ceilingRespectRate",
            "degenerateRate",
        ):
            self.assertIsNone(metrics[field])

    def test_nonstandard_or_ambiguous_json_is_not_json_valid(self):
        for response in (
            '{"conviction":NaN}',
            '{"conviction":Infinity}',
            '{"conviction":-Infinity}',
            '{"conviction":1e9999}',
            '{"conviction":0.99,"conviction":0.2}',
            '{"outer":{"conviction":0.99,"conviction":0.2}}',
            '{"conviction":0.2,"other":NaN}',
        ):
            with self.subTest(response=response):
                self.assertIsNone(extract_json_object(response))
                (metrics, _), _ = evaluate([response])
                self.assertEqual(metrics["jsonValidRate"], 0)
        self.assertEqual(
            extract_json_object('prefix {"action":"ABSTAIN","conviction":0.2} suffix'),
            {"action": "ABSTAIN", "conviction": 0.2},
        )


class EvaluationSchemaTests(unittest.TestCase):
    def test_schema_matches_nondisabled_rate_boundaries(self):
        from jsonschema.validators import Draft202012Validator

        schema = json.loads(
            (ROOT / "schema/jobspec.v2.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema["properties"]["eval"])
        validator.validate(config())
        for field, value in (
            ("minJsonValidRate", 0),
            ("minRequiredKeysRate", 0),
            ("minCeilingRespectRate", 0),
            ("maxDegenerateRate", 1),
        ):
            evaluation = config()
            evaluation[field] = value
            with self.subTest(field=field):
                self.assertFalse(validator.is_valid(evaluation))


if __name__ == "__main__":
    unittest.main()
