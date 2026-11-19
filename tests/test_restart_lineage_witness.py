"""Adversarial, provider-free restart-lineage witness tests."""

from datetime import datetime, timedelta, timezone
import json
import unittest

from tools.salad.restart_lineage_witness import witness

NOW = datetime(2026, 11, 19, 9, 20, tzinfo=timezone.utc)
PINS = {
    "source": "a" * 40,
    "base": "b" * 40,
    "dataset": "c" * 40,
    "command": "d" * 64,
}
FILES = {
    k: True
    for k in (
        "adapter",
        "trainer_state",
        "optimizer",
        "rng",
        "scheduler",
        "dataloader",
        "tokenizer",
    )
}


def candidate():
    return {
        "schema": "szl.restart-lineage-input/v1",
        "pins": dict(PINS),
        "target_steps": 2000,
        "prior_step": 313,
        "observed_at_utc": "2026-11-19T09:19:00Z",
        "intent": "resume",
        "checkpoint": {
            "step": 300,
            "pins": dict(PINS),
            "commit": "e" * 40,
            "manifest_sha256": "f" * 64,
            "state_files": dict(FILES),
        },
    }


def judge(obj):
    return witness(json.dumps(obj, sort_keys=True).encode(), now=NOW)


class RestartLineageTests(unittest.TestCase):
    def test_candidate_only_no_dispatch_authority(self):
        out = judge(candidate())
        self.assertEqual(out["state"], "REVIEW_CANDIDATE_ONLY")
        self.assertEqual(out["replay_steps_conditional"], 13)
        self.assertEqual(
            out["replay_fraction_conditional"],
            {"numerator": 13, "denominator": 2000},
        )
        self.assertFalse(out["gpu_training_authorized"])
        self.assertFalse(out["owner_generation_authenticated"])

    def test_no_checkpoint_is_hard_hold_with_replay_accounting(self):
        obj = candidate()
        obj["checkpoint"] = None
        out = judge(obj)
        self.assertEqual(out["reason"], "RESUME_WITHOUT_CHECKPOINT")
        self.assertEqual(out["replay_steps_conditional"], 313)
        self.assertFalse(out["checkpoint_bytes_verified_here"])

    def test_genesis_with_prior_steps_refused(self):
        obj = candidate()
        obj["intent"] = "new"
        obj["checkpoint"] = None
        self.assertEqual(judge(obj)["reason"], "GENESIS_REPLAYS_RECORDED_WORK")

    def test_fresh_no_prior_work_cannot_authorize_dispatch(self):
        obj = candidate()
        obj["intent"] = "new"
        obj["prior_step"] = 0
        obj["checkpoint"] = None
        out = judge(obj)
        self.assertEqual(out["reason"], "FRESH_NO_PRIOR_STEPS_UNATTESTED")
        self.assertFalse(out["gpu_training_authorized"])

    def test_unknown_prior_step_refuses(self):
        obj = candidate()
        obj["prior_step"] = None
        self.assertEqual(judge(obj)["reason"], "PRIOR_TRAINER_STEP_UNOBSERVED")

    def test_checkpoint_ahead_is_rejected(self):
        obj = candidate()
        obj["checkpoint"]["step"] = 400
        self.assertEqual(judge(obj)["reason"], "CHECKPOINT_AHEAD_OF_TRAINER")

    def test_source_mismatch(self):
        obj = candidate()
        obj["checkpoint"]["pins"]["dataset"] = "0" * 40
        self.assertEqual(judge(obj)["reason"], "CHECKPOINT_RUN_IDENTITY_MISMATCH")

    def test_every_full_state_member_required(self):
        for key in FILES:
            with self.subTest(key=key):
                obj = candidate()
                obj["checkpoint"]["state_files"][key] = False
                self.assertEqual(judge(obj)["reason"], "INCOMPLETE_RESUME_STATE")

    def test_missing_key_and_extra_state_rejected(self):
        obj = candidate()
        del obj["checkpoint"]["state_files"]["rng"]
        self.assertEqual(judge(obj)["reason"], "STATE_COMPONENT_COVERAGE_MISSING")
        obj = candidate()
        obj["checkpoint"]["state_files"]["unknown"] = True
        self.assertEqual(judge(obj)["reason"], "STATE_COMPONENT_COVERAGE_MISSING")

    def test_stale_and_future_clock_refused(self):
        obj = candidate()
        obj["observed_at_utc"] = (NOW - timedelta(hours=2)).isoformat()
        self.assertEqual(judge(obj)["reason"], "OBSERVATION_STALE_OR_CLOCK_SKEW")
        obj["observed_at_utc"] = (NOW + timedelta(minutes=2)).isoformat()
        self.assertEqual(judge(obj)["reason"], "OBSERVATION_STALE_OR_CLOCK_SKEW")

    def test_bool_step_not_integer(self):
        obj = candidate()
        obj["prior_step"] = True
        self.assertEqual(judge(obj)["reason"], "STEP_UNAVAILABLE_OR_INVALID")

    def test_duplicate_keys(self):
        raw = b'{"schema":"x","schema":"y"}'
        self.assertEqual(witness(raw, now=NOW)["reason"], "DUPLICATE_INPUT_KEY")

    def test_nonfinite_and_invalid_json(self):
        self.assertEqual(witness(b'{"x":NaN}', now=NOW)["reason"], "NONFINITE_NUMBER")
        self.assertEqual(witness(b"{", now=NOW)["reason"], "INVALID_JSON")

    def test_input_byte_budget(self):
        self.assertEqual(witness(b"x" * 32769, now=NOW)["reason"], "INPUT_BYTE_BUDGET")

    def test_replay_fraction_never_calls_it_quality(self):
        out = judge(candidate())
        self.assertFalse(out["model_completion_qualified"])
        self.assertFalse(out["backup_verified_here"])
        self.assertFalse(out["controller_transition_authorized"])


if __name__ == "__main__":
    unittest.main()
