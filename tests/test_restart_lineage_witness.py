"""Adversarial, provider-free restart-lineage witness tests."""

from datetime import datetime, timedelta, timezone
import random
import json
import unittest

from tools.salad.restart_lineage_witness import SCHEMA, SCHEMA_V2, witness

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
        "schema": SCHEMA_V2,
        "pins": dict(PINS),
        "target_steps": 2000,
        "prior_step": 313,
        "observed_at_utc": "2026-11-19T09:19:00Z",
        "intent": "resume",
        "resume_topology": {
            "saved_dp_size": 2,
            "requested_dp_size": 2,
            "dataloader_checkpoint_step": 300,
            "rank_digests": [
                {"rank": 0, "checkpoint_step": 300, "state_sha256": "a" * 64},
                {"rank": 1, "checkpoint_step": 300, "state_sha256": "b" * 64},
            ],
        },
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
        obj["resume_topology"] = None
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

    def test_v1_resume_cannot_claim_full_state_review_without_rank_topology(self):
        obj = candidate()
        obj["schema"] = SCHEMA
        del obj["resume_topology"]
        result = judge(obj)
        self.assertEqual(result["reason"], "TOPOLOGY_EVIDENCE_REQUIRED")
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["replay_steps_conditional"], 13)
        self.assertFalse(result["gpu_training_authorized"])

    def test_v1_fresh_schema_remains_parseable_and_non_authorizing(self):
        obj = candidate()
        obj["schema"] = SCHEMA
        del obj["resume_topology"]
        obj["intent"] = "new"
        obj["prior_step"] = 0
        obj["checkpoint"] = None
        result = judge(obj)
        self.assertEqual(result["state"], "REVIEW_CANDIDATE_ONLY")
        self.assertFalse(result["controller_transition_authorized"])

    def test_v2_resume_requires_exact_topology_field(self):
        obj = candidate()
        del obj["resume_topology"]
        self.assertEqual(judge(obj)["reason"], "INPUT_SCHEMA_MISMATCH")
        obj = candidate()
        obj["resume_topology"] = None
        self.assertEqual(judge(obj)["reason"], "TOPOLOGY_EVIDENCE_REQUIRED")

    def test_v2_new_run_forbids_stale_topology(self):
        obj = candidate()
        obj["intent"] = "new"
        obj["prior_step"] = 0
        obj["checkpoint"] = None
        malformed = judge(obj)
        self.assertEqual(malformed["reason"], "NEW_RUN_WITH_TOPOLOGY")
        self.assertIsNone(malformed["replay_steps_conditional"])
        obj["resume_topology"] = None
        self.assertEqual(judge(obj)["state"], "REVIEW_CANDIDATE_ONLY")

    def test_dp_size_change_is_hard_hold(self):
        obj = candidate()
        obj["resume_topology"]["requested_dp_size"] = 1
        result = judge(obj)
        self.assertEqual(result["reason"], "DP_TOPOLOGY_CHANGED")
        self.assertFalse(result["dataloader_resume_verified_here"])

    def test_size_types_and_bounds_are_strict(self):
        for bad in (True, False, 0, -1, 129, 1.0, "2", None):
            for key in ("saved_dp_size", "requested_dp_size"):
                with self.subTest(key=key, bad=bad):
                    obj = candidate()
                    obj["resume_topology"][key] = bad
                    self.assertEqual(judge(obj)["reason"], "DP_SIZE_INVALID")

    def test_rank_lists_must_be_complete_unique_and_in_range(self):
        for rank_list in (
            [],
            [candidate()["resume_topology"]["rank_digests"][0]],
            [candidate()["resume_topology"]["rank_digests"][0]] * 2,
            candidate()["resume_topology"]["rank_digests"]
            + [{"rank": 2, "checkpoint_step": 300, "state_sha256": "c" * 64}],
        ):
            with self.subTest(rank_list=rank_list):
                obj = candidate()
                obj["resume_topology"]["rank_digests"] = rank_list
                self.assertEqual(judge(obj)["reason"], "RANK_STATE_INCOMPLETE")

    def test_rank_type_must_be_real_integer(self):
        for rank in (True, False, "0", -1, 2, 0.0, None):
            with self.subTest(rank=rank):
                obj = candidate()
                obj["resume_topology"]["rank_digests"][0]["rank"] = rank
                self.assertEqual(judge(obj)["reason"], "RANK_STATE_INCOMPLETE")

    def test_rank_schema_rejects_unknown_or_missing_keys(self):
        obj = candidate()
        del obj["resume_topology"]["rank_digests"][0]["state_sha256"]
        self.assertEqual(judge(obj)["reason"], "RANK_STATE_MALFORMED")
        obj = candidate()
        obj["resume_topology"]["rank_digests"][0]["unknown"] = True
        self.assertEqual(judge(obj)["reason"], "RANK_STATE_MALFORMED")

    def test_topology_schema_rejects_unknown_or_missing_keys(self):
        obj = candidate()
        del obj["resume_topology"]["saved_dp_size"]
        self.assertEqual(judge(obj)["reason"], "TOPOLOGY_SCHEMA_MISMATCH")
        obj = candidate()
        obj["resume_topology"]["extra"] = "x"
        self.assertEqual(judge(obj)["reason"], "TOPOLOGY_SCHEMA_MISMATCH")

    def test_selected_loader_step_must_bind_full_checkpoint_step(self):
        obj = candidate()
        obj["resume_topology"]["dataloader_checkpoint_step"] = 299
        self.assertEqual(judge(obj)["reason"], "DATALOADER_STATE_STEP_MISMATCH")
        obj = candidate()
        obj["resume_topology"]["dataloader_checkpoint_step"] = True
        self.assertEqual(judge(obj)["reason"], "STEP_UNAVAILABLE_OR_INVALID")
        obj = candidate()
        obj["resume_topology"]["rank_digests"][1]["checkpoint_step"] = 299
        self.assertEqual(judge(obj)["reason"], "RANK_STATE_STEP_MISMATCH")

    def test_rank_digest_syntax_and_untrusted_hash_rejection(self):
        for digest in (None, True, "A" * 64, "f" * 63, "g" * 64, 123):
            with self.subTest(digest=digest):
                obj = candidate()
                obj["resume_topology"]["rank_digests"][0]["state_sha256"] = digest
                self.assertEqual(judge(obj)["reason"], "RANK_STATE_HASH_INVALID")
        obj = candidate()
        obj["resume_topology"]["rank_digests"][0]["state_sha256"] = "f" * 64
        result = judge(obj)
        self.assertEqual(result["reason"], "FULL_STATE_DECLARED_UNATTESTED")
        self.assertFalse(result["topology_bytes_verified_here"])

    def test_rank_order_is_explicit_not_inferred_from_list_position(self):
        obj = candidate()
        obj["resume_topology"]["rank_digests"].reverse()
        self.assertEqual(judge(obj)["state"], "REVIEW_CANDIDATE_ONLY")

    def test_rank_checkpoint_boolean_not_accepted(self):
        obj = candidate()
        obj["resume_topology"]["rank_digests"][0]["checkpoint_step"] = True
        self.assertEqual(judge(obj)["reason"], "RANK_STATE_STEP_MISMATCH")

    def test_structurally_admitted_resume_has_no_cloud_authority(self):
        result = judge(candidate())
        for field in (
            "resume_source_authenticated",
            "checkpoint_bytes_verified_here",
            "backup_verified_here",
            "owner_generation_authenticated",
            "gpu_training_authorized",
            "controller_transition_authorized",
            "model_completion_qualified",
            "topology_bytes_verified_here",
            "dataloader_resume_verified_here",
        ):
            with self.subTest(field=field):
                self.assertFalse(result[field])

    def test_oversized_integer_json_and_fuzz_do_not_crash(self):
        raw = (
            b'{"schema":"szl.restart-lineage-input/v2","prior_step":'
            + b"9" * 5000
            + b"}"
        )
        result = witness(raw, now=NOW)
        self.assertEqual(result["state"], "HOLD")
        self.assertFalse(result["gpu_training_authorized"])
        rng = random.Random(20261119)
        alphabet = '}{[]:,"az01-+NaN '
        for _ in range(300):
            fuzz = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 140)))
            out = witness(fuzz.encode("utf-8"), now=NOW)
            self.assertEqual(out["state"], "HOLD")
            self.assertFalse(out["gpu_training_authorized"])

    def test_replay_fraction_never_calls_it_quality(self):
        out = judge(candidate())
        self.assertFalse(out["model_completion_qualified"])
        self.assertFalse(out["backup_verified_here"])
        self.assertFalse(out["controller_transition_authorized"])


if __name__ == "__main__":
    unittest.main()
