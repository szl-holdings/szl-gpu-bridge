"""Credential-free synthetic tests; no provider or trainer imported."""

import unittest

from tools.salad.stop_preflight import SCHEMA, assess_stop_preflight

NOW = "2026-11-19T15:00:00Z"
SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40
SHA_D = "d" * 40
D_A = "a" * 64
D_B = "b" * 64
D_C = "c" * 64


def valid():
    intent = dict(
        schema=SCHEMA,
        group="szl-test-training-r1",
        owner_lease_digest=D_A,
        owner_generation=2,
        source_revision=SHA_A,
        base_revision=SHA_B,
        dataset_revision=SHA_C,
        model_revision=SHA_D,
        target_steps=200,
    )
    evidence = dict(
        group=intent["group"],
        owner_lease_digest=D_A,
        owner_generation=2,
        observed_at_utc=NOW,
        model_metadata_http_status=200,
        source_revision=SHA_A,
        base_revision=SHA_B,
        dataset_revision=SHA_C,
        model_revision=SHA_D,
        trainer_state=dict(
            revision=SHA_D,
            source_revision=SHA_A,
            global_step=200,
            max_steps=200,
            trainer_state_sha256=D_B,
            bytes_verified=True,
        ),
        artifact=dict(
            revision=SHA_D,
            bytes_verified=True,
            complete_config_and_tokenizer=True,
            adapter_sha256=D_C,
            manifest_sha256=D_A,
        ),
        backup=dict(
            independent_readback=True,
            adapter_sha256=D_C,
            manifest_sha256=D_B,
            verified_at_utc=NOW,
        ),
        provider=dict(
            group=intent["group"],
            observed_at_utc=NOW,
            desired_replicas=1,
            instance_counts=dict(
                allocating_count=0, creating_count=0, running_count=1, stopping_count=0
            ),
            update_pending=False,
        ),
    )
    return intent, evidence


class StopPreflightTests(unittest.TestCase):
    def run_witness(self, change=None, *, now=NOW):
        intent, evidence = valid()
        if change:
            change(intent, evidence)
        return assess_stop_preflight(intent=intent, evidence=evidence, now_utc=now)

    def test_complete_synthetic_claims_never_authorize_stop(self):
        result = self.run_witness()
        self.assertEqual(result["state"], "READY_FOR_OWNER_REVIEW")
        self.assertTrue(result["ready_for_owner_review"])
        for flag in (
            "shutdown_authorized",
            "training_complete_verified",
            "provider_mutation_performed",
        ):
            self.assertFalse(result[flag])

    def test_hub_404_is_not_artifact_absence_proof(self):
        result = self.run_witness(lambda i, e: e.update(model_metadata_http_status=404))
        self.assertEqual(result["reason"], "MODEL_METADATA_NOT_VERIFIED")
        self.assertFalse(result["ready_for_owner_review"])

    def test_auth_denials_and_rate_limit_stop(self):
        for code in (401, 403, 429):
            with self.subTest(code=code):
                r = self.run_witness(
                    lambda i, e: e.update(model_metadata_http_status=code)
                )
                self.assertTrue(r["access_stop_required"])
                self.assertEqual(r["state"], "HOLD")

    def test_unknown_http_and_boolean_200(self):
        for status in (None, True, "200"):
            with self.subTest(status=status):
                r = self.run_witness(
                    lambda i, e: e.update(model_metadata_http_status=status)
                )
                self.assertEqual(r["reason"], "MODEL_METADATA_STATUS_UNOBSERVED")

    def test_deadline_does_not_override_bad_evidence(self):
        r = self.run_witness(lambda i, e: e["trainer_state"].update(global_step=199))
        self.assertEqual(r["reason"], "TRAINER_STATE_NOT_TERMINAL_AND_PINNED")

    def test_immutable_binding_mismatch(self):
        r = self.run_witness(lambda i, e: e.update(dataset_revision=SHA_A))
        self.assertEqual(r["reason"], "IMMUTABLE_BINDING_UNVERIFIED")

    def test_owner_epoch_mismatch(self):
        r = self.run_witness(lambda i, e: e.update(owner_generation=3))
        self.assertEqual(r["reason"], "OWNER_EPOCH_MISMATCH")

    def test_bool_generation_and_step_fail(self):
        self.assertEqual(
            self.run_witness(lambda i, e: i.update(owner_generation=True))["reason"],
            "OWNER_IDENTITY_UNVERIFIED",
        )
        r = self.run_witness(lambda i, e: e["trainer_state"].update(global_step=True))
        self.assertEqual(r["reason"], "TRAINER_STATE_NOT_TERMINAL_AND_PINNED")

    def test_bad_artifact_file_or_tokenizer(self):
        r = self.run_witness(
            lambda i, e: e["artifact"].update(complete_config_and_tokenizer=False)
        )
        self.assertEqual(r["reason"], "FINAL_ARTIFACT_UNVERIFIED")

    def test_nonindependent_or_divergent_backup(self):
        for k, v in (("independent_readback", False), ("adapter_sha256", D_B)):
            with self.subTest(k=k):
                self.assertEqual(
                    self.run_witness(lambda i, e: e["backup"].update({k: v}))["reason"],
                    "INDEPENDENT_BACKUP_UNVERIFIED",
                )

    def test_expired_and_future_clock(self):
        r = self.run_witness(
            lambda i, e: e.update(observed_at_utc="2026-11-19T14:30:00Z")
        )
        self.assertEqual(r["reason"], "EVIDENCE_STALE_OR_CLOCK_UNCERTAIN")
        clock = self.run_witness(now="local-now")
        self.assertEqual(clock["reason"], "CLOCK_UNOBSERVED")

    def test_provider_counts_and_pending(self):
        r = self.run_witness(
            lambda i, e: e["provider"]["instance_counts"].update(running_count=True)
        )
        self.assertEqual(r["reason"], "PROVIDER_INSTANCE_COUNTS_INVALID")
        r = self.run_witness(lambda i, e: e["provider"].update(update_pending=True))
        self.assertEqual(r["reason"], "PROVIDER_PENDING_CHANGE_UNCERTAIN")

    def test_provider_staleness(self):
        r = self.run_witness(
            lambda i, e: e["provider"].update(observed_at_utc="2026-11-19T14:40:00Z")
        )
        self.assertEqual(r["reason"], "PROVIDER_READBACK_STALE")

    def test_no_secret_or_private_identity_echo(self):
        intent, evidence = valid()
        evidence["owner_private_note"] = "do-not-echo-private-owner-text"
        evidence["provider"]["private_note"] = "sensitive-internal-state"
        result = assess_stop_preflight(intent=intent, evidence=evidence, now_utc=NOW)
        self.assertNotIn("do-not-echo-private-owner-text", str(result))
        self.assertNotIn("sensitive-internal-state", str(result))


if __name__ == "__main__":
    unittest.main()
