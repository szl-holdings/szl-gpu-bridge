"""Synthetic, offline regression tests for SaladCloud transition readback."""
import unittest
from tools.salad.provider_transition_witness import witness


def sample(**override):
    group = {
        "name": "szl-train-test-r1", "replicas": 1, "pending_change": False,
        "current_state": {"status": "running", "instance_status_counts": {
            "allocating_count": 0, "creating_count": 0, "running_count": 1,
            "stopping_count": 0}},
    }
    v = dict(transport_exit=0, http_status=200, body=b"{}", group=group,
             instances={"instances": [{"state": "running"}]},
             name="szl-train-test-r1", requested_replicas=1)
    v.update(override)
    return witness(**v)


class WitnessTests(unittest.TestCase):
    def test_provider_running_not_training(self):
        r = sample()
        self.assertEqual(r["state"], "PROVIDER_STATE_OBSERVED_NOT_TRAINING")
        self.assertTrue(r["group_state_reconciled"])
        self.assertFalse(r["trainer_process_verified"])
        self.assertFalse(r["training_launch_authorized"])
        self.assertFalse(r["shutdown_authorized"])

    def test_curl_zero_with_http_400_fails(self):
        r = sample(http_status=400, body=b'{"type":"pending_update_in_progress"}')
        self.assertEqual(r["state"], "PROVIDER_NON_SUCCESS")
        self.assertFalse(r["provider_write_acknowledged"])

    def test_2xx_with_error_envelope_fails(self):
        r = sample(body=b'{"status":400,"title":"Bad Request"}')
        self.assertEqual(r["state"], "PROVIDER_ERROR_ENVELOPE")

    def test_2xx_masked_auth_denial_stops(self):
        r = sample(http_status=200, body=b'{"status":403,"detail":"hidden"}')
        self.assertEqual(r["state"], "ACCESS_DENIED_403")
        self.assertTrue(r["access_stop_required"])

    def test_masked_500_and_pending_update_link_fails(self):
        self.assertEqual(sample(body=b'{"status":500}')["state"], "PROVIDER_ERROR_ENVELOPE")
        self.assertEqual(sample(body=b'{"type":"https://example.invalid/pending_update_in_progress"}')["state"], "PROVIDER_ERROR_ENVELOPE")

    def test_unknown_http_status_fails(self):
        self.assertEqual(sample(http_status=None)["state"], "HTTP_STATUS_UNOBSERVED")
        self.assertEqual(sample(http_status=True)["state"], "HTTP_STATUS_UNOBSERVED")

    def test_transport_failure_even_with_http_200(self):
        self.assertEqual(sample(transport_exit=7)["state"], "TRANSPORT_FAILED_OR_UNKNOWN")

    def test_auth_and_rate_limit_stop(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                r = sample(http_status=status)
                self.assertTrue(r["access_stop_required"])
                self.assertFalse(r["provider_write_acknowledged"])

    def test_no_invented_readback(self):
        self.assertEqual(sample(group=None)["state"], "READBACK_UNOBSERVED")
        self.assertEqual(sample(instances=None)["state"], "READBACK_UNOBSERVED")

    def test_readback_target_mismatch(self):
        self.assertEqual(sample(name="other-group")["state"], "READBACK_GROUP_MISMATCH")
        self.assertEqual(sample(requested_replicas=0)["state"], "READBACK_REPLICA_MISMATCH")

    def test_counts_missing_or_bool_are_invalid(self):
        d = sample()["actual_counts"]
        del d["running_count"]
        g = sample()["actual_counts"]
        g["running_count"] = True
        for counts in (d, g):
            group = sample_group(counts=counts)
            self.assertEqual(sample(group=group)["state"], "READBACK_COUNTS_INVALID")

    def test_pending_change_refuses(self):
        group = sample_group()
        group["pending_change"] = True
        self.assertEqual(sample(group=group)["state"], "PROVIDER_UPDATE_PENDING")

    def test_zero_counts_do_not_prove_completion(self):
        group = sample_group(replicas=0, counts={k: 0 for k in (
            "allocating_count", "creating_count", "running_count", "stopping_count")})
        r = sample(group=group, instances={"instances": []}, requested_replicas=0)
        self.assertEqual(r["state"], "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION")
        self.assertFalse(r["checkpoint_or_final_verified"])

    def test_allocating_not_training(self):
        group = sample_group(counts={"allocating_count": 0, "creating_count": 1,
                                     "running_count": 0, "stopping_count": 0})
        r = sample(group=group, instances={"instances": [{"state": "creating"}]})
        self.assertEqual(r["state"], "NO_RUNNING_INSTANCE_PROVIDER_REPORTED")

    def test_duplicate_json_and_nonfinite_refused(self):
        self.assertEqual(sample(body=b'{"status":200,"status":200}')["state"], "PROVIDER_BODY_INVALID")
        self.assertEqual(sample(body=b'{"value":NaN}')["state"], "PROVIDER_BODY_INVALID")

    def test_no_provider_mutations(self):
        self.assertFalse(sample()["provider_mutation_performed"])
        self.assertNotIn("token", sample())


def sample_group(*, replicas=1, counts=None):
    return {"name": "szl-train-test-r1", "replicas": replicas, "pending_change": False,
            "current_state": {"status": "running", "instance_status_counts": counts or {
                "allocating_count": 0, "creating_count": 0, "running_count": 1,
                "stopping_count": 0}}}


if __name__ == "__main__":
    unittest.main()
