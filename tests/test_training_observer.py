"""Read-only telemetry must distinguish metadata, provider status and evidence."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest import mock

from test_artifact_qualification import FakeApi, fixture
from tools.salad import training_observer as observer


class Response(io.BytesIO):
    status = 200


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeApi(fixture())
        self.candidate = {"repo": "SZLHOLDINGS/example", "target_steps": 200}

    def observe(self):
        return observer.observe_training(self.api, self.api.fetch, self.candidate)

    def test_terminal_metadata_never_claims_completion(self):
        result = self.observe()
        self.assertEqual(result["state"], "TERMINAL_METADATA_PRESENT")
        self.assertFalse(result["training_complete_verified"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["trainer_process_liveness_verified"])
        self.assertEqual(len(self.api.calls), 1)

    def test_checkpoint_without_final_is_not_complete(self):
        self.api = FakeApi(
            {k: v for k, v in fixture().items() if not k.startswith("final/")}
        )
        self.assertEqual(
            self.observe()["state"], "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE"
        )

    def test_partial_final_tokenizer_is_not_complete_metadata(self):
        for missing in ("final/tokenizer.json", "final/tokenizer_config.json"):
            with self.subTest(missing=missing):
                blobs = fixture()
                del blobs[missing]
                self.api = FakeApi(blobs)
                result = self.observe()
                self.assertTrue(result["snapshot"]["final_tokenizer"])
                self.assertEqual(
                    result["state"], "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE"
                )
                self.assertFalse(result["training_complete_verified"])
                self.assertFalse(result["promotion_authorized"])
                self.assertFalse(result["trainer_process_liveness_verified"])

    def test_training_progress_is_not_process_liveness(self):
        self.api = FakeApi(fixture(step=175))
        result = self.observe()
        self.assertEqual(result["state"], "CHECKPOINT_PROGRESS_OBSERVED")
        self.assertFalse(result["trainer_process_liveness_verified"])

    def test_overshoot_distinguished(self):
        self.api = FakeApi(fixture(step=201))
        self.assertEqual(self.observe()["state"], "TERMINAL_STEP_OVERSHOOT")

    def test_target_contract_distinguished(self):
        self.api = FakeApi(fixture(target=250))
        self.assertEqual(self.observe()["state"], "TARGET_CONTRACT_MISMATCH")

    def test_repo_presence_does_not_imply_checkpoint(self):
        self.api = FakeApi({"README.md": b"present"})
        self.assertEqual(self.observe()["state"], "NO_CHECKPOINT_OBSERVED")

    def test_unavailable_does_not_reuse_old_success(self):
        self.assertEqual(self.observe()["state"], "TERMINAL_METADATA_PRESENT")
        self.api.info.sha = "main"
        result = self.observe()
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertFalse(result["training_complete_verified"])

    def test_provider_empty_success_distinguished(self):
        result = observer.provider_observation(
            "szl-example",
            "not-a-real-token",
            opener=lambda *a, **k: Response(b'{"instances":[]}'),
        )
        self.assertEqual(result["state"], "OBSERVED")
        self.assertEqual(result["instance_count"], 0)

    def test_provider_unauthorized_is_unavailable_not_empty(self):
        def denied(*args, **kwargs):
            raise urllib.error.HTTPError(
                "https://example.invalid", 401, "unauthorized", {}, None
            )

        result = observer.provider_observation(
            "szl-example", "not-a-real-token", opener=denied
        )
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertNotIn("instance_count", result)
        self.assertEqual(result["reason"], "PROVIDER_HTTP_401")

    def test_provider_malformed_is_unavailable(self):
        for raw in [
            b"{}",
            b'{"instances":null}',
            b'{"instances":[null]}',
            b'{"instances":[{}]}',
        ]:
            with self.subTest(raw=raw):
                result = observer.provider_observation(
                    "szl-example",
                    "not-a-real-token",
                    opener=lambda *a, **k: Response(raw),
                )
                self.assertEqual(result["state"], "UNAVAILABLE")

    def test_provider_running_never_proves_trainer_liveness(self):
        def opened(request, timeout):
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(
                request.get_header("User-agent"), "szl-training-observer/1.0"
            )
            self.assertEqual(request.get_header("Accept"), "application/json")
            self.assertEqual(timeout, 12)
            return Response(b'{"instances":[{"state":"running","ready":true}]}')

        result = observer.provider_observation(
            "szl-example", "not-a-real-token", opener=opened
        )
        self.assertEqual(result["ready_count"], 1)
        self.assertFalse(result["trainer_process_liveness_verified"])
        self.assertNotIn("not-a-real-token", json.dumps(result))

    def test_provider_body_budget_enforced(self):
        result = observer.provider_observation(
            "szl-example",
            "not-a-real-token",
            opener=lambda *a, **k: Response(b" " * 1_000_001),
        )
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertEqual(result["reason"], "PROVIDER_RESPONSE_BYTE_BUDGET")

    def test_redirect_cannot_forward_provider_credentials(self):
        handler = observer.NoRedirect()
        request = observer.urllib.request.Request(
            "https://api.salad.com/example",
            headers={"Salad-Api-Key": "not-a-real-token"},
        )
        self.assertIsNone(
            handler.redirect_request(
                request, None, 302, "Found", {}, "https://example.invalid/elsewhere"
            )
        )

    def test_no_token_makes_no_request(self):
        result = observer.provider_observation(
            "szl-example", None, opener=lambda *a, **k: self.fail("unexpected request")
        )
        self.assertEqual(result["state"], "UNAVAILABLE")

    def test_unsafe_group_makes_no_request(self):
        result = observer.provider_observation(
            "../elsewhere",
            "not-a-real-token",
            opener=lambda *a, **k: self.fail("unexpected request"),
        )
        self.assertEqual(result["state"], "UNAVAILABLE")

    def test_atomic_write_preserves_valid_old_status_on_invalid_new_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "status.json"
            observer.atomic_status(path, {"state": "OLD"})
            with self.assertRaises(ValueError):
                observer.atomic_status(path, {"state": float("nan")})
            self.assertEqual(json.loads(path.read_bytes()), {"state": "OLD"})
            self.assertEqual(len(list(Path(temp).iterdir())), 1)
            observer.atomic_status(path, {"state": "NEW"})
            self.assertEqual(json.loads(path.read_bytes()), {"state": "NEW"})

    def test_salad_auth_or_rate_limit_denial_stops_provider_lane(self):
        for code in (401, 403, 429):
            with self.subTest(code=code):

                def denied(*args, **kwargs):
                    raise urllib.error.HTTPError(
                        "https://example.invalid", code, "denied", {}, None
                    )

                result = observer.provider_observation(
                    "szl-example", "not-a-real-token", opener=denied
                )
                self.assertTrue(result["access_stop_required"])
                self.assertEqual(result["http_status"], code)
                self.assertEqual(result["reason"], "PROVIDER_HTTP_" + str(code))
                self.assertNotIn("not-a-real-token", json.dumps(result))

    def test_salad_status_only_denial_is_not_a_generic_error(self):
        response = Response(b"{}")
        response.status = 429
        result = observer.provider_observation(
            "szl-example",
            "not-a-real-token",
            opener=lambda *args, **kwargs: response,
        )
        self.assertEqual(result["state"], "UNAVAILABLE")
        self.assertTrue(result["access_stop_required"])
        self.assertEqual(result["http_status"], 429)

    def test_salad_non_denial_does_not_stop_other_reads(self):
        def not_found(*args, **kwargs):
            raise urllib.error.HTTPError(
                "https://example.invalid", 404, "not found", {}, None
            )

        result = observer.provider_observation(
            "szl-example", "not-a-real-token", opener=not_found
        )
        self.assertFalse(result["access_stop_required"])
        self.assertEqual(result["reason"], "PROVIDER_HTTP_404")

    def test_hub_auth_or_rate_limit_denial_stops_hub_lane(self):
        for code in (401, 403, 429):
            with self.subTest(code=code):

                class DeniedApi:
                    def model_info(self, *args, **kwargs):
                        raise urllib.error.HTTPError(
                            "https://example.invalid", code, "denied", {}, None
                        )

                result = observer.observe_training(
                    DeniedApi(), self.api.fetch, self.candidate
                )
                self.assertEqual(result["state"], "UNAVAILABLE")
                self.assertTrue(result["access_stop_required"])
                self.assertEqual(result["http_status"], code)
                self.assertEqual(result["reason"], "HUB_HTTP_" + str(code))
                self.assertFalse(result["promotion_authorized"])
                self.assertFalse(result["training_complete_verified"])

    def test_multi_candidate_multi_cycle_denials_are_not_retried(self):
        # Integration check: both provider lanes stop for the entire bounded
        # run. Other resources are not restarted, promoted, or mutated.
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            spec = folder / "spec.json"
            status = folder / "status.json"
            history = folder / "history.jsonl"
            candidates = [
                {
                    "repo": "SZLHOLDINGS/example-a",
                    "target_steps": 5,
                    "salad_group": "szl-example-a",
                },
                {
                    "repo": "SZLHOLDINGS/example-b",
                    "target_steps": 5,
                    "salad_group": "szl-example-b",
                },
            ]
            spec.write_text(json.dumps({"candidates": candidates}))
            hub_calls = []
            salad_calls = []

            def denied_hub(api, fetch, candidate):
                hub_calls.append(candidate["repo"])
                return {
                    "repo": candidate["repo"],
                    "state": "UNAVAILABLE",
                    "reason": "HUB_HTTP_429",
                    "access_stop_required": True,
                    "promotion_authorized": False,
                    "training_complete_verified": False,
                    "trainer_process_liveness_verified": False,
                }

            def denied_salad(group, token):
                salad_calls.append(group)
                return {
                    "state": "UNAVAILABLE",
                    "reason": "PROVIDER_HTTP_403",
                    "access_stop_required": True,
                    "trainer_process_liveness_verified": False,
                }

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.HfApi = lambda: object()
            fake_hub.hf_hub_download = lambda **kwargs: None
            argv = [
                "training_observer",
                "--spec",
                str(spec),
                "--status",
                str(status),
                "--history",
                str(history),
                "--cycles",
                "2",
                "--interval",
                "60",
            ]
            with (
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(observer, "observe_training", side_effect=denied_hub),
                mock.patch.object(
                    observer, "provider_observation", side_effect=denied_salad
                ),
                mock.patch.object(observer.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(observer.main(), 0)

            self.assertEqual(hub_calls, ["SZLHOLDINGS/example-a"])
            self.assertEqual(salad_calls, ["szl-example-a"])
            output = json.loads(status.read_bytes())
            self.assertEqual(output["cycle"], 2)
            self.assertEqual(
                output["provider_lane_stops"], {"huggingface": True, "salad": True}
            )
            self.assertTrue(
                all(
                    entry["reason"] == "HUB_ACCESS_STOPPED_AFTER_DENIAL"
                    and entry["provider"]["reason"]
                    == "PROVIDER_ACCESS_STOPPED_AFTER_DENIAL"
                    and not entry["promotion_authorized"]
                    for entry in output["candidates"]
                )
            )
            self.assertEqual(len(history.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
