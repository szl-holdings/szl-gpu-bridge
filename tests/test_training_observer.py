"""Read-only telemetry must distinguish metadata, provider status and evidence."""

import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

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

    def test_provider_identifies_observer_at_edge(self):
        def edge(request, timeout):
            headers = {key.lower(): value for key, value in request.header_items()}
            if headers.get("user-agent") != "szl-training-observer/1.0":
                raise urllib.error.HTTPError(
                    request.full_url, 403, "client identification required", {}, None
                )
            self.assertEqual(headers.get("accept"), "application/json")
            self.assertEqual(headers.get("salad-api-key"), "not-a-real-token")
            self.assertEqual(
                request.full_url, observer.SALAD_BASE + "szl-example/instances"
            )
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(timeout, 12)
            return Response(b'{"instances":[]}')

        result = observer.provider_observation(
            "szl-example", "not-a-real-token", opener=edge
        )
        self.assertEqual(result["state"], "OBSERVED")
        self.assertEqual(result["instance_count"], 0)
        self.assertNotIn("not-a-real-token", json.dumps(result))

    def test_provider_redirect_cannot_forward_credentials(self):
        request = observer.urllib.request.Request(
            observer.SALAD_BASE + "szl-example/instances",
            headers={"Salad-Api-Key": "not-a-real-token"},
        )
        handler = observer.NoRedirect()
        for code in [301, 302, 303, 307, 308]:
            with self.subTest(code=code):
                self.assertIsNone(
                    handler.redirect_request(
                        request, None, code, "redirect", {}, "https://example.invalid/"
                    )
                )

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
            self.assertEqual(timeout, 12)
            return Response(b'{"instances":[{"state":"running","ready":true}]}')

        result = observer.provider_observation(
            "szl-example", "not-a-real-token", opener=opened
        )
        self.assertEqual(result["ready_count"], 1)
        self.assertFalse(result["trainer_process_liveness_verified"])
        self.assertNotIn("not-a-real-token", json.dumps(result))

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


if __name__ == "__main__":
    unittest.main()
