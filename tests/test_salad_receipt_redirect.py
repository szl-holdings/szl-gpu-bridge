"""Synthetic HTTPS receipt-readback redirect and HTTP-error regression gates.

Tests run with the existing fake Salad worker fixture: no live credentials,
network requests, Hub writes, GPUs, training, or controller mutations.
"""

import importlib.util
import urllib.error
import urllib.request
import unittest
from unittest.mock import Mock, patch

from test_salad_worker_terminal import TerminalContract, WORKER

_REVISION = "a" * 40
_DIGEST = "b" * 64
_URL = (
    "https://huggingface.co/datasets/fixture/receipts/resolve/"
    + _REVISION
    + "/mesh/salad-"
    + _DIGEST
    + ".json"
)


class ReceiptRedirectSafety(unittest.TestCase):
    def setUp(self):
        self.fixture = TerminalContract(
            "test_unconfigured_or_missing_token_cannot_emit_local_success"
        )
        self.fixture.setUp()
        # The existing terminal tests deliberately replace _readback_fetch
        # with a deterministic fake. Load a second source-pinned worker module
        # with the same mocked dependencies but without that transport seam
        # override, so THESE tests evaluate the actual redirect implementation.
        spec = importlib.util.spec_from_file_location(
            "salad_receipt_transport_under_test", WORKER
        )
        assert spec is not None and spec.loader is not None
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

    def tearDown(self):
        self.fixture.doCleanups()

    def _origin_request(self):
        return urllib.request.Request(
            _URL,
            headers={
                "Authorization": "Bearer SYNTHETIC_NOT_A_CREDENTIAL",
                "Cookie": "synthetic-session",
                "Proxy-Authorization": "Basic SYNTHETIC",
            },
        )

    def _header_names(self, request):
        return {name.lower() for name, _ in request.header_items()}

    def test_cross_origin_hf_cdn_strips_all_credentials(self):
        redirected = self.worker._SafeReceiptRedirect().redirect_request(
            self._origin_request(),
            None,
            302,
            "Found",
            {},
            "https://cas-bridge.xethub.hf.co/signed-blob?synthetic=yes",
        )
        self.assertEqual(
            redirected.full_url,
            "https://cas-bridge.xethub.hf.co/signed-blob?synthetic=yes",
        )
        self.assertFalse(
            self._header_names(redirected)
            & {"authorization", "proxy-authorization", "cookie"}
        )

    def test_cross_origin_second_redirect_cannot_restore_credentials(self):
        handler = self.worker._SafeReceiptRedirect()
        first = handler.redirect_request(
            self._origin_request(),
            None,
            302,
            "Found",
            {},
            "https://cas-bridge.xethub.hf.co/signed-blob",
        )
        second = handler.redirect_request(
            first, None, 302, "Found", {}, "https://cdn-lfs.hf.co/blob"
        )
        for request in (first, second):
            self.assertFalse(
                self._header_names(request)
                & {"authorization", "proxy-authorization", "cookie"}
            )

    def test_same_origin_https_keeps_initial_bearer(self):
        # A private Git-backed receipt may use a same-origin HTTPS redirect.
        redirect = self.worker._SafeReceiptRedirect().redirect_request(
            self._origin_request(),
            None,
            302,
            "Found",
            {},
            "https://huggingface.co/datasets/fixture/receipts/resolve/"
            + _REVISION
            + "/mesh/salad-"
            + _DIGEST
            + ".json?synthetic=yes",
        )
        self.assertIn("authorization", self._header_names(redirect))

    def test_downgrade_userinfo_lookalike_and_unknown_hosts_hold(self):
        forbidden = (
            "http://cdn-lfs.hf.co/blob",
            "https://huggingface.co.attacker.invalid/blob",
            "https://attacker.invalid/blob",
            "https://user@cdn-lfs.hf.co/blob",
            "https://cdn-lfs.hf.co:8443/blob",
            "https://cas-bridge.xethub.hf.co.attacker.invalid/blob",
            "https://cdn-lfs.hf.co/blob#fragment",
        )
        for target in forbidden:
            with self.subTest(category=target.split("/")[2]):
                with self.assertRaises(self.worker.ReceiptError):
                    self.worker._SafeReceiptRedirect().redirect_request(
                        self._origin_request(), None, 302, "Found", {}, target
                    )

    def test_invalid_initial_origin_never_starts_network(self):
        with patch.object(
            self.worker.urllib.request,
            "build_opener",
            side_effect=AssertionError("NO_NETWORK_EXPECTED"),
        ):
            for target in (
                "http://huggingface.co/datasets/fixture/receipts/resolve/"
                + _REVISION
                + "/mesh/salad-"
                + _DIGEST
                + ".json",
                "https://huggingface.co.attacker.invalid/evil",
                _URL + "?unbound=1",
                "https://huggingface.co/datasets/fixture/receipts/resolve/"
                + "main/mesh/salad-"
                + _DIGEST
                + ".json",
            ):
                with self.subTest(target=target[:54]):
                    with self.assertRaises(self.worker.ReceiptError):
                        self.fixture.real_readback_fetch(
                            target, "SYNTHETIC_NOT_A_CREDENTIAL", 256
                        )

    def test_initial_https_uses_guarded_opener_and_exact_byte_bound(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, size):
                self.requested_bytes = size
                return b"expected"

        response = Response()
        opener = Mock()
        opener.open.return_value = response
        with patch.object(
            self.worker.urllib.request, "build_opener", return_value=opener
        ) as build:
            result = self.fixture.real_readback_fetch(_URL, "SYNTHETIC_NOT_A_CREDENTIAL", 9)
        self.assertEqual(result, b"expected")
        self.assertEqual(response.requested_bytes, 9)
        self.assertIsInstance(build.call_args.args[0], self.worker._SafeReceiptRedirect)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, _URL)
        self.assertIn("authorization", self._header_names(request))

    def test_missing_status_is_not_success(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, n):
                return b"expected"

        opener = Mock()
        opener.open.return_value = Response()
        with patch.object(
            self.worker.urllib.request, "build_opener", return_value=opener
        ):
            with self.assertRaises(self.worker.ReceiptError):
                self.fixture.real_readback_fetch(_URL, "SYNTHETIC_NOT_A_CREDENTIAL", 9)

    def test_urllib_401_403_404_fail_without_retry_sleep(self):
        for status in (401, 403, 404):
            with self.subTest(http=status):
                exc = urllib.error.HTTPError(_URL, status, "test", {}, None)
                operation = Mock(side_effect=exc)
                self.fixture.sleep.reset_mock()
                with self.assertRaises(self.worker.ReceiptError):
                    self.worker.receipt_retry(operation, "receipt readback")
                self.assertEqual(operation.call_count, 1)
                self.fixture.sleep.assert_not_called()

    def test_urllib_429_retries_only_within_budget(self):
        operation = Mock(
            side_effect=urllib.error.HTTPError(_URL, 429, "test", {}, None)
        )
        self.fixture.sleep.reset_mock()
        with self.assertRaises(self.worker.ReceiptError):
            self.worker.receipt_retry(operation, "receipt readback")
        self.assertEqual(operation.call_count, 6)
        self.assertEqual(self.fixture.sleep.call_count, 5)


if __name__ == "__main__":
    unittest.main()
