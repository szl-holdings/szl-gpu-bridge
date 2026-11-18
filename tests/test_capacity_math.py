"""Synthetic, adversarial capacity math regression; no GPU, network or provider calls."""

from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import unittest

from tools.salad import capacity_math as subject

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests" / "capacity_synthetic_sample.json"
T = datetime(2026, 11, 17, 19, 2, tzinfo=timezone.utc)


def fixture():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def evaluate(value=None, at=T):
    return subject.evaluate(fixture() if value is None else value, as_of=at)


class CapacityMathTests(unittest.TestCase):
    def test_reference_model_exact_algebra_is_not_a_scale_command(self):
        d = evaluate()
        self.assertEqual(d["schema"], subject.OUTPUT_SCHEMA)
        self.assertEqual(d["state"], "SYNTHETIC_ANALYSIS")
        self.assertAlmostEqual(d["observation"]["arrival_rate_per_second"], 0.75)
        self.assertAlmostEqual(d["observation"]["completion_rate_per_second"], 0.70)
        self.assertEqual(d["observation"]["backlog_delta_count"], 3)
        self.assertAlmostEqual(d["queue"]["mean_service_seconds"], 1.0)
        self.assertAlmostEqual(d["queue"]["arrival_cv_squared"], 0.09)
        self.assertAlmostEqual(d["queue"]["service_cv_squared"], 0.04)
        self.assertAlmostEqual(d["queue"]["rho_per_ready_replica"], 0.75)
        self.assertEqual(d["queue"]["kingman"]["state"], "G_G_1_APPROXIMATION")
        self.assertAlmostEqual(d["queue"]["kingman"]["mean_wait_seconds"], 0.195)
        self.assertAlmostEqual(
            d["queue"]["kingman_mean_response_residual_seconds"],
            (20 * 1.5 + 10 * 1.9) / 30 - (1.0 + 0.195),
        )
        self.assertEqual(
            d["queue"]["residual_scope"],
            "UNVERIFIED_SAMPLE_VS_APPROXIMATION_NO_SIGNIFICANCE_TEST",
        )
        self.assertAlmostEqual(
            d["queue"]["netflix_vegas_queue_signal"], 12 * (1 - 0.055 / 0.085)
        )
        self.assertAlmostEqual(
            d["queue"]["mean_offered_concurrency_little_identity"],
            0.75 * d["queue"]["mean_response_seconds"],
        )
        self.assertEqual(d["capacity"]["hpa_like_raw_replicas"], 2)
        self.assertEqual(d["capacity"]["bounded_planning_replicas"], 2)
        self.assertEqual(d["capacity"]["scale_action"], "NONE")
        self.assertFalse(d["capacity"]["proposed_gpu_replica_change_authorized"])
        self.assertEqual(d["authority"]["provider_writes"], 0)
        self.assertFalse(d["authority"]["promotion_authorized"])
        self.assertFalse(d["authority"]["model_quality_verified"])
        self.assertFalse(d["source"]["source_revision_verified"])
        self.assertFalse(d["source"]["telemetry_authenticity_verified"])
        self.assertFalse(d["observation"]["current_eligible"])

    def test_kingman_is_not_applied_as_ggc(self):
        value = fixture()
        value["capacity"]["ready_replicas"] = 3
        d = evaluate(value)
        self.assertEqual(d["queue"]["kingman"]["state"], "MULTISERVER_NOT_G_G_1")
        self.assertIsNone(d["queue"]["kingman"]["mean_wait_seconds"])
        self.assertIsNone(d["queue"]["kingman_mean_response_residual_seconds"])
        self.assertAlmostEqual(d["queue"]["rho_per_ready_replica"], 0.25)

    def test_at_or_above_single_server_saturation_is_explicit(self):
        value = fixture()
        value["requests"]["arrivals"] = 65
        d = evaluate(value)
        self.assertGreater(d["queue"]["rho_per_ready_replica"], 1)
        self.assertEqual(
            d["queue"]["kingman"]["state"], "NO_STATIONARY_FINITE_WAIT_AT_OR_ABOVE_ONE"
        )
        self.assertIsNone(d["queue"]["kingman"]["mean_wait_seconds"])
        self.assertNotEqual(d["capacity"]["scale_action"], "START")

    def test_kingman_matches_mm1_ideal_variability_only_under_assumptions(self):
        value = fixture()
        # Artificial descriptive CV^2 = 1 (alternating positive values with
        # near-zero entries) and rho 0.4: Kingman approaches M/M/1 Wq.
        value["requests"]["arrivals"] = 24
        value["requests"]["service_time_seconds"] = [0.000001, 1.999999] * 15
        value["requests"]["interarrival_seconds"] = [0.000001, 1.999999] * 10
        d = evaluate(value)
        self.assertAlmostEqual(d["queue"]["rho_per_ready_replica"], 0.4)
        self.assertAlmostEqual(
            d["queue"]["kingman"]["mean_wait_seconds"],
            0.4 / (1 - 0.4),
            places=5,
        )
        self.assertEqual(
            d["queue"]["kingman"]["assumption"],
            "ONE_STATIONARY_SINGLE_SERVER_QUEUE_NOT_VERIFIED",
        )

    def test_synthetic_mm1_reference_simulation_calibrates_ideal_formula(self):
        """Deterministic M/M/1 discrete-event benchmark, not an SZL GPU trace."""
        rng = random.Random(20261117)
        wait = 0.0
        previous_service = rng.expovariate(1.0)
        observed_waits = []
        observed_arrivals = []
        observed_services = []
        for i in range(80000):
            gap = rng.expovariate(0.7)
            wait = max(0.0, wait + previous_service - gap)
            previous_service = rng.expovariate(1.0)
            observed_arrivals.append(gap)
            observed_services.append(previous_service)
            if i >= 5000:
                observed_waits.append(wait)
        rho = 0.7
        empirical = statistics.fmean(observed_waits)
        kingman = (
            rho
            / (1 - rho)
            * (subject._cv2(observed_arrivals) + subject._cv2(observed_services))
            / 2
            * statistics.fmean(observed_services)
        )
        mm1_exact = rho / (1.0 - rho)
        self.assertAlmostEqual(mm1_exact, 7.0 / 3.0, places=10)
        self.assertLess(abs(empirical - mm1_exact), 0.20)
        self.assertLess(abs(kingman - empirical), 0.20)

    def test_wilson_upper_bound_never_uses_zero_failures_as_zero_risk(self):
        self.assertAlmostEqual(
            subject._wilson_upper(0, 100), 0.03699349820698569, places=10
        )
        self.assertGreater(subject._wilson_upper(0, 100), 0)
        self.assertGreaterEqual(subject._wilson_upper(99, 100), 0.99)
        self.assertEqual(subject._wilson_upper(1, 1), 1.0)

    def test_empirical_quantile_is_nearest_rank(self):
        self.assertEqual(subject._percentile(list(range(1, 21)), 0.95), 19)
        self.assertEqual(subject._percentile([1, 1, 10, 20], 0.95), 20)

    def test_hpa_like_ceiling_is_not_silent_max_saturation(self):
        value = fixture()
        value["capacity"]["max_replicas"] = 1
        d = evaluate(value)
        self.assertEqual(d["capacity"]["planning_state"], "CAPACITY_CEILING_EXCEEDED")
        self.assertEqual(d["capacity"]["hpa_like_raw_replicas"], 2)
        self.assertEqual(d["capacity"]["bounded_planning_replicas"], 1)
        self.assertFalse(d["capacity"]["proposed_gpu_replica_change_authorized"])

    def test_downscale_is_not_auto_authorized(self):
        value = fixture()
        value["capacity"]["ready_replicas"] = 5
        d = evaluate(value)
        self.assertEqual(d["capacity"]["planning_state"], "DOWNSCALE_NOT_EVALUATED")
        self.assertEqual(d["capacity"]["scale_action"], "NONE")

    def test_completed_more_than_arrivals_is_valid_backlog_drain(self):
        value = fixture()
        value["requests"]["arrivals"] = 32
        value["requests"]["completed"] = 40
        d = evaluate(value)
        self.assertEqual(d["observation"]["backlog_delta_count"], -8)

    def test_nonuniform_samples_and_unverified_sources_are_not_promoted(self):
        sample = fixture()
        sample["source"]["kind"] = "OPERATOR_REPORTED_UNVERIFIED"
        output = evaluate(sample, at=T + timedelta(days=40))
        self.assertEqual(output["state"], "UNVERIFIED_OPERATOR_SAMPLE")
        self.assertFalse(output["observation"]["current_eligible"])
        self.assertGreater(output["observation"]["seconds_since_window"], 3600)
        self.assertFalse(output["authority"]["signed_evidence"])

    def test_unknown_keys_and_source_labels_fail_closed(self):
        cases = [
            ("INVALID_INPUT_FIELDS", lambda d: d.__setitem__("x402_token", "secret")),
            (
                "SOURCE_BINDING_FIELDS",
                lambda d: d["source"].__setitem__("secret", "value"),
            ),
            (
                "SOURCE_BINDING_UNVERIFIED",
                lambda d: d["source"].__setitem__("repository", "foreign"),
            ),
            (
                "SOURCE_BINDING_UNVERIFIED",
                lambda d: d["source"].__setitem__("revision", "main"),
            ),
            (
                "SOURCE_BINDING_UNVERIFIED",
                lambda d: d["source"].__setitem__("kind", ["SYNTHETIC"]),
            ),
            (
                "SOURCE_BINDING_UNVERIFIED",
                lambda d: d["source"].__setitem__("sample_sha256", "0" * 64),
            ),
            ("WINDOW_FIELDS", lambda d: d["window"].__setitem__("tenant", "private")),
            (
                "CAPACITY_FIELDS",
                lambda d: d["capacity"].__setitem__("gpu_key", "secret"),
            ),
            (
                "REQUEST_FIELDS",
                lambda d: d["requests"].__setitem__("model_url", "evil"),
            ),
        ]
        for reason, change in cases:
            with self.subTest(reason=reason, change=change):
                data = fixture()
                change(data)
                with self.assertRaisesRegex(subject.Hold, reason):
                    evaluate(data)

    def test_numeric_type_coercions_and_nonfinite_are_denied(self):
        cases = [
            ("REPLICA_COUNT_INVALID", "capacity", "ready_replicas", True),
            ("REPLICA_COUNT_INVALID", "capacity", "ready_replicas", 1.0),
            ("INFLIGHT_LIMIT_INVALID", "capacity", "inflight_limit", 0),
            ("UTILIZATION_TARGET_INVALID", "capacity", "target_utilization", 1),
            (
                "UTILIZATION_TARGET_INVALID",
                "capacity",
                "target_utilization",
                float("nan"),
            ),
            (
                "UTILIZATION_TARGET_INVALID",
                "capacity",
                "target_utilization",
                float("inf"),
            ),
            ("ARRIVALS_INVALID", "requests", "arrivals", False),
            ("COMPLETIONS_INVALID", "requests", "completed", -1),
            ("FAILURES_INVALID", "requests", "failed", 99),
            ("BASELINE_RTT_INVALID", "requests", "baseline_rtt_seconds", 0),
            ("RTT_BASELINE_EXCEEDS_SAMPLE", "requests", "baseline_rtt_seconds", 0.1),
        ]
        for code, section, key, value in cases:
            with self.subTest(key=key, value=value):
                data = fixture()
                data[section][key] = value
                with self.assertRaisesRegex(subject.Hold, code):
                    evaluate(data)

    def test_sample_bounds_are_enforced(self):
        cases = [
            ("SERVICE_SAMPLE_INVALID", "service_time_seconds", [1.0] * 19),
            ("SERVICE_SAMPLE_INVALID", "service_time_seconds", [1.0] * 257),
            ("SERVICE_SAMPLE_INVALID", "service_time_seconds", [True] * 25),
            ("SERVICE_SAMPLE_INVALID", "service_time_seconds", [float("inf")] * 25),
            ("RESPONSE_SAMPLE_INVALID", "response_time_seconds", [0.0] * 30),
            ("INTERARRIVAL_SAMPLE_INVALID", "interarrival_seconds", [-1.0] * 30),
            ("SAMPLE_COUNT_INCONSISTENT", "interarrival_seconds", [1.0] * 50),
        ]
        for code, key, value in cases:
            with self.subTest(key=key, code=code):
                data = fixture()
                data["requests"][key] = value
                with self.assertRaisesRegex(subject.Hold, code):
                    evaluate(data)

    def test_bad_windows_and_future_source_are_held(self):
        for begin, end, expected in [
            ("2026-11-17T19:00:00", "2026-11-17T19:01:00Z", "WINDOW_TIME_INVALID"),
            ("2026-11-17T19:00:00Z", "2026-11-17T19:00:02Z", "WINDOW_DURATION_INVALID"),
            ("2026-11-17T19:00:00Z", "2026-11-18T19:00:01Z", "WINDOW_DURATION_INVALID"),
            ("2026-11-18T19:00:00Z", "2026-11-18T19:01:00Z", "WINDOW_IN_FUTURE"),
        ]:
            with self.subTest(begin=begin):
                data = fixture()
                data["window"] = {"started_at": begin, "ended_at": end}
                with self.assertRaisesRegex(subject.Hold, expected):
                    evaluate(data)

    def test_strict_json_refuses_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            for data, code in [
                (b'{"a":1,"a":2}', "DUPLICATE_JSON_KEY"),
                (b'{"value":NaN}', "NONFINITE_JSON"),
                (b'{"value":Infinity}', "NONFINITE_JSON"),
                (b'{"x":"', "SOURCE_JSON_INVALID"),
                (b"X" * (subject.MAX_BYTES + 2), "SOURCE_FILE_BYTE_BUDGET"),
            ]:
                with self.subTest(code=code):
                    p.write_bytes(data)
                    with self.assertRaisesRegex(subject.Hold, code):
                        subject.read_input(p)

    def test_file_symlinks_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            p.write_text("{}", encoding="utf-8")
            link = Path(tmp) / "link.json"
            try:
                link.symlink_to(p)
            except OSError:
                self.skipTest("Windows symlink access not permitted")
            with self.assertRaisesRegex(subject.Hold, "SOURCE_FILE_UNAVAILABLE"):
                subject.read_input(link)

    def test_renderer_escapes_and_denies_promoted_authority(self):
        result = evaluate()
        rendered = subject.render_html(result).decode()
        self.assertIn("NO AUTOSCALING AUTHORIZED", rendered)
        self.assertNotIn("<script", rendered)
        self.assertNotIn("<form", rendered)
        self.assertIn("script-src 'none'", rendered)
        polluted = copy.deepcopy(result)
        polluted["state"] = '<svg onload="alert(1)">'
        rendered = subject.render_html(polluted).decode()
        self.assertNotIn("<svg ", rendered)
        self.assertIn("&lt;svg", rendered)
        promoted = copy.deepcopy(result)
        promoted["authority"]["promotion_authorized"] = True
        with self.assertRaisesRegex(subject.Hold, "RENDER_CONTRACT_INVALID"):
            subject.render_html(promoted)

    def test_cli_json_and_html_output_are_provider_free(self):
        tool = ROOT / "tools" / "salad" / "capacity_math.py"
        args = [sys.executable, "-X", "utf8", "-B", str(tool), "--input", str(SAMPLE)]
        with tempfile.TemporaryDirectory() as tmp:
            a = subprocess.run(
                args, cwd=tmp, capture_output=True, text=True, timeout=10, check=False
            )
            self.assertEqual(a.returncode, 0, a.stderr)
            d = json.loads(a.stdout)
            self.assertEqual(d["schema"], subject.OUTPUT_SCHEMA)
            self.assertEqual(
                d["source"]["input_sha256"],
                hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
            )
            self.assertEqual(d["authority"]["provider_writes"], 0)
            b = subprocess.run(
                args + ["--format", "html"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(b.returncode, 0, b.stderr)
            self.assertIn("<table", b.stdout)
            self.assertNotIn("Authorization: Bearer", b.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_cli_rejects_invalid_file_without_source_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "unsafe.json"
            p.write_text('{"token":"SECRET","schema":"oops"}', encoding="utf-8")
            tool = ROOT / "tools" / "salad" / "capacity_math.py"
            run = subprocess.run(
                [sys.executable, "-B", str(tool), "--input", str(p)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(run.returncode, 2)
            self.assertIn("CAPACITY_HOLD:", run.stderr)
            self.assertNotIn("SECRET", run.stderr + run.stdout)
            self.assertEqual(run.stdout, "")


if __name__ == "__main__":
    unittest.main()
