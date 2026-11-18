"""Adversarial, offline capacity trajectory tests; no provider/GPU I/O."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.evidence import storage_horizon as h

NOW = datetime(2026, 11, 18, 7, 0, tzinfo=timezone.utc)
GIB = h.GIB


def make_receipt(
    start: datetime,
    *,
    min_free_gib: float = 70,
    swing_gib: float = 0,
    revision: str = h.EXPECTED_VERSION,
) -> bytes:
    total = 950 * GIB
    lows = [
        min_free_gib,
        min_free_gib + swing_gib,
        min_free_gib,
        min_free_gib,
        min_free_gib,
    ]
    snapshots = [
        {
            "observed_utc": (start + timedelta(seconds=i * 2)).isoformat(),
            "total_bytes": total,
            "free_bytes": int(free * GIB),
            "used_bytes": total - int(free * GIB),
        }
        for i, free in enumerate(lows)
    ]
    minimum = min(s["free_bytes"] for s in snapshots)
    maximum = max(s["free_bytes"] for s in snapshots)
    ok = minimum >= 50 * GIB and maximum - minimum <= 2 * GIB
    record = {
        "schema": h.INPUT_SCHEMA,
        "source_version": revision,
        "generated_utc": (start + timedelta(seconds=9)).isoformat(),
        "target_drive_root": "not-exported",
        "observed_samples": snapshots,
        "observation_error_type": None,
        "training_job_started": False,
        "authorizes_flagship_dispatch": False,
        "verdict": {
            "state": "CAPACITY_GATE_PASS_ONLY" if ok else "HOLD",
            "sample_count": len(snapshots),
            "total_capacity_bytes": total,
            "minimum_observed_free_bytes": minimum,
            "maximum_observed_free_bytes": maximum,
            "observed_free_swing_bytes": maximum - minimum,
            "last_observed_free_bytes": snapshots[-1]["free_bytes"],
            "storage_only_passed": ok,
            "other_training_qualification_gates_checked": False,
            "policy": {"minimum_free_gib": 50, "maximum_observed_swing_gib": 2},
        },
    }
    return json.dumps(record, sort_keys=True).encode()


def history(levels=(70, 70, 70)):
    return [
        make_receipt(
            NOW - timedelta(minutes=20 * (len(levels) - i - 1), seconds=10),
            min_free_gib=v,
        )
        for i, v in enumerate(levels)
    ]


class StorageHorizonTests(unittest.TestCase):
    def test_three_stable_windows_still_never_admit_training(self):
        result = h.assess(history(), now=NOW)
        self.assertEqual(result["observed_window_count"], 3)
        self.assertEqual(result["observed_sample_count"], 15)
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["training_launch_authorized"])
        self.assertFalse(result["dispatch_owner_verified"])
        self.assertFalse(result["receipt_chain_is_signed_or_origin_verified"])
        self.assertIn(
            "DISPATCH_OWNER_NOT_ATTESTED_BY_STORAGE_RECEIPTS", result["reasons"]
        )
        self.assertEqual(result["max_observed_depletion_bytes_per_second"], 0)

    def test_shortfall_is_hold_even_when_stable(self):
        result = h.assess(history((51, 49, 44)), now=NOW)
        self.assertEqual(result["state"], "HOLD")
        self.assertIn("LOCAL_DISK_BELOW_50_GIB", result["reasons"])
        self.assertIn("HISTORICAL_CAPACITY_HOLD_PRESENT", result["reasons"])

    def test_rising_pressure_holds_even_when_latest_above_reserve(self):
        result = h.assess(history((100, 85, 70)), now=NOW)
        self.assertGreater(result["max_observed_depletion_bytes_per_second"], 0)
        self.assertIn("OBSERVED_TREND_EXHAUSTS_4H_PLANNING_RESERVE", result["reasons"])
        self.assertFalse(result["projection_is_statistical_confidence_bound"])

    def test_high_swing_is_not_hidden_by_minimum(self):
        raw = history((70, 70, 70))
        raw[-1] = make_receipt(
            NOW - timedelta(seconds=10), min_free_gib=70, swing_gib=3
        )
        result = h.assess(raw, now=NOW)
        self.assertIn("LOCAL_SWING_ABOVE_2_GIB", result["reasons"])

    def test_count_and_unsorted_reject(self):
        with self.assertRaisesRegex(h.Hold, "INSUFFICIENT"):
            h.assess(history()[:2], now=NOW)
        with self.assertRaisesRegex(h.Hold, "WINDOW_OVERLAP"):
            h.assess(list(reversed(history())), now=NOW)

    def test_stale_and_future_reject(self):
        with self.assertRaisesRegex(h.Hold, "LATEST_WINDOW_STALE"):
            h.assess(history(), now=NOW + timedelta(minutes=7))
        with self.assertRaisesRegex(h.Hold, "LATEST_WINDOW_STALE"):
            h.assess(history(), now=NOW - timedelta(seconds=100))

    def test_unobserved_history_gap_reject(self):
        raw = [make_receipt(NOW - timedelta(hours=3)), *history((75, 74))]
        with self.assertRaisesRegex(h.Hold, "HISTORY_GAP_UNOBSERVED"):
            h.assess(raw, now=NOW)

    def test_samples_must_be_independent_and_valid(self):
        items = history()
        value = json.loads(items[0])
        value["observed_samples"][1]["observed_utc"] = value["observed_samples"][0][
            "observed_utc"
        ]
        items[0] = json.dumps(value).encode()
        with self.assertRaisesRegex(h.Hold, "SAMPLE_SPACING_INVALID"):
            h.assess(items, now=NOW)

    def test_volume_size_drift_fails_closed(self):
        items = history()
        value = json.loads(items[1])
        value["observed_samples"] = [
            {
                **sample,
                "total_bytes": sample["total_bytes"] + GIB,
                "used_bytes": sample["used_bytes"] + GIB,
            }
            for sample in value["observed_samples"]
        ]
        value["verdict"]["total_capacity_bytes"] += GIB
        items[1] = json.dumps(value).encode()
        with self.assertRaisesRegex(h.Hold, "VOLUME_IDENTITY_OR_SIZE_CHANGED"):
            h.assess(items, now=NOW)

    def test_verdict_forgery_does_not_promote(self):
        raw = history((75, 75, 44))
        value = json.loads(raw[-1])
        value["verdict"]["state"] = "CAPACITY_GATE_PASS_ONLY"
        value["verdict"]["storage_only_passed"] = True
        raw[-1] = json.dumps(value).encode()
        with self.assertRaisesRegex(h.Hold, "SOURCE_VERDICT_CONTRADICTION"):
            h.assess(raw, now=NOW)

    def test_source_only_verification_requires_flag(self):
        raw = history()
        value = json.loads(raw[-1])
        value["authorizes_flagship_dispatch"] = True
        raw[-1] = json.dumps(value).encode()
        with self.assertRaisesRegex(h.Hold, "SOURCE_AUTHORITY_CONTRADICTION"):
            h.assess(raw, now=NOW)

    def test_invalid_types_policy_and_version(self):
        for mutate, code in [
            (
                lambda d: d["verdict"]["policy"].__setitem__("minimum_free_gib", True),
                "SOURCE_POLICY_DRIFT",
            ),
            (
                lambda d: d["observed_samples"][0].__setitem__("free_bytes", True),
                "FREE_BYTES_INVALID",
            ),
            (
                lambda d: d.__setitem__("source_version", "unreviewed"),
                "SOURCE_VERSION_MISMATCH",
            ),
            (
                lambda d: d["observed_samples"][0].__setitem__("used_bytes", -1),
                "USED_BYTES_INVALID",
            ),
        ]:
            with self.subTest(code=code):
                raw = history()
                value = json.loads(raw[0])
                mutate(value)
                raw[0] = json.dumps(value).encode()
                with self.assertRaisesRegex(h.Hold, code):
                    h.assess(raw, now=NOW)

    def test_duplicate_keys_nan_and_bytes_budget_rejected(self):
        cases = [
            (b'{"x":1,"x":2}', "DUPLICATE_RECEIPT_FIELD"),
            (b'{"a": NaN}', "NONFINITE_NUMBER"),
            (b"Z" * 65000, "RECEIPT_BYTE_BUDGET"),
        ]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(h.Hold, expected):
                    h.parse_receipt(raw)

    def test_deterministic_chain_safeguards_provenance(self):
        original = history()
        out1 = h.assess(original, now=NOW)
        out2 = h.assess(original, now=NOW)
        self.assertEqual(out1["receipt_chain_sha256"], out2["receipt_chain_sha256"])
        altered = history()
        value = json.loads(altered[0])
        value["target_drive_root"] = "changed private path not emitted"
        altered[0] = json.dumps(value).encode()
        out3 = h.assess(altered, now=NOW)
        self.assertNotEqual(out1["receipt_chain_sha256"], out3["receipt_chain_sha256"])
        self.assertNotIn("target_drive_root", out3)

    def test_cli_is_read_only_and_returns_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            stale = datetime(2001, 1, 1, tzinfo=timezone.utc)
            for i, free in enumerate((80, 70, 44)):
                data = make_receipt(
                    stale + timedelta(minutes=20 * i), min_free_gib=free
                )
                path = Path(temp) / f"win{i}.json"
                path.write_bytes(data)
                paths.append(path)
            cmd = [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[1]
                    / "tools/evidence/storage_horizon.py"
                ),
            ]
            for path in paths:
                cmd += ["--receipt", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 78)
            body = json.loads(result.stdout)
            self.assertEqual(body["state"], "HOLD_UNOBSERVED")
            self.assertNotIn(str(paths[0]), result.stdout)
            self.assertEqual(len(list(Path(temp).iterdir())), 3)


if __name__ == "__main__":
    unittest.main()
