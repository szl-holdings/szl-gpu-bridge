"""Offline proof of bounds, holds, source pins and append-only receipts."""

from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from tools.proving import model_recipe_loop_v0 as g

SOURCES = dict(g.SOURCE_OWNERS)


def plan():
    return {
        "schema": g.SCHEMA,
        "source_pins": dict(SOURCES),
        "training_dispatch_enabled": False,
        "provider_write_enabled": False,
        "private_graph_hydration": False,
        "models": [
            {
                "id": "SZLHOLDINGS/A11OY-MINI",
                "observed_sha": "a" * 40,
                "role": "offline",
                "admission": "REVIEW_REQUIRED",
            }
        ],
    }


def observations():
    return (
        {
            "schema": g.OBS_SCHEMA,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "rows": [
                {
                    "id": "SZLHOLDINGS/A11OY-MINI",
                    "state": "PUBLIC_METADATA_OBSERVED_NOT_ARTIFACT_VERIFIED",
                    "sha": "a" * 40,
                    "gated": "False",
                }
            ],
        },
        {
            "schema": g.DEPENDENCY_SCHEMA,
            "rows": [
                {
                    "surface": "SZLHOLDINGS/szl-second-brain",
                    "path": "/health",
                    "http_status": 404,
                },
                {
                    "surface": "SZLHOLDINGS/szl-second-brain",
                    "path": "/readyz",
                    "http_status": 404,
                },
                {
                    "surface": "Hatun product",
                    "path": "https://a-11-oy.com/wires",
                    "http_status": 403,
                },
            ],
            "hatun_runtime_tools_list_authenticated": "NOT_RUN_NO_HOST_OR_CREDENTIAL_BINDING",
        },
    )


class TestLoop(unittest.TestCase):
    def test_plan_exact(self):
        self.assertEqual(len(g.verify_plan(plan())), 1)

    def test_source_owner_drift(self):
        p = plan()
        p["source_pins"]["szl-holdings/hatun-mcp"] = "b" * 40
        with self.assertRaises(g.Hold):
            g.verify_plan(p)

    def test_duplicate_model_refused(self):
        p = plan()
        p["models"].append(copy.deepcopy(p["models"][0]))
        with self.assertRaises(g.Hold):
            g.verify_plan(p)

    def test_training_enable_refused(self):
        p = plan()
        p["training_dispatch_enabled"] = True
        with self.assertRaises(g.Hold):
            g.verify_plan(p)

    def test_missing_approval_refused(self):
        p = plan()
        p["models"][0]["admission"] = "PROMOTED"
        with self.assertRaises(g.Hold):
            g.verify_plan(p)

    def test_wrong_revision_hold(self):
        a, b = observations()
        a["rows"][0]["sha"] = "b" * 40
        records, gate = g.classify(g.verify_plan(plan()), a, b, {"state": "NOT_STAGED"})
        self.assertIn("MODEL_SOURCE_SHA_CHANGED", records[0]["reasons"])
        self.assertFalse(records[0]["training_or_inference_dispatched"])

    def test_second_brain_404_hold(self):
        a, b = observations()
        r, gate = g.classify(g.verify_plan(plan()), a, b, {"state": "NOT_STAGED"})
        self.assertIn("SECOND_BRAIN_READINESS_UNAVAILABLE", r[0]["reasons"])
        self.assertFalse(gate["second_brain_health_ready"])

    def test_hatun_unconfigured_hold_even_brain_200(self):
        a, b = observations()
        for x in b["rows"][:2]:
            x["http_status"] = 200
        r, gate = g.classify(
            g.verify_plan(plan()), a, b, {"state": "SOURCE_BYTES_HASHED_TRAIN_ONLY"}
        )
        self.assertIn("HATUN_AUTH_SIGNER_UNVERIFIED", r[0]["reasons"])
        self.assertFalse(r[0]["training_or_inference_dispatched"])

    def test_empty_metadata_not_green(self):
        a, b = observations()
        a["rows"] = []
        r, _ = g.classify(g.verify_plan(plan()), a, b, {"state": "NOT_STAGED"})
        self.assertIn("MODEL_METADATA_UNAVAILABLE", r[0]["reasons"])

    def test_stale_metadata_hold(self):
        a, b = observations()
        a["captured_utc"] = "2020-01-01T00:00:00Z"
        r, _ = g.classify(g.verify_plan(plan()), a, b, {"state": "NOT_STAGED"})
        self.assertIn("HF_OBSERVATION_STALE_OR_UNDATED", r[0]["reasons"])

    def test_wrong_train_zip_fails(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "bad.zip"
            p.write_bytes(b"PK\x03\x04BAD")
            with self.assertRaisesRegex(g.Hold, "TRAIN_ZIP_SHA_DRIFT"):
                g.verify_training_zip(p)

    def test_no_dataset_is_not_verified(self):
        self.assertEqual(g.verify_training_zip(None)["state"], "NOT_STAGED")

    def test_cycle_limit(self):
        a, b = observations()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(g.Hold, "CYCLE_CAP"):
                g.run(plan(), a, b, {"state": "NOT_STAGED"}, Path(t) / "out", cycles=4)

    def test_step_limit(self):
        a, b = observations()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(g.Hold, "STEP_CAP"):
                g.run(
                    plan(), a, b, {"state": "NOT_STAGED"}, Path(t) / "out", max_steps=0
                )

    def test_two_cycles_hash_chained_and_never_train(self):
        a, b = observations()
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "fresh"
            result = g.run(plan(), a, b, {"state": "NOT_STAGED"}, out, cycles=2)
            self.assertEqual(result["cycles_completed"], 2)
            self.assertEqual(result["training_jobs_created"], 0)
            first = (out / "cycle-01.json").read_bytes()
            second = (out / "cycle-02.json").read_bytes()
            self.assertEqual(g.digest(first), result["cycle_receipt_sha256"][0])
            self.assertEqual(g.digest(second), result["cycle_receipt_sha256"][1])
            self.assertEqual(
                json.loads(second)["previous_receipt_sha256"],
                result["cycle_receipt_sha256"][0],
            )
            self.assertEqual(json.loads(second)["training_jobs_created"], 0)
            with self.assertRaisesRegex(g.Hold, "EVIDENCE_OUTPUT_EXISTS"):
                g.run(plan(), a, b, {"state": "NOT_STAGED"}, out)

    def test_no_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "dup.json"
            p.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(g.Hold):
                g.read_json(p)


if __name__ == "__main__":
    unittest.main()
