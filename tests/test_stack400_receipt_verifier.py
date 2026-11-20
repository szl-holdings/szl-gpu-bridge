"""Adversarial network-free tests for public Stack400 receipt integrity claims."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.salad import stack400_receipt_verifier as v


def fixtures():
    run = "stack400-canary-test"
    receipt = {
        "schema": "szl.receipt/v1",
        "seq": 1,
        "actor": "szl-stack400-canary-worker/v1",
        "repo": "SZLHOLDINGS/stack400-runs",
        "head": "salad:" + run,
        "base": "Qwen/Qwen2.5-0.5B@snapshot",
        "policy_digest": "sha256:" + "a" * 64,
        "verifier_digest": "sha256:" + "b" * 64,
        "artifact_digest": "sha256:" + "c" * 64,
        "effects": [
            {"path": f"receipts/{run}/receipt.json", "action": "create"},
            {"path": f"receipts/{run}/markers.txt", "action": "create"},
        ],
        "prev": "genesis",
        "training": {
            "job_seed": 1337,
            "no_op_inject": False,
            "mean_delta": 0.8,
            "max_delta": 1.0,
            "adapter_trainable_l1": 5.0,
            "eval_score": 1.0,
            "loss_first": 4.0,
            "loss_last": 0.5,
        },
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    recorded = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    receipt["recorded_hash"] = recorded
    markers = [
        {"tag": "BOOT_OK", "run": run},
        {"tag": "IMPORTS_OK"},
        {"tag": "FETCH_OK"},
        {"tag": "LOAD_OK"},
        {"tag": "GPU_ATTEST", "name": "SYNTHETIC_GPU"},
        {"tag": "TRAIN_OK", "steps": 12, "loss_first": 4.0, "loss_last": 0.5},
        {"tag": "PROBE_MEASURED", "mean_delta": 0.8, "max_delta": 1.0, "no_op": False},
        {"tag": "EVAL_OK", "score": 1.0},
        {"tag": "RECEIPT_MINTED", "recorded_hash": recorded},
        {"tag": "UPLOAD_OK", "path": f"receipts/{run}/receipt.json"},
    ]
    return receipt, markers


def make_bytes(receipt, markers):
    raw_receipt = json.dumps(receipt, ensure_ascii=True, indent=2).encode()
    raw_markers = "".join(json.dumps(m) + "\n" for m in markers).encode()
    return raw_receipt, raw_markers


def rehash(receipt):
    obj = {k: val for k, val in receipt.items() if k != "recorded_hash"}
    raw = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    receipt["recorded_hash"] = "sha256:" + hashlib.sha256(raw).hexdigest()


class Stack400ReviewTests(unittest.TestCase):
    def verify(self, receipt=None, markers=None):
        if receipt is None or markers is None:
            rec, rows = fixtures()
            receipt = rec if receipt is None else receipt
            markers = rows if markers is None else markers
        return v.verify(*make_bytes(receipt, markers))

    def test_consistent_fixture_is_still_not_training_admission(self):
        result = self.verify()
        self.assertEqual(result["verdict"], "REVIEW_ONLY_HOLD")
        self.assertEqual(result["worker_claimed_steps"], 12)
        self.assertTrue(result["public_receipt_self_hash_match"])
        self.assertFalse(result["54h_training_complete_verified"])
        self.assertFalse(result["reward_eligible"])
        self.assertFalse(result["signed_receipt_verified"])
        self.assertFalse(result["provider_mutations"])
        self.assertIn("BASE_IMMUTABLE_REVISION_UNAVAILABLE", result["reasons"])

    def test_tampered_loss_rejected_even_with_original_markers(self):
        receipt, markers = fixtures()
        receipt["training"]["loss_last"] = 0.25
        with self.assertRaisesRegex(v.ReceiptHold, "RECEIPT_SELF_HASH_MISMATCH"):
            self.verify(receipt, markers)

    def test_no_self_hash_rejected(self):
        receipt, markers = fixtures()
        receipt.pop("recorded_hash")
        with self.assertRaisesRegex(v.ReceiptHold, "RECEIPT_SCHEMA"):
            self.verify(receipt, markers)

    def test_empty_or_oversized_receipt_rejected(self):
        with self.assertRaises(v.ReceiptHold):
            v.verify(b"", b"{}\n")
        with self.assertRaises(v.ReceiptHold):
            v.verify(b"x" * (v.MAX_RECEIPT_BYTES + 1), b"{}\n")

    def test_duplicate_json_key_rejected(self):
        receipt, markers = fixtures()
        raw = json.dumps(receipt)
        raw = raw.replace('"seq": 1', '"seq": 1, "seq": 2', 1)
        with self.assertRaisesRegex(v.ReceiptHold, "DUPLICATE_JSON_KEY"):
            v.verify(raw.encode(), make_bytes(receipt, markers)[1])

    def test_missing_upload_is_not_training_completion(self):
        receipt, markers = fixtures()
        markers = markers[:-1]
        with self.assertRaisesRegex(v.ReceiptHold, "STAGE_MISSING_OR_DUPLICATE"):
            self.verify(receipt, markers)

    def test_reordered_stages_rejected(self):
        receipt, markers = fixtures()
        markers[5], markers[6] = markers[6], markers[5]
        with self.assertRaisesRegex(v.ReceiptHold, "STAGE_ORDER_INVALID"):
            self.verify(receipt, markers)

    def test_marker_loss_mismatch_rejected(self):
        receipt, markers = fixtures()
        markers[5]["loss_last"] = 0.1
        with self.assertRaisesRegex(v.ReceiptHold, "RECEIPT_MARKER_LOSS_MISMATCH"):
            self.verify(receipt, markers)

    def test_zero_movement_with_false_noop_flag_denied(self):
        receipt, markers = fixtures()
        receipt["training"].update(
            mean_delta=0.0, max_delta=0.0, adapter_trainable_l1=0.0
        )
        markers[6].update(mean_delta=0.0, max_delta=0.0, no_op=False)
        rehash(receipt)
        markers[-2]["recorded_hash"] = receipt["recorded_hash"]
        with self.assertRaisesRegex(v.ReceiptHold, "TRAIN_MOVEMENT_INCONSISTENT"):
            self.verify(receipt, markers)

    def test_invalid_movement_bounds_are_not_consistent_training(self):
        for changes in (
            {"mean_delta": -0.1},
            {"max_delta": 0.1, "mean_delta": 0.8},
            {"adapter_trainable_l1": 0.0},
        ):
            with self.subTest(changes=changes):
                receipt, markers = fixtures()
                receipt["training"].update(changes)
                rehash(receipt)
                markers[-2]["recorded_hash"] = receipt["recorded_hash"]
                with self.assertRaisesRegex(
                    v.ReceiptHold, "TRAIN_MOVEMENT_INCONSISTENT"
                ):
                    self.verify(receipt, markers)

    def test_zero_mean_with_positive_max_stays_review_only(self):
        receipt, markers = fixtures()
        receipt["training"]["mean_delta"] = 0.0
        markers[6]["mean_delta"] = 0.0
        rehash(receipt)
        markers[-2]["recorded_hash"] = receipt["recorded_hash"]
        result = self.verify(receipt, markers)
        self.assertEqual(result["verdict"], "REVIEW_ONLY_HOLD")
        self.assertTrue(result["source_markers_consistent"])
        self.assertFalse(result["reward_eligible"])

    def test_noop_quarantined_even_with_matching_self_hash(self):
        receipt, markers = fixtures()
        markers[6]["no_op"] = True
        with self.assertRaisesRegex(v.ReceiptHold, "NOOP_QUARANTINE"):
            self.verify(receipt, markers)

    def test_forged_uploaded_path_rejected(self):
        receipt, markers = fixtures()
        markers[-1]["path"] = "receipts/other/receipt.json"
        with self.assertRaisesRegex(v.ReceiptHold, "UPLOAD_PATH_MISMATCH"):
            self.verify(receipt, markers)

    def test_recorded_hash_and_marker_link_must_match(self):
        receipt, markers = fixtures()
        markers[-2]["recorded_hash"] = "sha256:" + "d" * 64
        with self.assertRaisesRegex(v.ReceiptHold, "MARKER_SELF_HASH_MISMATCH"):
            self.verify(receipt, markers)

    def test_forged_signature_field_cannot_auto_approve(self):
        receipt, markers = fixtures()
        receipt["signature"] = "self_reported_signed"
        rehash(receipt)
        with self.assertRaisesRegex(v.ReceiptHold, "RECEIPT_SCHEMA"):
            self.verify(receipt, markers)

    def test_pinned_base_syntax_still_does_not_prove_adapter_bytes(self):
        receipt, markers = fixtures()
        receipt["base"] = "Qwen/Qwen2.5-0.5B@" + "e" * 40
        rehash(receipt)
        markers[-2]["recorded_hash"] = receipt["recorded_hash"]
        result = self.verify(receipt, markers)
        self.assertFalse(result["adapter_bytes_verified"])
        self.assertFalse(result["signed_receipt_verified"])
        self.assertFalse(result["54h_training_complete_verified"])
        self.assertFalse(result["reward_eligible"])

    def test_missing_provider_instance_proof_not_inferred_from_gpu_marker(self):
        result = self.verify()
        self.assertFalse(result["gpu_instance_verified"])
        self.assertIn("SALAD_GROUP_INSTANCE_NOT_AUTHENTICATED", result["reasons"])

    def test_output_is_create_only(self):
        report = self.verify()
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "review.json"
            digest = v.create_report(p, report)
            self.assertEqual(len(digest), 64)
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                v.create_report(p, report)
            self.assertEqual(original, p.read_bytes())

    def test_no_real_gpu_or_network_is_called(self):
        receipt, markers = fixtures()
        for _ in range(2):
            result = self.verify(copy.deepcopy(receipt), copy.deepcopy(markers))
            self.assertEqual(result["provider_mutations"], 0)
            self.assertEqual(result["verdict"], "REVIEW_ONLY_HOLD")


if __name__ == "__main__":
    unittest.main()
