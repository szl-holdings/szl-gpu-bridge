"""CPU dry-run that reproduces the pre-training failures from attempts 8-15.

The integration step imports the pinned TRL 0.23.1 stack when it is installed.
Signature failures are also covered without that stack in
``test_nemo_v3_sftconfig_compat.py``.  No signing key is opened, and no GPU
job is launched.
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_cpu_dry_run  # noqa: E402
from nemo_v3_cpu_dry_run import (  # noqa: E402
    DsseVerificationError,
    verify_synthetic_trainer_envelope,
)


def _verify(envelope: dict) -> dict:
    if envelope.get("payloadType") != "szl-nemo-v3-dry-run":
        raise RuntimeError("payload type rejected")
    if envelope.get("tampered"):
        raise RuntimeError("DSSE payload bytes do not match")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("DSSE payload is not an object")
    return payload


def _envelope(**payload_updates: object) -> dict:
    payload = {
        "recipe": {"maxSeqLength": 32, "packingStrategy": "ffd"},
        "dataset": {"field": "messages"},
    }
    payload.update(payload_updates)
    return {"payloadType": "szl-nemo-v3-dry-run", "payload": payload}


class NemoV3CpuDryRunTests(unittest.TestCase):
    def test_source_does_not_reach_keys_or_dispatch(self) -> None:
        source = (ROOT / "laptop" / "nemo_v3_cpu_dry_run.py").read_text(
            encoding="utf-8"
        )
        lowered = source.lower()
        for forbidden in (
            "keys/",
            "laptop_key",
            "engine_key",
            "repository_dispatch",
            "hfapi(",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_mocked_dsse_accepts_a_chat_dataset_and_rejects_tampering(self) -> None:
        accepted = verify_synthetic_trainer_envelope(_envelope(), _verify)
        self.assertEqual(accepted["dataset"]["field"], "messages")
        tampered = _envelope()
        tampered["tampered"] = True
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            verify_synthetic_trainer_envelope(tampered, _verify)
        with self.assertRaises(DsseVerificationError):
            verify_synthetic_trainer_envelope(_envelope(), None)  # type: ignore[arg-type]
        bad = _envelope()
        bad["payload"] = {
            "recipe": {"maxSeqLength": 32, "packingStrategy": "ffd"},
            "dataset": {"field": "text"},
        }
        with self.assertRaisesRegex(DsseVerificationError, "chat column"):
            verify_synthetic_trainer_envelope(bad, _verify)

    def test_cpu_dry_run_trains_a_few_steps(self) -> None:
        try:
            import torch
            import trl
        except ImportError as exc:
            self.skipTest(f"pinned trainer stack is not installed: {exc}")
        del torch, trl
        result = nemo_v3_cpu_dry_run.run_cpu_dry_run(_envelope(), _verify)
        self.assertGreaterEqual(result["global_steps"], 1)
        self.assertTrue(result["chat_template_present"])
        self.assertGreater(result["supervised_tokens"], 0)
        self.assertEqual(result["eval_strategy"], "epoch")
        self.assertEqual(result["max_length"], 32)
        self.assertEqual(result["packing_strategy"], "bfd")
        self.assertFalse(result["bf16"])
        self.assertFalse(result["fp16"])
        self.assertEqual(result["cpu_dtype"], "torch.float32")
        self.assertIn("4-bit", result["quantization_refused"])
        self.assertEqual(result["dataset_field"], "messages")
        self.assertEqual(result["lm_head_device"], "cpu")
        self.assertFalse(result["gpu_job_launched"])
        self.assertTrue(math.isfinite(result["training_loss"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
