"""CPU admission tests for the additional NeMo Framework backend.

No training, no network, no image build. The nemo v3 contract file is
read as text so this test cannot loosen it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.nemo_framework.backend import (
    CAPS,
    FAMILIES,
    HF_PEFT_BACKEND,
    HF_PEFT_MODULE,
    NAMED_OPEN_LICENSE_NAMES,
    PINNED_TRAIN_IMAGE,
    RECIPE_INTEGER_RANGES,
    USD_TO_UUSD,
    V3_LICENSE_ID,
    AdmissionError,
    admit,
    framework_draft,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "laptop" / "nemo_v3_contract.py"
DRAFT = ROOT / "jobspecs" / "drafts" / "nemo-framework-trainer.v0.json"
BACKEND_SOURCE = ROOT / "tools" / "nemo_framework" / "backend.py"

CONTRACT_LOCKS = (
    'and base["licenseId"] != "nvidia-nemotron-open-model-license"',
    '"maxSeqLength": (256, 4096)',
    '"loraR": (1, 64)',
    '"loraAlpha": (1, 256)',
    '"batchSize": (1, 1)',
    '"gradAccum": (1, 64)',
    'or not 0 < recipe["epochs"] <= 8',
    'or not 0 < recipe["learningRate"] <= 0.001',
    'if recipe["optimizer"] not in {"adamw_8bit", "paged_adamw_8bit"}:',
    'if recipe["gradientCheckpointing"] not in {"unsloth", "true"}:',
    'if recipe["lrSchedulerType"] not in {"linear", "cosine"}:',
    'or gates["minFreeVramGb"] < 5',
)


class NeMoFrameworkBackendTests(unittest.TestCase):
    def test_contract_bounds_and_license_gate_stay_closed(self) -> None:
        text = CONTRACT.read_text(encoding="utf-8")
        for snippet in CONTRACT_LOCKS:
            self.assertIn(snippet, text)
        self.assertEqual(RECIPE_INTEGER_RANGES["maxSeqLength"], (256, 4096))
        self.assertEqual(RECIPE_INTEGER_RANGES["loraR"], (1, 64))
        self.assertEqual(RECIPE_INTEGER_RANGES["batchSize"], (1, 1))
        self.assertEqual(V3_LICENSE_ID, "nvidia-nemotron-open-model-license")
        source = BACKEND_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("import nemo_v3_contract", source)
        self.assertNotIn("from nemo_v3_contract", source)
        self.assertNotIn("import nemo", source)
        self.assertNotIn("SFTTrainer", source)

    def test_draft_file_matches_builder_and_is_not_queued(self) -> None:
        draft = framework_draft()
        self.assertEqual(json.loads(DRAFT.read_text(encoding="utf-8")), draft)
        self.assertFalse((ROOT / "queue" / "pending" / DRAFT.name).exists())
        receipt = admit(draft)
        self.assertFalse(receipt["launchAuthorized"])
        self.assertFalse(receipt["trainingStarted"])
        self.assertFalse(receipt["imagePushed"])
        self.assertEqual(receipt["existingBackendModule"], HF_PEFT_MODULE)
        self.assertEqual(receipt["caps"], CAPS)
        self.assertEqual(CAPS["perRunUusd"], 100 * USD_TO_UUSD)
        self.assertEqual(CAPS["perDayUusd"], 500 * USD_TO_UUSD)
        self.assertEqual(CAPS["perMonthUusd"], 5000 * USD_TO_UUSD)

    def test_family_license_register_blocks_every_family(self) -> None:
        receipt = admit(framework_draft())
        by_family = {item["family"]: item for item in receipt["families"]}
        nano4 = by_family["nano-4b"]
        self.assertEqual(nano4["hfApiLicense"], "other")
        self.assertEqual(nano4["hfApiLicenseName"], V3_LICENSE_ID)
        self.assertEqual(nano4["hfRepoId"], "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        self.assertFalse(nano4["admittedForTraining"])
        self.assertEqual(nano4["amaruLicenseCheck"], "PENDING")
        nano9 = by_family["nano-9b-v2"]
        self.assertEqual(nano9["hfApiLicenseName"], "nvidia-open-model-license")
        self.assertIn("LICENSE_NAME_DIFFERS_FROM_V3_GATE", nano9["reasons"])
        self.assertNotEqual(nano9["hfApiLicenseName"], V3_LICENSE_ID)
        illapa = by_family["illapa"]
        self.assertEqual(illapa["hfRepoId"], "UNKNOWN")
        self.assertIn("HF_REPO_UNKNOWN", illapa["reasons"])
        self.assertEqual(FAMILIES["illapa"]["hfApiLicense"], "UNKNOWN")
        for candidate in FAMILIES["illapa"]["unboundCandidatesObserved"]:
            self.assertFalse(candidate["adopted"])
        self.assertNotIn("openmdw-1.1", NAMED_OPEN_LICENSE_NAMES)

    def test_hf_peft_backend_stays_the_existing_module(self) -> None:
        draft = framework_draft()
        draft["backend"] = HF_PEFT_BACKEND
        draft.pop("families")
        receipt = admit(draft)
        self.assertEqual(receipt["backend"], HF_PEFT_BACKEND)
        self.assertEqual(receipt["existingBackendModule"], HF_PEFT_MODULE)
        self.assertFalse(receipt["launchAuthorized"])
        self.assertIn("HF_PEFT_PATH_UNCHANGED", receipt["reasons"])

    def test_launch_and_policy_widening_are_refused(self) -> None:
        draft = framework_draft()
        launched = dict(draft, launch=True)
        with self.assertRaises(AdmissionError) as raised:
            admit(launched)
        self.assertEqual(raised.exception.code, "LAUNCH_REFUSED")
        foreign = dict(draft, launcher="other")
        with self.assertRaises(AdmissionError) as raised:
            admit(foreign)
        self.assertEqual(raised.exception.code, "LAUNCHER_NOT_SAYWA")
        salad_only = dict(draft, providers=["salad"])
        with self.assertRaises(AdmissionError) as raised:
            admit(salad_only)
        self.assertEqual(raised.exception.code, "PROVIDER_NOT_ALLOWED")
        raised_cap = dict(draft, caps=dict(CAPS, perRunUsd=101))
        with self.assertRaises(AdmissionError) as raised:
            admit(raised_cap)
        self.assertEqual(raised.exception.code, "CAP_MISMATCH")
        enterprise = dict(draft, baseImage="nvcr.io/nvidia/nemo:24.01")
        with self.assertRaises(AdmissionError) as raised:
            admit(enterprise)
        self.assertEqual(raised.exception.code, "ENTERPRISE_IMAGE_FORBIDDEN")
        moved = dict(draft, baseImage=PINNED_TRAIN_IMAGE + "extra")
        with self.assertRaises(AdmissionError) as raised:
            admit(moved)
        self.assertEqual(raised.exception.code, "BASE_IMAGE_NOT_PINNED")
        built = dict(draft, successorImageDigest="a" * 64)
        with self.assertRaises(AdmissionError) as raised:
            admit(built)
        self.assertEqual(raised.exception.code, "SUCCESSOR_DIGEST_NOT_UNKNOWN")
        wide = json.loads(json.dumps(draft))
        wide["recipe"]["loraR"] = 65
        with self.assertRaises(AdmissionError) as raised:
            admit(wide)
        self.assertEqual(raised.exception.code, "RECIPE_RANGE")
        flagged = json.loads(json.dumps(draft))
        flagged["recipe"]["batchSize"] = True
        with self.assertRaises(AdmissionError) as raised:
            admit(flagged)
        self.assertEqual(raised.exception.code, "RECIPE_RANGE")
        with self.assertRaises(AdmissionError) as raised:
            admit(dict(draft, backend="megatron"))
        self.assertEqual(raised.exception.code, "BACKEND_UNKNOWN")


if __name__ == "__main__":
    unittest.main()
