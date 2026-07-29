import base64
import hashlib
import json
import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

from frontier_contract import ContractError, pae, verify_envelope  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    NEMO_V3_PAYLOAD_TYPE,
    record_ids_sha256,
    validate_nemo_v3_spec,
)


def pinned(path: str, *, name: str | None = None, ids: list[str] | None = None):
    value = {"path": path, "sha256": "a" * 64, "bytes": 100}
    if name is not None:
        assert ids
        value.update(
            {"name": name, "recordIds": ids, "recordIdsSha256": record_ids_sha256(ids)}
        )
    return value


def valid_spec():
    now = datetime.now(timezone.utc)
    return {
        "jobId": "job-2026-nemo-v3-governed",
        "kind": "szl-nemo-governed-v3",
        "createdAt": now.isoformat().replace("+00:00", "Z"),
        "expiresAt": (now + timedelta(hours=12)).isoformat().replace("+00:00", "Z"),
        "source": {
            "repoId": "szl-holdings/a11oy",
            "revision": "1" * 40,
            "licenseId": "apache-2.0",
        },
        "base": {
            "repoId": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "revision": "2" * 40,
            "licenseId": "nvidia-open-model-license",
            "licenseAcknowledgement": "I accept the pinned NVIDIA upstream model license and preserve attribution.",
            "trustRemoteCode": True,
        },
        "dataset": {
            "provenance": "Every row is independently project-authored, rights-admitted, and separated from all frozen evaluation suites.",
            "rightsBasis": "PROJECT_AUTHORED_SCENARIOS",
            "train": pinned("model_release/szl-nemo-v3/train.jsonl"),
            "holdouts": [
                pinned(
                    "model_release/szl-nemo-v3/holdout-original-v2.jsonl",
                    name="original-v2",
                    ids=["eval:a", "eval:b"],
                ),
                pinned(
                    "model_release/szl-nemo-v3/holdout-shadow-v2.jsonl",
                    name="shadow-v2",
                    ids=["shadow:a"],
                ),
                pinned(
                    "model_release/szl-nemo-v3/holdout-challenge-v3.jsonl",
                    name="challenge-v3",
                    ids=["challenge:a"],
                ),
            ],
            "preregistration": pinned("model_release/szl-nemo-v3/preregistration.json"),
        },
        "recipe": {
            "maxSeqLength": 2048,
            "loraR": 16,
            "loraAlpha": 32,
            "loraDropout": 0,
            "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "batchSize": 1,
            "gradAccum": 8,
            "epochs": 2,
            "learningRate": 0.0001,
            "optimizer": "adamw_8bit",
            "gradientCheckpointing": "unsloth",
            "seed": 3407,
            "warmupRatio": 0.05,
            "weightDecay": 0.01,
            "lrSchedulerType": "linear",
        },
        "gates": {
            "minFreeVramGb": 6.5,
            "minFreeDiskGb": 50,
            "maxWallclockMinutes": 240,
            "maxDatasetRows": 500,
            "maxTemperatureC": 78,
            "maxUtilizationPct": 15,
        },
        "outputs": {
            "candidateId": "SZL-Nemo-v3-Nemotron-4B-Adapter",
            "receiptsRepoId": "SZLHOLDINGS/szl-training-receipts",
            "private": True,
            "publishCandidate": False,
        },
        "evaluation": {
            "requiredPassRate": 1.0,
            "maxDegenerateRate": 0.0,
            "maxNewTokens": 192,
            "requireExactRecordOrder": True,
        },
        "notes": "one governed attempt",
    }


def owner_dispatch():
    return {
        "workflowIdentity": (
            "szl-holdings/a11oy/.github/workflows/"
            "nemo-v3-isolated-owner-dispatch.yml@refs/heads/main"
        ),
        "workflowBlob": "7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33",
        "workflowVersion": "nemo-v3-owner-dispatch.v2",
        "trainingImage": (
            "unsloth/unsloth@sha256:"
            "9cc97606fc386b4b13455285eb7bd2668f51530988a9c2578707fe6cdfc46123"
        ),
        "candidateUpload": False,
        "modelCardUpload": False,
        "datasetUpload": False,
        "receiptsRepoId": "SZLHOLDINGS/szl-training-receipts",
    }


class NemoV3ContractTests(unittest.TestCase):
    def test_valid_contract(self):
        spec = valid_spec()
        self.assertIs(validate_nemo_v3_spec(spec), spec)

    def test_holdout_order_and_identity_are_immutable(self):
        spec = valid_spec()
        spec["dataset"]["holdouts"].reverse()
        with self.assertRaisesRegex(ContractError, "order"):
            validate_nemo_v3_spec(spec)
        spec = valid_spec()
        spec["dataset"]["holdouts"][0]["recordIdsSha256"] = "b" * 64
        with self.assertRaisesRegex(ContractError, "does not bind"):
            validate_nemo_v3_spec(spec)

    def test_candidate_publication_and_threshold_weakening_are_refused(self):
        spec = valid_spec()
        spec["outputs"]["publishCandidate"] = True
        with self.assertRaisesRegex(ContractError, "publication"):
            validate_nemo_v3_spec(spec)
        spec = valid_spec()
        spec["evaluation"]["requiredPassRate"] = 0.9
        with self.assertRaisesRegex(ContractError, "all holdouts"):
            validate_nemo_v3_spec(spec)
        spec = valid_spec()
        spec["gates"]["minFreeVramGb"] = 1
        with self.assertRaisesRegex(ContractError, "weakened"):
            validate_nemo_v3_spec(spec)

    def test_rights_and_source_are_fixed(self):
        spec = valid_spec()
        spec["dataset"]["rightsBasis"] = "HARVESTED_BRAIN_ROWS"
        with self.assertRaisesRegex(ContractError, "rights"):
            validate_nemo_v3_spec(spec)
        spec = valid_spec()
        spec["source"]["revision"] = "main"
        with self.assertRaisesRegex(ContractError, "40"):
            validate_nemo_v3_spec(spec)

    def test_owner_dispatch_is_exact_and_receipt_only(self):
        spec = valid_spec()
        spec["jobId"] = "job-2026-nemo-v3-governed-attempt-2"
        spec["ownerDispatch"] = owner_dispatch()
        self.assertIs(validate_nemo_v3_spec(spec), spec)

        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            widened = json.loads(json.dumps(spec))
            widened["ownerDispatch"][field] = True
            with self.assertRaisesRegex(ContractError, field):
                validate_nemo_v3_spec(widened)

        mutable_image = json.loads(json.dumps(spec))
        mutable_image["ownerDispatch"]["trainingImage"] = "unsloth/unsloth:latest"
        with self.assertRaisesRegex(ContractError, "training image"):
            validate_nemo_v3_spec(mutable_image)

        extra = json.loads(json.dumps(spec))
        extra["ownerDispatch"]["unreviewed"] = False
        with self.assertRaisesRegex(ContractError, "unsupported fields"):
            validate_nemo_v3_spec(extra)

    def test_exact_dsse_bytes_verify_and_tamper_fails(self):
        try:
            from nacl.signing import SigningKey
        except ImportError:
            self.skipTest("PyNaCl not installed")
        key = SigningKey.generate()
        spki = b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + bytes(
            key.verify_key
        )
        key_id = hashlib.sha256(spki).hexdigest()[:16]
        spec = valid_spec()
        payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        signature = key.sign(pae(NEMO_V3_PAYLOAD_TYPE, payload)).signature
        envelope = {
            "payloadType": NEMO_V3_PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode(),
            "publicKeySpkiBase64": base64.b64encode(spki).decode(),
            "signatures": [
                {"keyid": key_id, "sig": base64.b64encode(signature).decode()}
            ],
        }
        pin = {"keyId": key_id, "publicKeySpkiBase64": envelope["publicKeySpkiBase64"]}
        observed, exact, payload_type = verify_envelope(
            envelope, pin, allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,)
        )
        self.assertEqual(observed, spec)
        self.assertEqual(exact, payload)
        self.assertEqual(payload_type, NEMO_V3_PAYLOAD_TYPE)
        tampered = dict(envelope)
        tampered["payload"] = base64.b64encode(
            payload.replace(b"governed", b"untrusted", 1)
        ).decode()
        with self.assertRaisesRegex(ContractError, "verification failed"):
            verify_envelope(
                tampered, pin, allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,)
            )


if __name__ == "__main__":
    unittest.main()
