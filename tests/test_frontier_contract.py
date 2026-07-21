import base64
import json
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

from frontier_contract import (  # noqa: E402
    ContractError, V2_PAYLOAD_TYPE, pae, validate_v2_spec, verify_envelope,
)
from frontier_runtime import (  # noqa: E402
    artifact_manifest,
    chat_template_evidence,
    export_plan,
    extract_json_object,
    is_degenerate_text,
    manifest_digest,
    normalize_conversation,
    prompt_messages,
)


def valid_spec():
    now = datetime.now(timezone.utc)
    return {
        "jobId": "job-2026-frontier-sft",
        "kind": "unsloth-frontier-sft-v2",
        "createdAt": now.isoformat().replace("+00:00", "Z"),
        "expiresAt": (now + timedelta(hours=12)).isoformat().replace("+00:00", "Z"),
        "base": {
            "repoId": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
            "revision": "a" * 40,
            "licenseId": "apache-2.0",
            "trustRemoteCode": False,
        },
        "dataset": {
            "repoId": "SZLHOLDINGS/szl-quant-sft-v1",
            "revision": "b" * 40,
            "file": "train/messages.jsonl",
            "sha256": "c" * 64,
            "provenance": "Every row cites a signed receipt and content-addressed source archive.",
            "format": "messages-jsonl",
            "licenseId": "apache-2.0",
        },
        "recipe": {
            "maxSeqLength": 2048,
            "loraR": 32,
            "loraAlpha": 32,
            "loraDropout": 0,
            "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "batchSize": 2,
            "gradAccum": 4,
            "epochs": 1,
            "learningRate": 0.0002,
            "optimizer": "adamw_8bit",
            "gradientCheckpointing": "unsloth",
            "seed": 3407,
            "packing": False,
            "packingStrategy": "ffd",
            "assistantOnlyLoss": False,
            "useRsLoRA": True,
            "warmupRatio": 0.05,
            "weightDecay": 0.01,
            "lrSchedulerType": "linear",
            "expectedChatTemplateSha256": "d" * 64,
        },
        "gates": {
            "minFreeVramGb": 6,
            "minFreeDiskGb": 50,
            "maxWallclockMinutes": 720,
            "maxDatasetRows": 100000,
            "abortOnNanLoss": True,
        },
        "outputs": {
            "modelRepoId": "SZLHOLDINGS/SZL-Frontier-1.7B-Adapter",
            "receiptsRepoId": "SZLHOLDINGS/szl-training-receipts",
            "checkpointBucketId": "SZLHOLDINGS/szl-training-working-set",
            "private": True,
            "exports": {
                "adapter": True,
                "merged16bit": True,
                "ggufQuantizations": ["q4_k_m"],
                "requireReloadSmoke": True,
            },
        },
        "eval": {
            "suite": "frontier-heldout-v2",
            "heldOutFraction": 0.1,
            "seed": 7,
            "maxGenerations": 20,
            "maxNewTokens": 400,
            "requiredJsonKeys": ["action", "conviction"],
            "convictionCeiling": 0.97,
            "maxDegenerateRate": 0,
            "minJsonValidRate": 0.95,
            "minRequiredKeysRate": 0.95,
            "minCeilingRespectRate": 1.0,
        },
        "notes": "test",
    }


class ContractTests(unittest.TestCase):
    def test_valid_contract(self):
        spec = valid_spec()
        self.assertIs(validate_v2_spec(spec), spec)

    def test_dsse_exact_bytes_verify_and_tamper_rejects(self):
        try:
            from nacl.signing import SigningKey
        except ImportError:
            self.skipTest("PyNaCl not installed")
        signing_key = SigningKey.generate()
        spki = b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + bytes(signing_key.verify_key)
        key_id = __import__("hashlib").sha256(spki).hexdigest()[:16]
        spec = valid_spec()
        payload = json.dumps(spec, separators=(",", ":"), ensure_ascii=False).encode()
        signature = signing_key.sign(pae(V2_PAYLOAD_TYPE, payload)).signature
        envelope = {
            "payloadType": V2_PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode(),
            "publicKeySpkiBase64": base64.b64encode(spki).decode(),
            "signatures": [{"keyid": key_id, "sig": base64.b64encode(signature).decode()}],
        }
        pin = {"keyId": key_id, "publicKeySpkiBase64": envelope["publicKeySpkiBase64"]}
        observed, exact, observed_type = verify_envelope(envelope, pin)
        self.assertEqual(observed, spec)
        self.assertEqual(exact, payload)
        self.assertEqual(observed_type, V2_PAYLOAD_TYPE)
        tampered = dict(envelope)
        tampered_payload = payload.replace(b"frontier-sft", b"frontier-xft", 1)
        tampered["payload"] = base64.b64encode(tampered_payload).decode()
        with self.assertRaisesRegex(ContractError, "verification failed"):
            verify_envelope(tampered, pin)

    def test_floating_revision_is_refused(self):
        spec = valid_spec()
        spec["base"]["revision"] = "main"
        with self.assertRaisesRegex(ContractError, "immutable"):
            validate_v2_spec(spec)

    def test_path_traversal_is_refused(self):
        spec = valid_spec()
        spec["dataset"]["file"] = "../secret.jsonl"
        with self.assertRaises(ContractError):
            validate_v2_spec(spec)

    def test_unvalidated_export_is_refused(self):
        spec = valid_spec()
        spec["outputs"]["exports"]["requireReloadSmoke"] = False
        with self.assertRaisesRegex(ContractError, "reload smoke"):
            validate_v2_spec(spec)

    def test_duplicate_quant_is_refused(self):
        spec = valid_spec()
        spec["outputs"]["exports"]["ggufQuantizations"] = ["q4_k_m", "q4_k_m"]
        with self.assertRaisesRegex(ContractError, "unique"):
            validate_v2_spec(spec)

    def test_export_plan_is_deterministic(self):
        plan = export_plan(valid_spec())
        self.assertEqual(
            plan,
            [
                {"kind": "adapter", "path": "adapter"},
                {"kind": "merged_16bit", "path": "merged-16bit"},
                {"kind": "gguf", "quantization": "q4_k_m", "path": "gguf/q4_k_m"},
            ],
        )


class RuntimeEvidenceTests(unittest.TestCase):
    def test_chat_template_evidence(self):
        tokenizer = SimpleNamespace(chat_template="{% generation %}{{ x }}{% endgeneration %}")
        evidence = chat_template_evidence(tokenizer)
        self.assertTrue(evidence["present"])
        self.assertTrue(evidence["hasGenerationBlocks"])
        self.assertEqual(len(evidence["sha256"]), 64)

    def test_artifact_manifest_hashes_and_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "a.txt").write_text("a")
            (root / "nested").mkdir()
            (root / "nested" / "b.bin").write_bytes(b"bb")
            entries = artifact_manifest(root)
            self.assertEqual([e["path"] for e in entries], ["a.txt", "nested/b.bin"])
            self.assertEqual(len(manifest_digest(entries)), 64)
            try:
                (root / "link").symlink_to(root / "a.txt")
            except (OSError, NotImplementedError):
                return
            with self.assertRaisesRegex(ValueError, "symlink"):
                artifact_manifest(root)

    def test_prompt_excludes_held_out_assistant(self):
        result = prompt_messages(
            [
                {"role": "system", "content": "be honest"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "target"},
            ]
        )
        self.assertEqual(result[-1]["role"], "user")

    def test_conversation_requires_final_assistant_target(self):
        with self.assertRaisesRegex(ValueError, "assistant target"):
            normalize_conversation(
                [{"role": "user", "content": "question"}],
                require_final_assistant=True,
            )
        normalized = normalize_conversation(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            require_final_assistant=True,
        )
        self.assertEqual(normalized[-1]["role"], "assistant")

    def test_dataset_license_is_required(self):
        spec = valid_spec()
        del spec["dataset"]["licenseId"]
        with self.assertRaisesRegex(ContractError, "licenseId"):
            validate_v2_spec(spec)

    def test_evaluation_threshold_is_bounded(self):
        spec = valid_spec()
        spec["eval"]["minJsonValidRate"] = 1.1
        with self.assertRaisesRegex(ContractError, "minJsonValidRate"):
            validate_v2_spec(spec)

    def test_json_and_degeneracy_checks(self):
        self.assertEqual(extract_json_object('prefix {"action":"ABSTAIN"} suffix')["action"], "ABSTAIN")
        self.assertIsNone(extract_json_object("no object"))
        self.assertTrue(is_degenerate_text("@@@@@@@@@@@@@@@@@@@@@@@@@@@@"))
        self.assertFalse(is_degenerate_text('{"action":"ABSTAIN","conviction":0.2}'))


if __name__ == "__main__":
    unittest.main()
