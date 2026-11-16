"""CPU/network-free negative controls for revision-bound artifact qualification."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

from tools.salad import artifact_qualification as q


class FakeApi:
    def __init__(self, blobs, *, step=200):
        self.blobs = dict(blobs)
        self.calls = []
        entries = []
        for name, raw in blobs.items():
            lfs = (
                {"sha256": hashlib.sha256(raw).hexdigest()}
                if name.endswith("safetensors")
                else None
            )
            entries.append(
                NS(
                    rfilename=name,
                    size=len(raw),
                    lfs=lfs,
                    blob_id=hashlib.sha1(
                        b"blob " + str(len(raw)).encode() + b"\0" + raw
                    ).hexdigest(),
                )
            )
        self.info = NS(sha="a" * 40, private=True, siblings=entries, last_modified=None)

    def model_info(self, repo, **kwargs):
        self.calls.append((repo, kwargs))
        return self.info

    def fetch(self, *, repo_id, filename, revision, local_dir, etag_timeout):
        if revision != "a" * 40:
            raise AssertionError("moving branch read")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.blobs[filename])
        return str(path)


def fixture(step=200, target=200):
    data = b"finite-tensor-fixture-not-real-safetensors"
    return {
        f"checkpoint-{step}/trainer_state.json": json.dumps(
            {"global_step": step, "max_steps": target}
        ).encode(),
        f"checkpoint-{step}/adapter_model.safetensors": data,
        "final/adapter_config.json": json.dumps(
            {
                "base_model_name_or_path": "org/base",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 32,
                "target_modules": ["q_proj"],
            }
        ).encode(),
        "final/adapter_model.safetensors": data,
        "final/tokenizer.json": b'{"version":"1.0"}',
        "final/tokenizer_config.json": b'{"tokenizer_class":"Fixture"}',
    }


def fake_inspect(_):
    return {"all_finite": True, "tensor_count": 2}


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "repo": "SZLHOLDINGS/example",
            "base": "org/base",
            "target_steps": 200,
        }
        self.api = FakeApi(fixture())

    def observe(self, *, inspect=fake_inspect, fetch=None):
        return q.observe_candidate(
            self.api, fetch or self.api.fetch, self.candidate, inspect=inspect
        )

    def hold(self, reason, **kwargs):
        result = self.observe(**kwargs)
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["ready_for_evaluation"])
        self.assertFalse(result["promotion_authorized"])
        self.assertIn(reason, result["hold_reasons"])
        return result

    def test_pinned_artifact_is_not_release_or_signature(self):
        result = self.observe()
        self.assertEqual(result["status"], "ARTIFACT_VERIFIED")
        self.assertTrue(result["ready_for_evaluation"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["model_quality_measured"])
        self.assertFalse(result["training_execution_provenance_verified"])
        self.assertEqual(result["signature_status"], "NOT_SIGNED")
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(len(result["gates"]["files"]), 5)
        self.assertEqual(result["release_blockers"], q.RELEASE_BLOCKERS)

    def test_progress_is_not_completion(self):
        self.api = FakeApi(fixture(step=175))
        self.hold("TRAINING_IN_PROGRESS")

    def test_overshoot_is_not_completion(self):
        self.api = FakeApi(fixture(step=201))
        self.hold("TERMINAL_STEP_OVERSHOOT")

    def test_target_contract_must_match(self):
        self.api = FakeApi(fixture(target=201))
        self.hold("TARGET_CONTRACT_MISMATCH")

    def test_filename_step_must_match_state(self):
        blobs = fixture()
        blobs["checkpoint-200/trainer_state.json"] = (
            b'{"global_step":199,"max_steps":200}'
        )
        self.api = FakeApi(blobs)
        self.hold("CHECKPOINT_STEP_MISMATCH")

    def test_boolean_is_not_integer_step(self):
        blobs = fixture()
        blobs["checkpoint-200/trainer_state.json"] = (
            b'{"global_step":true,"max_steps":200}'
        )
        self.api = FakeApi(blobs)
        self.hold("CHECKPOINT_STEP_MISMATCH")

    def test_directory_presence_cannot_qualify(self):
        self.api = FakeApi({"checkpoint-200/README.md": b"not proof"})
        self.hold("TRAINER_STATE_MISSING")

    def test_readme_is_not_a_checkpoint(self):
        self.api = FakeApi({"README.md": b"complete"})
        self.hold("NO_CHECKPOINT_OBSERVED")

    def test_each_final_component_required(self):
        for name in [n for n in fixture() if n.startswith("final/")]:
            with self.subTest(name=name):
                blobs = fixture()
                del blobs[name]
                self.api = FakeApi(blobs)
                self.hold("FINAL_BUNDLE_INCOMPLETE")

    def test_final_must_match_terminal_checkpoint(self):
        blobs = fixture()
        blobs["final/adapter_model.safetensors"] = b"different-weights"
        self.api = FakeApi(blobs)
        self.hold("FINAL_TERMINAL_ADAPTER_MISMATCH")

    def test_changed_same_size_bytes_rejected(self):
        def fetch(**kwargs):
            p = Path(self.api.fetch(**kwargs))
            if p.name == "adapter_model.safetensors":
                p.write_bytes(b"x" * p.stat().st_size)
            return str(p)

        self.hold("FILE_HASH_MISMATCH", fetch=fetch)

    def test_truncated_file_rejected(self):
        def fetch(**kwargs):
            p = Path(self.api.fetch(**kwargs))
            p.write_bytes(p.read_bytes()[:-1])
            return str(p)

        self.hold("FILE_SIZE_MISMATCH", fetch=fetch)

    def test_base_model_contract_must_match(self):
        self.candidate["base"] = "org/wrong"
        self.hold("ADAPTER_CONFIG_CONTRACT_MISMATCH")

    def test_duplicate_config_key_rejected(self):
        blobs = fixture()
        blobs["final/adapter_config.json"] = b'{"r":32,"r":64}'
        self.api = FakeApi(blobs)
        self.hold("DUPLICATE_JSON_KEY")

    def test_nonfinite_tensor_is_hold(self):
        def inspect(_):
            raise q.EvidenceHold("NONFINITE_ADAPTER_TENSOR")

        self.hold("NONFINITE_ADAPTER_TENSOR", inspect=inspect)

    def test_incomplete_tensor_inspection_is_hold(self):
        self.hold(
            "TENSOR_INSPECTION_INCOMPLETE",
            inspect=lambda _: {"all_finite": True, "tensor_count": 0},
        )

    def test_local_change_after_inspection_is_hold(self):
        def inspect(path):
            Path(path).write_bytes(b"x" * Path(path).stat().st_size)
            return fake_inspect(path)

        self.hold("FILE_HASH_MISMATCH", inspect=inspect)

    def test_unknown_visibility_is_hold(self):
        self.api.info.private = None
        self.hold("VISIBILITY_UNOBSERVED")

    def test_duplicate_metadata_is_hold(self):
        self.api.info.siblings.append(self.api.info.siblings[0])
        self.hold("DUPLICATE_FILE_METADATA")

    def test_nonimmutable_revision_is_hold(self):
        self.api.info.sha = "main"
        self.hold("IMMUTABLE_REVISION_UNAVAILABLE")

    def test_network_error_is_not_absent_or_complete(self):
        def fetch(**_):
            raise OSError("secret-token-do-not-log")

        result = self.hold("OBSERVATION_UNAVAILABLE:OSError", fetch=fetch)
        self.assertNotIn("secret-token", json.dumps(result))

    def test_oversized_metadata_rejected_before_download(self):
        self.api.info.siblings[0].size = q.MAX_JSON_BYTES + 1
        self.hold(
            "TRAINER_JSON_BYTE_BUDGET", fetch=lambda **_: self.fail("must not download")
        )

    def test_download_escape_rejected(self):
        self.hold("DOWNLOAD_PATH_ESCAPE", fetch=lambda **_: __file__)

    def test_no_pickle_downloads(self):
        requested = []

        def fetch(**kwargs):
            requested.append(kwargs["filename"])
            return self.api.fetch(**kwargs)

        self.observe(fetch=fetch)
        self.assertTrue(all(n.endswith((".json", ".safetensors")) for n in requested))

    def test_invalid_candidate_contract_rejected(self):
        for target in [True, "200", -1, 0]:
            with self.subTest(target=target):
                self.candidate["target_steps"] = target
                with self.assertRaises(q.EvidenceHold):
                    self.observe()

    def test_strict_json_controls(self):
        for raw in [b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}', b"[]", b"null"]:
            with self.subTest(raw=raw), self.assertRaises(q.EvidenceHold):
                q.strict_object(raw)

    def test_bundle_is_exclusive_and_self_consistent_not_signed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "observation"
            q.write_bundle(directory, [self.observe()])
            manifest = json.loads((directory / "manifest.json").read_bytes())
            self.assertFalse(manifest["signed"])
            for name, expected in manifest["sha256"].items():
                self.assertEqual(
                    hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                    expected,
                )
            with self.assertRaises(FileExistsError):
                q.write_bundle(directory, [])


if __name__ == "__main__":
    unittest.main()
