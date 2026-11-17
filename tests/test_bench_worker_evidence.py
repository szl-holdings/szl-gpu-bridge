"""Offline only: bench publication contracts; no GPU, provider, network or model import."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
import types
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "tools/salad/salad_bench_worker.py"
if not SOURCE.is_file():  # isolated private staging on BETTERWITHAGE
    SOURCE = Path(__file__).with_name("salad_bench_worker.py")
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
ALLOWED = {
    "ReceiptError",
    "hub_revision",
    "valid_lane",
    "sanitized_public_results",
    "publish_bench_results",
    "resolve_bench_model_pins",
}
nodes = [
    node
    for node in TREE.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ALLOWED
]
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
scope = {"argparse": argparse, "hashlib": hashlib, "re": re, "json": json}
exec(compile(module, str(SOURCE), "exec"), scope)


class API:
    def __init__(self):
        self.sha = "a" * 40
        self.calls = []
        self.raw = None
        self.fail_upload = False

    def repo_info(self, repo, repo_type):
        self.calls.append(("repo_info", repo, repo_type))
        return types.SimpleNamespace(sha=self.sha)

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))
        if self.fail_upload:
            raise RuntimeError("synthetic write failure")
        self.raw = kw["path_or_fileobj"]
        return types.SimpleNamespace(oid="b" * 40)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.api = API()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "readback.json"
        self.raw = b'{"kind":"synthetic","no_verdict":true}\n'
        self.original_download = scope.get("hf_hub_download")

        def download(**kwargs):
            self.api.calls.append(("readback", kwargs))
            self.path.write_bytes(self.api.raw or b"")
            return str(self.path)

        scope["hf_hub_download"] = download

    def publish(self, lane="ring-bench-v3", run_id="c" * 32):
        return scope["publish_bench_results"](
            self.api, "test/receipts", lane, run_id, self.raw
        )

    def test_unique_immutable_attempt_and_exact_parent(self):
        result = self.publish()
        self.assertEqual(
            result["path"], "mesh/bench/ring-bench-v3-" + ("c" * 32) + "-diff-eval.json"
        )
        self.assertEqual(result["revision"], "b" * 40)
        self.assertEqual(result["file_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertTrue(result["publication_verified"])
        upload = self.api.calls[1][1]
        self.assertEqual(upload["repo_type"], "dataset")
        self.assertEqual(upload["parent_commit"], "a" * 40)
        self.assertEqual(upload["path_in_repo"], result["path"])
        readback = self.api.calls[-1][1]
        self.assertEqual(readback["revision"], "b" * 40)
        self.assertIs(readback["force_download"], True)

    def test_invalid_path_component_never_uploads(self):
        for lane in ("../run", "other/run", "", ".", "..", "x y", "\\evil", "é"):
            with self.subTest(lane=lane):
                self.api.calls.clear()
                with self.assertRaises(argparse.ArgumentTypeError):
                    self.publish(lane=lane)
                self.assertEqual(self.api.calls, [])

    def test_invalid_run_id_never_uploads(self):
        for run_id in ("main", "../bad", "a" * 31):
            with self.subTest(run_id=run_id):
                self.api.calls.clear()
                with self.assertRaises(scope["ReceiptError"]):
                    self.publish(run_id=run_id)
                self.assertEqual(self.api.calls, [])

    def test_no_mutable_parent(self):
        self.api.sha = "main"
        with self.assertRaises(scope["ReceiptError"]):
            self.publish()
        self.assertEqual(len(self.api.calls), 1)

    def test_mismatched_readback_fails_closed(self):
        scope["hf_hub_download"] = lambda **kw: str(self.path)
        self.path.write_bytes(b"bad")
        with self.assertRaises(scope["ReceiptError"]):
            self.publish()

    def test_failed_upload_has_no_completion(self):
        self.api.fail_upload = True
        with self.assertRaises(RuntimeError):
            self.publish()

    def test_public_allowlist_withholds_raw_outputs_and_private_ids(self):
        private = {
            "kind": "fixture",
            "run_id": "c" * 32,
            "lane": "ring",
            "base": "PRIVATE_BASE_REPO_ID",
            "adapter": "PRIVATE_ADAPTER_REPO_ID",
            "pairs": [
                {
                    "prompt_index": 0,
                    "prompt": "PRIVATE_PROMPT",
                    "base_output": "PRIVATE_BASE_OUTPUT",
                    "adapter_output": "PRIVATE_ADAPTER_OUTPUT",
                    "identical": False,
                    "doctrine_markers_base": 0,
                    "doctrine_markers_adapter": 1,
                }
            ],
            "no_verdict": True,
        }
        result = scope["sanitized_public_results"](private)
        txt = json.dumps(result)
        for secret in (
            "PRIVATE_BASE_REPO_ID",
            "PRIVATE_ADAPTER_REPO_ID",
            "PRIVATE_BASE_OUTPUT",
            "PRIVATE_ADAPTER_OUTPUT",
            "PRIVATE_PROMPT",
        ):
            self.assertNotIn(secret, txt)
        self.assertEqual(
            result["privacy"]["binding_status"], "MODEL_REVISIONS_NOT_VERIFIED"
        )
        self.assertTrue(result["no_verdict"])

    def test_smoke_namespace_not_rebound_after_parse(self):
        main = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        assignments = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "a" for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertIn("parse_args", ast.unparse(assignments[0]))

    def test_pin_base_and_adapter_revisions_before_hydration(self):
        class ModelApi:
            def __init__(self):
                self.calls = []

            def model_info(self, repo, files_metadata=False):
                self.calls.append((repo, files_metadata))
                if repo == "test/base":
                    return types.SimpleNamespace(sha="a" * 40)
                if repo == "test/adapter":
                    entries = [
                        types.SimpleNamespace(rfilename=p)
                        for p in (
                            "final/adapter_config.json",
                            "final/adapter_model.safetensors",
                        )
                    ]
                    return types.SimpleNamespace(sha="b" * 40, siblings=entries)
                raise AssertionError("unexpected model")

        api = ModelApi()
        pins = scope["resolve_bench_model_pins"](api, "test/base", "test/adapter")
        self.assertEqual(api.calls, [("test/base", False), ("test/adapter", True)])
        self.assertEqual(pins["base"]["revision"], "a" * 40)
        self.assertEqual(pins["adapter"]["revision"], "b" * 40)
        self.assertEqual(len(pins["revision_commitment_sha256"]), 64)
        public = scope["sanitized_public_results"](
            {
                "kind": "synthetic",
                "pairs": [],
                "no_verdict": True,
                "model_binding": {
                    "revision_commitment_sha256": pins["revision_commitment_sha256"]
                },
            }
        )
        self.assertNotIn("test/base", json.dumps(public))
        self.assertNotIn("test/adapter", json.dumps(public))
        self.assertEqual(
            public["privacy"]["binding_status"],
            "IMMUTABLE_MODEL_REVISIONS_PINNED_WEIGHT_BYTES_NOT_VERIFIED",
        )

    def test_model_revision_pin_missing_rejects(self):
        for stage in ("base", "adapter"):

            class BrokenApi:
                def model_info(self, repo, files_metadata=False):
                    if (stage == "base" and repo == "test/base") or (
                        stage == "adapter" and repo == "test/adapter"
                    ):
                        return types.SimpleNamespace(sha="main", siblings=[])
                    return types.SimpleNamespace(sha="a" * 40)

            with self.subTest(stage=stage):
                with self.assertRaises(scope["ReceiptError"]):
                    scope["resolve_bench_model_pins"](
                        BrokenApi(), "test/base", "test/adapter"
                    )

    def test_unobserved_final_files_reject_before_download(self):
        class MissingFinal:
            def model_info(self, repo, files_metadata=False):
                return types.SimpleNamespace(sha="a" * 40, siblings=[])

        with self.assertRaisesRegex(scope["ReceiptError"], "INCOMPLETE"):
            scope["resolve_bench_model_pins"](
                MissingFinal(), "test/base", "test/adapter"
            )

    def test_metadata_denial_makes_no_fallback_request(self):
        class Denied:
            def __init__(self):
                self.calls = 0

            def model_info(self, *args, **kwargs):
                self.calls += 1
                raise PermissionError("synthetic 403")

        api = Denied()
        with self.assertRaisesRegex(scope["ReceiptError"], "UNOBSERVED"):
            scope["resolve_bench_model_pins"](api, "test/base", "test/adapter")
        self.assertEqual(api.calls, 1)

    def test_all_model_loaders_use_pinned_revisions(self):
        main = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        named = {
            ast.unparse(node.func): node
            for node in calls
            if ast.unparse(node.func)
            in (
                "AutoTokenizer.from_pretrained",
                "AutoModelForCausalLM.from_pretrained",
                "snapshot_download",
            )
        }
        self.assertEqual(len(named), 3)
        for method, role in (
            ("AutoTokenizer.from_pretrained", "base"),
            ("AutoModelForCausalLM.from_pretrained", "base"),
            ("snapshot_download", "adapter"),
        ):
            kwargs = {x.arg: ast.unparse(x.value) for x in named[method].keywords}
            self.assertEqual(kwargs["revision"], f"pins['{role}']['revision']")
            self.assertEqual(kwargs["token"], "a.hf_token")
        for method in (
            "AutoTokenizer.from_pretrained",
            "AutoModelForCausalLM.from_pretrained",
        ):
            kwargs = {x.arg: ast.unparse(x.value) for x in named[method].keywords}
            self.assertEqual(kwargs["trust_remote_code"], "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
