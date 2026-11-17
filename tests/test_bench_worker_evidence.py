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
ALLOWED = {"ReceiptError", "hub_revision", "valid_lane",
           "sanitized_public_results", "publish_bench_results"}
nodes = [node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
         and node.name in ALLOWED]
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
    def publish(self, lane="ring-bench-v3", run_id="c"*32):
        return scope["publish_bench_results"](self.api, "test/receipts",
                                               lane, run_id, self.raw)
    def test_unique_immutable_attempt_and_exact_parent(self):
        result = self.publish()
        self.assertEqual(result["path"], "mesh/bench/ring-bench-v3-"+("c"*32)+"-diff-eval.json")
        self.assertEqual(result["revision"], "b"*40)
        self.assertEqual(result["file_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertTrue(result["publication_verified"])
        upload = self.api.calls[1][1]
        self.assertEqual(upload["repo_type"], "dataset")
        self.assertEqual(upload["parent_commit"], "a"*40)
        self.assertEqual(upload["path_in_repo"], result["path"])
        readback = self.api.calls[-1][1]
        self.assertEqual(readback["revision"], "b"*40)
        self.assertIs(readback["force_download"], True)
    def test_invalid_path_component_never_uploads(self):
        for lane in ("../run", "other/run", "", ".", "..", "x y", "\\evil", "é"):
            with self.subTest(lane=lane):
                self.api.calls.clear()
                with self.assertRaises(argparse.ArgumentTypeError):
                    self.publish(lane=lane)
                self.assertEqual(self.api.calls, [])
    def test_invalid_run_id_never_uploads(self):
        for run_id in ("main", "../bad", "a"*31):
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
            "kind": "fixture", "run_id": "c"*32, "lane": "ring",
            "base": "PRIVATE_BASE_REPO_ID", "adapter": "PRIVATE_ADAPTER_REPO_ID",
            "pairs": [{"prompt_index": 0,"prompt": "PRIVATE_PROMPT",
                       "base_output": "PRIVATE_BASE_OUTPUT",
                       "adapter_output": "PRIVATE_ADAPTER_OUTPUT",
                       "identical": False, "doctrine_markers_base": 0,
                       "doctrine_markers_adapter": 1}],
            "no_verdict": True,
        }
        result = scope["sanitized_public_results"](private)
        txt = json.dumps(result)
        for secret in ("PRIVATE_BASE_REPO_ID","PRIVATE_ADAPTER_REPO_ID",
                       "PRIVATE_BASE_OUTPUT","PRIVATE_ADAPTER_OUTPUT","PRIVATE_PROMPT"):
            self.assertNotIn(secret, txt)
        self.assertEqual(result["privacy"]["binding_status"],
                         "MODEL_REVISIONS_NOT_VERIFIED")
        self.assertTrue(result["no_verdict"])
    def test_smoke_namespace_not_rebound_after_parse(self):
        main = next(node for node in TREE.body
                    if isinstance(node, ast.FunctionDef) and node.name=="main")
        assignments = [node for node in ast.walk(main)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id=="a" for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertIn("parse_args", ast.unparse(assignments[0]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
