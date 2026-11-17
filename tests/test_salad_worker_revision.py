"""Pinned inputs must remain pinned through the dataset fallback path."""

import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

WORKER = Path(__file__).resolve().parents[1] / "tools" / "salad" / "salad_worker.py"


class DatasetRevision(unittest.TestCase):
    def setUp(self):
        source = ast.parse(WORKER.read_text(encoding="utf-8"))
        functions = [
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("load_train_split", "transport_label")
        ]
        self.assertEqual(
            {function.name for function in functions},
            {"load_train_split", "transport_label"},
        )
        self.load = Mock(return_value="SYNTHETIC_DATASET")
        namespace = {"HfApi": object, "load_dataset": self.load, "receipt": Mock()}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(WORKER), "exec"),
            namespace,
        )
        self.load_split = namespace["load_train_split"]
        self.args = types.SimpleNamespace(
            dataset="fixture/private",
            dataset_revision="a" * 40,
            dataset_train_file="train.jsonl",
            dataset_config="",
            hf_token="unit-test-not-a-credential",
        )

    def test_initial_load_selects_pinned_repository_file(self):
        self.assertEqual(self.load_split(self.args, None), "SYNTHETIC_DATASET")
        self.assertEqual(self.load.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(
            self.load.call_args.kwargs["data_files"], {"train": "train.jsonl"}
        )

    def test_fallback_uses_pinned_hub_bytes_instead_of_working_directory_file(self):
        self.load.side_effect = [RuntimeError("split schema"), "SYNTHETIC_PINNED_DATA"]
        download = Mock(return_value="/fixture/pinned-cache/train.jsonl")
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = download
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            self.assertEqual(self.load_split(self.args, None), "SYNTHETIC_PINNED_DATA")
        self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(
            self.load.call_args.kwargs["data_files"],
            {"train": "/fixture/pinned-cache/train.jsonl"},
        )


if __name__ == "__main__":
    unittest.main()
