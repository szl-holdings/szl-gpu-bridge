"""Distill lane recipe: base image, max_length, completion honesty. CPU only."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "distill"))

import distill_corpus as dc  # noqa: E402
import render_recipe as rr  # noqa: E402
import train_lane as tl  # noqa: E402


class RecipeTests(unittest.TestCase):
    def test_rendered_recipe_passes_lint(self):
        self.assertEqual(rr.lint_recipe(rr.BASE_IMAGE, rr.render_command()), [])

    def test_base_image_is_the_working_one(self):
        self.assertIn("pytorch:2.14.1-cuda12.6", rr.BASE_IMAGE)
        self.assertTrue(
            rr.lint_recipe(
                "docker.io/pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
                rr.render_command(),
            )
        )

    def test_no_pip_step_touches_torch(self):
        script = rr.render_command()[-1]
        for line in script.splitlines():
            if line.startswith("pip install"):
                self.assertNotRegex(line, r"\b(torch|torchvision|torchaudio)\b")
        bad = script.replace("pip install -q ", "pip install -q torch==2.15.0 ")
        self.assertTrue(rr.lint_recipe(rr.BASE_IMAGE, ["bash", "-lc", bad]))

    def test_bash_never_reports_completion(self):
        script = rr.render_command()[-1]
        self.assertNotIn("LANE_COMPLETE", script)
        bad = script + "\necho LANE_COMPLETE"
        self.assertTrue(rr.lint_recipe(rr.BASE_IMAGE, ["bash", "-lc", bad]))

    def test_script_is_fail_fast_and_ends_in_trainer(self):
        script = rr.render_command()[-1]
        self.assertIn("set -euo pipefail", script)
        self.assertNotIn("|| true", script)
        self.assertIn(
            "torchvision", script, "import smoke test must exercise torchvision"
        )
        self.assertTrue(script.strip().splitlines()[-1].startswith("exec python3"))

    def test_shipped_files_roundtrip_exactly(self):
        script = rr.render_command()[-1]
        for name in rr.SHIPPED_FILES:
            m = re.search(
                r"echo (\S+) \| base64 -d > /workspace/lane/%s" % re.escape(name),
                script,
            )
            self.assertIsNotNone(m, name)
            self.assertEqual(
                base64.b64decode(m.group(1)), (Path(rr.HERE) / name).read_bytes()
            )

    def test_cli_check(self):
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "distill" / "render_recipe.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


class OldTrl:
    """Stands in for trl 1.17.1 SFTConfig: accepts max_length only."""

    def __init__(self, output_dir=None, max_length=None, max_steps=None):
        pass


class LegacyTrl:
    def __init__(self, output_dir=None, max_seq_length=None, max_steps=None):
        pass


class NeitherTrl:
    def __init__(self, output_dir=None, max_steps=None):
        pass


class SftKwargsTests(unittest.TestCase):
    def test_uses_max_length_when_accepted(self):
        kw = tl.sft_config_kwargs(OldTrl, 1024, output_dir="x", max_steps=5)
        self.assertEqual(kw["max_length"], 1024)
        self.assertNotIn("max_seq_length", kw)
        OldTrl(**kw)  # the real constructor accepts it

    def test_falls_back_to_legacy_name_only_if_that_is_all_there_is(self):
        kw = tl.sft_config_kwargs(LegacyTrl, 512, output_dir="x", max_steps=5)
        self.assertEqual(kw["max_seq_length"], 512)

    def test_fails_instead_of_silently_dropping(self):
        with self.assertRaises(tl.LaneError):
            tl.sft_config_kwargs(NeitherTrl, 1024, output_dir="x")

    def test_unknown_kwarg_fails(self):
        with self.assertRaises(tl.LaneError):
            tl.sft_config_kwargs(OldTrl, 1024, output_dir="x", bogus=1)


GOOD_ENV = {
    "SZL_TRAIN_APPROVAL_ID": "approval-test-1",
    "LANE_NAME": "distill-1.7b-r1",
    "BASE": "Qwen/Qwen3-1.7B",
    "CANDIDATE": "SZLHOLDINGS/test-adapter",
    "CORPUS_DATASET": "SZLHOLDINGS/test-corpus",
    "MAX_STEPS": "10",
}


class FakeDeps:
    def __init__(
        self, corpus_src, workdir, write_final=True, remote=None, train_error=None
    ):
        self.corpus_src = Path(corpus_src)
        self.workdir = Path(workdir)
        self.write_final = write_final
        self.remote = remote if remote is not None else list(tl.REQUIRED_ADAPTER_FILES)
        self.train_error = train_error
        self.calls = []

    def fetch_corpus(self, repo, dest):
        self.calls.append("fetch")
        Path(dest).mkdir(parents=True, exist_ok=True)
        for p in self.corpus_src.iterdir():
            (Path(dest) / p.name).write_bytes(p.read_bytes())

    def train(self, cfg, records, out_dir):
        self.calls.append("train")
        if self.train_error:
            raise self.train_error
        final = Path(out_dir) / "final"
        final.mkdir(parents=True, exist_ok=True)
        if self.write_final:
            for name in tl.REQUIRED_ADAPTER_FILES:
                (final / name).write_bytes(b"x")
        return {"train_loss": 0.5, "records_seen": float(len(records))}

    def create_private_repo(self, repo_id):
        self.calls.append("create_private")

    def upload_folder(self, folder, repo_id):
        self.calls.append("upload")

    def list_repo_files(self, repo_id):
        self.calls.append("list")
        return self.remote


class LaneRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus = Path(cls.tmp.name) / "corpus"
        dc.build(cls.corpus, seed=3, n=2100)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def env(self, work, **over):
        e = dict(
            GOOD_ENV,
            OUT_DIR=str(Path(work) / "out"),
            CORPUS_DIR=str(Path(work) / "corpus"),
        )
        e.update(over)
        return e

    def test_happy_path_uploads_private_and_verifies_before_receipt(self):
        with tempfile.TemporaryDirectory() as w:
            deps = FakeDeps(self.corpus, w)
            receipt = tl.run(self.env(w), deps)
            self.assertEqual(
                deps.calls, ["fetch", "train", "create_private", "upload", "list"]
            )
            self.assertTrue(receipt["private"])
            self.assertEqual(receipt["heldout_eval"], "NOT_RUN")
            self.assertIn("no quality claim", receipt["claims"])
            self.assertTrue((Path(w) / "out" / "completion_receipt.json").is_file())

    def test_no_approval_id_means_no_run(self):
        with tempfile.TemporaryDirectory() as w:
            deps = FakeDeps(self.corpus, w)
            with self.assertRaisesRegex(tl.LaneError, "SZL_TRAIN_APPROVAL_ID"):
                tl.run(self.env(w, SZL_TRAIN_APPROVAL_ID=""), deps)
            self.assertEqual(deps.calls, [])

    def test_max_steps_bounded(self):
        with tempfile.TemporaryDirectory() as w:
            for bad in ("0", "999999", "abc"):
                with self.assertRaises(tl.LaneError):
                    tl.read_config(self.env(w, MAX_STEPS=bad))

    def test_label_must_match_base_size(self):
        with self.assertRaisesRegex(tl.LaneError, "names size"):
            tl.check_label_matches_base("distill-4b-v2", "Qwen/Qwen3-1.7B")
        tl.check_label_matches_base("distill-1.7b-r1", "Qwen/Qwen3-1.7B")
        tl.check_label_matches_base("distill-27b-r1", "Qwen/Qwen3.8-27B")
        tl.check_label_matches_base("distill-r1", "Qwen/Qwen3-1.7B")

    def test_tiny_corpus_refused_before_any_training(self):
        with tempfile.TemporaryDirectory() as w:
            small = Path(w) / "small"
            small.mkdir()
            recs = dc._read(self.corpus / "train.jsonl")[:4]
            (small / "train.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in recs)
            )
            (small / "heldout.jsonl").write_text("")
            manifest = json.loads((self.corpus / "manifest.json").read_text())
            (small / "manifest.json").write_text(json.dumps(manifest))
            deps = FakeDeps(small, w)
            with self.assertRaisesRegex(tl.LaneError, "corpus refused"):
                tl.run(self.env(w), deps)
            self.assertNotIn("train", deps.calls)

    def test_training_crash_means_no_completion(self):
        with tempfile.TemporaryDirectory() as w:
            deps = FakeDeps(
                self.corpus,
                w,
                train_error=RuntimeError("torchvision::nms does not exist"),
            )
            with self.assertRaises(RuntimeError):
                tl.run(self.env(w), deps)
            self.assertNotIn("upload", deps.calls)

    def test_missing_final_dir_means_no_completion(self):
        with tempfile.TemporaryDirectory() as w:
            deps = FakeDeps(self.corpus, w, write_final=False)
            with self.assertRaisesRegex(tl.LaneError, "final/"):
                tl.run(self.env(w), deps)
            self.assertNotIn("upload", deps.calls)

    def test_unverified_remote_upload_means_no_completion(self):
        with tempfile.TemporaryDirectory() as w:
            deps = FakeDeps(self.corpus, w, remote=["README.md"])
            with self.assertRaisesRegex(tl.LaneError, "upload not verified"):
                tl.run(self.env(w), deps)
            self.assertFalse((Path(w) / "out" / "completion_receipt.json").exists())

    def test_main_prints_failed_and_exits_nonzero_without_token(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "distill" / "train_lane.py")],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("LANE_FAILED", r.stderr)
        self.assertNotIn("LANE_COMPLETE", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
