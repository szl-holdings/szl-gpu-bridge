"""Distill corpus builder: size, split, ABSTAIN, dedup, provenance, fail-closed.

CPU only. No network, no Hugging Face, no paid generation.
"""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "distill"))

import distill_corpus as dc  # noqa: E402


class CorpusBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "corpus"
        cls.manifest = dc.build(cls.out, seed=7, n=2200)
        cls.train = dc._read(cls.out / "train.jsonl")
        cls.heldout = dc._read(cls.out / "heldout.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_at_least_two_thousand_distinct_prompts(self):
        recs = self.train + self.heldout
        self.assertGreaterEqual(len(recs), 2000)
        keys = {dc.normalise(r["messages"][1]["content"]) for r in recs}
        self.assertEqual(len(keys), len(recs))

    def test_heldout_exists_and_is_disjoint(self):
        self.assertGreaterEqual(len(self.heldout), dc.MIN_HELDOUT)
        tp = {r["messages"][1]["content"] for r in self.train}
        hp = {r["messages"][1]["content"] for r in self.heldout}
        self.assertFalse(tp & hp)
        tg = {r["meta"]["group"] for r in self.train}
        hg = {r["meta"]["group"] for r in self.heldout}
        self.assertFalse(tg & hg)

    def test_abstain_examples_in_both_splits_and_all_kinds(self):
        for recs in (self.train, self.heldout):
            ab = [r for r in recs if r["meta"]["label"] == "ABSTAIN"]
            frac = len(ab) / len(recs)
            self.assertGreaterEqual(frac, dc.MIN_ABSTAIN_FRAC)
            self.assertLessEqual(frac, dc.MAX_ABSTAIN_FRAC)
            self.assertEqual(
                {r["meta"]["abstain_kind"] for r in ab}, set(dc.ABSTAIN_KINDS)
            )
            for r in ab:
                self.assertEqual(r["messages"][2]["content"], "ABSTAIN")

    def test_answers_only_use_identifiers_from_prompt(self):
        for r in self.train + self.heldout:
            prompt, answer = r["messages"][1]["content"], r["messages"][2]["content"]
            if r["meta"]["label"] == "ANSWER":
                self.assertNotEqual(answer, "ABSTAIN")
                self.assertTrue(dc.identifiers(answer))
            self.assertFalse(dc.ungrounded_identifiers(prompt, answer))

    def test_every_record_has_provenance(self):
        for r in self.train + self.heldout:
            m = r["meta"]
            self.assertEqual(m["source"], "synthetic-template")
            self.assertEqual(m["generator"], dc.GENERATOR)
            self.assertEqual(m["seed"], 7)
            self.assertIn(m["split"], ("train", "heldout"))
            self.assertTrue(m["template_id"] and m["group"])
        prov = self.manifest["provenance"]
        self.assertIsNone(prov["teacher_model"])
        self.assertFalse(prov["paid_generation"])
        self.assertFalse(prov["network_used"])

    def test_template_diversity(self):
        self.assertGreaterEqual(self.manifest["stats"]["families"], dc.MIN_TEMPLATES)
        self.assertLessEqual(
            self.manifest["stats"]["max_family_share"], dc.MAX_TEMPLATE_SHARE
        )

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as other:
            m2 = dc.build(Path(other) / "c", seed=7, n=2200)
        self.assertEqual(m2["files"], self.manifest["files"])

    def test_verify_roundtrip_passes(self):
        self.assertEqual(dc.verify(self.out)["train"]["records"], len(self.train))

    def test_cli_build_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            script = str(ROOT / "tools" / "distill" / "distill_corpus.py")
            b = subprocess.run(
                [sys.executable, script, "build", "--out", d, "--n", "2100"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(b.returncode, 0, b.stderr)
            v = subprocess.run(
                [sys.executable, script, "verify", d], capture_output=True, text=True
            )
            self.assertEqual(v.returncode, 0, v.stderr)


class FailClosedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "c"
        dc.build(cls.out, seed=11, n=2100)
        cls.train = dc._read(cls.out / "train.jsonl")
        cls.heldout = dc._read(cls.out / "heldout.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh(self):
        return copy.deepcopy(self.train), copy.deepcopy(self.heldout)

    def test_tiny_corpora_like_the_stopped_lanes_are_refused(self):
        for size in (1, 2, 4, 46):
            with self.assertRaisesRegex(dc.CorpusError, "too small"):
                dc.validate_corpus(self.train[:size], [])

    def test_build_refuses_to_generate_too_few(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(dc.CorpusError, "too small"):
                dc.build(Path(d) / "c", seed=1, n=500)
            self.assertFalse(
                (Path(d) / "c").exists(), "nothing may be written on failure"
            )

    def test_all_answer_corpus_is_refused(self):
        train, held = self.fresh()
        for r in train + held:
            if r["meta"]["label"] == "ABSTAIN":
                r["messages"][2]["content"] = "Some answer."
                r["meta"]["label"] = "ANSWER"
                r["meta"]["abstain_kind"] = None
        with self.assertRaises(dc.CorpusError):
            dc.validate_corpus(train, held)

    def test_missing_heldout_is_refused(self):
        train, held = self.fresh()
        with self.assertRaisesRegex(dc.CorpusError, "held-out split too small"):
            dc.validate_corpus(train + held, [])

    def test_leakage_is_refused(self):
        train, held = self.fresh()
        held[0] = copy.deepcopy(train[0])
        with self.assertRaisesRegex(dc.CorpusError, "leakage|duplicate"):
            dc.validate_corpus(train, held)

    def test_group_leakage_is_refused(self):
        train, held = self.fresh()
        held[0]["meta"]["group"] = train[0]["meta"]["group"]
        with self.assertRaisesRegex(dc.CorpusError, "group leakage"):
            dc.validate_corpus(train, held)

    def test_duplicate_prompt_is_refused(self):
        train, held = self.fresh()
        train[1] = copy.deepcopy(train[0])
        with self.assertRaisesRegex(dc.CorpusError, "duplicate prompt"):
            dc.validate_corpus(train, held)

    def test_ungrounded_identifier_is_refused(self):
        train, held = self.fresh()
        idx = next(i for i, r in enumerate(train) if r["meta"]["label"] == "ANSWER")
        train[idx]["messages"][2]["content"] += " See CTR-99999999."
        with self.assertRaisesRegex(dc.CorpusError, "ungrounded"):
            dc.validate_corpus(train, held)

    def test_abstain_must_be_exact_token(self):
        train, held = self.fresh()
        idx = next(i for i, r in enumerate(train) if r["meta"]["label"] == "ABSTAIN")
        train[idx]["messages"][2]["content"] = "ABSTAIN."
        with self.assertRaisesRegex(dc.CorpusError, "mismatch"):
            dc.validate_corpus(train, held)

    def test_wrong_system_prompt_is_refused(self):
        train, held = self.fresh()
        train[0]["messages"][0]["content"] = "You are helpful."
        with self.assertRaisesRegex(dc.CorpusError, "grounding gate"):
            dc.validate_corpus(train, held)

    def test_missing_provenance_is_refused(self):
        train, held = self.fresh()
        del train[0]["meta"]["template_id"]
        with self.assertRaisesRegex(dc.CorpusError, "provenance"):
            dc.validate_corpus(train, held)

    def test_low_template_diversity_is_refused(self):
        train, held = self.fresh()
        for r in train + held:
            r["meta"]["template_id"] = "one/template#f0"
        with self.assertRaisesRegex(dc.CorpusError, "template"):
            dc.validate_corpus(train, held)

    def test_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "c"
            dc.build(work, seed=11, n=2100)
            dc.verify(work)
            path = work / "train.jsonl"
            path.write_text(path.read_text().replace("ABSTAIN", "ANSWER?", 1))
            with self.assertRaisesRegex(dc.CorpusError, "sha256"):
                dc.verify(work)

    def test_verify_requires_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(dc.CorpusError, "manifest"):
                dc.verify(Path(d))

    def test_cli_exits_nonzero_on_failure(self):
        script = str(ROOT / "tools" / "distill" / "distill_corpus.py")
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, script, "build", "--out", d, "--n", "100"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 2)
            self.assertIn("FAIL-CLOSED", r.stderr)


class DedupTests(unittest.TestCase):
    def rec(self, text):
        return {"messages": [{}, {"content": text}, {}]}

    def test_exact_and_normalised_duplicates_dropped(self):
        kept, stats = dc.dedup(
            [
                self.rec("What is CTR-12345?"),
                self.rec("what is  ctr-12345 ??"),
                self.rec("Other thing entirely"),
            ]
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["exact"], 1)

    def test_near_duplicates_dropped_but_different_ids_kept(self):
        base = " ".join("word%d" % i for i in range(60))
        kept, stats = dc.dedup(
            [self.rec(base + " CTR-10001"), self.rec(base + " CTR-10001 x")]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["near"], 1)
        short = [
            self.rec("What is the status of CTR-10001?"),
            self.rec("What is the status of CTR-10002?"),
        ]
        self.assertEqual(len(dc.dedup(short)[0]), 2)


if __name__ == "__main__":
    unittest.main()
