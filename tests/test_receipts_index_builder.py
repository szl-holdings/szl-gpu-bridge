# SPDX-License-Identifier: Apache-2.0
"""Offline tests for tools/receipts_index/build_receipts_index.py (no network)."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_receipts_index", ROOT / "tools/receipts_index/build_receipts_index.py"
)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def sealed(body, field="receipt_sha256", compact=False):
    dump = (
        (lambda o: json.dumps(o, sort_keys=True, separators=(",", ":")))
        if compact
        else (lambda o: json.dumps(o, sort_keys=True))
    )
    return {**body, field: hashlib.sha256(dump(body).encode()).hexdigest()}


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def write(self, rel, obj=None, raw=None):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw if raw is not None else json.dumps(obj), encoding="utf-8")


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.tmp.cleanup)

    def rows(self):
        return {(r["path"], r["line"]): r for r in B.build_rows(self.fx.root)}

    def test_verified_receipt_copies_fields(self):
        self.fx.write(
            "benchmark/b.json",
            sealed({"kind": "k/v1", "state": "MEASURED", "ts": "2026-11-12T00:00:00Z"}),
        )
        r = self.rows()[("benchmark/b.json", "UNKNOWN")]
        self.assertEqual(
            (r["kind"], r["detail"], r["ts"]),
            ("k/v1", "MEASURED", "2026-11-12T00:00:00Z"),
        )
        self.assertEqual(r["hash_status"], "VERIFIED")
        self.assertEqual(r["hash_rule"], "sorted-json-without-hash-field")

    def test_missing_fields_are_unknown_not_invented(self):
        self.fx.write("census/c.json", {"note": "no kind, no ts, no hash"})
        r = self.rows()[("census/c.json", "UNKNOWN")]
        for f in ("kind", "detail", "ts", "sha256", "hash_field", "hash_rule"):
            self.assertEqual(r[f], "UNKNOWN", f)
        self.assertEqual(r["hash_status"], "NO_HASH")

    def test_tampered_receipt_is_not_verified(self):
        good = sealed({"kind": "k", "value": 1})
        good["value"] = 2
        self.fx.write("mesh/t.json", good)
        r = self.rows()[("mesh/t.json", "UNKNOWN")]
        self.assertEqual(r["hash_status"], "UNVERIFIED")
        self.assertEqual(
            r["sha256"], good["receipt_sha256"]
        )  # claimed hash is reported, not replaced

    def test_signed_officer_rule(self):
        body = {
            "kind": "szl.auditor-receipt/v2",
            "ts": "2026-11-11T02:00:27Z",
            "cycle": "11",
        }
        rec = sealed(body, field="body_sha256", compact=True)
        rec.update(signature="ab" * 32, signature_length="64")
        self.fx.write("officers/o.json", rec)
        r = self.rows()[("officers/o.json", "UNKNOWN")]
        self.assertEqual(r["hash_status"], "VERIFIED")
        self.assertEqual(
            r["hash_rule"], "sorted-compact-json-without-hash-and-signature"
        )

    def test_jsonl_one_row_per_line_and_invalid_lines_flagged(self):
        a = sealed({"kind": "pg/v1", "detail": "start", "ts": "t1"})
        self.fx.write("proving-ground/r.jsonl", raw=json.dumps(a) + "\n\nnot json\n")
        rows = self.rows()
        self.assertEqual(rows[("proving-ground/r.jsonl", 1)]["hash_status"], "VERIFIED")
        self.assertEqual(
            rows[("proving-ground/r.jsonl", 3)]["hash_status"], "INVALID_JSON"
        )
        self.assertEqual(len([k for k in rows if k[0] == "proving-ground/r.jsonl"]), 2)

    def test_symlinked_receipts_do_not_escape_or_duplicate_the_snapshot(self):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "outside.json"
            outside.write_text(json.dumps({"kind": "outside-snapshot"}))
            self.fx.write("valid.json", sealed({"kind": "local"}))
            (self.fx.root / "escape.json").symlink_to(outside)
            (self.fx.root / "alias.json").symlink_to(self.fx.root / "valid.json")
            self.assertEqual([row["path"] for row in B.build_rows(self.fx.root)], ["valid.json"])

    def test_index_itself_and_hidden_dirs_are_excluded(self):
        self.fx.write("data/receipts-index.jsonl", raw='{"path": "x"}\n')
        self.fx.write(".cache/huggingface/x.json", {"kind": "cache"})
        self.fx.write("lean/l.json", sealed({"kind": "lean"}))
        self.assertEqual([k[0] for k in self.rows()], ["lean/l.json"])

    def test_output_is_deterministic_and_sorted(self):
        for name in ("z/b.json", "a/a.json", "m/m.json"):
            self.fx.write(name, sealed({"kind": name}))
        one = B.serialise(B.build_rows(self.fx.root))
        two = B.serialise(B.build_rows(self.fx.root))
        self.assertEqual(one, two)
        self.assertEqual(
            [json.loads(line)["path"] for line in one.splitlines()],
            ["a/a.json", "m/m.json", "z/b.json"],
        )

    def test_check_mode_detects_stale_index(self):
        self.fx.write("benchmark/b.json", sealed({"kind": "k"}))
        out = (
            self.fx.root / "custom" / "out.jsonl"
        )  # inside root: must not index itself
        self.assertEqual(B.main(["--root", str(self.fx.root), "--output", str(out)]), 0)
        self.assertEqual(
            B.main(["--root", str(self.fx.root), "--output", str(out), "--check"]), 0
        )
        self.fx.write("benchmark/new.json", sealed({"kind": "k2"}))
        self.assertEqual(
            B.main(["--root", str(self.fx.root), "--output", str(out), "--check"]), 1
        )

    def test_compare_counts_added_rows_and_normalises_v1_placeholders(self):
        self.fx.write("benchmark/old.json", sealed({"note": "no kind"}))
        self.fx.write("benchmark/new.json", sealed({"kind": "k"}))
        rows = B.build_rows(self.fx.root)
        old_row = next(r for r in rows if r["path"] == "benchmark/old.json")
        v1 = [
            {
                "path": "benchmark/old.json",
                "kind": "?",
                "detail": "",
                "ts": "",
                "sha256": old_row["sha256"],
            }
        ]
        rep = B.compare(v1, rows)
        self.assertEqual(rep["added"], 1)
        self.assertEqual(rep["added_paths"], ["benchmark/new.json"])
        self.assertEqual(rep["changed_fields"], {})
        self.assertEqual(rep["removed"], 0)


if __name__ == "__main__":
    unittest.main()
