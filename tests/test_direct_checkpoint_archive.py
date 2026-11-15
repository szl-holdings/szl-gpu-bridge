import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "direct_checkpoint_archive.py"
if not SOURCE.exists():
    SOURCE = HERE.parent / "tools" / "salad" / "direct_checkpoint_archive.py"
spec = importlib.util.spec_from_file_location("direct_checkpoint_archive", SOURCE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rev = "a" * 40
        self.step = 10500
        self.data = {}
        self.entries = []
        for leaf in g.LEAVES:
            name = f"checkpoint-{self.step}/{leaf}"
            data = (
                json.dumps({"global_step": self.step, "max_steps": g.TARGET}).encode()
                if leaf == "trainer_state.json"
                else b"opaque test fixture"
            )
            self.data[name] = data
            self.entries.append(
                NS(
                    rfilename=name,
                    size=len(data),
                    lfs=NS(sha256=hashlib.sha256(data).hexdigest()),
                    blob_id="b" * 40,
                )
            )
        self.info = NS(sha=self.rev, private=True, siblings=self.entries)
        self.api = Mock()
        self.api.model_info.return_value = self.info
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.calls = []

    def download(self, repo, name, **kw):
        self.calls.append((repo, name, kw))
        p = self.cache / Path(name).name
        p.write_bytes(self.data[name])
        return str(p)

    def run_it(self, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return g.run_archive(
                self.api,
                self.download,
                self.root / "archive",
                self.rev,
                self.step,
                **kw,
            )

    def test_valid_snapshot(self):
        self.assertEqual(len(g.select(self.info, self.rev, self.step)), 11)

    def test_pin_required(self):
        with self.assertRaises(g.Hold):
            g.select(self.info, "main", self.step)

    def test_public_denied(self):
        self.info.private = False
        with self.assertRaises(g.Hold):
            self.run_it()
        self.assertEqual(self.calls, [])

    def test_unknown_visibility_denied(self):
        self.info.private = None
        with self.assertRaises(g.Hold):
            self.run_it()

    def test_revision_mismatch(self):
        self.info.sha = "b" * 40
        with self.assertRaises(g.Hold):
            self.run_it()

    def test_missing_optimizer_denied(self):
        self.info.siblings = [
            x for x in self.entries if not x.rfilename.endswith("/optimizer.pt")
        ]
        with self.assertRaises(g.Hold):
            self.run_it()

    def test_unexpected_script_denied(self):
        self.info.siblings.append(
            NS(
                rfilename=f"checkpoint-{self.step}/run.py",
                size=1,
                lfs=None,
                blob_id="a" * 40,
            )
        )
        with self.assertRaises(g.Hold):
            self.run_it()

    def test_boolean_step_denied(self):
        with self.assertRaises(g.Hold):
            g.select(self.info, self.rev, True)

    def test_overshoot_denied(self):
        with self.assertRaises(g.Hold):
            g.select(self.info, self.rev, g.TARGET + 1)

    def test_unsafe_paths(self):
        for name in (
            "../x",
            "/x",
            "C:/x",
            "a\\b",
            "a//b",
            "a/./b",
            "a/../b",
            "a/x ",
            "a/x.",
        ):
            with self.subTest(name=name), self.assertRaises(g.Hold):
                g.relative(name)

    def test_hash_and_size(self):
        entry = self.entries[0]
        p = self.root / "blob"
        p.write_bytes(self.data[entry.rfilename])
        self.assertEqual(g.verify(p, entry)["hub_oid_algorithm"], "sha256")
        p.write_bytes(b"corrupt")
        with self.assertRaises(g.Hold):
            g.verify(p, entry)

    def test_same_size_corruption(self):
        entry = self.entries[0]
        p = self.root / "blob"
        p.write_bytes(b"x" * entry.size)
        with self.assertRaises(g.Hold):
            g.verify(p, entry)

    def test_git_oid(self):
        p = self.root / "blob"
        p.write_bytes(b"hello")
        entry = NS(size=5, lfs=None, blob_id=hashlib.sha1(b"blob 5\0hello").hexdigest())
        self.assertEqual(g.verify(p, entry)["hub_oid_algorithm"], "git-blob-sha1")

    def test_target_mismatch(self):
        with self.assertRaises(g.Hold):
            g.validate_state({"global_step": self.step, "max_steps": 1}, self.step)

    def test_boolean_target(self):
        with self.assertRaises(g.Hold):
            g.validate_state({"global_step": self.step, "max_steps": True}, self.step)

    def test_step_mismatch(self):
        with self.assertRaises(g.Hold):
            g.validate_state({"global_step": 9, "max_steps": g.TARGET}, self.step)

    def test_exact_archive_and_no_mutations(self):
        result = self.run_it()
        self.assertEqual(result["files"], 11)
        self.assertEqual(result["cloud_mutations"], 0)
        self.assertFalse(result["training_complete"])
        self.assertEqual(result["gate1_status"], "UNOBSERVED")
        self.assertTrue(all(c[2]["revision"] == self.rev for c in self.calls))
        manifest = json.loads((Path(result["archive"]) / "manifest.json").read_bytes())
        self.assertFalse(manifest["pickle_deserialized"])
        self.assertEqual(manifest["total_bytes"], sum(map(len, self.data.values())))
        self.assertEqual({c[0] for c in self.api.method_calls}, {"model_info"})

    def test_repeat_rehashes_without_redownload(self):
        first = self.run_it()
        self.calls.clear()
        second = self.run_it()
        self.assertEqual(self.calls, [])
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_existing_corruption_not_overwritten(self):
        result = self.run_it()
        p = Path(result["archive"]) / f"checkpoint-{self.step}/optimizer.pt"
        p.write_bytes(b"corrupt")
        with self.assertRaises(g.Hold):
            self.run_it()
        self.assertEqual(p.read_bytes(), b"corrupt")

    def test_existing_lock_not_removed(self):
        path = self.root / "taken.lock"
        path.write_text("another-writer")
        # Exercise entry directly: a rejected lock has no reachable body.
        manager = g.exclusive_lock(path)
        self.assertRaises(g.Hold, manager.__enter__)
        self.assertEqual(path.read_text(), "another-writer")

    def test_lock_released_on_error(self):
        path = self.root / "owned.lock"

        def fail_inside_lock():
            with g.exclusive_lock(path):
                raise ValueError("synthetic locked-operation failure")

        self.assertRaises(ValueError, fail_inside_lock)
        self.assertFalse(path.exists())

    def test_no_manifest_on_download_failure(self):
        self.download = Mock(side_effect=OSError("synthetic"))
        with self.assertRaises(OSError):
            self.run_it()
        self.assertFalse((self.root / "archive" / self.rev / "manifest.json").exists())

    def test_no_manifest_on_visibility_change(self):
        self.api.model_info.side_effect = [self.info, NS(sha=self.rev, private=False)]
        with self.assertRaises(g.Hold):
            self.run_it()
        self.assertFalse((self.root / "archive" / self.rev / "manifest.json").exists())

    def test_no_manifest_on_deadline(self):
        with self.assertRaises(g.Hold):
            self.run_it(deadline_seconds=-1)
        self.assertFalse((self.root / "archive" / self.rev / "manifest.json").exists())

    def test_low_disk_blocks_download(self):
        with patch.object(g.shutil, "disk_usage", return_value=NS(free=0)):
            with self.assertRaises(g.Hold):
                self.run_it()
        self.assertFalse(self.calls)

    def test_symlink_denied(self):
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(g.Hold):
                g.no_links(self.root / "linked")

    def test_no_pickle_or_model_execution(self):
        src = SOURCE.read_text()
        for text in (
            "torch.load(",
            "pickle.load(",
            "trust_remote_code=True",
            "subprocess.",
        ):
            self.assertNotIn(text, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
