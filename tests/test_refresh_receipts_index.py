# SPDX-License-Identifier: Apache-2.0
"""Offline tests for tools/receipts_index/refresh_receipts_index.py (no network)."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refresh_receipts_index", ROOT / "tools/receipts_index/refresh_receipts_index.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def report(**kw):
    base = {
        "old_rows": 10,
        "new_rows": 10,
        "added": 0,
        "removed": 0,
        "changed_fields": {},
        "hash_status": {"VERIFIED": 8, "UNVERIFIED": 1, "NO_HASH": 1},
    }
    base.update(kw)
    return base


class RefreshTests(unittest.TestCase):
    def test_ref_and_snapshot_authentication_respect_mode(self):
        revision = "a" * 40
        refs = SimpleNamespace(
            branches=[SimpleNamespace(name="main", target_commit=revision)]
        )
        api = Mock()
        api.list_repo_refs.return_value = refs
        snapshot_download = Mock()
        hub = SimpleNamespace(
            HfApi=Mock(return_value=api), snapshot_download=snapshot_download
        )

        for read_only in (True, False):
            with (
                self.subTest(read_only=read_only),
                patch.dict(sys.modules, {"huggingface_hub": hub}),
                patch.dict(os.environ, {"HF_TOKEN": "synthetic-fixture-only"}),
            ):
                api.reset_mock()
                snapshot_download.reset_mock()
                self.assertEqual(
                    R.resolve_revision(R.DEFAULT_REPO, read_only=read_only), revision
                )
                R.snapshot(
                    R.DEFAULT_REPO, revision, Path("unused"), read_only=read_only
                )
                ref_options = {"token": False} if read_only else {}
                api.list_repo_refs.assert_called_once_with(
                    R.DEFAULT_REPO, repo_type="dataset", **ref_options
                )
                download = snapshot_download.call_args.kwargs
                self.assertEqual(download["revision"], revision)
                if read_only:
                    self.assertIs(download["token"], False)
                    self.assertEqual(download["max_workers"], 2)
                else:
                    self.assertEqual(download["token"], "synthetic-fixture-only")
                    self.assertNotIn("max_workers", download)

    def test_completed_check_reports_fresh_after_pinned_snapshot(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "github-output"
            report_path = root / "report"
            snap = root / "snapshot"
            snap.mkdir()
            with (
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
                patch.object(R, "resolve_revision", return_value="a" * 40) as resolve,
                patch.object(R, "snapshot") as snapshot,
                patch.object(R, "provably_stale_json_paths", return_value=None),
                patch.object(R, "publish") as publish,
            ):
                self.assertEqual(
                    R.main(["--workdir", temp, "--report", str(report_path)]), 0
                )
            resolve.assert_called_once_with(R.DEFAULT_REPO, read_only=True)
            snapshot.assert_called_once_with(
                R.DEFAULT_REPO, "a" * 40, snap, read_only=True
            )
            publish.assert_not_called()
            self.assertIn("check_status=FRESH\n", output.read_text())
            self.assertIn("stale=false\n", output.read_text())
            self.assertIn("Stale: **no**", report_path.read_text())

    def test_429_before_or_during_download_reports_unavailable(self):
        response_error = RuntimeError("provider details not for the alert")
        response_error.response = SimpleNamespace(status_code=429)
        wrapper = RuntimeError("snapshot wrapper")
        wrapper.__cause__ = response_error
        for stage in ("resolve_revision", "snapshot"):
            with self.subTest(stage=stage), TemporaryDirectory() as temp:
                root = Path(temp)
                output = root / "github-output"
                report_path = root / "report"
                with (
                    patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
                    patch.object(R, "resolve_revision", return_value="a" * 40),
                    patch.object(R, "snapshot"),
                    patch.object(R, "provably_stale_json_paths", return_value=None),
                    patch.object(R, stage, side_effect=wrapper),
                    patch.object(R.builder, "build_rows") as build,
                    patch.object(R, "publish") as publish,
                ):
                    self.assertEqual(
                        R.main(["--workdir", temp, "--report", str(report_path)]), 3
                    )
                self.assertEqual(output.read_text(), "check_status=CHECK_UNAVAILABLE\n")
                self.assertFalse((root / "new-index.jsonl").exists())
                self.assertIn("freshness was not determined", report_path.read_text())
                self.assertNotIn("provider details", report_path.read_text())
                build.assert_not_called()
                publish.assert_not_called()

    def test_repeated_429_on_same_pinned_snapshot_never_creates_index(self):
        error = RuntimeError("rate limited")
        error.response = SimpleNamespace(status_code=429)
        wrapper = RuntimeError("snapshot wrapper")
        wrapper.__cause__ = error
        revision = "a" * 40
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "github-output"
            with (
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
                patch.object(R, "resolve_revision", return_value=revision),
                patch.object(R, "snapshot", side_effect=[error, wrapper]) as snapshot,
                patch.object(R, "provably_stale_json_paths", return_value=None),
                patch.object(R, "publish") as publish,
            ):
                for _ in range(2):
                    self.assertEqual(R.main(["--workdir", temp]), 3)
            self.assertEqual(snapshot.call_count, 2)
            for call in snapshot.call_args_list:
                self.assertEqual(
                    call.args, (R.DEFAULT_REPO, revision, root / "snapshot")
                )
                self.assertTrue(call.kwargs["read_only"])
            self.assertEqual(output.read_text(), "check_status=CHECK_UNAVAILABLE\n" * 2)
            self.assertFalse((root / "new-index.jsonl").exists())
            publish.assert_not_called()

    def test_metadata_preflight_proves_only_unindexed_json_paths(self):
        class RepoFile:
            def __init__(self, path):
                self.path = path

        class RepoFolder:
            def __init__(self, path):
                self.path = path

        with TemporaryDirectory() as temp:
            index = Path(temp) / "index.jsonl"
            index.write_text('{"path":"old/a.json"}\n', encoding="utf-8")
            api = Mock()
            api.list_repo_tree.return_value = iter(
                [
                    RepoFolder("skip.json"),
                    RepoFile("old/a.json"),
                    RepoFile("data/receipts-index.jsonl"),
                    RepoFile(".cache/private.json"),
                    RepoFile("new/empty.jsonl"),
                    RepoFile("new/b.json"),
                    RepoFile("new/c.json"),
                    RepoFile("new/d.json"),  # early stop after threshold+1
                ]
            )
            download = Mock(return_value=str(index))
            hub = SimpleNamespace(
                HfApi=Mock(return_value=api), hf_hub_download=download
            )
            hf_api = SimpleNamespace(RepoFile=RepoFile)
            with patch.dict(
                sys.modules,
                {"huggingface_hub": hub, "huggingface_hub.hf_api": hf_api},
            ):
                self.assertEqual(
                    R.provably_stale_json_paths(R.DEFAULT_REPO, "b" * 40, 1), 2
                )
            download.assert_called_once_with(
                repo_id=R.DEFAULT_REPO,
                repo_type="dataset",
                filename=R.builder.INDEX_RELPATH,
                revision="b" * 40,
                token=False,
            )
            api.list_repo_tree.assert_called_once_with(
                repo_id=R.DEFAULT_REPO,
                repo_type="dataset",
                revision="b" * 40,
                recursive=True,
                token=False,
            )

    def test_missing_published_index_falls_back_to_full_comparison(self):
        class EntryNotFoundError(Exception):
            pass

        missing = EntryNotFoundError("synthetic missing index at pinned revision")
        hub = SimpleNamespace(HfApi=Mock(), hf_hub_download=Mock(side_effect=missing))
        with patch.dict(
            sys.modules,
            {
                "huggingface_hub": hub,
                "huggingface_hub.hf_api": SimpleNamespace(
                    RepoFile=type("RepoFile", (), {})
                ),
                "huggingface_hub.errors": SimpleNamespace(
                    EntryNotFoundError=EntryNotFoundError
                ),
            },
        ):
            self.assertIsNone(R.provably_stale_json_paths(R.DEFAULT_REPO, "b" * 40, 1))
        hub.hf_hub_download.assert_called_once()
        hub.HfApi.assert_not_called()

    def test_absent_index_full_check_emits_stale_not_unavailable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "github-output"
            report_file = root / "report.md"

            def synthetic_snapshot(repo, revision, dest, *, read_only):
                self.assertTrue(read_only)
                (dest / "new").mkdir(parents=True)
                (dest / "new" / "receipt.json").write_text(
                    '{"kind":"synthetic-fixture"}', encoding="utf-8"
                )

            with (
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
                patch.object(R, "resolve_revision", return_value="b" * 40),
                patch.object(R, "provably_stale_json_paths", return_value=None),
                patch.object(R, "snapshot", side_effect=synthetic_snapshot),
                patch.object(R, "publish") as publish,
            ):
                self.assertEqual(
                    R.main(
                        [
                            "--workdir",
                            temp,
                            "--report",
                            str(report_file),
                            "--max-added",
                            "0",
                        ]
                    ),
                    1,
                )
            self.assertIn("check_status=STALE", output.read_text())
            self.assertIn("Stale: **yes**", report_file.read_text())
            publish.assert_not_called()

    def test_inconclusive_metadata_must_not_mark_fresh(self):
        class RepoFile:
            def __init__(self, path):
                self.path = path

        with TemporaryDirectory() as temp:
            index = Path(temp) / "index.jsonl"
            index.write_text('{"path":"old/a.json"}\n', encoding="utf-8")
            api = Mock()
            api.list_repo_tree.return_value = iter(
                [RepoFile("new/empty.jsonl"), RepoFile("new/b.json")]
            )
            hub = SimpleNamespace(
                HfApi=Mock(return_value=api),
                hf_hub_download=Mock(return_value=str(index)),
            )
            with patch.dict(
                sys.modules,
                {
                    "huggingface_hub": hub,
                    "huggingface_hub.hf_api": SimpleNamespace(RepoFile=RepoFile),
                },
            ):
                self.assertIsNone(
                    R.provably_stale_json_paths(R.DEFAULT_REPO, "b" * 40, 1)
                )

    def test_proven_stale_check_never_snapshots_or_publishes(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "github-output"
            with (
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}),
                patch.object(R, "resolve_revision", return_value="b" * 40),
                patch.object(R, "provably_stale_json_paths", return_value=51),
                patch.object(R, "snapshot") as snapshot,
                patch.object(R, "publish") as publish,
            ):
                self.assertEqual(R.main(["--workdir", temp, "--max-added", "50"]), 1)
            snapshot.assert_not_called()
            publish.assert_not_called()
            self.assertEqual(output.read_text().splitlines()[0], "check_status=STALE")
            self.assertFalse((Path(temp) / "new-index.jsonl").exists())

    def test_complete_stale_check_is_nonzero_not_cosmetic_green(self):
        with TemporaryDirectory() as temp:
            with (
                patch.object(R, "resolve_revision", return_value="b" * 40),
                patch.object(R, "provably_stale_json_paths", return_value=None),
                patch.object(R, "snapshot"),
                patch.object(
                    R.builder,
                    "build_rows",
                    return_value=[
                        R.builder.row_for(
                            "new/a.json", {"kind": "test"}, b'{"kind":"test"}', None
                        )
                    ],
                ),
                patch.object(R, "publish") as publish,
            ):
                self.assertEqual(R.main(["--workdir", temp, "--max-added", "0"]), 1)
            publish.assert_not_called()

    def test_other_failures_and_publish_429_still_raise(self):
        for mode, status in (("check", 403), ("publish", 429)):
            error = RuntimeError("HTTP response")
            error.response = SimpleNamespace(status_code=status)
            with (
                self.subTest(mode=mode, status=status),
                patch.object(R, "resolve_revision", side_effect=error),
                patch.object(R, "write_report") as write_report,
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP response"):
                    R.main(["--workdir", "unused", "--mode", mode])
                write_report.assert_not_called()

    def test_fresh_within_threshold(self):
        self.assertFalse(R.is_stale(report(added=R.DEFAULT_MAX_ADDED)))

    def test_stale_when_many_unindexed(self):
        self.assertTrue(R.is_stale(report(added=R.DEFAULT_MAX_ADDED + 1)))

    def test_stale_when_row_removed_or_changed(self):
        self.assertTrue(R.is_stale(report(removed=1)))
        self.assertTrue(R.is_stale(report(changed_fields={"sha256": 1})))

    def test_publish_refuses_removed_rows(self):
        self.assertTrue(R.publish_blockers(report(removed=2), []))

    def test_publish_refuses_invalid_json(self):
        rows = [{"hash_status": "INVALID_JSON"}]
        self.assertTrue(R.publish_blockers(report(), rows))
        self.assertTrue(R.publish_blockers(report(hash_status={"INVALID_JSON": 1}), []))

    def test_publish_allows_additions_only(self):
        self.assertEqual(
            R.publish_blockers(report(added=5), [{"hash_status": "VERIFIED"}]), []
        )

    def test_commit_message_cites_builder_revision_and_counts(self):
        title, body = R.commit_message(report(new_rows=12), "abc123", "6c5a0dc7ffff")
        self.assertIn("6c5a0dc7", title)
        self.assertIn("abc123", body)
        self.assertIn("10 -> 12", body)
        self.assertIn("VERIFIED 8", body)
        self.assertIn("Only this file is changed", body)

    def test_report_states_check_is_read_only(self):
        md = R.markdown_report(report(added=99), "abc", True, 50)
        self.assertIn("Stale: **yes**", md)
        self.assertIn("never writes to the Hub", md)


if __name__ == "__main__":
    unittest.main()
