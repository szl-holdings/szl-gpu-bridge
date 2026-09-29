from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cloud.sync_training_receipts_card import (
    publish,
    render_card,
    validate_card_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


SOURCE_SHA = "b" * 40


class TrainingReceiptsCardTests(unittest.TestCase):
    def test_card_is_private_evidence_not_training_data(self) -> None:
        card = render_card(SOURCE_SHA).decode("utf-8")
        self.assertIn(f"Source revision: `{SOURCE_SHA}`", card)
        self.assertIn("not a training corpus", card)
        self.assertIn("missing evidence", card)
        self.assertIn("license: other", card)
        self.assertIn(
            "license_name: szl-governed-operational-evidence-no-blanket-reuse",
            card,
        )
        self.assertIn("do not receive a blanket data-reuse grant", card)
        self.assertIn(
            "license_link: https://github.com/szl-holdings/szl-gpu-bridge/blob/main/"
            "hf/szl-training-receipts/README.md#license-and-data-handling",
            card,
        )
        self.assertNotIn("__SOURCE_REVISION__", card)

    def test_card_enforces_hugging_face_license_name_constraint(self) -> None:
        card = render_card(SOURCE_SHA).decode("utf-8")
        validate_card_metadata(card)
        for invalid in (
            "SZL-governed-operational-evidence-no-blanket-reuse",
            "szl governed operational evidence no blanket reuse",
            "szl_governed_operational_evidence_no_blanket_reuse",
        ):
            with self.subTest(license_name=invalid):
                with self.assertRaisesRegex(ValueError, "Hugging Face pattern"):
                    validate_card_metadata(
                        card.replace(
                            "szl-governed-operational-evidence-no-blanket-reuse",
                            invalid,
                            1,
                        )
                    )

    def test_card_rejects_removed_no_blanket_reuse_terms(self) -> None:
        card = render_card(SOURCE_SHA).decode("utf-8")
        with self.assertRaisesRegex(ValueError, "no-blanket-reuse terms"):
            validate_card_metadata(
                card.replace(
                    "do not receive a blanket data-reuse grant", "are reusable"
                )
            )

    def test_card_rejects_mutable_source_reference(self) -> None:
        for value in ("main", "b" * 12, "G" * 40):
            with self.subTest(source_sha=value):
                with self.assertRaises(ValueError):
                    render_card(value)


class _Info:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class _Commit:
    def __init__(self, oid: str) -> None:
        self.oid = oid


class _FakeHub:
    """In-memory stand-in for HfApi: one dataset, README.md per revision."""

    def __init__(
        self,
        *,
        head: str = "a" * 40,
        created: str = "c" * 40,
        readback_sha: str | None = None,
        stored: bytes | None = None,
    ) -> None:
        self.head = head
        self.created = created
        self.readback_sha = readback_sha
        self.stored = stored
        self.files: dict[str, bytes] = {}
        self.commits: list[dict] = []

    def dataset_info(self, repo_id, revision=None, token=None):
        assert repo_id == "SZLHOLDINGS/szl-training-receipts"
        if revision is None:
            return _Info(self.head)
        return _Info(self.readback_sha or revision)

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        (op,) = kwargs["operations"]
        body = op["path_or_fileobj"].read()
        self.files[self.created] = self.stored if self.stored is not None else body
        return _Commit(self.created)

    def download(
        self, *, repo_id, repo_type, filename, revision, token, force_download
    ):
        assert (repo_type, filename, force_download) == ("dataset", "README.md", True)
        path = Path(self.tmp) / f"{revision}.md"
        path.write_bytes(self.files[revision])
        return str(path)


def _operation(**kwargs):
    return kwargs


class PublishReadbackTests(unittest.TestCase):
    def _publish(self, hub: _FakeHub) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            hub.tmp = tmp
            return publish(
                api=hub,
                downloader=hub.download,
                operation=_operation,
                source_sha=SOURCE_SHA,
                token="t",
            )

    def test_publish_commits_on_observed_head_and_reads_back_exact_bytes(self) -> None:
        hub = _FakeHub()
        result = self._publish(hub)
        self.assertEqual(result["status"], "PUBLISHED_AND_READ_BACK")
        self.assertEqual(result["hf_revision"], "c" * 40)
        self.assertEqual(result["previous_hf_revision"], "a" * 40)
        self.assertTrue(result["readback"]["matches_rendered_card"])
        (commit,) = hub.commits
        self.assertEqual(commit["parent_commit"], "a" * 40)
        self.assertEqual(
            [op["path_in_repo"] for op in commit["operations"]], ["README.md"]
        )

    def test_publish_fails_closed_when_hub_bytes_differ(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "differs from the rendered card"):
            self._publish(_FakeHub(stored=b"tampered"))

    def test_publish_fails_closed_when_revision_readback_differs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not match the created commit"):
            self._publish(_FakeHub(readback_sha="d" * 40))

    def test_publish_fails_closed_without_a_new_exact_revision(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no new revision"):
            self._publish(_FakeHub(created="a" * 40))
        with self.assertRaisesRegex(RuntimeError, "40-hex commit id"):
            self._publish(_FakeHub(created="main"))

    def test_invalid_source_revision_is_rejected_before_any_hub_call(self) -> None:
        hub = _FakeHub()
        with self.assertRaises(ValueError):
            publish(
                api=hub,
                downloader=hub.download,
                operation=_operation,
                source_sha="main",
                token="t",
            )
        self.assertEqual(hub.commits, [])


class WorkflowLockTests(unittest.TestCase):
    def test_card_writer_holds_the_per_asset_lock(self) -> None:
        text = (
            ROOT / ".github" / "workflows" / "hf-training-receipts-card.yml"
        ).read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r"\n    concurrency:\n      group: hf-write/dataset/SZLHOLDINGS/szl-training-receipts\n"
            r"      cancel-in-progress: false\n",
        )
        self.assertNotIn("event_name", text)


if __name__ == "__main__":
    unittest.main()
