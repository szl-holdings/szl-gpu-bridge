"""Card generation must preserve existing documents and unknown evidence."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_all_cards", ROOT / "tools/hf_cards/build_all.py"
)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class CardTests(unittest.TestCase):
    def card(self, kind="model", **metadata):
        return build.generate(
            kind, {"id": "SZLHOLDINGS/szl-nemo", "cardData": metadata}
        )

    def metadata(self, card):
        header = card.split("---")[1].strip()
        return (
            {
                json.loads(key): json.loads(value)
                for key, value in (line.split(": ", 1) for line in header.splitlines())
            }
            if header
            else {}
        )

    def test_unknown_license_and_framework_are_not_fabricated(self):
        card = self.card()
        self.assertNotIn("license", self.metadata(card))
        self.assertNotIn("library_name", self.metadata(card))
        self.assertNotIn("Apache", card)
        self.assertNotIn("frontier-class", card)
        self.assertNotIn("served from", card)
        self.assertIn("UNVERIFIED", card)
        self.assertNotIn("links verified at build time", card)

    def test_license_and_upstream_metadata_survive(self):
        data = {
            "license": "other",
            "license_name": "upstream-terms",
            "license_link": "https://example.test/license",
            "base_model": ["owner/base"],
            "library_name": "peft",
        }
        self.assertEqual(self.metadata(self.card(**data)), data)

    def test_space_sdk_is_preserved_and_required(self):
        self.assertEqual(
            self.metadata(self.card("space", sdk="gradio", sdk_version="6.0.0"))["sdk"],
            "gradio",
        )
        with self.assertRaises(ValueError):
            self.card("space")

    def test_yaml_special_characters_cannot_change_metadata(self):
        value = "x\nlicense: apache-2.0"
        self.assertEqual(self.metadata(self.card(title=value)), {"title": value})

    def test_plain_readme_without_frontmatter_is_not_missing(self):
        with (
            patch.object(
                build, "list_assets", return_value=[("model", {"id": "org/repo"})]
            ),
            patch.object(
                build, "get", return_value={"siblings": [{"rfilename": "README.md"}]}
            ),
        ):
            self.assertEqual(build.no_readme(), [])

    def test_missing_file_inventory_is_unknown(self):
        for payload in [{}, {"siblings": None}, {"siblings": [{}]}]:
            with (
                patch.object(
                    build, "list_assets", return_value=[("model", {"id": "org/repo"})]
                ),
                patch.object(build, "get", return_value=payload),
            ):
                self.assertEqual(build.no_readme(), [])

    def test_verified_absence_preserves_returned_metadata(self):
        payload = {"id": "org/repo", "siblings": [], "cardData": {"license": "mit"}}
        with (
            patch.object(
                build, "list_assets", return_value=[("model", {"id": "org/repo"})]
            ),
            patch.object(build, "get", return_value=payload),
        ):
            self.assertEqual(build.no_readme(), [("model", payload)])

    def api(self, siblings):
        api = Mock()
        api.repo_info.return_value = types.SimpleNamespace(
            sha="a" * 40, siblings=siblings
        )
        return api

    def test_upload_rechecks_readme_and_refuses_overwrite(self):
        api = self.api([types.SimpleNamespace(rfilename="ReadMe.md")])
        with patch.dict(
            sys.modules,
            {"huggingface_hub": types.SimpleNamespace(HfApi=lambda **kw: api)},
        ):
            with self.assertRaises(RuntimeError):
                build.upload("model", "org/repo", "draft")
        api.upload_file.assert_not_called()

    def test_upload_uses_parent_commit_to_reject_races(self):
        api = self.api([])
        with patch.dict(
            sys.modules,
            {"huggingface_hub": types.SimpleNamespace(HfApi=lambda **kw: api)},
        ):
            build.upload("dataset", "org/repo", "draft")
        self.assertEqual(api.upload_file.call_args.kwargs["parent_commit"], "a" * 40)
        self.assertEqual(api.upload_file.call_args.kwargs["repo_type"], "dataset")


if __name__ == "__main__":
    unittest.main()
