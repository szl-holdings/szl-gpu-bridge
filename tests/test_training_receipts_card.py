from __future__ import annotations

import unittest

from cloud.sync_training_receipts_card import render_card


SOURCE_SHA = "b" * 40


class TrainingReceiptsCardTests(unittest.TestCase):
    def test_card_is_private_evidence_not_training_data(self) -> None:
        card = render_card(SOURCE_SHA).decode("utf-8")
        self.assertIn(f"Source revision: `{SOURCE_SHA}`", card)
        self.assertIn("not a training corpus", card)
        self.assertIn("missing evidence", card)
        self.assertIn("license: other", card)
        license_name = "szl-governed-operational-evidence-no-blanket-reuse-grant"
        self.assertIn(f"license_name: {license_name}", card)
        self.assertTrue(
            all(
                character in "abcdefghijklmnopqrstuvwxyz0123456789-."
                for character in license_name
            )
        )
        self.assertIn(
            "license_link: https://github.com/szl-holdings/szl-gpu-bridge/blob/main/"
            "hf/szl-training-receipts/README.md#license-and-data-handling",
            card,
        )
        self.assertNotIn("__SOURCE_REVISION__", card)

    def test_card_rejects_mutable_source_reference(self) -> None:
        for value in ("main", "b" * 12, "G" * 40):
            with self.subTest(source_sha=value):
                with self.assertRaises(ValueError):
                    render_card(value)


if __name__ == "__main__":
    unittest.main()
