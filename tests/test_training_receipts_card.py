from __future__ import annotations

import unittest

from cloud.sync_training_receipts_card import render_card, validate_card_metadata


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
                card.replace("do not receive a blanket data-reuse grant", "are reusable")
            )

    def test_card_rejects_mutable_source_reference(self) -> None:
        for value in ("main", "b" * 12, "G" * 40):
            with self.subTest(source_sha=value):
                with self.assertRaises(ValueError):
                    render_card(value)


if __name__ == "__main__":
    unittest.main()
