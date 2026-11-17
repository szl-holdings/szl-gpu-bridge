"""Exercise training serialization without importing GPU or Hub dependencies."""

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock


WORKER = Path(__file__).resolve().parents[1] / "tools" / "salad" / "salad_worker.py"
HELPER_NAMES = {
    "training_format_contract",
    "format_training_example",
    "verify_resume_training_format",
    "save_training_format",
}
TEMPLATE = "synthetic template: {{ messages }}\n\u03bb"
BASE = "fixture/synthetic-base"
REVISION = "a" * 40


def load_helpers():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES
    ]
    if {node.name for node in functions} != HELPER_NAMES:
        raise AssertionError("worker training-format helpers are missing")
    namespace = {"hashlib": hashlib, "json": json, "os": os}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(WORKER), "exec"),
        namespace,
    )
    return namespace


HELPERS = load_helpers()


class TrainingFormat(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Mock(spec=["get_chat_template", "apply_chat_template"])
        self.tokenizer.get_chat_template.return_value = TEMPLATE
        self.tokenized = {"input_ids": [101, 202, 303], "attention_mask": [1, 1, 1]}
        self.tokenizer.apply_chat_template.return_value = self.tokenized
        self.format_example = HELPERS["format_training_example"]
        self.make_contract = HELPERS["training_format_contract"]
        self.verify_resume = HELPERS["verify_resume_training_format"]
        self.save_contract = HELPERS["save_training_format"]

    def chat_contract(self):
        return self.make_contract(
            self.tokenizer, ["prompt", "completion", "source"], "chat", BASE, REVISION
        )

    def test_verbatim_rejects_actual_by_plus_by_join(self):
        example = {
            "prompt": "Prove this: theorem synthetic (n : Nat) : n = n := by",
            "completion": "by rfl",
        }
        with self.assertRaisesRegex(
            ValueError, "boundary requires whitespace"
        ) as error:
            self.format_example(example, self.tokenizer, "verbatim")
        self.assertNotIn(example["prompt"], str(error.exception))
        self.assertNotIn(example["completion"], str(error.exception))
        self.tokenizer.apply_chat_template.assert_not_called()

    def test_verbatim_preserves_leading_and_trailing_whitespace_bytes(self):
        pairs = [
            ("\t synthetic instruction := by\r\n", "  exact h\n\n"),
            ("  synthetic instruction := by", "\n\t exact h\r\n"),
            ("\n\u03bb instruction \t", "\r\n  \u03bb completion  "),
        ]
        for prompt, completion in pairs:
            with self.subTest(prompt=repr(prompt), completion=repr(completion)):
                result = self.format_example(
                    {"prompt": prompt, "completion": completion},
                    self.tokenizer,
                    "verbatim",
                )
                self.assertEqual(
                    result["text"].encode("utf-8"),
                    prompt.encode("utf-8") + completion.encode("utf-8"),
                )
        self.tokenizer.apply_chat_template.assert_not_called()

    def test_pair_fields_must_be_nonempty_strings(self):
        for mode in ("verbatim", "chat"):
            for field in ("prompt", "completion"):
                for invalid in (None, 17, [], {}, "", " \r\n\t"):
                    with self.subTest(mode=mode, field=field, invalid=invalid):
                        example = {
                            "prompt": "synthetic prompt\n",
                            "completion": "answer",
                        }
                        example[field] = invalid
                        with self.assertRaisesRegex(ValueError, "nonempty strings"):
                            self.format_example(example, self.tokenizer, mode)
        self.tokenizer.apply_chat_template.assert_not_called()

    def test_unknown_pair_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown.*serialization mode"):
            self.format_example(
                {"prompt": "prompt\n", "completion": "answer"},
                self.tokenizer,
                "invented-mode",
            )

    def test_chat_preserves_exact_turns_and_disables_generation_prompt(self):
        prompt = "  Synthetic theorem := by\r\n\t"
        completion = "\nby\n  exact synthetic_hypothesis  \r\n"
        result = self.format_example(
            {"prompt": prompt, "completion": completion, "source": "fixture/source"},
            self.tokenizer,
            "chat",
        )
        self.assertEqual(result, self.tokenized)
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )

    def test_chat_contract_records_template_digest_and_immutable_tokenizer_pin(self):
        self.assertEqual(
            self.chat_contract(),
            {
                "schema": "szl.training-format/v1",
                "formatter_version": "v1",
                "mode": "prompt_completion_chat",
                "chat_template_sha256": hashlib.sha256(
                    TEMPLATE.encode("utf-8")
                ).hexdigest(),
                "tokenizer_base": BASE,
                "tokenizer_revision": REVISION,
            },
        )

    def test_contract_contains_no_raw_template_or_example_fields(self):
        contract = self.chat_contract()
        self.assertEqual(
            set(contract),
            {
                "schema",
                "formatter_version",
                "mode",
                "chat_template_sha256",
                "tokenizer_base",
                "tokenizer_revision",
            },
        )
        self.assertNotIn(TEMPLATE, json.dumps(contract))
        for field in ("prompt", "completion", "messages", "text", "source"):
            self.assertNotIn(field, contract)

    def test_chat_contract_rejects_missing_or_blank_templates(self):
        for columns in (["prompt", "completion"], ["messages"]):
            for template in (None, "", " \r\n\t", {}):
                with self.subTest(columns=columns, template=template):
                    self.tokenizer.get_chat_template.return_value = template
                    with self.assertRaisesRegex(
                        ValueError, "explicit tokenizer template"
                    ):
                        self.make_contract(
                            self.tokenizer, columns, "chat", BASE, REVISION
                        )

    def test_missing_tokenizer_template_error_propagates(self):
        self.tokenizer.get_chat_template.side_effect = ValueError(
            "no template configured"
        )
        with self.assertRaisesRegex(ValueError, "no template configured"):
            self.chat_contract()

    def test_nonchat_contracts_do_not_require_a_template(self):
        self.tokenizer.get_chat_template.side_effect = AssertionError(
            "must not be called"
        )
        for columns, expected_mode in (
            (["text"], "text"),
            (["prompt", "completion"], "prompt_completion_verbatim"),
            (["arbitrary"], "json"),
        ):
            with self.subTest(columns=columns):
                contract = self.make_contract(
                    self.tokenizer, columns, "verbatim", BASE, ""
                )
                self.assertEqual(contract["mode"], expected_mode)
                self.assertIsNone(contract["chat_template_sha256"])
                self.assertIsNone(contract["tokenizer_revision"])

    def test_existing_text_behavior_and_precedence_are_unchanged(self):
        text = "\r\n  synthetic \u03bb text\t\n"
        example = {"text": text, "messages": [], "prompt": "", "completion": ""}
        self.assertEqual(
            self.format_example(example, self.tokenizer, "chat"), {"text": text}
        )
        self.assertEqual(
            self.make_contract(self.tokenizer, list(example), "chat", BASE, REVISION)[
                "mode"
            ],
            "text",
        )
        self.tokenizer.apply_chat_template.assert_not_called()
        self.tokenizer.get_chat_template.assert_not_called()

    def test_messages_preserve_turns_and_precedence_with_single_tokenization(self):
        messages = [
            {"role": "system", "content": "synthetic system"},
            {"role": "user", "content": " synthetic question "},
            {"role": "assistant", "content": "\nsynthetic answer\n"},
        ]
        example = {"messages": messages, "prompt": "", "completion": ""}
        self.assertEqual(
            self.format_example(example, self.tokenizer, "verbatim"),
            self.tokenized,
        )
        self.tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True
        )
        contract = self.make_contract(
            self.tokenizer, list(example), "verbatim", BASE, REVISION
        )
        self.assertEqual(contract["mode"], "messages_chat")
        self.assertEqual(
            contract["chat_template_sha256"],
            hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest(),
        )

    def test_existing_json_fallback_is_unchanged(self):
        example = {"synthetic_metadata": [1, "\u03bb"], "enabled": True}
        self.assertEqual(
            self.format_example(example, self.tokenizer, "chat"),
            {"text": json.dumps(example)},
        )
        self.tokenizer.apply_chat_template.assert_not_called()

    def test_genesis_chat_run_does_not_require_resume_metadata(self):
        for resume in (None, ""):
            with self.subTest(resume=resume):
                self.assertIsNone(self.verify_resume(resume, self.chat_contract()))

    def test_both_chat_formats_are_blocked_for_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as resume:
            for columns in (["prompt", "completion"], ["messages"]):
                with self.subTest(columns=columns):
                    contract = self.make_contract(
                        self.tokenizer, columns, "chat", BASE, REVISION
                    )
                    with self.assertRaisesRegex(
                        ValueError, "legacy checkpoint lacks.*contract"
                    ):
                        self.verify_resume(resume, contract)
            self.assertFalse((Path(resume) / "training_format.json").exists())

    def test_unchanged_legacy_serializers_can_resume_without_metadata(self):
        with tempfile.TemporaryDirectory() as resume:
            for columns in (
                ["text"],
                ["prompt", "completion"],
                ["arbitrary"],
            ):
                with self.subTest(columns=columns):
                    contract = self.make_contract(
                        self.tokenizer, columns, "verbatim", BASE, REVISION
                    )
                    self.assertIsNone(self.verify_resume(resume, contract))

    def test_identical_saved_contract_allows_resume(self):
        contract = self.chat_contract()
        with tempfile.TemporaryDirectory() as resume:
            self.save_contract(resume, contract)
            self.assertIsNone(self.verify_resume(resume, contract))

    def test_changed_contract_rejects_resume(self):
        contract = self.chat_contract()
        changes = {
            "schema": "szl.training-format/v2",
            "formatter_version": "v2",
            "mode": "prompt_completion_verbatim",
            "chat_template_sha256": "b" * 64,
            "tokenizer_base": "fixture/different-base",
            "tokenizer_revision": "b" * 40,
        }
        with tempfile.TemporaryDirectory() as resume:
            self.save_contract(resume, contract)
            for field, changed_value in changes.items():
                with self.subTest(field=field):
                    changed = dict(contract, **{field: changed_value})
                    with self.assertRaisesRegex(ValueError, "format differs"):
                        self.verify_resume(resume, changed)

    def test_saved_contract_roundtrips_in_a_new_nested_directory(self):
        contract = self.chat_contract()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "new" / "checkpoint-50"
            self.save_contract(directory, contract)
            data = (directory / "training_format.json").read_bytes()
            self.assertTrue(data.endswith(b"\n"))
            self.assertEqual(json.loads(data), contract)
            self.assertIsNone(self.verify_resume(directory, contract))
            self.assertNotIn(TEMPLATE.encode("utf-8"), data)

    def test_malformed_or_wrong_shape_resume_metadata_fails_closed(self):
        contract = self.chat_contract()
        malformed = [
            "{truncated",
            "",
            "null",
            "[]",
            '"not an object"',
            "{}",
            json.dumps(
                {key: value for key, value in contract.items() if key != "mode"}
            ),
            json.dumps(dict(contract, unexpected="unrecognized metadata")),
        ]
        with tempfile.TemporaryDirectory() as resume:
            path = Path(resume) / "training_format.json"
            for metadata in malformed:
                with self.subTest(metadata=metadata):
                    path.write_text(metadata, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.verify_resume(resume, contract)


if __name__ == "__main__":
    unittest.main()
