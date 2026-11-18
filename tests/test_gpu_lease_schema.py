"""Cross-language JSON schema for SZL's CPU-only simulated GPU lease API."""

import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema/gpu-lease-replay.v1.json"
FIXTURE_PATH = ROOT / "docs/fixtures/gpu-lease-synthetic.json"


def fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class GPUReplaySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_current_synthetic_fixture_passes_all_specified_fields(self):
        self.validator.validate(fixture())

    def test_schema_cannot_be_coerced_to_production_or_forged_source(self):
        for field, value in (
            ("repository", "Multi-V-VM/GPUOS"),
            ("upstream_revision", "a" * 40),
            ("evidence_kind", "VERIFIED_PROVIDER"),
            ("revision", "malformed"),
        ):
            with self.subTest(field=field):
                data = fixture()
                data["source"][field] = value
                self.assertFalse(self.validator.is_valid(data))

    def test_tail_coverage_is_a_hard_minimum_not_one_convenient_batch(self):
        data = fixture()
        data["tail"]["samples"] = data["tail"]["samples"][:299]
        self.assertFalse(self.validator.is_valid(data))
        data = fixture()
        data["tail"]["samples"][0]["shared_us"] = "PRIVATE_TOKEN"
        self.assertFalse(self.validator.is_valid(data))

    def test_no_extra_privileged_fields_or_future_event_ops(self):
        extra_values = (
            ("top", lambda x: x.update({"execute_gpu": True})),
            ("source", lambda x: x["source"].update({"token": "PRIVATE_TOKEN"})),
            ("event", lambda x: x["events"][0].update({"cmd": "rm -rf /"})),
            ("url", lambda x: x["events"].append({"op": "launch_kernel"})),
        )
        for label, mutation in extra_values:
            with self.subTest(label=label):
                data = fixture()
                mutation(data)
                self.assertFalse(self.validator.is_valid(data))

    def test_ack_cannot_disguise_resource_return_as_client_claim(self):
        data = fixture()
        ack = next(row for row in data["events"] if row["op"] == "ack")
        ack["cta_returned"] = True
        self.assertFalse(self.validator.is_valid(data))

    def test_boolean_or_nonfinite_integer_arguments_fail(self):
        for field, bad in (("memory_limit_mib", True), ("memory_limit_mib", -1)):
            with self.subTest(field=field, value=bad):
                data = fixture()
                data[field] = bad
                self.assertFalse(self.validator.is_valid(data))

    def test_every_event_has_an_explicit_bounded_variant(self):
        data = fixture()
        for i, event in enumerate(data["events"]):
            with self.subTest(index=i, operation=event["op"]):
                self.assertTrue(self.validator.is_valid(data))
        data["events"][1]["extent"] = 0
        self.assertFalse(self.validator.is_valid(data))

    def test_fixture_cannot_publish_multiple_code_epochs_while_live(self):
        # JSON schema covers shape. Runtime ledger must enforce temporal safety.
        data = copy.deepcopy(fixture())
        self.assertTrue(self.validator.is_valid(data))
        data["events"].insert(1, {"op": "publish_epoch", "next_epoch": 2})
        self.assertTrue(
            self.validator.is_valid(data),
            "Temporal/hardware conditions live in ledger, not JSON schema",
        )


if __name__ == "__main__":
    unittest.main()
