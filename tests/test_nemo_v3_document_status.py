"""Fail-closed, offline consistency guard for Nemotron attempt-17 documentation."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def verify_attempt_17_docs(runbook: str, readme: str, execution: str) -> None:
    """Reject false terminal/dispatch claims without requiring matching prose."""
    rows = [
        line
        for line in runbook.splitlines()
        if line.startswith("| Nemotron v3 chain |")
    ]
    if len(rows) != 1:
        raise AssertionError("expected exactly one Nemotron v3 runbook row")
    row = rows[0]

    for marker in (
        "EXPIRED_AWAITING_ENGINE_SIGNATURE",
        "non-terminal",
        "NOT SETTLED",
        "attempt 17",
        "no terminal receipt",
        "no automated retry",
        "dispatch",
        "quarantine override",
    ):
        if marker not in row:
            raise AssertionError(f"missing attempt-17 runbook constraint: {marker}")
    if "**SETTLED" in row or "attempt-17 terminal receipt" in row:
        raise AssertionError("attempt 17 cannot be promoted to a terminal receipt")

    for marker in (
        "No attempt-17 queue envelope exists",
        "EXPIRED_AWAITING_ENGINE_SIGNATURE",
        "non-terminal",
        "Nothing was dispatched",
    ):
        if marker not in readme:
            raise AssertionError(
                f"README contradicts/misses attempt-17 state: {marker}"
            )

    if (
        "| `EXPIRED_AWAITING_ENGINE_SIGNATURE` |" not in execution
        or "expired without a queue envelope" not in execution
    ):
        raise AssertionError("execution status contract lacks expired unsigned state")


class NemoV3DocumentStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runbook = (ROOT / "docs" / "NONSTOP_TRAINING_RUNBOOK.md").read_text(
            encoding="utf-8"
        )
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.execution = (ROOT / "docs" / "NEMO_V3_EXECUTION.md").read_text(
            encoding="utf-8"
        )

    def test_canonical_attempt_17_state_is_consistent(self) -> None:
        verify_attempt_17_docs(self.runbook, self.readme, self.execution)

    def test_rejects_false_settled_runbook_row(self) -> None:
        tampered = self.runbook.replace(
            "**EXPIRED_AWAITING_ENGINE_SIGNATURE**",
            "**SETTLED — attempt-17 terminal receipt**",
        )
        self.assertNotEqual(tampered, self.runbook)
        with self.assertRaises(AssertionError):
            verify_attempt_17_docs(tampered, self.readme, self.execution)

    def test_rejects_auto_retry_promotion(self) -> None:
        tampered = self.runbook.replace("no automated retry", "automated retry")
        self.assertNotEqual(tampered, self.runbook)
        with self.assertRaises(AssertionError):
            verify_attempt_17_docs(tampered, self.readme, self.execution)


if __name__ == "__main__":
    unittest.main()
