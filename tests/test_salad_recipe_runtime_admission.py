"""Source-only regression tests for paid Salad recipe admission.

No cloud/network calls, training imports, credentials or provider mutations.
A passing static recipe gate is never a live training-readiness verdict.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "salad" / "recipe_lint.py"
spec = importlib.util.spec_from_file_location("salad_recipe_lint_under_test", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("source-only recipe linter unavailable")
linter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = linter
spec.loader.exec_module(linter)

PINNED = (
    "https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/"
    + "a" * 40
    + "/tools/salad/salad_worker.py"
)
MOVING_MAIN = (
    "https://raw.githubusercontent.com/szl-holdings/"
    "szl-gpu-bridge/main/tools/salad/salad_worker.py"
)
MOVING_REFS = (
    "https://raw.githubusercontent.com/szl-holdings/"
    "szl-gpu-bridge/refs/heads/main/tools/salad/salad_worker.py"
)


def _command(pin_set: str, worker: str) -> str:
    return (
        " ".join(linter.PIN_SETS[pin_set])
        + " PIP_DEFAULT_TIMEOUT=45 PIP_RETRIES=2 "
        + "1.1.1.1 8.8.8.8 HF_HUB_ENABLE_HF_TRANSFER=0 "
        + "hf_hub_download STAGE0 INNER_EXIT "
        + worker
    )


def _fails(command: str) -> set[str]:
    findings = linter.lint(
        "ring admission test", command, "pytorch:reviewed",
        {"HF_TOKEN": "injected-by-owner"}, None,
    )
    return {f["rule"] for f in findings if f["severity"] == "FAIL"}


class RecipeRuntimeAdmissionTests(unittest.TestCase):
    def test_historical_legacy_worker_pair_is_refused(self) -> None:
        fails = _fails(_command("LEGACY", PINNED))
        self.assertIn("R5_KNOWN_FATAL_TRL_WORKER_PAIR", fails)
        self.assertNotIn("R5_PINSET", fails)

    def test_modern_main_fetched_worker_is_refused(self) -> None:
        fails = _fails(_command("MODERN", MOVING_MAIN))
        self.assertIn("R5_MOVING_WORKER_REVISION", fails)

    def test_refs_heads_main_is_refused(self) -> None:
        fails = _fails(_command("MODERN", MOVING_REFS))
        self.assertIn("R5_MOVING_WORKER_REVISION", fails)

    def test_pinned_modern_recipe_is_not_rejected_by_static_checks(self) -> None:
        self.assertFalse(_fails(_command("MODERN", PINNED)))

    def test_legacy_non_worker_use_is_not_newly_prohibited(self) -> None:
        self.assertNotIn(
            "R5_KNOWN_FATAL_TRL_WORKER_PAIR",
            _fails(_command("LEGACY", "run_other_legacy_tool.py")),
        )

    def test_static_pass_cannot_claim_training(self) -> None:
        # A deliberately nonexistent commit digest is enough to demonstrate
        # that linting is only static syntax and not provider/source attestation.
        self.assertFalse(_fails(_command("MODERN", PINNED)))
        self.assertNotIn("saladcloud", sys.modules)


if __name__ == "__main__":
    unittest.main()
