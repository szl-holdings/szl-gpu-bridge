"""Exercise the census workflow's real admission condition without provider calls."""

import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/census-anthropic.yml"
SCRIPT = ROOT / "tools/census/anthropic_leg.py"


def job_block(source, job):
    match = re.search(
        rf"(?ms)^  {re.escape(job)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        source.split("\njobs:\n", 1)[1],
    )
    if match is None:
        raise AssertionError(f"missing job: {job}")
    return match.group(1)


def evaluate_expression(expression, context):
    """Evaluate the actual Boolean/string GitHub expression, not a copied policy.

    Only the syntax used by this workflow is supported. Unknown syntax fails the
    contract instead of executing arbitrary source from a workflow under review.
    GitHub compares strings without regard to case.
    """
    tree = ast.parse(
        "(" + expression.replace("&&", " and ").replace("||", " or ") + ")",
        mode="eval",
    )

    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id == "github":
            return context
        if isinstance(node, ast.Attribute):
            parent = value(node.value)
            return parent.get(node.attr, "") if isinstance(parent, dict) else ""
        if isinstance(node, ast.Name) and node.id in ("true", "false"):
            return node.id == "true"
        if isinstance(node, ast.BoolOp):
            # GitHub's && and || return an operand, including the PR number
            # selected by the real concurrency expression.
            result = ""
            for item in node.values:
                result = value(item)
                if isinstance(node.op, ast.And) and not result:
                    return result
                if isinstance(node.op, ast.Or) and result:
                    return result
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = value(node.left), value(node.comparators[0])
            if isinstance(left, str) and isinstance(right, str):
                left, right = left.casefold(), right.casefold()
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
        raise AssertionError(f"unsupported admission syntax: {ast.dump(node)}")

    return value(tree)


def evaluate_condition(expression, context):
    return bool(evaluate_expression(expression, context))


class CensusWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.contract = job_block(cls.workflow, "contract")
        cls.census = job_block(cls.workflow, "census")
        match = re.search(
            r"(?ms)^    if: >-\n\s+\$\{\{(.*?)\}\}\n",
            cls.census,
        )
        if match is None:
            raise AssertionError("live census has no explicit job admission condition")
        cls.condition = match.group(1)

    def admitted(self, **overrides):
        context = {
            "repository": "szl-holdings/szl-gpu-bridge",
            "ref": "refs/heads/main",
            "ref_protected": True,
            "actor": "reviewed-maintainer",
            "event_name": "push",
        }
        context.update(overrides)
        return evaluate_condition(self.condition, context)

    def test_protected_main_live_events_are_admitted(self):
        for event in ("push", "schedule", "workflow_dispatch"):
            with self.subTest(event=event):
                self.assertTrue(self.admitted(event_name=event))

    def test_dependency_branches_tags_and_pull_refs_are_denied(self):
        for event in ("push", "schedule", "workflow_dispatch"):
            for ref in (
                "refs/heads/dependabot/github_actions/update",
                "refs/heads/feature",
                "refs/heads/main-extra",
                "refs/tags/main",
                "refs/pull/205/merge",
                "",
            ):
                with self.subTest(event=event, ref=ref):
                    self.assertFalse(self.admitted(event_name=event, ref=ref))

    def test_pr_and_unrelated_events_cannot_borrow_main_authority(self):
        for event in (
            "pull_request",
            "pull_request_target",
            "workflow_run",
            "repository_dispatch",
            "merge_group",
            "check_suite",
            "",
        ):
            with self.subTest(event=event):
                self.assertFalse(self.admitted(event_name=event))

    def test_unprotected_refs_forks_and_missing_context_are_denied(self):
        for event in ("push", "schedule", "workflow_dispatch"):
            for overrides in (
                {"ref_protected": False},
                {"ref_protected": ""},
                {"repository": "untrusted/szl-gpu-bridge"},
                {"repository": ""},
                {"actor": ""},
            ):
                with self.subTest(event=event, overrides=overrides):
                    self.assertFalse(self.admitted(event_name=event, **overrides))
        self.assertFalse(evaluate_condition(self.condition, {}))

    def test_dependabot_is_denied_even_after_a_maintainer_rerun(self):
        for event in ("push", "schedule", "workflow_dispatch"):
            for actor in ("dependabot[bot]", "Dependabot[bot]"):
                with self.subTest(event=event, actor=actor):
                    self.assertFalse(
                        self.admitted(
                            event_name=event,
                            actor=actor,
                            triggering_actor="reviewed-maintainer",
                        )
                    )

    def test_pr_contract_is_offline_and_precedes_live_execution(self):
        self.assertNotIn("secrets.", self.contract)
        self.assertNotIn("ANTHROPIC_API_KEY", self.contract)
        self.assertIn("persist-credentials: false", self.contract)
        self.assertIn(
            "python3 -m unittest discover -s tests -p "
            "'test_census_workflow_contract.py' -v",
            self.contract,
        )
        self.assertIn("    needs: contract\n", self.census)
        self.assertNotIn("continue-on-error", self.census)
        self.assertIn("persist-credentials: false", self.census)
        self.assertIn("secrets.ANTHROPIC_API_KEY", self.census)
        self.assertIn("secrets.ANTHROPIC_WORKSPACE_ID", self.census)

    def test_main_push_filter_and_pr_validation_are_both_present(self):
        trigger_source = self.workflow.split("\npermissions:\n", 1)[0]
        push = re.search(
            r"(?ms)^  push:\n(.*?)(?=^  [A-Za-z0-9_-]+:|\Z)", trigger_source
        ).group(1)
        pull_request = re.search(
            r"(?ms)^  pull_request:\n(.*?)(?=^  [A-Za-z0-9_-]+:|\Z)",
            trigger_source,
        ).group(1)
        self.assertRegex(push, r"(?m)^    branches: \[main\]$")
        for trigger in (push, pull_request):
            self.assertIn('"tools/census/**"', trigger)
            self.assertIn('".github/workflows/census-anthropic.yml"', trigger)
            self.assertIn('"tests/test_census_workflow_contract.py"', trigger)
        self.assertNotIn("pull_request_target:", trigger_source)

    def test_pr_and_live_concurrency_groups_are_disjoint_and_stable(self):
        block = self.workflow.split("\nconcurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        template = re.search(r"(?m)^  group: (.+)$", block).group(1)
        parts = re.fullmatch(r"(.*?)\$\{\{(.*?)\}\}(.*)", template)
        self.assertIsNotNone(
            parts, "a constant group lets PRs replace live pending runs"
        )

        def group(event_name, ref="refs/heads/main", pr_number=None, action=""):
            event = {"action": action}
            if pr_number is not None:
                event["pull_request"] = {"number": pr_number}
            selected = evaluate_expression(
                parts.group(2),
                {"event_name": event_name, "ref": ref, "event": event},
            )
            return parts.group(1) + str(selected) + parts.group(3)

        live_groups = {
            group(event) for event in ("push", "schedule", "workflow_dispatch")
        }
        self.assertEqual(len(live_groups), 1, "live runs must share one serial queue")
        pr_group = group("pull_request", "refs/pull/205/merge", 205, "opened")
        self.assertNotIn(pr_group, live_groups)
        self.assertEqual(
            pr_group,
            group("pull_request", "refs/pull/205/merge", 205, "synchronize"),
            "updates to one PR must stay in that PR's queue",
        )
        self.assertNotEqual(
            pr_group,
            group("pull_request", "refs/pull/206/merge", 206, "opened"),
        )
        self.assertNotIn(group("workflow_dispatch", "refs/heads/feature"), live_groups)
        self.assertIn("  cancel-in-progress: false", block)

    def test_missing_key_fails_before_transport_or_receipt_creation(self):
        spec = importlib.util.spec_from_file_location("census_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(sys, "argv", [str(SCRIPT), "--out", str(receipt)]),
                mock.patch.object(
                    module.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("provider transport is forbidden"),
                ) as transport,
                contextlib.redirect_stderr(stderr),
            ):
                result = module.main()
            self.assertEqual(result, 2)
            transport.assert_not_called()
            self.assertFalse(receipt.exists())
            self.assertIn("FAIL-CLOSED", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
