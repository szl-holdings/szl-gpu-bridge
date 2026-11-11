"""Exercise the health watch's real issue writer with an offline GitHub stub."""

import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/main-health-watch.yml"


def run_writer(report, *, existing=False, raw=False):
    workflow = Path(os.environ.get("HEALTH_WORKFLOW", WORKFLOW)).read_text()
    script = workflow.split("          script: |\n", 1)[1]
    script = "\n".join(line[12:] for line in script.splitlines())
    serialized = report if raw else json.dumps(report)
    # Emulate Actions' original expression expansion for the baseline replay.
    script = script.replace("${{ steps.health.outputs.report }}", serialized)
    harness = r"""
import { readFileSync } from 'node:fs';
const input = JSON.parse(readFileSync(0, 'utf8'));
process.env.MAIN_HEALTH_REPORT = input.report;
const calls = [];
const github = { rest: { issues: {
  listForRepo: async () => ({ data: input.existing ? [{
    number: 153, title: 'Main health watch: red workflows detected'
  }] : [] }),
  create: async (args) => calls.push({ method: 'create', args }),
  update: async (args) => calls.push({ method: 'update', args }),
}}};
const context = {repo: {owner: 'fixture-owner', repo: 'fixture-repo'}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
try {
  await new AsyncFunction('github', 'context', input.script)(github, context);
  process.stdout.write(JSON.stringify({calls}));
} catch (error) {
  process.stdout.write(JSON.stringify({calls, error: error.name}));
}
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        input=json.dumps(
            {"report": serialized, "existing": existing, "script": script}
        ),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


class MainHealthReportTests(unittest.TestCase):
    def assert_report(self, expected, result, method):
        self.assertNotIn("error", result)
        self.assertEqual(len(result["calls"]), 1)
        call = result["calls"][0]
        self.assertEqual(call["method"], method)
        self.assertEqual(call["args"]["owner"], "fixture-owner")
        self.assertEqual(call["args"]["repo"], "fixture-repo")
        body = call["args"]["body"]
        self.assertNotIn("[object Object]", body)
        encoded = body.split("```json\n", 1)[1].split("\n```", 1)[0]
        self.assertEqual(json.loads(encoded), expected)

    def test_create_preserves_failure_and_unreachable_evidence(self):
        report = {
            "szl-gpu-bridge": ["GPU bridge contract CI: failure"],
            "platform": ["UNREACHABLE(HTTPError)"],
        }
        result = run_writer(report)
        self.assert_report(report, result, "create")
        self.assertEqual(
            result["calls"][0]["args"]["labels"], ["main-health", "automated"]
        )

    def test_update_preserves_escaped_text(self):
        report = {"fixture": ['quoted "name"; newline\nnext; slash\\; λ']}
        result = run_writer(report, existing=True)
        self.assert_report(report, result, "update")
        self.assertEqual(result["calls"][0]["args"]["issue_number"], 153)

    def test_invalid_json_cannot_publish_partial_report(self):
        result = run_writer("{invalid", raw=True)
        self.assertEqual(result["calls"], [])
        self.assertEqual(result["error"], "SyntaxError")


if __name__ == "__main__":
    unittest.main()
