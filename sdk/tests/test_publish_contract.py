"""Release publisher must preserve exact PEP 740 provenance source contract.

These are deterministic source-only tests. A passing fixture neither publishes
a package nor proves the contents of existing PyPI releases.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "sdk" / "tools" / "check_publish_contract.py"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"

spec = importlib.util.spec_from_file_location("szl_publish_contract", CHECK)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class PublisherSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release = RELEASE.read_text(encoding="utf-8")
        cls.ci = CI.read_text(encoding="utf-8")

    def assert_rejected(self, source: str) -> None:
        with self.assertRaises(contract.ContractHold):
            contract.inspect(source)

    def test_main_workflow_declares_source_policy_but_not_release_proof(self) -> None:
        result = contract.inspect(self.release)
        self.assertEqual(result["status"], "SOURCE_PUBLISH_CONTRACT_PASS")
        self.assertEqual(result["environment"], "release")
        self.assertTrue(result["pep740_attestations_declared"])
        self.assertFalse(result["post_publication_signatures_verified"])
        self.assertFalse(result["published_distribution_identity_verified"])
        self.assertFalse(result["latest_version_publish_authorized"])
        self.assertRegex(result["official_action_commit"], r"^[0-9a-f]{40}$")

    def test_ci_and_tag_build_both_enforce_the_contract(self) -> None:
        self.assertIn("run: python sdk/tools/check_publish_contract.py", self.release)
        self.assertIn("run: python sdk/tools/check_publish_contract.py", self.ci)
        self.assertIn("python -m unittest discover -s", self.ci)

    def test_historical_uv_trusted_publish_without_attestations_fails(self) -> None:
        # Historical v0.1.7 tag source used this direct uv command. Reproducing
        # the source shape must never pass a future PEP 740 admission check.
        body = re.sub(
            r"(?m)^        uses: pypa/gh-action-pypi-publish@[0-9a-f]{40}.*$",
            "        run: uv publish --trusted-publishing always dist/*.whl dist/*.tar.gz",
            self.release,
        ).replace("          attestations: true\n", "")
        self.assertNotEqual(body, self.release)
        self.assert_rejected(body)

    def test_missing_signature_attestations_fails(self) -> None:
        self.assert_rejected(
            self.release.replace("          attestations: true\n", "", 1)
        )

    def test_malformed_publisher_action_pin_fails(self) -> None:
        action = re.search(r"pypa/gh-action-pypi-publish@[0-9a-f]{40}", self.release)
        self.assertIsNotNone(action)
        self.assert_rejected(
            self.release.replace(action.group(0), "pypa/gh-action-pypi-publish@main")
        )

    def test_missing_signed_deployment_environment_fails(self) -> None:
        self.assert_rejected(
            self.release.replace("    environment: release", "    environment: staging")
        )

    def test_oidc_permission_missing_fails(self) -> None:
        self.assert_rejected(
            self.release.replace("      id-token: write", "      id-token: none")
        )

    def test_tag_gating_removed_fails(self) -> None:
        self.assert_rejected(
            self.release.replace(
                "if: startsWith(github.ref, 'refs/tags/szl-v')",
                "if: success()",
            )
        )

    def test_main_ancestry_gate_removed_fails(self) -> None:
        self.assert_rejected(
            self.release.replace(
                'run: bash sdk/tools/check_release_ref.sh "$GITHUB_REF"',
                'run: echo "skipping tag ancestry"',
            )
        )

    def test_skip_existing_or_api_token_backdoor_fails(self) -> None:
        for token in ("skip-existing", "TWINE_PASSWORD", "PYPI_API_TOKEN"):
            with self.subTest(token=token):
                self.assert_rejected(
                    self.release.replace(
                        "          attestations: true",
                        "          attestations: true\n          " + token + ": true",
                    )
                )

    def test_package_dir_mismatch_fails(self) -> None:
        self.assert_rejected(
            self.release.replace(
                "          packages-dir: dist/",
                "          packages-dir: external-dist/",
            )
        )

    def test_removed_source_check_itself_fails(self) -> None:
        self.assert_rejected(
            self.release.replace(
                "run: python sdk/tools/check_publish_contract.py",
                'run: echo "skip publisher gate"',
            )
        )

    def test_comment_cannot_fake_live_attestations_policy(self) -> None:
        mutated = self.release.replace(
            "          attestations: true",
            "          # attestations: true",
        )
        self.assert_rejected(mutated)

    def test_duplicate_publisher_ambiguity_fails(self) -> None:
        action = re.search(
            r"(?m)^        uses: pypa/gh-action-pypi-publish@[0-9a-f]{40}.*$",
            self.release,
        )
        self.assertIsNotNone(action)
        self.assert_rejected(
            self.release.replace(
                action.group(0), action.group(0) + "\n" + action.group(0)
            )
        )

    def test_cli_rejects_historical_fixture_without_network(self) -> None:
        legacy = self.release.replace(
            "          attestations: true",
            "          # no uploaded attestations at this tag",
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "historical-release.yml"
            p.write_text(legacy, encoding="utf-8")
            r = subprocess.run(
                [sys.executable, str(CHECK), "--workflow", str(p)],
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
            self.assertEqual(r.returncode, 2)
            self.assertIn('"status": "HOLD"', r.stdout)


if __name__ == "__main__":
    unittest.main()
