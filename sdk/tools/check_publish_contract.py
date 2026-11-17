"""Fail-closed source contract for future SZL SDK Trusted Publishing.

This is a static workflow *configuration* check. It does not contact PyPI,
validate signatures, claim a published artifact, authorize a tag or change CI.
Separate PEP 740 cryptographic readback is required after publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
SCHEMA = "szl.sdk-publish-source-contract/v1"


class ContractHold(ValueError):
    """Fixed evidence code; no workflow/environment secrets are emitted."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ContractHold(code)


def job(source: str, name: str) -> str:
    """Isolate one expected indentation-scoped YAML job, fail on ambiguity."""
    marker = re.compile(r"(?m)^  " + re.escape(name) + r":\s*$")
    matches = list(marker.finditer(source))
    require(len(matches) == 1, "JOB_NOT_UNIQUE_" + name.upper())
    start = matches[0].end()
    other = re.search(r"(?m)^  [a-zA-Z][a-zA-Z0-9_-]*:\s*$", source[start:])
    end = start + other.start() if other else len(source)
    return source[start:end]


def inspect(source: str) -> dict[str, object]:
    if not isinstance(source, str) or len(source) > 100_000:
        raise ContractHold("WORKFLOW_NOT_BOUNDED_TEXT")

    # Ignore comment-only lines; a comment must never satisfy an admission gate.
    content = (
        "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        + "\n"
    )
    require(
        'tags: ["szl-v*"]' in content,
        "PUBLISH_TAG_TRIGGER_UNOBSERVED",
    )
    require("workflow_dispatch" in content, "BRANCH_VALIDATION_ENTRY_UNOBSERVED")

    build = job(content, "build")
    publish = job(content, "publish")

    for token, code in (
        ("bash sdk/tools/check_release_ref.sh", "TAG_MAIN_ANCESTRY_GATE_MISSING"),
        ("sdk/tools/check_publish_contract.py", "PUBLISH_SOURCE_GATE_NOT_CALLED"),
        ("python sdk/tools/verify_release.py", "ARTIFACT_CONTRACT_NOT_CALLED"),
        ("python -m unittest discover -s", "SDK_TESTS_MISSING"),
        ("sdk-source", "SDIST_REBUILD_GATE_MISSING"),
    ):
        require(token in build, code)

    require(
        "needs: build" in publish,
        "BUILD_DEPENDENCY_MISSING",
    )
    require(
        "if: startsWith(github.ref, 'refs/tags/szl-v')" in publish,
        "PUBLISH_TAG_CONDITION_MISSING",
    )
    require(
        re.search(r"(?m)^    environment: release\s*$", publish) is not None,
        "SIGNED_RELEASE_ENVIRONMENT_MISSING",
    )
    require(
        re.search(r"(?m)^      id-token: write\s*$", publish) is not None,
        "OIDC_PERMISSION_MISSING",
    )
    require(
        re.search(r"(?m)^      contents: read\s*$", publish) is not None,
        "PUBLISH_JOB_CONTENTS_SCOPE_MISSING",
    )

    # This source lane deliberately permits ONE official PyPA publisher action.
    # No direct token/manual uv/twine publishing or skip-existing fallback.
    for prohibited in (
        "uv publish",
        "twine upload",
        "PYPI_API_TOKEN",
        "TWINE_PASSWORD",
        "skip-existing",
        "skip_existing",
        "password:",
        "verify-metadata: false",
    ):
        require(prohibited not in publish, "UNAPPROVED_PUBLISH_PATH")

    action = re.compile(
        r"(?m)^        uses: pypa/gh-action-pypi-publish@([a-f0-9]{40})"
        r"(?:\s+#.*)?$"
    )
    matches = list(action.finditer(publish))
    require(len(matches) == 1, "OFFICIAL_PUBLISH_ACTION_NOT_PINNED_ONCE")
    start = matches[0].end()
    next_step = re.search(r"(?m)^      - ", publish[start:])
    tail = publish[start : start + next_step.start()] if next_step else publish[start:]
    require(
        re.search(r"(?m)^        with:\s*$", tail) is not None,
        "PUBLISH_ACTION_WITH_BLOCK_MISSING",
    )
    require(
        re.search(r"(?m)^          packages-dir: dist/\s*$", tail) is not None,
        "TESTED_BUILD_OUTPUT_NOT_USED",
    )
    require(
        re.search(r"(?m)^          attestations: true\s*$", tail) is not None,
        "PEP740_ATTESTATIONS_NOT_ENABLED",
    )
    require(
        len(re.findall(r"(?m)^          attestations:\s*", publish)) == 1,
        "ATTESTATIONS_POLICY_AMBIGUOUS",
    )
    require(
        not re.search(r"(?m)^        run:\s*", publish),
        "UNREVIEWED_SHELL_PUBLISH_STEP",
    )
    return {
        "schema": SCHEMA,
        "status": "SOURCE_PUBLISH_CONTRACT_PASS",
        "publisher": "GitHub",
        "repository": "szl-holdings/szl-gpu-bridge",
        "workflow": "release.yml",
        "environment": "release",
        "official_action_commit": matches[0].group(1),
        "pep740_attestations_declared": True,
        "release_tag_pattern": "szl-v*",
        "post_publication_signatures_verified": False,
        "published_distribution_identity_verified": False,
        "latest_version_publish_authorized": False,
        "cloud_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=RELEASE)
    args = parser.parse_args()
    try:
        with args.workflow.open("rb") as stream:
            data = stream.read(100_001)
        if len(data) > 100_000:
            raise ContractHold("WORKFLOW_SOURCE_BYTE_LIMIT")
        result = inspect(data.decode("utf-8"))
    except (ContractHold, UnicodeError, OSError) as error:
        code = str(error) if isinstance(error, ContractHold) else type(error).__name__
        print(json.dumps({"status": "HOLD", "reason": code}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
