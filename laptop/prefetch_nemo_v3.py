#!/usr/bin/env python3
"""Prefetch one signed Nemo v3 job without executing model repository code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from frontier_contract import load_pin, verify_envelope
from frontier_job import _verify_card_license
from nemo_v3_contract import (
    NEMO_V3_PAYLOAD_TYPE,
    expected_engine_key_id,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def raw_source_url(spec: dict[str, Any], path: str) -> str:
    owner, name = spec["source"]["repoId"].split("/", 1)
    encoded_path = "/".join(
        urllib.parse.quote(part, safe="") for part in path.split("/")
    )
    return (
        "https://raw.githubusercontent.com/"
        f"{urllib.parse.quote(owner, safe='')}/"
        f"{urllib.parse.quote(name, safe='')}/"
        f"{spec['source']['revision']}/{encoded_path}"
    )


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_descriptor(
    spec: dict[str, Any], descriptor: dict[str, Any], input_cache: pathlib.Path
) -> dict[str, Any]:
    target = input_cache / descriptor["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(
            raw_source_url(spec, descriptor["path"]),
            headers={
                "User-Agent": "szl-gpu-bridge-prefetch/1.0",
                "Cache-Control": "no-cache",
            },
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 200:
                raise RuntimeError(f"source download returned HTTP {response.status}")
            data = response.read()
        target.write_bytes(data)

    observed_bytes = target.stat().st_size
    observed_sha256 = file_sha256(target)
    if observed_bytes != descriptor["bytes"] or observed_sha256 != descriptor["sha256"]:
        raise RuntimeError(
            f"pinned input mismatch for {descriptor['path']}: "
            f"bytes={observed_bytes}, sha256={observed_sha256}"
        )
    return {
        "path": descriptor["path"],
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def load_verified_job(
    spec_path: pathlib.Path,
    engine_key_path: pathlib.Path,
    *,
    expected_job_id: str,
    expected_source_revision: str,
    expected_workflow_blob: str,
    expected_execution_bridge_revision: str,
) -> tuple[dict[str, Any], bytes]:
    envelope = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    pin = load_pin(engine_key_path)
    spec, exact_payload, payload_type = verify_envelope(
        envelope, pin, allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,)
    )
    if payload_type != NEMO_V3_PAYLOAD_TYPE:
        raise ValueError("signed job is not a Nemo v3 payload")
    validate_nemo_v3_spec(spec)
    if spec.get("jobId") != expected_job_id:
        raise ValueError("signed job differs from the selected job identity")
    if spec.get("source", {}).get("revision") != expected_source_revision:
        raise ValueError("signed job differs from the selected A11oy source")
    if spec.get("ownerDispatch", {}).get("workflowBlob") != expected_workflow_blob:
        raise ValueError("signed job differs from the selected owner workflow")
    require_nemo_v3_dispatchable(
        spec,
        expected_execution_bridge_revision=expected_execution_bridge_revision,
    )
    if "authorization" in spec and pin.get("keyId") != expected_engine_key_id(spec):
        raise ValueError("Nemo v3 engine authorization key mismatch")
    return spec, exact_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--engine-key", type=pathlib.Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workflow-blob", required=True)
    parser.add_argument("--execution-bridge-revision", required=True)
    parser.add_argument("--hf-cache", type=pathlib.Path, required=True)
    parser.add_argument("--input-cache", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for authenticated prefetch")

    spec, exact_payload = load_verified_job(
        args.spec,
        args.engine_key,
        expected_job_id=args.job_id,
        expected_source_revision=args.source_revision,
        expected_workflow_blob=args.workflow_blob,
        expected_execution_bridge_revision=args.execution_bridge_revision,
    )
    if datetime.fromisoformat(spec["expiresAt"].replace("Z", "+00:00")) < datetime.now(
        timezone.utc
    ):
        raise RuntimeError(f"signed job expired at {spec['expiresAt']}")

    descriptors = [
        spec["dataset"]["train"],
        spec["dataset"]["preregistration"],
        *spec["dataset"]["holdouts"],
    ]
    inputs = [
        fetch_descriptor(spec, descriptor, args.input_cache)
        for descriptor in descriptors
    ]

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    model = api.model_info(spec["base"]["repoId"], revision=spec["base"]["revision"])
    if model.sha != spec["base"]["revision"]:
        raise RuntimeError(
            f"model revision drift: {model.sha} != {spec['base']['revision']}"
        )
    snapshot = pathlib.Path(
        snapshot_download(
            repo_id=spec["base"]["repoId"],
            revision=spec["base"]["revision"],
            cache_dir=args.hf_cache,
            token=token,
        )
    )
    readme = snapshot / "README.md"
    if not readme.is_file():
        raise RuntimeError("pinned model snapshot has no README.md")
    base_license = _verify_card_license(
        readme=readme,
        repo_id=spec["base"]["repoId"],
        revision=model.sha,
        expected=spec["base"]["licenseId"],
        repo_type="model",
    )

    receipt = {
        "kind": "szl-nemo-v3-prefetch",
        "v": 1,
        "jobId": spec["jobId"],
        "at": now_iso(),
        "signedJobPayloadSha256": hashlib.sha256(exact_payload).hexdigest(),
        "model": {
            "repoId": spec["base"]["repoId"],
            "revision": model.sha,
            "license": base_license,
        },
        "inputs": inputs,
        "remoteCodeExecuted": False,
        "credentialPersisted": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "jobId": spec["jobId"],
                "modelRevision": model.sha,
                "inputs": len(inputs),
                "verdict": "VERIFIED_PREFETCH_NO_REMOTE_CODE_EXECUTED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
