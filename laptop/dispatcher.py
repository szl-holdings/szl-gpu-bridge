#!/usr/bin/env python3
"""Verify a bridge job once, then dispatch only to an allowed local runner."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

from frontier_contract import (
    ContractError,
    V1_PAYLOAD_TYPE,
    V2_PAYLOAD_TYPE,
    load_pin,
    validate_v2_spec,
    verify_envelope,
)
from nemo_v3_contract import (
    NEMO_V3_KIND,
    NEMO_V3_PAYLOAD_TYPE,
    validate_nemo_v3_spec,
)

ROOT = pathlib.Path(__file__).resolve().parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def refuse(spec_path: str, reason: str, *, verified: dict | None = None) -> int:
    path = ROOT / "logs" / "refused-specs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": now_iso(), "file": spec_path, "reason": reason}
    if verified:
        record["verifiedJobId"] = verified.get("jobId")
        record["verifiedKind"] = verified.get("kind")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"REFUSED: {reason}")
    return 3


def main(spec_path: str) -> int:
    try:
        envelope = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8-sig"))
        pin = load_pin(ROOT / "keys" / "engine_pubkey.json")
        spec, _, payload_type = verify_envelope(
            envelope,
            pin,
            allowed_payload_types=(V1_PAYLOAD_TYPE, V2_PAYLOAD_TYPE, NEMO_V3_PAYLOAD_TYPE),
        )
    except Exception as exc:  # noqa: BLE001
        return refuse(spec_path, f"envelope verification failed: {exc}")

    if payload_type == V1_PAYLOAD_TYPE and spec.get("kind") == "unsloth-qlora-sft-v1":
        runner = ROOT / "runjob.py"
    elif payload_type == V2_PAYLOAD_TYPE and spec.get("kind") == "unsloth-frontier-sft-v2":
        try:
            validate_v2_spec(spec)
        except ContractError as exc:
            return refuse(spec_path, f"v2 contract invalid: {exc}", verified=spec)
        runner = ROOT / "runjob_frontier.py"
    elif payload_type == NEMO_V3_PAYLOAD_TYPE and spec.get("kind") == NEMO_V3_KIND:
        try:
            validate_nemo_v3_spec(spec)
        except ContractError as exc:
            return refuse(spec_path, f"Nemo v3 contract invalid: {exc}", verified=spec)
        runner = ROOT / "runjob_nemo_v3.py"
    else:
        return refuse(
            spec_path,
            f"verified but unsupported payload/kind pair: {payload_type!r}/{spec.get('kind')!r}",
            verified=spec,
        )

    if not runner.is_file():
        return refuse(spec_path, f"local runner missing: {runner.name}", verified=spec)
    completed = subprocess.run([sys.executable, str(runner), spec_path], check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: dispatcher.py <signed-job-spec.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
