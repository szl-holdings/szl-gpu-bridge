#!/usr/bin/env python3
"""Read published receipts from their declared Hub repositories, once.

No provider controls, publication, model execution or qualification. Repository
revision binds an observation, not proof that these bytes ran a training job.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

from tools.salad.legacy_receipt_audit import MAX_BYTES, audit, strict_json

TARGETS = {
    "heldout-v4": {
        "repo_id": "SZLHOLDINGS/salad-frontier-v4-r1",
        "repo_type": "model",
        "path": "receipts/salad-v4-frontier-r1-heldout/result.json",
        "source_path": "receipts/salad-v4-frontier-r1-heldout/heldout_eval.py",
    },
    "heldout-v5": {
        "repo_id": "SZLHOLDINGS/salad-frontier-v5-r1",
        "repo_type": "model",
        "path": "receipts/salad-v5-frontier-r1-heldout/result.json",
        "source_path": "receipts/salad-v5-frontier-r1-heldout/heldout_eval.py",
    },
    "frontier-v5-r7": {
        "repo_id": "SZLHOLDINGS/frontier-54-runs",
        "repo_type": "dataset",
        "path": "receipts/salad-frontier-v5-r7/result.json",
    },
}
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _read_bounded(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("evidence exceeds byte budget")
    return raw


def receipt_summary(receipt):
    """Preserve producer status; publication alone never qualifies a run."""
    status = receipt.get("status")
    steps = receipt.get("new_steps_this_boot")
    zero_steps = type(steps) is int and steps == 0
    if status != "OK":
        classification = "PRODUCER_ERROR" if status == "ERROR" else "UNQUALIFIED"
    elif zero_steps:
        classification = "NO_NEW_TRAINING"
    elif receipt.get("completion_qualified") is False:
        classification = "UNQUALIFIED"
    else:
        classification = "PUBLISHED_RECEIPT"
    fields = (
        "schema",
        "status",
        "kind",
        "completed_utc",
        "steps",
        "new_steps_this_boot",
        "completion_qualified",
        "scope",
        "heldout_quality_all_pass",
    )
    return {
        "producer": {key: receipt[key] for key in fields if key in receipt},
        "classification": classification,
        "producer_error_present": bool(receipt.get("error")),
        "model_qualified": False,
        "energy_independently_verified": False,
        "current_process_liveness_verified": False,
    }


def _snapshot(api, target):
    info = api.repo_info(
        target["repo_id"],
        repo_type=target["repo_type"],
        files_metadata=True,
        timeout=30,
    )
    revision = str(getattr(info, "sha", ""))
    if not FULL_SHA.fullmatch(revision):
        raise ValueError("Hub observation has no immutable revision")
    siblings = getattr(info, "siblings", None)
    if siblings is None:
        raise ValueError("Hub file inventory unavailable")
    names = {item.rfilename for item in siblings}
    return revision, names


def observe_receipt(api, downloader, name, target):
    """Absence is scoped to an explicit repo/type/path at one immutable revision."""
    base = {"target": name, **target}
    revision, names = _snapshot(api, target)
    base["repo_revision"] = revision
    if target["path"] not in names:
        return {**base, "observation": "ABSENT_AT_OBSERVED_REVISION"}

    def download(path):
        return _read_bounded(
            downloader(
                repo_id=target["repo_id"],
                repo_type=target["repo_type"],
                filename=path,
                revision=revision,
            )
        )

    raw = download(target["path"])
    result = {
        **base,
        "observation": "PUBLISHED",
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        **receipt_summary(strict_json(raw)),
    }
    source_path = target.get("source_path")
    if source_path:
        if source_path not in names:
            result["legacy_audit"] = {
                "status": "UNAVAILABLE",
                "reason": "archived source absent",
            }
        else:
            try:
                result["legacy_audit"] = audit(raw, download(source_path))
            except (ValueError, KeyError, TypeError) as exc:
                # Invalid/ERROR receipts stay published evidence, never a pass.
                result["legacy_audit"] = {
                    "status": "UNAVAILABLE",
                    "reason": type(exc).__name__,
                }
    return result


def heartbeat_selection(names, prefix):
    """Numeric sequence and terminal record are separate; neither proves liveness."""
    pattern = re.compile(re.escape(prefix) + r"-hb([0-9]+)\.log$")
    numbered = []
    for name in names:
        match = pattern.fullmatch(name)
        if match:
            numbered.append((int(match[1]), name))
    latest = max(numbered, default=None)
    terminal = prefix + "-hbfinal.log"
    return {
        "numbered_count": len(numbered),
        "highest_numbered_path": latest[1] if latest else None,
        "highest_sequence": latest[0] if latest else None,
        "terminal_path": terminal if terminal in names else None,
        "current_process_liveness_verified": False,
    }


def observe_heartbeats(api):
    target = {
        "repo_id": "SZLHOLDINGS/frontier-54-runs",
        "repo_type": "dataset",
    }
    revision, names = _snapshot(api, target)
    return {
        **target,
        "repo_revision": revision,
        **heartbeat_selection(names, "mesh/lanelogs/flagship54-v1"),
        "scope": "file catalog only; sequence is not a freshness timestamp",
    }


def main(argv=None):
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", choices=sorted(TARGETS))
    parser.add_argument("--skip-heartbeats", action="store_true")
    args = parser.parse_args(argv)
    api = HfApi()
    report = {
        "schema": "szl.receipt-observation/v1",
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "read_only": True,
        "receipts": [],
    }
    failed = False
    for name in args.target or TARGETS:
        try:
            item = observe_receipt(api, hf_hub_download, name, TARGETS[name])
        except Exception as exc:
            # Exception messages can contain signed URLs. Do not log their values.
            item = {
                "target": name,
                **TARGETS[name],
                "observation": "UNAVAILABLE",
                "error_type": type(exc).__name__,
            }
            failed = True
        report["receipts"].append(item)
        if item["observation"] == "UNAVAILABLE":
            # Do not continue a provider lane after denied/unavailable access.
            break
    if not failed and not args.skip_heartbeats:
        try:
            report["flagship_heartbeat_catalog"] = observe_heartbeats(api)
        except Exception as exc:
            report["flagship_heartbeat_catalog"] = {
                "observation": "UNAVAILABLE",
                "error_type": type(exc).__name__,
            }
            failed = True
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
