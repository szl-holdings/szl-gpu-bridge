#!/usr/bin/env python3
"""Publish one terminal Lean run record; qualification covers metadata, not quality."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

DEST = "SZLHOLDINGS/frontier-54-runs"
TARGET = "SZLHOLDINGS/salad-lean-math-v2-r1"
REMOTE_PATH = "mesh/lean/salad-lean-math-v2-r1-lane.json"
TERMINAL = {"exited", "stopped", "error", "refused"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=False).encode()


def verified(obj):
    body = {k: v for k, v in obj.items() if k != "receipt_sha256"}
    return obj.get("receipt_sha256") == sha(canonical(body))


def atomic(path, body):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_spec(spec):
    allowed = (
        "run_id",
        "source_commit",
        "source_files",
        "host",
        "base",
        "dataset",
        "adapter_repo",
        "max_steps",
    )
    result = {k: spec[k] for k in allowed}
    if (
        type(result["max_steps"]) is not int
        or result["max_steps"] != 200
        or result["adapter_repo"] != TARGET
    ):
        raise ValueError("unexpected lane or target")
    for key in ("run_id", "host"):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", result[key]):
            raise ValueError("invalid run or host label")
    if not re.fullmatch(r"[a-f0-9]{40}", result["source_commit"]):
        raise ValueError("source must be pinned")
    if not result["source_files"] or not all(
        re.fullmatch(r"[A-Za-z0-9_./-]{1,200}", k) and re.fullmatch(r"[a-f0-9]{64}", v)
        for k, v in result["source_files"].items()
    ):
        raise ValueError("source file hashes required")
    for key in ("base", "dataset"):
        item = result[key]
        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", item["repo"]
        ) or not re.fullmatch(r"[a-f0-9]{40}", item["revision"]):
            raise ValueError("model and dataset must be pinned")
        result[key] = {"repo": item["repo"], "revision": item["revision"]}
    return result


def worker_records(log):
    records, invalid = [], 0
    for line in log.decode("utf-8", errors="replace").splitlines():
        if '"szl.salad-worker/v1"' not in line:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(line[line.index("{") :])
            if obj.get("kind") != "szl.salad-worker/v1" or not verified(obj):
                raise ValueError("receipt hash mismatch")
            records.append(obj)
        except (ValueError, TypeError, AttributeError):
            invalid += 1
    return records, invalid


def terminal_evidence(spec, log, supervisor):
    if (
        supervisor.get("schema") != "szl.shared-gpu-supervisor/v1"
        or supervisor.get("status") not in TERMINAL
    ):
        raise ValueError("supervisor is not terminal")
    records, invalid = worker_records(log)
    last = records[-1] if records else {}
    boots = [r for r in records if r.get("event") == "boot"]
    reasons = []
    if invalid or len(boots) != 1:
        reasons.append("receipt_integrity_or_run_boundary_unverified")
    if (
        supervisor.get("status") != "exited"
        or type(supervisor.get("child_returncode")) is not int
        or supervisor["child_returncode"] != 0
    ):
        reasons.append("worker_did_not_exit_successfully")
    if last.get("event") != "complete" or last.get("completion_qualified") is not True:
        reasons.append("no_qualified_complete_event")
    if (
        type(last.get("actual_steps")) is not int
        or last["actual_steps"] != spec["max_steps"]
        or type(last.get("target_steps")) is not int
        or last.get("target_steps") != spec["max_steps"]
    ):
        reasons.append("observed_steps_do_not_match_target")
    if last.get("adapter") != spec["adapter_repo"]:
        reasons.append("adapter_binding_mismatch")
    if len(boots) == 1:
        boot = boots[0]
        if any(
            boot.get(k + "_revision") != spec[k]["revision"]
            for k in ("base", "dataset")
        ):
            reasons.append("input_revision_mismatch")
        try:
            start, first, final, end = [
                datetime.fromisoformat(v.replace("Z", "+00:00"))
                for v in (
                    supervisor["started_at"],
                    boot["ts"],
                    last["ts"],
                    supervisor["updated_at"],
                )
            ]
            if (
                not all(v.tzinfo is not None for v in (start, first, final, end))
                or not start <= first <= final <= end
            ):
                raise ValueError("stale receipt")
        except (ValueError, TypeError, KeyError, AttributeError):
            reasons.append("receipt_time_outside_supervisor_run")
    safe = {
        k: last.get(k)
        for k in (
            "event",
            "ts",
            "actual_steps",
            "target_steps",
            "resume_global_step",
            "receipt_sha256",
        )
    }
    safe["self_hash_verified"] = bool(last) and invalid == 0
    safe["original_receipt_payload_included"] = False
    return safe, reasons


def artifact_metadata(api, download, spec):
    info = api.repo_info(TARGET, repo_type="model")
    if info.private is not True or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("private pinned target required")
    files = api.list_repo_files(TARGET, repo_type="model", revision=info.sha)
    required = {
        "checkpoint-200/trainer_state.json",
        "final/adapter_config.json",
        "final/adapter_model.safetensors",
    }
    if not required.issubset(files):
        raise ValueError("required artifact metadata missing")

    def read(name):
        path = Path(download(TARGET, name, repo_type="model", revision=info.sha))
        if path.stat().st_size > 2_000_000:
            raise ValueError("metadata exceeds bound")
        return json.loads(path.read_bytes())

    state, config = (
        read("checkpoint-200/trainer_state.json"),
        read("final/adapter_config.json"),
    )
    if any(
        type(state.get(k)) is not int or state[k] != 200
        for k in ("global_step", "max_steps")
    ):
        raise ValueError("checkpoint trainer state does not prove the exact target")
    if config.get("base_model_name_or_path") != spec["base"]["repo"]:
        raise ValueError("adapter base metadata mismatch")
    return {
        "revision": info.sha,
        "private": True,
        "global_step": 200,
        "max_steps": 200,
        "final_files_present": True,
        "tensor_contents_verified": False,
    }


def publish(api, download, body, directory):
    marker = directory / "publication.json"
    progress = (
        json.loads(marker.read_bytes())
        if marker.exists()
        else {"payload_sha256": sha(body), "commit": None}
    )
    if progress["payload_sha256"] != sha(body):
        raise ValueError("publication payload changed")
    for attempt in range(3):
        try:
            if not progress["commit"]:
                # Recover a successful upload whose response was lost, avoiding
                # another commit when the immutable payload already reached Hub.
                head = api.repo_info(DEST, repo_type="dataset")
                if head.private is not False or not re.fullmatch(
                    r"[a-f0-9]{40}", head.sha
                ):
                    raise ValueError("public destination must have an exact commit")
                try:
                    existing = Path(
                        download(
                            DEST, REMOTE_PATH, repo_type="dataset", revision=head.sha
                        )
                    ).read_bytes()
                except Exception as exc:
                    if (
                        getattr(getattr(exc, "response", None), "status_code", None)
                        != 404
                    ):
                        raise
                    existing = None
                if existing == body:
                    progress["commit"] = head.sha
                else:
                    result = api.upload_file(
                        path_or_fileobj=body,
                        path_in_repo=REMOTE_PATH,
                        repo_id=DEST,
                        repo_type="dataset",
                        commit_message="Record Lean v2 terminal metadata",
                    )
                    progress["commit"] = result.oid
                if not re.fullmatch(r"[a-f0-9]{40}", progress["commit"]):
                    raise ValueError("publication returned no exact commit")
                atomic(marker, canonical(progress))
            elif not re.fullmatch(r"[a-f0-9]{40}", progress["commit"]):
                raise ValueError("invalid stored publication commit")
            downloaded = Path(
                download(
                    DEST, REMOTE_PATH, repo_type="dataset", revision=progress["commit"]
                )
            ).read_bytes()
            if downloaded != body or sha(downloaded) != progress["payload_sha256"]:
                raise ValueError("exact-commit publication readback mismatch")
            progress["verified"] = True
            atomic(marker, canonical(progress))
            return progress
        except Exception:
            if attempt == 2:
                raise
            time.sleep(15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("launch-spec", "worker-log", "supervisor-state"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--outputdir", "--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.outputdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        from huggingface_hub import HfApi, hf_hub_download

        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)

        def download(*a, **kw):
            return hf_hub_download(*a, token=token, **kw)

        inputs = {
            k: getattr(args, k).read_bytes()
            for k in ("launch_spec", "worker_log", "supervisor_state")
        }
        binding = {k + "_sha256": sha(v) for k, v in inputs.items()}
        path = args.outputdir / "lane_record.json"
        if path.exists():
            body = path.read_bytes()
            record = json.loads(body)
            if not verified(record) or record.get("input_digests") != binding:
                raise ValueError("existing local record does not bind these inputs")
        else:
            spec = source_spec(json.loads(inputs["launch_spec"]))
            supervisor = json.loads(inputs["supervisor_state"])
            terminal, reasons = terminal_evidence(
                spec, inputs["worker_log"], supervisor
            )
            metadata = None
            if not reasons:
                try:
                    metadata = artifact_metadata(api, download, spec)
                except Exception as exc:
                    reasons.append("artifact_metadata_unverified:" + type(exc).__name__)
            record = {
                "kind": "szl.lean-math-sft/v2-lane",
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": spec,
                "input_digests": binding,
                "worker_terminal": terminal,
                "supervisor": {
                    k: supervisor.get(k)
                    for k in ("status", "reason", "child_pid", "child_returncode")
                },
                "artifact_metadata": metadata,
                "status": "COMPLETE_METADATA_ONLY"
                if not reasons
                else "NOT_COMPLETION_QUALIFIED",
                "completion_qualified": not reasons,
                "completion_scope": "TERMINAL_AND_ARTIFACT_METADATA_ONLY",
                "model_quality_qualified": False,
                "provenance_qualified": False,
                "source_attestation": "LAUNCH_SPEC_DECLARATION_ONLY",
                "promotion_authorized": False,
                "energy_scope": "WHOLE_DEVICE_INCLUDES_CORESIDENT_WORKLOADS",
                "training_only_energy_qualified": False,
                "qualification_failures": reasons,
            }
            record["receipt_sha256"] = sha(canonical(record))
            body = (
                json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode()
                + b"\n"
            )
            atomic(path, body)
        publication = publish(api, download, body, args.outputdir)
        print(
            json.dumps(
                {
                    "publication": publication,
                    "completion_qualified": record["completion_qualified"],
                }
            )
        )
        return 0 if record["completion_qualified"] else 7
    except Exception as exc:
        error = {"status": "publisher_failed", "error_type": type(exc).__name__}
        atomic(args.outputdir / "publisher_error.json", canonical(error))
        print(json.dumps(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
