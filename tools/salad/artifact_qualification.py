"""Read-only, revision-bound qualification of terminal LoRA artifacts.

No model execution, pickle loading, signing, upload, cloud mutation or promotion.
A successful artifact check is eligibility for evaluation, never release approval.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from tools.salad.training_snapshot import (
    EvidenceHold,
    model_snapshot,
    verified_trainer_state,
)

SCHEMA = "szl.adapter-artifact-qualification/v1"
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_JSON_BYTES = 2_000_000
RELEASE_BLOCKERS = [
    "TRAINING_PROVENANCE_AND_SIGNATURE_NOT_VERIFIED",
    "HELDOUT_QUALITY_AND_CONTAMINATION_NOT_ASSESSED",
    "SAFETY_AND_NEUTRAL_CAPABILITY_NOT_ASSESSED",
    "SERVING_AUTH_ISOLATION_LATENCY_ROLLBACK_NOT_ASSESSED",
    "OWNER_PROMOTION_APPROVAL_NOT_BOUND",
]


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def strict_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EvidenceHold("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def constant(_):
        raise EvidenceHold("NONFINITE_JSON")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise EvidenceHold("NONFINITE_JSON")
        return result

    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_JSON_BYTES:
        raise EvidenceHold("JSON_BYTE_BUDGET")
    try:
        result = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceHold("INVALID_JSON") from exc
    if not isinstance(result, dict):
        raise EvidenceHold("JSON_NOT_OBJECT")
    return result


def lfs_sha(entry):
    lfs = getattr(entry, "lfs", None)
    value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EvidenceHold("LFS_SHA256_UNAVAILABLE")
    return value


def verify_file(path, entry):
    """Stream the bytes, checking observed size plus authoritative object identity."""
    path = Path(path)
    size = getattr(entry, "size", None)
    if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
        raise EvidenceHold("FILE_BYTE_BUDGET")
    if path.is_symlink() or not path.is_file():
        raise EvidenceHold("NONREGULAR_INPUT_FILE")
    sha, git = hashlib.sha256(), hashlib.sha1()
    git.update(b"blob " + str(size).encode() + b"\0")
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            if count > size:
                raise EvidenceHold("FILE_SIZE_MISMATCH")
            sha.update(chunk)
            git.update(chunk)
    if count != size:
        raise EvidenceHold("FILE_SIZE_MISMATCH")
    observed = sha.hexdigest()
    if getattr(entry, "lfs", None) is not None:
        if observed != lfs_sha(entry):
            raise EvidenceHold("FILE_HASH_MISMATCH")
    else:
        expected = getattr(entry, "blob_id", None)
        if (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{40}", expected)
            or git.hexdigest() != expected
        ):
            raise EvidenceHold("FILE_HASH_MISMATCH")
    return {
        "sha256": observed,
        "bytes": count,
        "integrity": "PINNED_OBJECT_BYTES_VERIFIED",
    }


def inspect_safetensors(path):
    """Inspect every floating tensor on CPU; no model or remote code is loaded."""
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    count = elements = 0
    dtypes = set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if not tensor.is_floating_point() or tensor.numel() <= 0:
                raise EvidenceHold("INVALID_ADAPTER_TENSOR")
            if not bool(torch.isfinite(tensor).all().item()):
                raise EvidenceHold("NONFINITE_ADAPTER_TENSOR")
            count += 1
            elements += tensor.numel()
            dtypes.add(str(tensor.dtype))
    if count == 0:
        raise EvidenceHold("EMPTY_ADAPTER")
    return {
        "tensor_count": count,
        "elements": elements,
        "dtypes": sorted(dtypes),
        "all_finite": True,
        "device": "cpu",
    }


def observe_candidate(api, fetch, candidate, *, inspect=inspect_safetensors):
    """One immutable metadata view, bounded allowlisted downloads, explicit holds."""
    repo, target, base = (
        candidate.get("repo"),
        candidate.get("target_steps"),
        candidate.get("base"),
    )
    if not isinstance(repo, str) or not re.fullmatch(
        r"SZLHOLDINGS/[A-Za-z0-9_.-]+", repo
    ):
        raise EvidenceHold("INVALID_CANDIDATE_ID")
    if type(target) is not int or target < 1 or not isinstance(base, str) or not base:
        raise EvidenceHold("INVALID_CANDIDATE_CONTRACT")
    result = {
        "repo": repo,
        "target_steps": target,
        "status": "HOLD",
        "ready_for_evaluation": False,
        "promotion_authorized": False,
        "training_execution_provenance_verified": False,
        "model_quality_measured": False,
        "signature_status": "NOT_SIGNED",
        "release_blockers": list(RELEASE_BLOCKERS),
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gates": {},
        "hold_reasons": [],
    }
    try:
        _, snapshot, entries = model_snapshot(api, repo)
        result["snapshot"] = snapshot
        revision = snapshot["revision"]
        step = snapshot["latest_checkpoint"]
        if step is None:
            raise EvidenceHold("NO_CHECKPOINT_OBSERVED")
        state_name = f"checkpoint-{step}/trainer_state.json"
        if state_name not in entries:
            raise EvidenceHold("TRAINER_STATE_MISSING")
        state_entry = entries[state_name]
        if (
            type(state_entry.size) is not int
            or not 0 < state_entry.size <= MAX_JSON_BYTES
        ):
            raise EvidenceHold("TRAINER_JSON_BYTE_BUDGET")
        with tempfile.TemporaryDirectory(prefix="szl-qualify-") as temp:

            def obtain(name):
                entry = entries[name]
                if type(entry.size) is not int or not 0 < entry.size <= MAX_FILE_BYTES:
                    raise EvidenceHold("FILE_BYTE_BUDGET")
                path = Path(
                    fetch(
                        repo_id=repo,
                        filename=name,
                        revision=revision,
                        local_dir=temp,
                        etag_timeout=12,
                    )
                )
                root = Path(temp).resolve()
                if path.is_symlink() or root not in path.resolve().parents:
                    raise EvidenceHold("DOWNLOAD_PATH_ESCAPE")
                proof = verify_file(path, entry)
                result["gates"].setdefault("files", {})[name] = proof
                return path

            raw_state = obtain(state_name).read_bytes()
            state = verified_trainer_state(raw_state, state_entry, step)
            result["trainer_state"] = state
            if state["max_steps"] != target:
                raise EvidenceHold("TARGET_CONTRACT_MISMATCH")
            if step < target:
                raise EvidenceHold("TRAINING_IN_PROGRESS")
            if step > target:
                raise EvidenceHold("TERMINAL_STEP_OVERSHOOT")
            result["gates"]["terminal_step_record_verified"] = True
            names = [
                "final/adapter_config.json",
                "final/adapter_model.safetensors",
                "final/tokenizer.json",
                "final/tokenizer_config.json",
            ]
            if any(name not in entries for name in names):
                raise EvidenceHold("FINAL_BUNDLE_INCOMPLETE")
            if sum(entries[n].size or 0 for n in names) > 192 * 1024 * 1024:
                raise EvidenceHold("CANDIDATE_BYTE_BUDGET")
            checkpoint_name = f"checkpoint-{target}/adapter_model.safetensors"
            if checkpoint_name not in entries or lfs_sha(
                entries[checkpoint_name]
            ) != lfs_sha(entries[names[1]]):
                raise EvidenceHold("FINAL_TERMINAL_ADAPTER_MISMATCH")
            config = strict_object(obtain(names[0]).read_bytes())
            if (
                config.get("base_model_name_or_path") != base
                or config.get("peft_type") != "LORA"
                or config.get("task_type") != "CAUSAL_LM"
            ):
                raise EvidenceHold("ADAPTER_CONFIG_CONTRACT_MISMATCH")
            if (
                type(config.get("r")) is not int
                or config["r"] <= 0
                or not isinstance(config.get("target_modules"), list)
                or not config["target_modules"]
            ):
                raise EvidenceHold("ADAPTER_CONFIG_INVALID")
            result["gates"]["adapter_config"] = {
                "base": base,
                "rank": config["r"],
                "target_modules": sorted(config["target_modules"]),
            }
            adapter = obtain(names[1])
            tensor_result = inspect(adapter)
            if (
                tensor_result.get("all_finite") is not True
                or type(tensor_result.get("tensor_count")) is not int
                or tensor_result["tensor_count"] <= 0
            ):
                raise EvidenceHold("TENSOR_INSPECTION_INCOMPLETE")
            result["gates"]["tensors"] = tensor_result
            # Verify tokenizer bytes, but do not execute tokenizer or custom code.
            obtain(names[2])
            strict_object(obtain(names[3]).read_bytes())
            # Rehash after inspection to detect local changes while checking.
            if (
                verify_file(adapter, entries[names[1]])
                != result["gates"]["files"][names[1]]
            ):
                raise EvidenceHold("LOCAL_ARTIFACT_CHANGED")
            result["gates"]["final_matches_terminal_checkpoint"] = True
            result.update(status="ARTIFACT_VERIFIED", ready_for_evaluation=True)
    except EvidenceHold as exc:
        result["hold_reasons"].append(str(exc))
    except Exception as exc:
        # Do not echo provider exception text (it may contain signed URLs or tokens).
        result["hold_reasons"].append("OBSERVATION_UNAVAILABLE:" + type(exc).__name__)
    return result


def write_bundle(directory, results):
    """Exclusive local bundle: old evidence is never replaced; hash is not signature."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    snapshot_source = Path(__file__).with_name("training_snapshot.py").read_bytes()
    receipt = {
        "schema": SCHEMA,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidates": results,
        "promotion_authorized": False,
        "signed": False,
        "scope": "bounded private artifact observation; not training attestation, quality or deployment",
        "code_sha256": {
            "artifact_qualification.py": hashlib.sha256(source).hexdigest(),
            "training_snapshot.py": hashlib.sha256(snapshot_source).hexdigest(),
        },
    }
    files = {
        "qualification.json": canonical(receipt),
        "artifact_qualification.py": source,
        "training_snapshot.py": snapshot_source,
    }
    manifest = {
        "schema": "szl.qualification-bundle/v1",
        "signed": False,
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    files["manifest.json"] = canonical(manifest)
    for name, raw in files.items():
        with (directory / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.output).is_symlink():
        parser.error("output already exists; choose a new observation directory")
    spec = strict_object(Path(args.spec).read_bytes())
    candidates = spec.get("candidates")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 6
        or any(not isinstance(c, dict) for c in candidates)
    ):
        parser.error("spec must contain 1-6 candidate objects")
    if len({c.get("repo") for c in candidates}) != len(candidates):
        parser.error("duplicate candidate identity")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    results = []
    for candidate in candidates:
        result = observe_candidate(api, hf_hub_download, candidate)
        results.append(result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("repo", "status", "hold_reasons", "ready_for_evaluation")
                }
            ),
            flush=True,
        )
    write_bundle(args.output, results)
    return 0 if all(r["ready_for_evaluation"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
