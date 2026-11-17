"""Bind an existing unsigned artifact observation to an exact evaluation input.

This module is standalone and standard-library only so an evaluation bundle can
archive it. It never imports or executes archived verifier sources, opens model
artifacts, contacts a provider, or grants compute or promotion authority.
"""

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

SCHEMA = "szl.evaluation-artifact-binding/v1"
QUALIFICATION_SCHEMA = "szl.adapter-artifact-qualification/v1"
BUNDLE_SCHEMA = "szl.qualification-bundle/v1"
MAX_INPUT_BYTES = 2_000_000
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
SOURCE_FILES = ("artifact_qualification.py", "training_snapshot.py")
BUNDLE_FILES = ("qualification.json", "manifest.json", *SOURCE_FILES)
RELEASE_BLOCKERS = (
    "TRAINING_PROVENANCE_AND_SIGNATURE_NOT_VERIFIED",
    "HELDOUT_QUALITY_AND_CONTAMINATION_NOT_ASSESSED",
    "SAFETY_AND_NEUTRAL_CAPABILITY_NOT_ASSESSED",
    "SERVING_AUTH_ISOLATION_LATENCY_ROLLBACK_NOT_ASSESSED",
    "OWNER_PROMOTION_APPROVAL_NOT_BOUND",
)
REASON_CODES = frozenset(
    {
        "QUALIFICATION_PIN_INVALID",
        "QUALIFICATION_PIN_MISMATCH",
        "QUALIFICATION_UNAVAILABLE",
        "QUALIFICATION_UNSAFE_PATH",
        "QUALIFICATION_BYTE_BUDGET",
        "QUALIFICATION_INVALID_JSON",
        "QUALIFICATION_DUPLICATE_JSON_KEY",
        "QUALIFICATION_NONFINITE_JSON",
        "QUALIFICATION_SOURCE_BYTES_INVALID",
        "QUALIFICATION_SCHEMA_INVALID",
        "QUALIFICATION_FILE_HASH_MISMATCH",
        "QUALIFICATION_SOURCE_HASH_MISMATCH",
        "QUALIFICATION_SCOPE_INVALID",
        "QUALIFICATION_CANDIDATES_INVALID",
        "QUALIFICATION_CANDIDATE_NOT_FOUND",
        "QUALIFICATION_CANDIDATE_NOT_READY",
        "QUALIFICATION_CANDIDATE_HOLD",
        "QUALIFICATION_CANDIDATE_UNAVAILABLE",
        "QUALIFICATION_CANDIDATE_INVALID",
        "QUALIFICATION_SNAPSHOT_INVALID",
        "QUALIFICATION_TERMINAL_MISMATCH",
        "QUALIFICATION_FILE_PROOF_INVALID",
        "QUALIFICATION_TENSOR_PROOF_INVALID",
        "QUALIFICATION_ADAPTER_CONFIG_INVALID",
        "QUALIFICATION_RELEASE_BLOCKERS_CHANGED",
        "QUALIFICATION_CONTRACT_INVALID",
        "QUALIFICATION_ADAPTER_REVISION_MISMATCH",
        "QUALIFICATION_ADAPTER_DIRECTORY_MISMATCH",
        "QUALIFICATION_BASE_MISMATCH",
        "QUALIFICATION_INTERNAL_ERROR",
    }
)


class BindingError(ValueError):
    """Only a fixed reason code is exposed; never provider or filesystem text."""

    def __init__(self, reason_code):
        self.reason_code = (
            reason_code
            if isinstance(reason_code, str) and reason_code in REASON_CODES
            else "QUALIFICATION_INTERNAL_ERROR"
        )
        self.code = self.reason_code
        super().__init__(self.reason_code)


def _require(condition, code):
    if not condition:
        raise BindingError(code)


def _keys(value, names, code):
    _require(isinstance(value, dict) and set(value) == set(names), code)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _repo(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) is not None
        and all(part not in {".", ".."} for part in value.split("/"))
    )


def _positive(value):
    return type(value) is int and value > 0


def _timestamp(value, code):
    _require(isinstance(value, str) and len(value) <= 64, code)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BindingError(code) from None
    _require(parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0), code)
    return parsed


def _encoded(value, code):
    """Exactly the heldout harness's canonical UTF-8 encoding, including newline."""
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise BindingError(code) from None
    _require(0 < len(raw) <= MAX_INPUT_BYTES, "QUALIFICATION_BYTE_BUDGET")
    return raw


def _strict_object(raw):
    _require(
        isinstance(raw, bytes) and 0 < len(raw) <= MAX_INPUT_BYTES,
        "QUALIFICATION_BYTE_BUDGET",
    )

    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, "QUALIFICATION_DUPLICATE_JSON_KEY")
            value[key] = item
        return value

    def constant(_):
        raise BindingError("QUALIFICATION_NONFINITE_JSON")

    def number(value):
        parsed = float(value)
        _require(math.isfinite(parsed), "QUALIFICATION_NONFINITE_JSON")
        return parsed

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=number,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, BindingError):
            raise
        raise BindingError("QUALIFICATION_INVALID_JSON") from None
    _require(isinstance(value, dict), "QUALIFICATION_INVALID_JSON")
    return value


def _unsafe(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _read_file(directory, name):
    path = directory / name
    try:
        before = path.lstat()
        _require(
            not _unsafe(before) and stat.S_ISREG(before.st_mode),
            "QUALIFICATION_UNSAFE_PATH",
        )
        _require(0 < before.st_size <= MAX_INPUT_BYTES, "QUALIFICATION_BYTE_BUDGET")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            _require(
                stat.S_ISREG(opened.st_mode)
                and (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
                "QUALIFICATION_UNSAFE_PATH",
            )
            raw = stream.read(MAX_INPUT_BYTES + 1)
        _require(
            len(raw) == before.st_size and 0 < len(raw) <= MAX_INPUT_BYTES,
            "QUALIFICATION_BYTE_BUDGET",
        )
        return raw
    except OSError:
        raise BindingError("QUALIFICATION_UNAVAILABLE") from None


def _report_schema(report):
    _keys(
        report,
        (
            "schema",
            "observed_at_utc",
            "candidates",
            "promotion_authorized",
            "signed",
            "scope",
            "code_sha256",
        ),
        "QUALIFICATION_SCHEMA_INVALID",
    )
    _require(report["schema"] == QUALIFICATION_SCHEMA, "QUALIFICATION_SCHEMA_INVALID")
    _require(
        report["promotion_authorized"] is False
        and report["signed"] is False
        and report["scope"]
        == (
            "bounded private artifact observation; not training attestation, "
            "quality or deployment"
        ),
        "QUALIFICATION_SCOPE_INVALID",
    )
    _timestamp(report["observed_at_utc"], "QUALIFICATION_SCHEMA_INVALID")
    _keys(report["code_sha256"], SOURCE_FILES, "QUALIFICATION_SCHEMA_INVALID")
    _require(
        all(_sha(value) for value in report["code_sha256"].values()),
        "QUALIFICATION_SCHEMA_INVALID",
    )
    candidates = report["candidates"]
    _require(
        isinstance(candidates, list)
        and 1 <= len(candidates) <= 6
        and all(isinstance(c, dict) and _repo(c.get("repo")) for c in candidates),
        "QUALIFICATION_CANDIDATES_INVALID",
    )
    _require(
        len({c["repo"] for c in candidates}) == len(candidates),
        "QUALIFICATION_CANDIDATES_INVALID",
    )


def load_qualification_bundle(directory, expected_sha256):
    """Read four bounded files and verify the operator's pin and hash links.

    A hash pin provides content identity, not source admission or authenticity.
    The returned raw files are suitable for archiving without re-reading inputs.
    """
    _require(_sha(expected_sha256), "QUALIFICATION_PIN_INVALID")
    try:
        directory = Path(directory).absolute()
        _require(".." not in directory.parts, "QUALIFICATION_UNSAFE_PATH")
        for part in (directory, *directory.parents):
            info = part.lstat()
            _require(
                not _unsafe(info) and stat.S_ISDIR(info.st_mode),
                "QUALIFICATION_UNSAFE_PATH",
            )
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, BindingError):
            raise
        raise BindingError("QUALIFICATION_UNAVAILABLE") from None
    files = {name: _read_file(directory, name) for name in BUNDLE_FILES}
    _require(
        hashlib.sha256(files["qualification.json"]).hexdigest() == expected_sha256,
        "QUALIFICATION_PIN_MISMATCH",
    )
    report = _strict_object(files["qualification.json"])
    manifest = _strict_object(files["manifest.json"])
    _report_schema(report)
    _keys(manifest, ("schema", "signed", "sha256"), "QUALIFICATION_SCHEMA_INVALID")
    _require(
        manifest["schema"] == BUNDLE_SCHEMA and manifest["signed"] is False,
        "QUALIFICATION_SCHEMA_INVALID",
    )
    _keys(
        manifest["sha256"],
        ("qualification.json", *SOURCE_FILES),
        "QUALIFICATION_SCHEMA_INVALID",
    )
    for name, expected in manifest["sha256"].items():
        _require(
            _sha(expected) and hashlib.sha256(files[name]).hexdigest() == expected,
            "QUALIFICATION_FILE_HASH_MISMATCH",
        )
    for name in SOURCE_FILES:
        _require(
            report["code_sha256"][name] == hashlib.sha256(files[name]).hexdigest(),
            "QUALIFICATION_SOURCE_HASH_MISMATCH",
        )
        try:
            source = files[name].decode("utf-8")
        except UnicodeError:
            raise BindingError("QUALIFICATION_SOURCE_BYTES_INVALID") from None
        _require("\x00" not in source, "QUALIFICATION_SOURCE_BYTES_INVALID")
    return report, files


def validate_candidate(candidate):
    """Return an allowlisted artifact observation summary, with no authority."""
    code = "QUALIFICATION_CANDIDATE_INVALID"
    _require(isinstance(candidate, dict), code)
    holds = candidate.get("hold_reasons")
    _require(isinstance(holds, list) and all(isinstance(x, str) for x in holds), code)
    if holds:
        unavailable = any(x.startswith("OBSERVATION_UNAVAILABLE:") for x in holds)
        raise BindingError(
            "QUALIFICATION_CANDIDATE_UNAVAILABLE"
            if unavailable
            else "QUALIFICATION_CANDIDATE_HOLD"
        )
    _require(
        candidate.get("status") == "ARTIFACT_VERIFIED"
        and candidate.get("ready_for_evaluation") is True,
        "QUALIFICATION_CANDIDATE_NOT_READY",
    )
    _keys(
        candidate,
        (
            "repo",
            "target_steps",
            "status",
            "ready_for_evaluation",
            "promotion_authorized",
            "training_execution_provenance_verified",
            "model_quality_measured",
            "signature_status",
            "release_blockers",
            "observed_at_utc",
            "gates",
            "hold_reasons",
            "snapshot",
            "trainer_state",
        ),
        code,
    )
    repo, target = candidate["repo"], candidate["target_steps"]
    _require(
        _repo(repo) and repo.startswith("SZLHOLDINGS/") and _positive(target), code
    )
    _timestamp(candidate["observed_at_utc"], code)
    _require(
        all(
            candidate[key] is False
            for key in (
                "promotion_authorized",
                "training_execution_provenance_verified",
                "model_quality_measured",
            )
        )
        and candidate["signature_status"] == "NOT_SIGNED",
        "QUALIFICATION_SCOPE_INVALID",
    )
    blockers = candidate["release_blockers"]
    _require(
        isinstance(blockers, list)
        and len(blockers) == len(RELEASE_BLOCKERS)
        and all(isinstance(x, str) for x in blockers)
        and set(blockers) == set(RELEASE_BLOCKERS),
        "QUALIFICATION_RELEASE_BLOCKERS_CHANGED",
    )
    snapshot = candidate["snapshot"]
    _require(isinstance(snapshot, dict), "QUALIFICATION_SNAPSHOT_INVALID")
    revision = snapshot.get("revision")
    _require(
        snapshot.get("id") == repo
        and snapshot.get("status") == "OBSERVED"
        and isinstance(revision, str)
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision) is not None
        and type(snapshot.get("private")) is bool
        and _positive(snapshot.get("file_count"))
        and snapshot["file_count"] >= 6
        and all(
            snapshot.get(key) is True
            for key in (
                "final_adapter",
                "final_config",
                "final_tokenizer",
                "latest_trainer_state_present",
            )
        )
        and snapshot.get("file_list_basis") == "same immutable metadata response"
        and snapshot.get("artifact_integrity") == "NOT_VERIFIED_BY_METADATA"
        and snapshot.get("training_complete") == "UNVERIFIED"
        and snapshot.get("process_liveness_verified") is False
        and snapshot.get("evaluation") == "NOT_ASSESSED",
        "QUALIFICATION_SNAPSHOT_INVALID",
    )
    state = candidate["trainer_state"]
    _keys(
        state,
        (
            "global_step",
            "max_steps",
            "epoch",
            "num_train_epochs",
            "trainer_state_sha256",
            "integrity",
        ),
        "QUALIFICATION_TERMINAL_MISMATCH",
    )
    _require(
        all(
            type(value) is int and value == target
            for value in (
                snapshot.get("latest_checkpoint"),
                state["global_step"],
                state["max_steps"],
            )
        )
        and state["integrity"] == "PINNED_OBJECT_BYTES_VERIFIED"
        and _sha(state["trainer_state_sha256"]),
        "QUALIFICATION_TERMINAL_MISMATCH",
    )
    gates = candidate["gates"]
    _keys(
        gates,
        (
            "adapter_config",
            "files",
            "final_matches_terminal_checkpoint",
            "tensors",
            "terminal_step_record_verified",
        ),
        "QUALIFICATION_FILE_PROOF_INVALID",
    )
    _require(
        gates["final_matches_terminal_checkpoint"] is True
        and gates["terminal_step_record_verified"] is True,
        "QUALIFICATION_TERMINAL_MISMATCH",
    )
    names = (
        f"checkpoint-{target}/trainer_state.json",
        "final/adapter_config.json",
        "final/adapter_model.safetensors",
        "final/tokenizer.json",
        "final/tokenizer_config.json",
    )
    files = gates["files"]
    _keys(files, names, "QUALIFICATION_FILE_PROOF_INVALID")
    for name in names:
        proof = files[name]
        _keys(
            proof, ("sha256", "bytes", "integrity"), "QUALIFICATION_FILE_PROOF_INVALID"
        )
        _require(
            _sha(proof["sha256"])
            and _positive(proof["bytes"])
            and proof["bytes"] <= MAX_ARTIFACT_BYTES
            and proof["integrity"] == "PINNED_OBJECT_BYTES_VERIFIED",
            "QUALIFICATION_FILE_PROOF_INVALID",
        )
        if name in (names[0], names[1], names[4]):
            _require(
                proof["bytes"] <= MAX_INPUT_BYTES, "QUALIFICATION_FILE_PROOF_INVALID"
            )
    _require(
        sum(files[name]["bytes"] for name in names[1:]) <= 192 * 1024 * 1024
        and files[names[0]]["sha256"] == state["trainer_state_sha256"],
        "QUALIFICATION_FILE_PROOF_INVALID",
    )
    config = gates["adapter_config"]
    _keys(
        config,
        ("base", "rank", "target_modules"),
        "QUALIFICATION_ADAPTER_CONFIG_INVALID",
    )
    modules = config["target_modules"]
    _require(
        _repo(config["base"])
        and _positive(config["rank"])
        and isinstance(modules, list)
        and 0 < len(modules) <= 1024
        and all(isinstance(x, str) and 0 < len(x) <= 512 for x in modules)
        and modules == sorted(set(modules)),
        "QUALIFICATION_ADAPTER_CONFIG_INVALID",
    )
    tensors = gates["tensors"]
    _keys(
        tensors,
        ("tensor_count", "elements", "dtypes", "all_finite", "device"),
        "QUALIFICATION_TENSOR_PROOF_INVALID",
    )
    dtypes = tensors["dtypes"]
    _require(
        tensors["all_finite"] is True
        and tensors["device"] == "cpu"
        and _positive(tensors["tensor_count"])
        and _positive(tensors["elements"])
        and tensors["elements"] >= tensors["tensor_count"]
        and isinstance(dtypes, list)
        and 0 < len(dtypes) <= 16
        and all(
            isinstance(x, str)
            and re.fullmatch(
                r"torch\.(?:float(?:16|32|64)|bfloat16|float8_[a-z0-9_]+)", x
            )
            is not None
            for x in dtypes
        )
        and dtypes == sorted(set(dtypes)),
        "QUALIFICATION_TENSOR_PROOF_INVALID",
    )
    return {
        "repo": repo,
        "revision": revision,
        "base_repo": config["base"],
        "target_steps": target,
        "artifact_sha256": files[names[2]]["sha256"],
        "observed_at_utc": candidate["observed_at_utc"],
        "release_blockers": list(RELEASE_BLOCKERS),
    }


def bind_report(report, contract, qualification_sha256, manifest_sha256):
    """Validate a report/contract association; loader must verify raw hash links.

    This deterministic operation performs no I/O. The full heldout contract must
    still pass the harness's own prepare() and external admission requirements.
    """
    _require(
        _sha(qualification_sha256) and _sha(manifest_sha256),
        "QUALIFICATION_PIN_INVALID",
    )
    _encoded(report, "QUALIFICATION_SCHEMA_INVALID")
    _report_schema(report)
    _keys(
        contract,
        (
            "schema",
            "attempt_id",
            "source",
            "base",
            "adapter",
            "fixtures",
            "training",
            "generation",
        ),
        "QUALIFICATION_CONTRACT_INVALID",
    )
    _require(
        contract["schema"] == "szl.heldout-paired/v2", "QUALIFICATION_CONTRACT_INVALID"
    )
    for key in ("base", "adapter"):
        _keys(
            contract[key],
            ("repo_id", "revision", "directory")
            if key == "adapter"
            else ("repo_id", "revision"),
            "QUALIFICATION_CONTRACT_INVALID",
        )
        identity = contract[key]
        _require(
            _repo(identity["repo_id"])
            and isinstance(identity["revision"], str)
            and re.fullmatch(r"[0-9a-f]{40}", identity["revision"]) is not None,
            "QUALIFICATION_CONTRACT_INVALID",
        )
    adapter = contract["adapter"]
    _require(
        adapter["directory"] == "final", "QUALIFICATION_ADAPTER_DIRECTORY_MISMATCH"
    )
    selected = [c for c in report["candidates"] if c["repo"] == adapter["repo_id"]]
    _require(len(selected) == 1, "QUALIFICATION_CANDIDATE_NOT_FOUND")
    observation = validate_candidate(selected[0])
    _require(
        observation["revision"] == adapter["revision"],
        "QUALIFICATION_ADAPTER_REVISION_MISMATCH",
    )
    _require(
        observation["base_repo"] == contract["base"]["repo_id"],
        "QUALIFICATION_BASE_MISMATCH",
    )
    _require(
        _timestamp(observation["observed_at_utc"], "QUALIFICATION_SCHEMA_INVALID")
        <= _timestamp(report["observed_at_utc"], "QUALIFICATION_SCHEMA_INVALID"),
        "QUALIFICATION_SCHEMA_INVALID",
    )
    contract_raw = _encoded(contract, "QUALIFICATION_CONTRACT_INVALID")
    return {
        "schema": SCHEMA,
        "state": "QUALIFICATION_BOUND",
        "artifact_observation_bound": True,
        "qualification_sha256": qualification_sha256,
        "manifest_sha256": manifest_sha256,
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "source_sha256": dict(report["code_sha256"]),
        "artifact": {
            "repo": observation["repo"],
            "revision": observation["revision"],
            "directory": "final",
            "sha256": observation["artifact_sha256"],
            "base_repo": observation["base_repo"],
            "target_steps": observation["target_steps"],
        },
        "observed_at_utc": observation["observed_at_utc"],
        "qualification_observed_at_utc": report["observed_at_utc"],
        "remaining_release_blockers": list(RELEASE_BLOCKERS),
        "authenticity_verified": False,
        "compute_authorized": False,
        "promotion_authorized": False,
        "training_complete_verified": False,
        "training_execution_provenance_verified": False,
        "base_revision_verified": False,
        "independent_holdout_verified": False,
        "source_admission_verified": False,
        "runtime_admission_verified": False,
        "quality_qualification": "NOT_ESTABLISHED",
        "model_quality_measured": False,
        "publication_performed": False,
        "production_ready": False,
        "scope": (
            "Unsigned artifact observation bound to one adapter revision; no artifact "
            "reinspection, base-revision attestation, training attestation, compute "
            "admission, scientific acceptance or promotion."
        ),
    }


def bind_qualification(bundle_dir, contract, expected_sha256):
    """Load pinned evidence and bind it without modifying any filesystem state."""
    report, files = load_qualification_bundle(bundle_dir, expected_sha256)
    return bind_report(
        report,
        contract,
        expected_sha256,
        hashlib.sha256(files["manifest.json"]).hexdigest(),
    )
