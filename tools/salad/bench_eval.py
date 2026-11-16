#!/usr/bin/env python3
# Copyright 2026 SZL Holdings -- SPDX-License-Identifier: Apache-2.0
"""Governed post-train bench for Salad-trained adapters (v2.0).

Same-method comparison of eval-split loss vs train-split reference loss
(v4full acceptance criterion 1: eval_repo loss <= final train loss + 1.0),
plus base-vs-adapter improvement on the identical pipeline, MEASURED energy
(NVML millijoule counters converted to joules), full pinning of base revision, adapter
revision, dataset file sha256s, and a fail-closed receipt to the Hub.

Mirrors tools/salad/salad_worker.py's exact text pipeline:
  text = tokenizer.apply_chat_template(messages, tokenize=False)
  truncate to max_length (worker SFTConfig max_length=2048), causal-LM loss
over the full rendered text. Both splits are computed with THIS script's one
method, so criterion 1 is internally consistent; the method is declared in
the receipt and never claimed to be TRL's internal loss verbatim.
"""

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

SCHEMA = "szl.salad-bench/v2"
FIT_CLASS = "TOKEN_LOSS_FIT_ONLY"
SPLIT_AUDIT_SCHEMA = "szl.scored-token-split-audit/v1"
SPLIT_AUDIT_SCOPE = (
    "exact input-token sequences after chat templating and truncation; "
    "not semantic, benchmark, or pretraining contamination detection"
)


def utcnow():
    return (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ]
        + "Z"
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Energy:
    def __init__(self):
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self.p = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self.h)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
            self.ok = True
        except Exception as e:
            self.name = "NVML_UNAVAILABLE: " + repr(e)[:120]

    def read(self):
        if not self.ok:
            return None
        try:
            # NVML documents this counter in mJ, not J. Keep raw reads for audit.
            raw = self.p.nvmlDeviceGetTotalEnergyConsumption(self.h)
            if type(raw) is not int or raw < 0:
                self.ok = False
                return None
            return raw
        except Exception:
            self.ok = False
            return None


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("dataset rows must be objects")
            yield row


def load_jsonl(path, max_rows=None):
    rows = []
    for row in iter_jsonl(path):
        rows.append(row)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def encode_row(row, tok, max_length):
    """One input pipeline for the preflight audit and measured model forwards."""
    msgs = row.get("messages")
    if not msgs:
        return None
    text = tok.apply_chat_template(msgs, tokenize=False)
    return tok(text, truncation=True, max_length=max_length, return_tensors="pt")


def audit_split_overlap(eval_rows, train_path, tok, max_length):
    """Scan the entire train file, even when train loss uses only a first-N sample.

    Matching effective token sequences catch metadata-only changes and collisions
    introduced by truncation. Fingerprints retain no prompt text. This establishes
    a narrow identity check, never independent sampling or model generalization.
    """
    summaries, fingerprints = {}, {}
    for name, source in (("eval", eval_rows), ("train", iter_jsonl(train_path))):
        seen = Counter()
        offered = 0
        for row in source:
            offered += 1
            enc = encode_row(row, tok, max_length)
            if enc is None or enc["input_ids"].shape[1] < 2:
                continue
            ids = enc["input_ids"].tolist()
            if (
                not isinstance(ids, list)
                or len(ids) != 1
                or not isinstance(ids[0], list)
                or len(ids[0]) != enc["input_ids"].shape[1]
                or any(type(x) is not int or x < 0 for x in ids[0])
            ):
                raise ValueError("tokenizer returned invalid input-token identities")
            raw = json.dumps(ids[0], separators=(",", ":")).encode("ascii")
            seen[hashlib.sha256(raw).hexdigest()] += 1
        scorable = sum(seen.values())
        summaries[name] = {
            "rows_scanned": offered,
            "scorable_rows": scorable,
            "unscorable_rows": offered - scorable,
            "unique_inputs": len(seen),
            "duplicate_rows": scorable - len(seen),
        }
        fingerprints[name] = seen
    common = fingerprints["train"].keys() & fingerprints["eval"].keys()
    available = all(s["scorable_rows"] > 0 for s in summaries.values())
    return {
        "schema": SPLIT_AUDIT_SCHEMA,
        "scope": SPLIT_AUDIT_SCOPE,
        "max_length": max_length,
        "status": "UNAVAILABLE"
        if not available
        else "OVERLAP_DETECTED"
        if common
        else "NO_EXACT_OVERLAP",
        "splits": summaries,
        "overlap_unique_inputs": len(common),
        "eval_rows_overlapping": sum(fingerprints["eval"][x] for x in common),
        "train_rows_overlapping": sum(fingerprints["train"][x] for x in common),
        "overlap_fingerprints": sorted(common)[:20],
        "overlap_fingerprint_limit": 20,
    }


def energy_fields(start_mj, end_mj):
    """A GPU-0 device-total observation, not isolated process or whole-node energy."""
    valid = type(start_mj) is int and type(end_mj) is int and 0 <= start_mj <= end_mj
    return {
        "energy_j": (end_mj - start_mj) / 1000.0 if valid else None,
        "energy_label": "MEASURED" if valid else "NOT_MEASURED",
        "energy_unit": "J",
        "energy_counter_unit": "mJ",
        "energy_counter_start": start_mj,
        "energy_counter_end": end_mj,
        "energy_scope": "NVML device index 0 total; not process-isolated or whole-node",
        "energy_reason": None if valid else "counter unavailable, invalid, or reset",
    }


def resolve_revision(api, repo, repo_type, revision=None):
    """Resolve once before reading bytes; every later read uses this immutable SHA."""
    info = api.repo_info(repo_id=repo, repo_type=repo_type, revision=revision)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("provider returned no immutable repository revision")
    return sha


def measure(model, rows, label, tok, en, max_length, torch_module):
    e0 = en.read()
    w0 = time.monotonic()
    total_loss, input_tokens, predicted_tokens, correct, count = 0.0, 0, 0, 0, 0
    model.eval()
    with torch_module.no_grad():
        for ex in rows:
            enc = encode_row(ex, tok, max_length)
            if enc is None:
                continue
            enc = enc.to(model.device)
            length = enc["input_ids"].shape[1]
            if length < 2:
                continue
            out = model(**enc, labels=enc["input_ids"])
            loss = float(out.loss.item())
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("non-finite or negative model loss")
            # Causal-LM labels are shifted: L input tokens produce L-1 targets.
            targets_n = length - 1
            total_loss += loss * targets_n
            input_tokens += length
            predicted_tokens += targets_n
            logits = out.logits[:, :-1]
            targets = enc["input_ids"][:, 1:]
            row_correct = int((logits.argmax(-1) == targets).sum())
            if not 0 <= row_correct <= targets_n:
                raise ValueError("invalid correct-token count")
            correct += row_correct
            count += 1
            if count % 200 == 0:
                print(
                    f"[{label}] {count}/{len(rows)} loss={total_loss / predicted_tokens:.4f}",
                    flush=True,
                )
    e1 = en.read()
    if not count or not predicted_tokens:
        raise ValueError("no scorable rows or predicted tokens; result is unavailable")
    mean = total_loss / predicted_tokens
    if not math.isfinite(mean):
        raise ValueError("non-finite aggregate loss")
    try:
        perplexity = math.exp(mean)
    except OverflowError:
        perplexity = None
    res = {
        "measurement_class": FIT_CLASS,
        "rows": count,
        "rows_offered": len(rows),
        "rows_skipped": len(rows) - count,
        "tokens": input_tokens,
        "predicted_tokens": predicted_tokens,
        "mean_loss": mean,
        "perplexity": perplexity,
        "perplexity_available": perplexity is not None,
        "correct_tokens": correct,
        "token_accuracy": correct / predicted_tokens,
        "seconds": round(time.monotonic() - w0, 1),
        **energy_fields(e0, e1),
    }
    print(f"[{label}] DONE {json.dumps(res, allow_nan=False)}", flush=True)
    return res


MAX_BUNDLE_FILE_BYTES = 2 * 1024 * 1024


def _strict_json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    def reject_constant(_):
        raise ValueError("non-finite JSON value")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("bundle JSON must be an object")
    return value


def _read_bundle_file(directory, name):
    path = directory / name
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("bundle members must be regular non-symlink files")
    if path.resolve().parent != directory.resolve():
        raise ValueError("bundle member escapes its directory")
    if path.stat().st_size > MAX_BUNDLE_FILE_BYTES:
        raise ValueError("bundle member exceeds byte bound")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BUNDLE_FILE_BYTES + 1)
    if len(raw) > MAX_BUNDLE_FILE_BYTES:
        raise ValueError("bundle member exceeds byte bound")
    return raw


def _finite(value, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("measurement must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError("measurement below bound")
    if maximum is not None and value > maximum:
        raise ValueError("measurement above bound")
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("measurement count must be an integer within bounds")
    return value


def _same(actual, expected):
    if not math.isclose(_finite(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("inconsistent derived measurement")


def validate_split_audit(result):
    """Validate reported coverage, not the truth of self-reported execution.

    Older v2 bundles may still be archived unchanged. They do not acquire an
    overlap audit or a new measurement-class field during publication retry.
    """
    if not any(
        key in result
        for key in (
            "split_audit",
            "measurement_class",
            "generalization_assessed",
            "overfitting_absence_established",
        )
    ):
        if any(
            "measurement_class" in result[name]
            for name in ("eval_base", "eval_adapter", "train_ref_adapter")
        ):
            raise ValueError("incomplete measurement scope")
        return
    if (
        result.get("measurement_class") != FIT_CLASS
        or result.get("generalization_assessed") is not False
        or result.get("overfitting_absence_established") is not False
    ):
        raise ValueError(
            "fit measurement cannot establish generalization or no overfit"
        )
    for name in ("eval_base", "eval_adapter", "train_ref_adapter"):
        if result[name].get("measurement_class") != FIT_CLASS:
            raise ValueError("each measurement must carry its fit classification")
    audit = result.get("split_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != SPLIT_AUDIT_SCHEMA
        or audit.get("status") != "NO_EXACT_OVERLAP"
        or audit.get("scope") != SPLIT_AUDIT_SCOPE
        or _integer(audit.get("max_length"), 2) != result["max_length"]
    ):
        raise ValueError("completed measurement requires a usable split audit")
    splits = audit.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("missing split audit coverage")
    for name in ("eval", "train"):
        entry = splits.get(name)
        if not isinstance(entry, dict):
            raise ValueError("missing split audit coverage")
        offered = _integer(entry.get("rows_scanned"), 1)
        scorable = _integer(entry.get("scorable_rows"), 1)
        skipped = _integer(entry.get("unscorable_rows"))
        unique = _integer(entry.get("unique_inputs"), 1)
        duplicates = _integer(entry.get("duplicate_rows"))
        if offered != scorable + skipped or scorable != unique + duplicates:
            raise ValueError("split audit coverage is inconsistent")
        measured = result["eval_base" if name == "eval" else "train_ref_adapter"]
        if name == "eval":
            if offered != measured["rows_offered"] or scorable != measured["rows"]:
                raise ValueError("audited and measured evaluation coverage differ")
        elif offered < measured["rows_offered"] or scorable < measured["rows"]:
            raise ValueError("train audit cannot cover less than the reference sample")
    for key in (
        "overlap_unique_inputs",
        "eval_rows_overlapping",
        "train_rows_overlapping",
    ):
        if _integer(audit.get(key)) != 0:
            raise ValueError("overlapping inputs cannot qualify a disjoint split")
    if audit.get("overlap_fingerprints") != []:
        raise ValueError("clean split audit carries overlapping fingerprints")
    if _integer(audit.get("overlap_fingerprint_limit"), 1) != 20:
        raise ValueError("unsupported split fingerprint sample bound")


def _safe_prefix(prefix):
    if (
        not isinstance(prefix, str)
        or not prefix
        or any(
            part in ("", ".", "..") or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None
            for part in prefix.split("/")
        )
    ):
        raise ValueError("unsafe receipt prefix")
    return prefix


def validate_completed_result(result, manifest):
    """Check semantics and destination binding, NOT origin authenticity or model quality.

    Hash consistency is not a signature or an independent execution witness.
    Consumers still need an admitted run/generation and trusted source binding.
    """
    if (
        result.get("schema") != SCHEMA
        or result.get("status") != "OK"
        or result.get("kind") != "MEASURED"
    ):
        raise ValueError("only completed v2 measurements may be published")
    if manifest.get("receipt_schema") != SCHEMA or manifest.get("status") != "OK":
        raise ValueError("manifest/receipt status or schema mismatch")
    if "source_revision" in result or "launch_contract_sha256" in result:
        for field, length in (("source_revision", 40), ("launch_contract_sha256", 64)):
            value = result.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None
            ):
                raise ValueError("incomplete launch provenance binding")
    repo = manifest.get("repo_id")
    if (
        not isinstance(repo, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo)
        is None
    ):
        raise ValueError("invalid receipt repository identity")
    for receipt_field, manifest_field in (
        ("receipts_repo", "repo_id"),
        ("receipts_repo_type", "repo_type"),
        ("receipt_prefix", "prefix"),
    ):
        if result.get(receipt_field) != manifest.get(manifest_field):
            raise ValueError("preserved receipt destination mismatch")
    run_id = result.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id) is None
    ):
        raise ValueError("missing or invalid attempt identity")
    if _safe_prefix(manifest["prefix"]).rsplit("/", 1)[-1] != run_id:
        raise ValueError("attempt/prefix identity mismatch")
    for field in (
        "base_revision",
        "adapter_repo_revision",
        "dataset_revision",
        "receipt_target_preflight_revision",
    ):
        if (
            not isinstance(result.get(field), str)
            or re.fullmatch(r"[0-9a-f]{40}", result[field]) is None
        ):
            raise ValueError("missing immutable input identity")
    for field in ("base", "adapter_repo", "adapter_dir", "dataset", "method"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError("missing measurement contract")
    stamps = []
    for field in ("started_utc", "completed_utc"):
        value = result.get(field)
        if not isinstance(value, str):
            raise ValueError("missing measurement timestamp")
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("measurement timestamp must include timezone")
        stamps.append(parsed)
    if stamps[1] < stamps[0]:
        raise ValueError("completion precedes start; clock reconciliation required")
    _finite(result.get("wall_seconds"), 0)
    max_length = _integer(result.get("max_length"), 2)
    _integer(result.get("seed"), 0)
    if result.get("heldout_quality_measured") is not False:
        raise ValueError("this harness cannot qualify heldout quality")
    pins = result.get("pins", {})
    for split in ("eval", "train"):
        observed = pins.get(split + "_sha256")
        expected = pins.get(split + "_sha256_expected")
        if (
            not isinstance(observed, str)
            or re.fullmatch(r"[0-9a-f]{64}", observed) is None
        ):
            raise ValueError("missing split byte identity")
        if expected is not None and expected != observed:
            raise ValueError("split identity mismatch")
    for field, row_field in (
        ("eval_base", "eval"),
        ("eval_adapter", "eval"),
        ("train_ref_adapter", "train_ref"),
    ):
        metrics = result.get(field)
        if not isinstance(metrics, dict):
            raise ValueError("incomplete benchmark measurements")
        rows = _integer(metrics.get("rows"), 1)
        offered = _integer(metrics.get("rows_offered"), rows)
        skipped = _integer(metrics.get("rows_skipped"))
        predicted = _integer(metrics.get("predicted_tokens"), rows)
        tokens = _integer(metrics.get("tokens"), 2 * rows)
        correct = _integer(metrics.get("correct_tokens"))
        if (
            offered != rows + skipped
            or tokens != predicted + rows
            or tokens > rows * max_length
            or correct > predicted
            or _integer(result.get("rows", {}).get(row_field), 1) != offered
        ):
            raise ValueError("inconsistent benchmark coverage")
        loss = _finite(metrics.get("mean_loss"), 0)
        _same(_finite(metrics.get("token_accuracy"), 0, 1), correct / predicted)
        _finite(metrics.get("seconds"), 0)
        try:
            expected_ppl = math.exp(loss)
        except OverflowError:
            expected_ppl = None
        if metrics.get("perplexity_available") is not (expected_ppl is not None):
            raise ValueError("inconsistent perplexity availability")
        if expected_ppl is None:
            if metrics.get("perplexity") is not None:
                raise ValueError("unavailable perplexity must remain null")
        else:
            _same(metrics.get("perplexity"), expected_ppl)
        start, end = (
            metrics.get("energy_counter_start"),
            metrics.get("energy_counter_end"),
        )
        for counter in (start, end):
            if counter is not None:
                _integer(counter)
        expected_energy = energy_fields(start, end)
        if expected_energy["energy_j"] is not None:
            _finite(metrics.get("energy_j"), 0)
        for name in (
            "energy_j",
            "energy_label",
            "energy_unit",
            "energy_counter_unit",
            "energy_scope",
        ):
            if metrics.get(name) != expected_energy[name]:
                raise ValueError("inconsistent energy units, scope or measurement")
    for field in ("rows", "rows_offered", "rows_skipped", "tokens", "predicted_tokens"):
        if result["eval_base"][field] != result["eval_adapter"][field]:
            raise ValueError("base/adapter comparison coverage mismatch")
    validate_split_audit(result)
    base = result["eval_base"]["mean_loss"]
    adapter = result["eval_adapter"]["mean_loss"]
    train = result["train_ref_adapter"]["mean_loss"]
    criterion, improvement = (
        result.get("criterion_1", {}),
        result.get("improvement_vs_base", {}),
    )
    _same(criterion.get("eval_loss"), adapter)
    _same(criterion.get("train_ref_loss"), train)
    if criterion.get("pass") is not (adapter <= train + 1.0):
        raise ValueError("inconsistent criterion verdict")
    for key, value in (
        ("eval_loss_base", base),
        ("eval_loss_adapter", adapter),
        ("delta", base - adapter),
    ):
        _same(improvement.get(key), value)


def save_bundle(rec, directory, repo, repo_type, prefix):
    """Preserve computed evidence before any upload; never overwrite prior artifacts."""
    directory = Path(directory)
    rec.update(receipts_repo=repo, receipts_repo_type=repo_type, receipt_prefix=prefix)
    source = Path(__file__).read_bytes()
    rec["harness_sha256"] = hashlib.sha256(source).hexdigest()
    result = (
        json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    entries = {"result.json": result, "bench_eval.py": source}
    manifest = {
        "schema": "szl.bench-local-bundle/v1",
        "receipt_schema": rec["schema"],
        "status": rec["status"],
        "repo_id": repo,
        "repo_type": repo_type,
        "prefix": prefix,
        "sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in entries.items()
        },
    }
    directory.mkdir(parents=True, exist_ok=False)
    for name, raw in entries.items():
        with (directory / name).open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    with (directory / "bundle.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    return directory


def publish_bundle(directory, api, download):
    """Publish a preserved bundle in one commit, then verify exact-commit bytes.

    This function does not load models or recompute measurements. A failed upload
    leaves the bundle available for an explicit owner retry. Existing remote bytes
    are reused only when identical; other receipts are never silently overwritten.
    """
    from huggingface_hub import CommitOperationAdd

    directory = Path(directory)
    manifest = _strict_json(_read_bundle_file(directory, "bundle.json"))
    if manifest.get("schema") != "szl.bench-local-bundle/v1":
        raise ValueError("unrecognized local bundle")
    if manifest.get("repo_type") not in ("model", "dataset", "space"):
        raise ValueError("invalid receipt repository type")
    prefix = _safe_prefix(manifest["prefix"])
    expected_names = {"result.json", "bench_eval.py"}
    if set(manifest.get("sha256", {})) != expected_names:
        raise ValueError("incomplete bundle manifest")
    entries = {}
    for name in sorted(expected_names):
        raw = _read_bundle_file(directory, name)
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"][name]:
            raise ValueError("local bundle hash mismatch")
        entries[prefix + "/" + name] = raw
    result = _strict_json(entries[prefix + "/result.json"])
    validate_completed_result(result, manifest)
    if result.get("harness_sha256") != manifest["sha256"]["bench_eval.py"]:
        raise ValueError("receipt/harness binding mismatch")
    repo, repo_type = manifest["repo_id"], manifest["repo_type"]
    parent = resolve_revision(api, repo, repo_type)
    present = [
        api.file_exists(
            repo_id=repo, filename=name, repo_type=repo_type, revision=parent
        )
        for name in entries
    ]
    if any(present) and not all(present):
        raise ValueError(
            "partial existing remote bundle; owner reconciliation required"
        )
    if all(present):
        revision = parent
    else:
        commit = api.create_commit(
            repo_id=repo,
            repo_type=repo_type,
            parent_commit=parent,
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw)
                for name, raw in entries.items()
            ],
            commit_message="Publish completed benchmark evidence bundle",
        )
        revision = getattr(commit, "oid", None)
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise ValueError("publication returned no immutable commit")
    for name, raw in entries.items():
        path = download(
            repo_id=repo, filename=name, repo_type=repo_type, revision=revision
        )
        if Path(path).read_bytes() != raw:
            raise ValueError(
                "remote bundle byte mismatch; no recovery/completion claim"
            )
    return revision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument(
        "--base-revision", default=None, help="pinned revision of the base model"
    )
    ap.add_argument("--adapter-repo")
    ap.add_argument("--adapter-dir", default="final")
    ap.add_argument("--dataset")
    ap.add_argument("--eval-file")
    ap.add_argument("--eval-sha256", default=None)
    ap.add_argument("--train-file")
    ap.add_argument("--train-sha256", default=None)
    ap.add_argument("--train-ref-rows", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--receipts-repo")
    ap.add_argument("--receipt-dir")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adapter-revision", default=None)
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument(
        "--receipts-repo-type", choices=("model", "dataset", "space"), default="dataset"
    )
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--publish-only",
        metavar="BUNDLE_DIR",
        help="retry saved evidence publication without inference",
    )
    ap.add_argument(
        "--run-id", default=None, help="controller-bound unique attempt; never reused"
    )
    ap.add_argument("--source-revision", default=None)
    ap.add_argument("--launch-contract-sha256", default=None)
    a = ap.parse_args()
    if a.run_id is not None and re.fullmatch(r"[0-9a-f]{32}", a.run_id) is None:
        ap.error("run-id must be a 32-character lowercase hexadecimal attempt identity")
    binding = (a.source_revision, a.launch_contract_sha256)
    if any(binding) and (
        not all(binding)
        or a.run_id is None
        or re.fullmatch(r"[0-9a-f]{40}", a.source_revision or "") is None
        or re.fullmatch(r"[0-9a-f]{64}", a.launch_contract_sha256 or "") is None
    ):
        ap.error(
            "launch binding requires run-id, immutable source revision and contract hash"
        )
    if a.publish_only and (a.run_id is not None or any(binding)):
        ap.error("publication-only must retain the original bundle identity")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if a.publish_only:
        try:
            revision = publish_bundle(a.publish_only, api, hf_hub_download)
        except Exception as exc:
            print("PUBLICATION_UNAVAILABLE:", type(exc).__name__, file=sys.stderr)
            return 7
        print("PUBLICATION_VERIFIED " + revision, flush=True)
        return 0
    required = (
        "base",
        "adapter_repo",
        "dataset",
        "eval_file",
        "train_file",
        "receipts_repo",
        "receipt_dir",
    )
    if any(not getattr(a, name) for name in required):
        ap.error(
            "measurement mode requires base, adapter, dataset, split files and receipt destination"
        )
    if a.max_length < 2 or a.train_ref_rows < 1:
        ap.error("max-length must be >=2 and train-ref-rows must be positive")
    run_id = a.run_id if a.run_id is not None else uuid.uuid4().hex
    output_dir = Path(a.output_dir) if a.output_dir else Path("bench-receipts") / run_id
    if output_dir.exists() or output_dir.is_symlink():
        ap.error(
            "output bundle already exists; reconcile or use --publish-only, never remeasure it"
        )
    # Separate every attempt; a result from an earlier run must not reap a newer job.
    receipt_prefix = a.receipt_dir.rstrip("/") + "/" + run_id
    import torch

    torch.manual_seed(a.seed)
    started = utcnow()
    t0 = time.monotonic()
    rec = {
        "schema": SCHEMA,
        "measurement_class": FIT_CLASS,
        "generalization_assessed": False,
        "overfitting_absence_established": False,
        "kind": "UNAVAILABLE",
        "run_id": run_id,
        "started_utc": started,
        "base": a.base,
        "base_revision": a.base_revision,
        "adapter_repo": a.adapter_repo,
        "adapter_dir": a.adapter_dir,
        "dataset": a.dataset,
        "max_length": a.max_length,
        "seed": a.seed,
        "method": (
            "text=apply_chat_template(messages, tokenize=False); "
            "tokenize+truncate to max_length; shifted-target-token-weighted causal-LM loss over full "
            "rendered text; identical pipeline for all three measurements"
        ),
    }

    if all(binding):
        # Operator-provided provenance references, not a signature or approval.
        rec["source_revision"] = a.source_revision
        rec["launch_contract_sha256"] = a.launch_contract_sha256

    def fail(msg, extra=None):
        rec.update({"status": "ERROR", "error": msg, "completed_utc": utcnow()})
        rec["kind"] = "PARTIAL" if "eval_base" in rec else "UNAVAILABLE"
        if extra:
            rec.update(extra)
        try:
            save_bundle(
                rec, output_dir, a.receipts_repo, a.receipts_repo_type, receipt_prefix
            )
        except Exception as exc:
            print("LOCAL_EVIDENCE_UNAVAILABLE:", type(exc).__name__, file=sys.stderr)
        print("FAIL-CLOSED:", msg, "local_bundle:", output_dir, file=sys.stderr)
        return 7

    try:
        # Verify the typed receipt target is readable before model hydration.
        # This is not proof of write permission or later publication success.
        receipt_target_revision = resolve_revision(
            api, a.receipts_repo, a.receipts_repo_type
        )
        rec["receipt_target_preflight_revision"] = receipt_target_revision
        a.base_revision = resolve_revision(api, a.base, "model", a.base_revision)
        adapter_revision = resolve_revision(
            api, a.adapter_repo, "model", a.adapter_revision
        )
        dataset_revision = resolve_revision(
            api, a.dataset, "dataset", a.dataset_revision
        )
    except Exception as exc:
        return fail("immutable input resolution failed: " + type(exc).__name__)
    rec.update(
        base_revision=a.base_revision,
        adapter_repo_revision=adapter_revision,
        dataset_revision=dataset_revision,
        receipt_prefix=receipt_prefix,
        receipts_repo_type=a.receipts_repo_type,
    )

    try:
        # --- energy meter (declared, never fabricated)
        en = Energy()
        rec["gpu"] = en.name

        # --- pins + fail-closed checks
        eval_path = hf_hub_download(
            a.dataset, a.eval_file, repo_type="dataset", revision=dataset_revision
        )
        train_path = hf_hub_download(
            a.dataset, a.train_file, repo_type="dataset", revision=dataset_revision
        )
        ev_sha, tr_sha = sha256_file(eval_path), sha256_file(train_path)
        rec["pins"] = {
            "eval_sha256": ev_sha,
            "train_sha256": tr_sha,
            "eval_sha256_expected": a.eval_sha256,
            "train_sha256_expected": a.train_sha256,
        }
        if a.eval_sha256 and ev_sha != a.eval_sha256:
            return fail("eval file sha256 mismatch", rec)
        if a.train_sha256 and tr_sha != a.train_sha256:
            return fail("train file sha256 mismatch", rec)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
        if tok.chat_template is None:
            return fail("base tokenizer has no chat template; pipeline undefined", rec)

        eval_rows = load_jsonl(eval_path)
        train_rows = load_jsonl(train_path, a.train_ref_rows)
        if not eval_rows or not train_rows:
            return fail(
                "empty split after load",
                {"eval_rows": len(eval_rows), "train_ref_rows": len(train_rows)},
            )
        rec["rows"] = {
            "eval": len(eval_rows),
            "train_ref": len(train_rows),
            "train_ref_note": "first-N deterministic rows of train file",
        }

        rec["split_audit"] = audit_split_overlap(
            eval_rows, train_path, tok, a.max_length
        )
        if sha256_file(eval_path) != ev_sha or sha256_file(train_path) != tr_sha:
            return fail("split files changed during the overlap audit")
        if rec["split_audit"]["status"] != "NO_EXACT_OVERLAP":
            return fail("train/eval split audit: " + rec["split_audit"]["status"])

        # Only hydrate model weights after the entire split audit has succeeded.
        try:
            adir = snapshot_download(
                a.adapter_repo,
                revision=adapter_revision,
                allow_patterns=[a.adapter_dir + "/*"],
            )
        except Exception as e:
            return fail("adapter snapshot failed: " + repr(e)[:200], rec)
        adapter_path = os.path.join(adir, a.adapter_dir)
        if not os.path.isfile(os.path.join(adapter_path, "adapter_model.safetensors")):
            return fail("adapter_model.safetensors missing under " + a.adapter_dir, rec)

        kw = {"dtype": torch.bfloat16, "device_map": "auto"}
        if a.base_revision:
            kw["revision"] = a.base_revision
        base = AutoModelForCausalLM.from_pretrained(a.base, **kw)
        rec["eval_base"] = measure(
            base, eval_rows, "eval/base", tok, en, a.max_length, torch
        )

        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        rec["eval_adapter"] = measure(
            model, eval_rows, "eval/adapter", tok, en, a.max_length, torch
        )
        rec["train_ref_adapter"] = measure(
            model, train_rows, "trainref/adapter", tok, en, a.max_length, torch
        )

    except Exception as exc:
        return fail("measurement unavailable: " + type(exc).__name__)

    rec["criterion_1"] = {
        "rule": "eval_repo loss (adapter) <= train-ref loss (adapter) + 1.0",
        "eval_loss": rec["eval_adapter"]["mean_loss"],
        "train_ref_loss": rec["train_ref_adapter"]["mean_loss"],
        "pass": rec["eval_adapter"]["mean_loss"]
        <= rec["train_ref_adapter"]["mean_loss"] + 1.0,
    }
    rec["improvement_vs_base"] = {
        "eval_loss_base": rec["eval_base"]["mean_loss"],
        "eval_loss_adapter": rec["eval_adapter"]["mean_loss"],
        "delta": rec["eval_base"]["mean_loss"] - rec["eval_adapter"]["mean_loss"],
    }
    rec["heldout_quality_measured"] = False
    rec["heldout_quality_note"] = (
        "criteria 2-5 (contamination official check, refusal "
        "preservation, neutral capability probe, invented-ID rate) "
        "not computed by this bench; heldout_quality stays false "
        "until all are computed and recorded (fail-closed)."
    )
    rec["status"] = "OK"
    rec["kind"] = "MEASURED"
    rec["completed_utc"] = utcnow()
    rec["wall_seconds"] = round(time.monotonic() - t0, 1)

    try:
        save_bundle(
            rec, output_dir, a.receipts_repo, a.receipts_repo_type, receipt_prefix
        )
        revision = publish_bundle(output_dir, api, hf_hub_download)
    except Exception as exc:
        print(
            "PUBLICATION_UNAVAILABLE:",
            type(exc).__name__,
            "local_bundle:",
            output_dir,
            file=sys.stderr,
        )
        return 7
    print(
        "BENCH_OK "
        + json.dumps(
            {
                "criterion_1": rec["criterion_1"],
                "publication_revision": revision,
                "receipt_prefix": receipt_prefix,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
