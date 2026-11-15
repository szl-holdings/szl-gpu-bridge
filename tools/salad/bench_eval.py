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


def load_jsonl(path, max_rows=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


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
            msgs = ex.get("messages")
            if not msgs:
                continue
            text = tok.apply_chat_template(msgs, tokenize=False)
            enc = tok(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(model.device)
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
        "rows": count,
        "rows_offered": len(rows),
        "rows_skipped": len(rows) - count,
        "tokens": input_tokens,
        "predicted_tokens": predicted_tokens,
        "mean_loss": mean,
        "perplexity": perplexity,
        "perplexity_available": perplexity is not None,
        "token_accuracy": correct / predicted_tokens,
        "seconds": round(time.monotonic() - w0, 1),
        **energy_fields(e0, e1),
    }
    print(f"[{label}] DONE {json.dumps(res, allow_nan=False)}", flush=True)
    return res


def save_bundle(rec, directory, repo, repo_type, prefix):
    """Preserve computed evidence before any upload; never overwrite prior artifacts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
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
    for name, raw in entries.items():
        with (directory / name).open("xb") as f:
            f.write(raw)
    with (directory / "bundle.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return directory


def publish_bundle(directory, api, download):
    """Publish a preserved bundle in one commit, then verify exact-commit bytes.

    This function does not load models or recompute measurements. A failed upload
    leaves the bundle available for an explicit owner retry. Existing remote bytes
    are reused only when identical; other receipts are never silently overwritten.
    """
    from huggingface_hub import CommitOperationAdd

    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "szl.bench-local-bundle/v1":
        raise ValueError("unrecognized local bundle")
    if manifest.get("repo_type") not in ("model", "dataset", "space"):
        raise ValueError("invalid receipt repository type")
    prefix = manifest["prefix"].strip("/")
    if (
        not prefix
        or any(x in ("", ".", "..") for x in prefix.split("/"))
        or "\\" in prefix
    ):
        raise ValueError("unsafe receipt prefix")
    expected_names = {"result.json", "bench_eval.py"}
    if set(manifest.get("sha256", {})) != expected_names:
        raise ValueError("incomplete bundle manifest")
    entries = {}
    for name in sorted(expected_names):
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"][name]:
            raise ValueError("local bundle hash mismatch")
        entries[prefix + "/" + name] = raw
    result = json.loads(entries[prefix + "/result.json"])
    if result.get("schema") != SCHEMA or result.get("status") != "OK":
        raise ValueError("only completed v2 measurements may be published")
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
    a = ap.parse_args()

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
    run_id = uuid.uuid4().hex
    output_dir = Path(a.output_dir) if a.output_dir else Path("bench-receipts") / run_id
    # Separate every attempt; a result from an earlier run must not reap a newer job.
    receipt_prefix = a.receipt_dir.rstrip("/") + "/" + run_id
    import torch

    torch.manual_seed(a.seed)
    started = utcnow()
    t0 = time.monotonic()
    rec = {
        "schema": SCHEMA,
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
