#!/usr/bin/env python3
# Copyright 2026 SZL Holdings -- SPDX-License-Identifier: Apache-2.0
"""Governed post-train bench for Salad-trained adapters (v1.0).

Same-method comparison of eval-split loss vs train-split reference loss
(v4full acceptance criterion 1: eval_repo loss <= final train loss + 1.0),
plus base-vs-adapter improvement on the identical pipeline, MEASURED energy
(NVML cumulative-counter joule deltas), full pinning of base revision, adapter
revision, dataset file sha256s, and a fail-closed receipt to the Hub.

Mirrors tools/salad/salad_worker.py's exact text pipeline:
  text = tokenizer.apply_chat_template(messages, tokenize=False)
  truncate to max_length (worker SFTConfig max_length=2048), causal-LM loss
over the full rendered text. Both splits are computed with THIS script's one
method, so criterion 1 is internally consistent; the method is declared in
the receipt and never claimed to be TRL's internal loss verbatim.
"""
import argparse, datetime, hashlib, json, os, sys, time
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

SCHEMA = "szl.salad-bench/v1"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


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
        return int(self.p.nvmlDeviceGetTotalEnergyConsumption(self.h))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision", default=None, help="pinned revision of the base model")
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--adapter-dir", default="final")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--eval-sha256", default=None)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--train-sha256", default=None)
    ap.add_argument("--train-ref-rows", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--receipts-repo", required=True)
    ap.add_argument("--receipt-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    started = utcnow()
    t0 = time.time()
    rec = {"schema": SCHEMA, "kind": "MEASURED", "started_utc": started,
           "base": a.base, "base_revision": a.base_revision,
           "adapter_repo": a.adapter_repo, "adapter_dir": a.adapter_dir,
           "dataset": a.dataset, "max_length": a.max_length, "seed": a.seed,
           "method": ("text=apply_chat_template(messages, tokenize=False); "
                      "tokenize+truncate to max_length; causal-LM loss over full "
                      "rendered text; identical pipeline for all three measurements")}

    def fail(msg, extra=None):
        rec.update({"status": "ERROR", "error": msg, "completed_utc": utcnow()})
        if extra:
            rec.update(extra)
        try:
            api.upload_file(path_or_fileobj=json.dumps(rec, indent=1).encode(),
                            path_in_repo=a.receipt_dir + "/result.json",
                            repo_id=a.receipts_repo, repo_type="dataset")
        except Exception as e:
            print("RECEIPT UPLOAD FAILED TOO:", repr(e), file=sys.stderr)
        print("FAIL-CLOSED:", msg, file=sys.stderr)
        return 7

    # --- energy meter (declared, never fabricated)
    en = Energy()
    rec["gpu"] = en.name

    # --- pins + fail-closed checks
    eval_path = hf_hub_download(a.dataset, a.eval_file, repo_type="dataset")
    train_path = hf_hub_download(a.dataset, a.train_file, repo_type="dataset")
    ev_sha, tr_sha = sha256_file(eval_path), sha256_file(train_path)
    rec["pins"] = {"eval_sha256": ev_sha, "train_sha256": tr_sha,
                   "eval_sha256_expected": a.eval_sha256,
                   "train_sha256_expected": a.train_sha256}
    if a.eval_sha256 and ev_sha != a.eval_sha256:
        return fail("eval file sha256 mismatch", rec)
    if a.train_sha256 and tr_sha != a.train_sha256:
        return fail("train file sha256 mismatch", rec)

    try:
        adir = snapshot_download(a.adapter_repo, allow_patterns=[a.adapter_dir + "/*"])
    except Exception as e:
        return fail("adapter snapshot failed: " + repr(e)[:200], rec)
    adapter_path = os.path.join(adir, a.adapter_dir)
    if not os.path.isfile(os.path.join(adapter_path, "adapter_model.safetensors")):
        return fail("adapter_model.safetensors missing under " + a.adapter_dir, rec)
    try:
        info = api.model_info(a.adapter_repo)
        rec["adapter_repo_revision"] = getattr(info, "sha", None)
    except Exception:
        rec["adapter_repo_revision"] = None

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    if tok.chat_template is None:
        return fail("base tokenizer has no chat template; pipeline undefined", rec)

    eval_rows = load_jsonl(eval_path)
    train_rows = load_jsonl(train_path, a.train_ref_rows)
    if not eval_rows or not train_rows:
        return fail("empty split after load", {"eval_rows": len(eval_rows), "train_ref_rows": len(train_rows)})
    rec["rows"] = {"eval": len(eval_rows), "train_ref": len(train_rows),
                   "train_ref_note": "first-N deterministic rows of train file"}

    def measure(model, rows, label):
        e0 = en.read()
        w0 = time.time()
        tot_loss, tot_tok, tot_correct, n = 0.0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for i, ex in enumerate(rows):
                msgs = ex.get("messages")
                if not msgs:
                    continue
                text = tok.apply_chat_template(msgs, tokenize=False)
                enc = tok(text, truncation=True, max_length=a.max_length,
                          return_tensors="pt").to(model.device)
                if enc["input_ids"].shape[1] < 2:
                    continue
                out = model(**enc, labels=enc["input_ids"])
                L = enc["input_ids"].shape[1]
                tot_loss += out.loss.item() * L
                tot_tok += L
                # token accuracy on next-token prediction (argmax vs label)
                logits = out.logits[:, :-1]
                targets = enc["input_ids"][:, 1:]
                tot_correct += int((logits.argmax(-1) == targets).sum())
                n += 1
                if n % 200 == 0:
                    print(f"[{label}] {n}/{len(rows)} loss={tot_loss/max(tot_tok,1):.4f}", flush=True)
        e1 = en.read()
        dt = time.time() - w0
        res = {"rows": n, "tokens": tot_tok, "mean_loss": tot_loss / max(tot_tok, 1),
               "perplexity": float(torch.exp(torch.tensor(tot_loss / max(tot_tok, 1)))),
               "token_accuracy": tot_correct / max(tot_tok - n, 1),
               "seconds": round(dt, 1),
               "energy_j": (e1 - e0) if (e0 is not None and e1 is not None) else None,
               "energy_label": "MEASURED" if (e0 is not None and e1 is not None) else "NOT_MEASURED"}
        print(f"[{label}] DONE {json.dumps(res)}", flush=True)
        return res

    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if a.base_revision:
        kw["revision"] = a.base_revision
    base = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    rec["eval_base"] = measure(base, eval_rows, "eval/base")

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    rec["eval_adapter"] = measure(model, eval_rows, "eval/adapter")
    rec["train_ref_adapter"] = measure(model, train_rows, "trainref/adapter")

    rec["criterion_1"] = {
        "rule": "eval_repo loss (adapter) <= train-ref loss (adapter) + 1.0",
        "eval_loss": rec["eval_adapter"]["mean_loss"],
        "train_ref_loss": rec["train_ref_adapter"]["mean_loss"],
        "pass": rec["eval_adapter"]["mean_loss"] <= rec["train_ref_adapter"]["mean_loss"] + 1.0}
    rec["improvement_vs_base"] = {
        "eval_loss_base": rec["eval_base"]["mean_loss"],
        "eval_loss_adapter": rec["eval_adapter"]["mean_loss"],
        "delta": rec["eval_base"]["mean_loss"] - rec["eval_adapter"]["mean_loss"]}
    rec["heldout_quality_measured"] = False
    rec["heldout_quality_note"] = ("criteria 2-5 (contamination official check, refusal "
                                   "preservation, neutral capability probe, invented-ID rate) "
                                   "not computed by this bench; heldout_quality stays false "
                                   "until all are computed and recorded (fail-closed).")
    rec["status"] = "OK"
    rec["completed_utc"] = utcnow()
    rec["wall_seconds"] = round(time.time() - t0, 1)

    api.upload_file(path_or_fileobj=json.dumps(rec, indent=1).encode(),
                    path_in_repo=a.receipt_dir + "/result.json",
                    repo_id=a.receipts_repo, repo_type="dataset")
    with open(__file__, "rb") as f:
        api.upload_file(path_or_fileobj=f.read(),
                        path_in_repo=a.receipt_dir + "/bench_eval.py",
                        repo_id=a.receipts_repo, repo_type="dataset")
    print("BENCH_OK " + json.dumps(rec["criterion_1"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
