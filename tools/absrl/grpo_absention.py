#!/usr/bin/env python3
"""grpo_absention.py - Abs-RL lane: GRPO abstention training for Salad (STAGED).

The novel play (leader-scan 2026-11-15): honest abstention as a TRAINED
behavior, scored on AbstentionBench, priced per-joule by the v3.2 worker
receipts. This file stages the lane; LAUNCH IS OWNER-GATED (Salad GPU cost) -
no launch, no dispatch, no Hub publish happens from this file existing.

Fail-closed posture (same doctrine as salad_worker.py):
- Requires a dataset with prompt / answer / answerable columns; anything
  else emits a fail-closed receipt and exits non-zero.
- Requires trl>=0.14 (GRPOTrainer); the salad image pins trl==0.13.0, so a
  dedicated absrl image decision is part of the owner-gated launch. A missing
  GRPOTrainer is an honest UNAVAILABLE receipt, never a fabricated run.
- The working repo for resumable checkpoints is created and verified PRIVATE
  before any upload; visibility flips remain human decisions.
- Receipts go to the chain (SZLHOLDINGS/frontier-54-runs); silence is an alarm.
- Energy is NOT reported by this lane yet: it says UNMEASURED rather than
  fabricating joules (the v3.2 zeus window covers the SFT worker lane).
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, sys, time

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/hf-cache")))

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards import abstention_reward

DS_RECEIPTS = "SZLHOLDINGS/frontier-54-runs"
REQUIRED_COLUMNS = ("prompt", "answer", "answerable")
PICKLE_PATTERNS = ["*.pt", "*.pth", "*.bin", "*.pkl", "*.pickle", "*.ckpt",
                   "optimizer.*", "scheduler.*", "rng_state*", "training_args.bin"]


def receipt(kind: str, detail: str, extra: dict | None = None):
    r = {"kind": "szl.absrl-grpo/v1", "detail": detail,
         "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         **(extra or {})}
    blob = json.dumps(r, sort_keys=True)
    r["receipt_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    print(json.dumps(r), flush=True)
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        try:
            HfApi(token=tok).upload_file(
                path_or_fileobj=json.dumps(r, indent=2).encode(),
                path_in_repo=f"mesh/absrl-{r['receipt_sha256'][:12]}.json",
                repo_id=DS_RECEIPTS, repo_type="dataset")
        except Exception as e:
            print(f"receipt upload failed (training continues): {str(e)[:80]}", flush=True)
    return r


class VisibilityError(RuntimeError):
    pass


def verify_private(api, repo_id: str, repo_type: str = "model") -> None:
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        raise VisibilityError(f"cannot verify visibility of {repo_id}: {type(e).__name__}") from None
    if getattr(info, "private", None) is not True:
        raise VisibilityError(
            f"{repo_id} is not private (private={getattr(info, 'private', None)!r}); refusing upload. "
            "Changing visibility is a human decision.")


def ensure_private_repo(api, repo_id: str, token: str, repo_type: str = "model") -> None:
    api.create_repo(repo_id, exist_ok=True, private=True, token=token, repo_type=repo_type)
    verify_private(api, repo_id, repo_type)


def load_reward_dataset(a, api: HfApi):
    kwargs = {"split": "train"}
    if a.dataset_config:
        kwargs["name"] = a.dataset_config
    try:
        ds = load_dataset(a.dataset, **kwargs)
    except Exception as e:
        receipt("fail-closed", f"dataset {a.dataset} failed to load: {str(e)[:100]}")
        raise SystemExit(5)
    missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        receipt("fail-closed",
                f"dataset {a.dataset} lacks required columns {missing} "
                f"(has {ds.column_names}); refusing to train on an unusable reward set")
        raise SystemExit(5)
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True,
                    help="reward dataset with prompt/answer/answerable columns")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--dataset-config", default="")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    a = ap.parse_args()
    if not a.hf_token:
        print("FAIL-CLOSED: HF_TOKEN required (env or --hf-token)", file=sys.stderr)
        return 2

    try:
        from trl import GRPOConfig, GRPOTrainer  # trl>=0.14
    except Exception as e:
        receipt("unavailable",
                f"GRPOTrainer not importable ({type(e).__name__}); the salad image pins "
                "trl==0.13.0 - an absrl image with trl>=0.14 is an owner-gated launch decision. "
                "NOT fabricating a run.")
        return 7

    api = HfApi(token=a.hf_token)
    try:
        ensure_private_repo(api, a.adapter_out, a.hf_token)
    except VisibilityError as e:
        print(f"FAIL-CLOSED: {e}", file=sys.stderr)
        return 3

    ds = load_reward_dataset(a, api)
    receipt("boot", f"absrl GRPO worker staged-boot - base={a.base} dataset={a.dataset}",
            {"rows": len(ds), "columns": sorted(ds.column_names),
             "worker_version": "absrl-v0.1-20261115",
             "energy": "UNMEASURED on this lane; joules are never fabricated"})

    if not torch.cuda.is_available():
        receipt("fail-closed", "CUDA unavailable - refusing to train on CPU")
        return 4

    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="auto")
    cfg = GRPOConfig(
        output_dir=os.path.expanduser("~/ckpt"),
        max_steps=a.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5, warmup_steps=10,
        bf16=True, report_to=[], logging_steps=5,
        max_completion_length=256, num_generations=4,
        save_steps=a.save_steps, save_total_limit=2)
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=abstention_reward,
                          args=cfg, train_dataset=ds, peft_config=peft)
    t0 = time.time()
    trainer.train()

    final = os.path.expanduser("~/ckpt/final")
    trainer.save_model(final)
    try:
        verify_private(api, a.adapter_out)
    except VisibilityError as e:
        print(f"FAIL-CLOSED: {e}; final adapter kept local only", file=sys.stderr)
        return 3
    api.upload_folder(folder_path=final, path_in_repo="final",
                      repo_id=a.adapter_out, repo_type="model",
                      ignore_patterns=PICKLE_PATTERNS)
    receipt("complete", f"absrl GRPO training complete in {round(time.time()-t0, 1)}s",
            {"steps": a.max_steps, "adapter": a.adapter_out,
             "energy": "UNMEASURED on this lane; joules are never fabricated"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
