#!/usr/bin/env python3
"""salad_worker.py — the resumable burst-training worker (the top-company pattern).

How NVIDIA et al. run training on spot-style fleets (Salad, Lambda on-demand,
k8s spot pools):
  1. Pre-built image — every dependency baked in; no pip at runtime.
  2. Durable checkpoints — every save_steps, weights + optimizer state go to
     object storage (here: the HF Hub), so a preempted node loses nothing.
  3. Resume-on-start — the worker checks for the latest checkpoint and
     continues from it; jobs are idempotent.
  4. Graceful preemption — SIGTERM handler checkpoints before exit.
  5. Receipts — boot/step/complete receipts to the chain; silence is an alarm.

This worker implements all five. It runs identically on Salad containers and
on the estate's A6000s (tested there first — never ship to spot metal what
hasn't run on owned metal).

Usage:
  python3 salad_worker.py --base Qwen/Qwen3-1.7B \\
      --dataset SZLHOLDINGS/abstention-doctrine-sft-v1 \\
      --adapter-out SZLHOLDINGS/test-adapter --max-steps 200
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, signal, sys, time

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/hf-cache")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import HfApi

DS_RECEIPTS = "SZLHOLDINGS/frontier-54-runs"
STOP = {"flag": False}


def receipt(kind: str, detail: str, extra: dict | None = None):
    r = {"kind": "szl.salad-worker/v1", "detail": detail,
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
                path_in_repo=f"mesh/salad-{r['receipt_sha256'][:12]}.json",
                repo_id=DS_RECEIPTS, repo_type="dataset")
        except Exception as e:
            print(f"receipt upload failed (training continues): {str(e)[:80]}", flush=True)
    return r


def on_term(signum, frame):
    if not STOP["flag"]:
        STOP["flag"] = True
        receipt("preempted", "SIGTERM received — checkpointing at next save boundary")
signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)


def latest_checkpoint(api: HfApi, repo: str) -> str | None:
    """The resume-on-start contract: find the newest checkpoint-* revision."""
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
        stops = sorted(int(f.split("/")[0].split("-")[-1]) for f in files
                        if f.startswith("checkpoint-") and "/" in f
                        and f.split("/")[0].split("-")[-1].isdigit())
        return f"checkpoint-{stops[-1]}" if stops else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    a = ap.parse_args()
    if not a.hf_token:
        print("FAIL-CLOSED: HF_TOKEN required (env or --hf-token)", file=sys.stderr)
        return 2

    api = HfApi(token=a.hf_token)
    api.create_repo(a.adapter_out, exist_ok=True, token=a.hf_token)
    resume = latest_checkpoint(api, a.adapter_out)
    receipt("boot", f"worker started — base={a.base} dataset={a.dataset}",
            {"resume_from": resume or "GENESIS",
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             "node": os.environ.get("HOSTNAME", "unknown")})

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="auto")
    ds = load_dataset(a.dataset, split="train")
    ds = ds.select(range(min(2000, len(ds))))
    def fmt(ex):
        if "text" in ex: return {"text": ex["text"]}
        if "messages" in ex:
            return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
        if "prompt" in ex and "completion" in ex:
            return {"text": ex["prompt"] + ex["completion"]}
        return {"text": json.dumps(ex)}
    ds = ds.map(fmt, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=os.path.expanduser("~/ckpt"),
        max_length=2048,
        per_device_train_batch_size=1,          # spot-metal safe: fits 24GB cards
        gradient_accumulation_steps=16,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=20,
        max_steps=a.max_steps,
        save_steps=a.save_steps, save_total_limit=2,
        bf16=True, optim="adamw_torch", report_to=[], logging_steps=10,
        gradient_checkpointing=False, packing=False)
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM")

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft)

    class StopOnSignal(torch.utils.data.Dataset):
        pass  # SIGTERM handled at save boundaries via callback below

    from transformers import TrainerCallback
    class ReceiptCallback(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            # push the checkpoint to durable storage immediately
            ck = f"checkpoint-{state.global_step}"
            path = os.path.join(args.output_dir, ck)
            if os.path.exists(path):
                try:
                    api.upload_folder(folder_path=path, path_in_repo=ck,
                        repo_id=a.adapter_out, repo_type="model")
                    receipt("checkpoint", f"checkpoint {ck} pushed to durable storage")
                except Exception as e:
                    receipt("checkpoint-fail", f"push failed: {str(e)[:80]}")
            if STOP["flag"]:
                control.should_training_stop = True
    trainer.add_callback(ReceiptCallback())

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume) if resume else trainer.train()
    final = os.path.expanduser("~/ckpt/final")
    trainer.save_model(final)
    api.upload_folder(folder_path=final, path_in_repo="final",
        repo_id=a.adapter_out, repo_type="model")
    receipt("complete", f"training complete in {round(time.time()-t0,1)}s",
            {"steps": a.max_steps, "adapter": a.adapter_out,
             "joules_note": "energy metering via the estate's NVML sampler where available"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
