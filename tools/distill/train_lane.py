#!/usr/bin/env python3
"""One governed-distill SFT/QLoRA lane where every failure is loud.

This replaces the inline recipes that the Salad distill groups ran. Those
recipes crashed on start (torch upgraded under an old torchvision), passed
``max_seq_length`` to a TRL that only accepts ``max_length``, trained on 1-46
records, published the adapter public, and printed ``LANE_COMPLETE`` and exited
0 right after crashing.

Rules enforced here:
  * refuses to start without an approval id (``SZL_TRAIN_APPROVAL_ID``);
  * refuses a corpus that fails ``distill_corpus.verify`` (too small, no
    ABSTAIN, no held-out split, leakage, hash mismatch);
  * refuses a lane label whose model size disagrees with the base model;
  * SFTConfig gets ``max_length`` or ``max_seq_length`` according to what the
    installed TRL accepts, and fails if it accepts neither (never silently drops);
  * the adapter repo is created PRIVATE;
  * ``LANE_COMPLETE`` is printed, and exit code 0 returned, only after final/
    exists locally, is uploaded, and the remote listing shows the adapter files.
Any other path prints ``LANE_FAILED`` to stderr and exits non-zero.

Heavy imports (torch, trl, peft, ...) happen inside functions so the gating
logic runs on CPU in tests without them.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
MAX_STEPS_CEILING = 5000


class LaneError(Exception):
    """Fatal, fail-closed lane error."""


def read_config(env: dict) -> dict:
    """Validate environment. No defaults for anything that gates spend."""
    missing = [
        k
        for k in (
            "SZL_TRAIN_APPROVAL_ID",
            "LANE_NAME",
            "BASE",
            "CANDIDATE",
            "CORPUS_DATASET",
            "MAX_STEPS",
        )
        if not env.get(k, "").strip()
    ]
    if missing:
        raise LaneError("missing required environment: %s" % ", ".join(missing))
    try:
        max_steps = int(env["MAX_STEPS"])
    except ValueError:
        raise LaneError("MAX_STEPS must be an integer") from None
    if not 1 <= max_steps <= MAX_STEPS_CEILING:
        raise LaneError("MAX_STEPS must be in 1..%d" % MAX_STEPS_CEILING)
    cfg = {
        "approval_id": env["SZL_TRAIN_APPROVAL_ID"].strip(),
        "lane": env["LANE_NAME"].strip(),
        "base": env["BASE"].strip(),
        "candidate": env["CANDIDATE"].strip(),
        "corpus_dataset": env["CORPUS_DATASET"].strip(),
        "max_steps": max_steps,
        "max_length": int(env.get("MAX_LENGTH", "1024")),
        "load_in_4bit": env.get("LOAD_IN_4BIT", "") == "1",
        "out_dir": env.get("OUT_DIR", "/workspace/out/lane"),
        "corpus_dir": env.get("CORPUS_DIR", "/workspace/corpus"),
    }
    check_label_matches_base(cfg["lane"], cfg["base"])
    return cfg


def _sizes(text: str) -> set[str]:
    return {
        m.group(1).lstrip("0") or "0"
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)b(?![a-z])", text.lower())
    }


def check_label_matches_base(lane: str, base: str) -> None:
    """The '4b' lanes actually trained a 1.7B model. Refuse that mismatch."""
    lane_sizes, base_sizes = _sizes(lane), _sizes(base)
    if lane_sizes and base_sizes and not (lane_sizes & base_sizes):
        raise LaneError(
            "lane %r names size %s but base %r is %s"
            % (lane, sorted(lane_sizes), base, sorted(base_sizes))
        )


def sft_config_kwargs(config_cls, max_length: int, **kwargs) -> dict:
    """Add the sequence-length kwarg the installed TRL accepts, or fail."""
    params = inspect.signature(config_cls).parameters
    if "max_length" in params:
        kwargs["max_length"] = max_length
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = max_length
    else:
        raise LaneError("SFTConfig accepts neither max_length nor max_seq_length")
    unknown = [k for k in kwargs if k not in params]
    if unknown:
        raise LaneError("SFTConfig does not accept: %s" % ", ".join(sorted(unknown)))
    return kwargs


def verify_remote_upload(list_repo_files, repo_id: str) -> list[str]:
    """Return remote files; raise unless the adapter files are really there."""
    remote = list(list_repo_files(repo_id))
    absent = [f for f in REQUIRED_ADAPTER_FILES if f not in remote]
    if absent:
        raise LaneError("upload not verified; %s missing from %s" % (absent, repo_id))
    return remote


def check_final_dir(final: Path) -> None:
    for name in REQUIRED_ADAPTER_FILES:
        f = final / name
        if not f.is_file() or f.stat().st_size == 0:
            raise LaneError("final/ is missing or empty: %s" % f)


def run(env: dict, deps) -> dict:
    """Orchestrate one lane. ``deps`` supplies the side-effecting steps.

    deps.fetch_corpus(repo, dest) -> None
    deps.train(cfg, train_records, out_dir) -> dict(metrics)   # writes out_dir/final
    deps.create_private_repo(repo_id) -> None
    deps.upload_folder(folder, repo_id) -> None
    deps.list_repo_files(repo_id) -> list[str]
    """
    cfg = read_config(env)
    corpus_dir = Path(cfg["corpus_dir"])
    deps.fetch_corpus(cfg["corpus_dataset"], corpus_dir)

    import distill_corpus  # shipped next to this file

    try:
        stats = distill_corpus.verify(corpus_dir)
    except distill_corpus.CorpusError as exc:
        raise LaneError("corpus refused: %s" % exc) from exc
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    train = [
        json.loads(line)
        for line in (corpus_dir / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]

    out_dir = Path(cfg["out_dir"])
    t0 = time.time()
    metrics = deps.train(cfg, train, out_dir)
    final = out_dir / "final"
    check_final_dir(final)

    deps.create_private_repo(cfg["candidate"])
    deps.upload_folder(str(final), cfg["candidate"])
    remote = verify_remote_upload(deps.list_repo_files, cfg["candidate"])

    receipt = {
        "lane": cfg["lane"],
        "approval_id": cfg["approval_id"],
        "base": cfg["base"],
        "candidate": cfg["candidate"],
        "private": True,
        "corpus_dataset": cfg["corpus_dataset"],
        "corpus_files": manifest["files"],
        "corpus_stats": stats,
        "max_steps": cfg["max_steps"],
        "metrics": metrics,
        "remote_files": sorted(remote),
        "wall_seconds": round(time.time() - t0, 1),
        "heldout_eval": "NOT_RUN",
        "claims": "adapter trained and uploaded; no quality claim without a held-out eval receipt",
    }
    (out_dir / "completion_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)
    )
    return receipt


class _RealDeps:  # pragma: no cover - needs GPU stack and network
    def __init__(self, token: str):
        self.token = token

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def fetch_corpus(self, repo, dest):
        from huggingface_hub import hf_hub_download

        Path(dest).mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "train.jsonl", "heldout.jsonl"):
            src = hf_hub_download(repo, name, repo_type="dataset", token=self.token)
            (Path(dest) / name).write_bytes(Path(src).read_bytes())

    def train(self, cfg, records, out_dir):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        tok = AutoTokenizer.from_pretrained(cfg["base"], token=self.token)
        kw = {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": self.token}
        if cfg["load_in_4bit"]:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(cfg["base"], **kw)
        ds = Dataset.from_list([{"messages": r["messages"]} for r in records])
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        sft = SFTConfig(
            **sft_config_kwargs(
                SFTConfig,
                cfg["max_length"],
                output_dir=str(out_dir),
                per_device_train_batch_size=1 if cfg["load_in_4bit"] else 4,
                gradient_accumulation_steps=16 if cfg["load_in_4bit"] else 8,
                learning_rate=1e-4 if cfg["load_in_4bit"] else 2e-4,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                bf16=True,
                max_steps=cfg["max_steps"],
                report_to=[],
                packing=False,
            )
        )
        trainer = SFTTrainer(
            model=model,
            args=sft,
            train_dataset=ds,
            peft_config=peft_cfg,
            processing_class=tok,
        )
        result = trainer.train()
        trainer.save_model(str(Path(out_dir) / "final"))
        tok.save_pretrained(str(Path(out_dir) / "final"))
        return {
            k: float(v)
            for k, v in result.metrics.items()
            if isinstance(v, (int, float))
        }

    def create_private_repo(self, repo_id):
        self._api().create_repo(repo_id, private=True, exist_ok=True)

    def upload_folder(self, folder, repo_id):
        self._api().upload_folder(folder_path=folder, repo_id=repo_id)

    def list_repo_files(self, repo_id):
        return self._api().list_repo_files(repo_id)


def main() -> int:  # pragma: no cover - thin wrapper
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
    try:
        if not token:
            raise LaneError("HF_TOKEN not set")
        receipt = run(dict(os.environ), _RealDeps(token))
    except Exception as exc:
        print("LANE_FAILED: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    print("LANE_COMPLETE", receipt["candidate"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
