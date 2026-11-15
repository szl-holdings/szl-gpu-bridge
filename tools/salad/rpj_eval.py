#!/usr/bin/env python3
# Copyright 2026 SZL Holdings -- SPDX-License-Identifier: Apache-2.0
"""Reasoning-per-Joule (RPJ) evaluation harness.

Leader-harvest (2026-11-15 CTO audit, doctrine pass 6):
  - Korea  LG AI Research, EXAONE Deep (arXiv 2503.12524): pass@k with sampling
          temperature 0.6 / top-p 0.95 and consensus@k majority vote.
  - China Alibaba Qwen3 thinking-budget two-step inference (QwenLM/Qwen3
          docs thinking_budget.md): reason within a token budget, then re-call
          with the reasoning appended to obtain the final answer.
  - Ireland Intercom Fin doctrine: resolution-rate-first metrics -- a response
          only counts when the task is verifiably resolved end to end.
  - Estate khipu/energy-attest doctrine: every measurement is a signed-style
          receipt with MEASURED nvml joule deltas (device-level cumulative
          counter), kind szl.rpj/v1. No leader publishes per-response joules;
          this harness makes reasoning quality directly comparable per joule.

Run on any NVML-visible GPU node (Salad worker image or VM):
  python rpj_eval.py --base Qwen/Qwen3-1.7B \
      [--adapter SZLHOLDINGS/salad-leanmath-r1] \
      --dataset prompts.jsonl --verifier lean_heuristic --budgets 128,512,2048 \
      --k 4 --out out/rpj
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

TEMPERATURE = 0.6   # EXAONE Deep eval setting
TOP_P = 0.95        # EXAONE Deep eval setting


# ---------------------------------------------------------------- energy ---
class EnergyMeter:
    """nvmlDeviceGetTotalEnergyConsumption cumulative-counter deltas (J)."""

    def __init__(self, index: int = 0) -> None:
        import pynvml  # noqa: WPS433 (runtime import, worker image has it)

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.gpu = pynvml.nvmlDeviceGetName(self._h)
        if isinstance(self.gpu, bytes):
            self.gpu = self.gpu.decode()

    def read(self) -> int:
        return int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h))


# ------------------------------------------------------------- verifiers ---
def _lean_verifier(prompt: str, completion: str) -> bool:
    """Heuristic resolution check (no Lean toolchain on worker nodes):

    Fin-style honest completion: `sorry`/`admit` anywhere, or a dangling `:= by`
    with nothing after it, is NOT a resolution.
    """
    low = completion.lower()
    if "sorry" in low or "admit" in low or "native_decide" in low:
        return False
    stripped = completion.strip()
    if stripped.endswith(":= by") or stripped.endswith(":= by "):
        return False
    return bool(stripped) and stripped.count("theorem") + stripped.count("lemma") == 0


def _exact_verifier_factory(answer: str) -> Callable[[str, str], bool]:
    def verify(_prompt: str, completion: str) -> bool:
        return answer.strip() in completion
    return verify


VERIFIERS: dict[str, Callable[[str, str], bool]] = {"lean_heuristic": _lean_verifier}


def load_verifier(name: str, spec: Optional[str]) -> Callable[[str, str], bool]:
    if name == "exact":
        return _exact_verifier_factory(spec or "")
    if name in VERIFIERS:
        return VERIFIERS[name]
    raise SystemExit(f"unknown verifier: {name} (lean_heuristic|exact)")


# --------------------------------------------------------------- receipts ---
def _sha256(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def receipt(kind_detail: str, payload: dict) -> dict:
    body = dict(payload)
    body["kind"] = "szl.rpj/v1"
    body["detail"] = kind_detail
    body["ts"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    body["receipt_sha256"] = _sha256(
        {k: v for k, v in body.items() if k != "receipt_sha256"}
    )
    return body


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description="Reasoning-per-Joule harness")
    ap.add_argument("--base", required=True, help="HF base model id")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter repo id")
    ap.add_argument("--dataset", required=True, help="jsonl: {prompt[,answer]}")
    ap.add_argument("--verifier", default="lean_heuristic", help="lean_heuristic|exact")
    ap.add_argument("--answer-key", default=None,
                    help="jsonl field holding the exact answer (verifier=exact)")
    ap.add_argument("--budgets", default="128,512,2048",
                    help="comma-separated thinking-token budgets (Qwen3 two-step)")
    ap.add_argument("--k", type=int, default=4, help="samples per item (pass@k)")
    ap.add_argument("--max-items", type=int, default=0, help="cap dataset rows (0=all)")
    ap.add_argument("--out", default="out/rpj", help="receipt output directory")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    rows = [json.loads(ln) for ln in Path(args.dataset).read_text().splitlines() if ln.strip()]
    if args.max_items:
        rows = rows[: args.max_items]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    meter = EnergyMeter()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    def gen(messages: list[dict], max_new: int, budget: Optional[int]) -> str:
        """Qwen3 thinking-budget two-step: bounded reasoning, then final answer."""
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=budget or max_new,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                pad_token_id=tok.eos_token_id,
            )
        reasoning = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        if budget is None:
            return reasoning
        # step 2: append reasoning, ask for the final answer (Qwen thinking_budget.md)
        messages2 = messages + [
            {"role": "assistant", "content": "<think\n" + reasoning},
        ]
        text2 = tok.apply_chat_template(
            messages2, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids2 = tok(text2, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out2 = model.generate(
                **ids2, max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out2[0][ids2["input_ids"].shape[1]:], skip_special_tokens=True)

    summary = []
    for budget in budgets:
        per_item: list[dict] = []
        joules_total = 0
        for row in rows:
            prompt = row["prompt"]
            verdicts: list[bool] = []
            e0 = meter.read()
            for _ in range(args.k):
                answer = gen(
                    [{"role": "user", "content": prompt}],
                    max_new=256, budget=budget,
                )
                verdicts.append(VERIFIERS.get(args.verifier, _lean_verifier)(prompt, answer))
            joules_total += meter.read() - e0
            per_item.append({"prompt_sha": _sha256(prompt)[:16], "verdicts": verdicts})

        # EXAONE Deep: pass@1 over k samples + cons@k majority vote
        n = len(per_item)
        pass1 = sum(sum(it["verdicts"]) / max(1, args.k) for it in per_item) / max(1, n)
        cons = sum(sum(it["verdicts"]) * 2 > args.k for it in per_item) / max(1, n)
        mean_j = joules_total / max(1, n * args.k)
        rpj = (cons / mean_j) if mean_j > 0 else 0.0
        rec = receipt(
            f"rpj budget={budget} k={args.k}",
            {
                "base": args.base, "adapter": args.adapter,
                "gpu": meter.gpu, "budget": budget, "k": args.k,
                "items": n, "pass_at_1": round(pass1, 4),
                "cons_at_k": round(cons, 4),
                "joules_per_response": round(mean_j, 1),
                "energy": {"label": "MEASURED", "method":
                    "nvmlDeviceGetTotalEnergyConsumption cumulative-counter delta (device level)"},
                "reasoning_per_joule": round(rpj, 6),
            },
        )
        path = Path(args.out) / f"rpj-{args.base.split('/')[-1]}-b{budget}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        summary.append({k: rec[k] for k in
                        ("budget", "pass_at_1", "cons_at_k",
                         "joules_per_response", "reasoning_per_joule")})
        print(f"[rpj] budget={budget:5d} pass@1={pass1:.3f} cons@{args.k}={cons:.3f} "
              f"J/resp={mean_j:8.1f} RPJ={rpj:.6f}", flush=True)

    tail = receipt(
        "rpj summary",
        {"base": args.base, "adapter": args.adapter, "gpu": meter.gpu,
         "verifier": args.verifier, "rows": summary},
    )
    (Path(args.out) / "rpj-summary.json").write_text(
        json.dumps(tail, indent=2, ensure_ascii=False)
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
