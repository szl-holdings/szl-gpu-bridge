#!/usr/bin/env python3
# Copyright 2026 SZL Holdings -- SPDX-License-Identifier: Apache-2.0
"""Reasoning-per-Joule (RPJ) evaluation harness, governed edition (v1.1).

Leader-harvest (2026-11-15 CTO audit, doctrine pass 6):
  - Korea  LG AI Research, EXAONE Deep (arXiv 2503.12524): pass@k with sampling
          temperature 0.6 / top-p 0.95 and consensus@k majority vote.
  - China Alibaba Qwen3 thinking-budget two-step inference (QwenLM/Qwen3
          docs thinking_budget.md): reason within a token budget, then re-call
          with the reasoning appended to obtain the final answer.
  - Ireland Intercom Fin doctrine: resolution-rate-first metrics -- a response
          only counts when the task is verifiably resolved end to end.

Estate math instilled (doctrine pass 7 -- the wedge):
  - Lambda (Lambda-Spine) aggregator, per szl-lambda-gate: the weighted
          GEOMETRIC MEAN of axis scores in [0,1]; any zeroed axis drives the
          aggregate to 0 (non-compensatory, fail-closed). ADVISORY ONLY:
          Lambda uniqueness = Conjecture 1, machine-checked open, never
          "green", never a theorem.
  - governed-receipt-spec core fields: decision (allow/review/abstain),
          lambda status, energy {joules, label} (never fabricated),
          payload_digest, and a prev/digest/seq hash chain (genesis = 64
          zeros at seq 0).
  - khipu/energy-attest doctrine: MEASURED nvml joule deltas per response
          (device-level cumulative counter), kind szl.rpj/v1.

Run on any NVML-visible GPU node (Salad worker image or VM):
  python rpj_eval.py --base Qwen/Qwen3-1.7B \
      [--adapter SZLHOLDINGS/salad-leanmath-r1] \
      --dataset prompts.jsonl --verifier lean_heuristic --budgets 128,512,2048 \
      --k 4 --out out/rpj
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

TEMPERATURE = 0.6   # EXAONE Deep eval setting
TOP_P = 0.95        # EXAONE Deep eval setting

# Lambda axis weights (sum to 1.0). Declared in every receipt.
LAMBDA_AXES = {
    "resolution": 0.5,   # cons@k: majority-verdict resolution rate
    "sample_stability": 0.2,  # pass@1: mean per-sample resolution rate
    "energy_measured": 0.3,   # 1.0 iff joules are MEASURED, else 0.0 -> Lambda = 0
}
LAMBDA_STATUS = (
    "advisory -- Lambda uniqueness = Conjecture 1 (machine-checked open); "
    "never green, never a theorem"
)


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
    """Heuristic resolution check (no Lean toolchain on worker nodes).

    Fin-style honest completion: sorry/admit anywhere, or a dangling := by
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


# ------------------------------------------------------------ math core ---
def lambda_aggregate(axes: dict[str, float], weights: dict[str, float]) -> float:
    """Lambda-Spine aggregator: weighted geometric mean, non-compensatory.

    Any axis at 0.0 drives the aggregate to 0.0 (fail-closed floor). All axis
    values are clamped to [0,1]. ADVISORY ONLY (Conjecture 1, never green).
    """
    score = 1.0
    for axis, weight in weights.items():
        value = max(0.0, min(1.0, axes.get(axis, 0.0)))
        score *= value ** weight
    return score


def _sha256_hex(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def governed_receipt(detail: str, payload: dict, prev_digest: str, seq: int) -> dict:
    """Emit a governed-receipt-spec-shaped szl.rpj/v1 record with hash chain."""
    body = dict(payload)
    body["kind"] = "szl.rpj/v1"
    body["detail"] = detail
    body["ts"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    body["prev"] = prev_digest
    body["seq"] = seq
    body["digest"] = _sha256_hex(json.dumps(body, sort_keys=True, ensure_ascii=False))
    return body


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description="Reasoning-per-Joule governed eval harness")
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

    verifier = VERIFIERS.get(args.verifier, _lean_verifier)
    summary = []
    prev_digest = "0" * 64
    seq = 0
    for budget in budgets:
        per_item: list[dict] = []
        joules_total = 0
        energy_label = "MEASURED"
        for row in rows:
            prompt = row["prompt"]
            verdicts: list[bool] = []
            e0 = meter.read()
            for _ in range(args.k):
                answer = gen(
                    [{"role": "user", "content": prompt}],
                    max_new=256, budget=budget,
                )
                verdicts.append(verifier(prompt, answer))
            try:
                delta = meter.read() - e0
                joules_total += delta
            except Exception:  # noqa: BLE001 (honest degradation, never fabricate)
                energy_label = "UNAVAILABLE"
            per_item.append({"prompt_sha": _sha256_hex(prompt)[:16], "verdicts": verdicts})

        # EXAONE Deep: pass@1 over k samples + cons@k majority vote
        n = len(per_item)
        pass1 = sum(sum(it["verdicts"]) / max(1, args.k) for it in per_item) / max(1, n)
        cons = sum(sum(it["verdicts"]) * 2 > args.k for it in per_item) / max(1, n)
        mean_j = (joules_total / max(1, n * args.k)) if energy_label == "MEASURED" else None

        # --- estate math: Lambda aggregation (advisory, non-compensatory) ---
        axes = {
            "resolution": cons,
            "sample_stability": pass1,
            "energy_measured": 1.0 if energy_label == "MEASURED" else 0.0,
        }
        lam = lambda_aggregate(axes, LAMBDA_AXES)
        rpj = (cons / mean_j) if (mean_j is not None and mean_j > 0) else 0.0

        # governed decision: allow only on majority resolution; abstain on zero
        if cons >= 0.5:
            decision = "allow"
        elif cons > 0.0 or pass1 > 0.0:
            decision = "review"
        else:
            decision = "abstain"

        payload_digest = _sha256_hex(json.dumps(per_item, sort_keys=True))
        seq += 1
        rec = governed_receipt(
            f"rpj budget={budget} k={args.k}",
            {
                "base": args.base, "adapter": args.adapter,
                "gpu": meter.gpu, "budget": budget, "k": args.k,
                "items": n, "pass_at_1": round(pass1, 4),
                "cons_at_k": round(cons, 4),
                "decision": decision,
                "lambda": {
                    "aggregate": round(lam, 6),
                    "axes": {k2: round(v, 4) for k2, v in axes.items()},
                    "weights": LAMBDA_AXES,
                    "status": LAMBDA_STATUS,
                },
                "energy": {
                    "label": energy_label,
                    "joules": round(mean_j, 1) if mean_j is not None else None,
                    "method": "nvmlDeviceGetTotalEnergyConsumption "
                              "cumulative-counter delta (device level)",
                },
                "reasoning_per_joule": round(rpj, 6),
                "payload_digest": payload_digest,
            },
            prev_digest, seq,
        )
        prev_digest = rec["digest"]
        path = Path(args.out) / f"rpj-{args.base.split('/')[-1]}-b{budget}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        summary.append({k3: rec[k3] for k3 in
                        ("budget", "decision", "pass_at_1", "cons_at_k",
                         "reasoning_per_joule")})
        print(f"[rpj] budget={budget:5d} decision={decision:7s} pass@1={pass1:.3f} "
              f"cons@{args.k}={cons:.3f} Lambda={lam:.4f} "
              f"J/resp={(mean_j if mean_j is not None else -1):8.1f} RPJ={rpj:.6f}",
              flush=True)

    seq += 1
    tail = governed_receipt(
        "rpj governed summary",
        {"base": args.base, "adapter": args.adapter, "gpu": meter.gpu,
         "verifier": args.verifier, "rows": summary,
         "lambda_status": LAMBDA_STATUS},
        prev_digest, seq,
    )
    (Path(args.out) / "rpj-summary.json").write_text(
        json.dumps(tail, indent=2, ensure_ascii=False)
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
