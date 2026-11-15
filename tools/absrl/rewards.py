"""Pure abstention reward core - importable without torch/trl/GPU (test surface).

Reward design (fail-closed, no fabricated rewards):
- correct decisive answer:        +1.0
- honest abstention on an
  unanswerable item:              +0.5
- wrong decisive answer:           0.0
- over-abstention (abstains when
  the item IS answerable):         0.0

Rewards are computed ONLY from dataset fields (answer, answerable) and the
completion text. Missing or misaligned columns raise - nothing is invented.
"""
from __future__ import annotations

ABSTAIN_MARK = "UNANSWERABLE"


def normalize_answerable(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1", "answerable")


def completion_text(comp) -> str:
    """Accept plain strings or conversational (role/content) completions."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, dict):
        return str(comp.get("content", ""))
    if isinstance(comp, (list, tuple)) and comp:
        return completion_text(comp[-1])
    return ""


def abstention_reward(completions, answer=None, answerable=None, **kw):
    """TRL GRPO reward function: (completions, **dataset_columns) -> list[float]."""
    n = len(completions)
    answers = list(answer) if answer is not None else [None] * n
    answerables = list(answerable) if answerable is not None else [None] * n
    if len(answers) != n or len(answerables) != n:
        raise ValueError(
            "reward columns must align with completions (fail-closed, no fabricated rewards)")
    out = []
    for comp, ans, ansb in zip(completions, answers, answerables):
        text = completion_text(comp)
        if ABSTAIN_MARK in text:
            out.append(0.5 if not normalize_answerable(ansb) else 0.0)
            continue
        gold = "" if ans is None else str(ans).strip()
        out.append(1.0 if (gold and gold.lower() in text.lower()) else 0.0)
    return out
