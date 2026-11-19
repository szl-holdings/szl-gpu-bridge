#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Candidate ABS-RL reward verifier v2: source-only and inactive.

This file is NOT imported by the currently staged GRPO worker. Its purpose is
to expose verifier failure modes before any new reward/prompt contract is
admitted to a training lane. No network, models, subprocesses or file writes.

Reward contract (intentionally stricter than rewards.py v1):
 * answerable=True + exact final answer: 1.0
 * answerable=False + exact abstention: 0.5
 * any other well-formed model response: 0.0
 * inconsistent or unknown dataset metadata/completion shape: ValueError

Positive rewards are conditional on caller-provided labels; their correctness,
license, source revision, privacy and held-out separation are NOT verified here.
"""
from __future__ import annotations

import re
import unicodedata

ABSTAIN_MARK = "UNANSWERABLE"
MAX_BATCH = 2048
MAX_COMPLETION_CHARS = 8192
MAX_GOLD_CHARS = 512
_FINAL = re.compile(r"FINAL:[ \t]+([^\r\n]+)\Z")
_TRUE = frozenset({"true", "yes", "1", "answerable"})
_FALSE = frozenset({"false", "no", "0", "unanswerable"})


def normalize_answerable(value) -> bool:
    """Unknown labels are invalid, not automatically unanswerable."""
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if type(value) is str:
        normalized = value.strip().casefold()
        if normalized in _TRUE:
            return True
        if normalized in _FALSE:
            return False
    raise ValueError("ANSWERABLE_LABEL_UNRECOGNIZED")


def completion_text(comp) -> str:
    """Read a model-authored message only; no arbitrary transcript recursion."""
    if type(comp) is str:
        text = comp
    else:
        if type(comp) in (list, tuple):
            if len(comp) != 1:
                raise ValueError("COMPLETION_SEQUENCE_AMBIGUOUS")
            comp = comp[0]
        if type(comp) is not dict:
            raise ValueError("COMPLETION_SHAPE_INVALID")
        if comp.get("role", "assistant") != "assistant":
            raise ValueError("COMPLETION_NOT_ASSISTANT")
        text = comp.get("content")
        if type(text) is not str:
            raise ValueError("COMPLETION_CONTENT_NOT_TEXT")
    if len(text) > MAX_COMPLETION_CHARS or "\x00" in text:
        raise ValueError("COMPLETION_TEXT_INVALID_OR_OVERSIZED")
    return text


def _gold(value, answerable: bool) -> str | None:
    if not answerable:
        if value is None or (type(value) is str and value.strip() == ""):
            return None
        raise ValueError("UNANSWERABLE_ROW_HAS_GOLD")
    if type(value) is not str:
        raise ValueError("ANSWERABLE_ROW_GOLD_MUST_BE_TEXT")
    if not 0 < len(value) <= MAX_GOLD_CHARS or any(c in value for c in "\r\n\x00"):
        raise ValueError("GOLD_INVALID_OR_OVERSIZED")
    result = unicodedata.normalize("NFC", value.strip())
    if not result or result == ABSTAIN_MARK or result.startswith("FINAL:"):
        raise ValueError("GOLD_RESERVED_OR_EMPTY")
    return result


def _final(text: str) -> str | None:
    """An entire final answer, never an arbitrary substring or embedded token."""
    candidate = text.strip()
    if not candidate or "\r" in candidate or "\n" in candidate:
        return None
    if candidate.startswith("FINAL:"):
        match = _FINAL.fullmatch(candidate)
        if match is None:
            return None
        candidate = match.group(1).strip()
    if not candidate:
        return None
    return unicodedata.normalize("NFC", candidate)


def abstention_reward_v2(completions, answer=None, answerable=None, **kwargs):
    """Optional future TRL reward callback; NEVER an authorization or quality proof.

    Unused kwargs support TRL's additional prompt/metadata callback fields.
    The full batch is validated before scoring a single completion.
    """
    if type(completions) not in (list, tuple):
        raise ValueError("COMPLETIONS_MUST_BE_SEQUENCE")
    n = len(completions)
    if not 1 <= n <= MAX_BATCH:
        raise ValueError("REWARD_BATCH_SIZE_INVALID")
    if type(answer) not in (list, tuple) or type(answerable) not in (list, tuple):
        raise ValueError("REQUIRED_REWARD_COLUMNS_MISSING_OR_INVALID")
    if len(answer) != n or len(answerable) != n:
        raise ValueError("REWARD_COLUMNS_MISALIGNED")

    validated = []
    for comp, gold_value, raw_label in zip(completions, answer, answerable):
        label = normalize_answerable(raw_label)
        gold = _gold(gold_value, label)
        terminal = _final(completion_text(comp))
        validated.append((label, gold, terminal))

    return [
        (1.0 if terminal is not None and terminal == gold else 0.0)
        if label
        else (0.5 if terminal == ABSTAIN_MARK else 0.0)
        for label, gold, terminal in validated
    ]


def shadow_summary_v2(completions, answer=None, answerable=None, *, group_size=1):
    """Return aggregate counts only; no raw generations, calls or provider effect.

    group_size is caller-provided, not trusted proof of actual GRPO grouping.
    Rewards are represented by integer half-units for exact finite comparisons.
    Even a strong result does NOT verify labels, prompts or training readiness.
    """
    if type(group_size) is not int or not 1 <= group_size <= MAX_BATCH:
        raise ValueError("SHADOW_GROUP_SIZE_INVALID")
    scores = abstention_reward_v2(
        completions, answer=answer, answerable=answerable
    )
    if len(scores) % group_size:
        raise ValueError("SHADOW_GROUP_DENOMINATOR_MISALIGNED")
    labels = [normalize_answerable(value) for value in answerable]
    terminals = [_final(completion_text(value)) for value in completions]
    answerable_count = sum(labels)
    unanswerable_count = len(labels) - answerable_count
    correct = scores.count(1.0)
    honest_abstain = scores.count(0.5)
    wrong_abstain = sum(
        label and terminal == ABSTAIN_MARK
        for label, terminal in zip(labels, terminals)
    )
    groups = [
        scores[i : i + group_size] for i in range(0, len(scores), group_size)
    ]
    half_units = 2 * correct + honest_abstain
    return {
        "schema": "szl.absrl-reward-v2-shadow/v1",
        "samples": len(scores),
        "groups": len(groups),
        "group_size": group_size,
        "answerable_rows": answerable_count,
        "unanswerable_rows": unanswerable_count,
        "decisive_correct_rows": correct,
        "honest_abstention_rows": honest_abstain,
        "overabstention_rows": wrong_abstain,
        "all_zero_groups": sum(not any(group) for group in groups),
        "observed_total_half_reward_units": half_units,
        "constant_abstain_half_reward_units": unanswerable_count,
        "beats_constant_abstain_on_this_sample": half_units > unanswerable_count,
        "labels_independently_verified": False,
        "prompt_format_verified": False,
        "dataset_rights_verified": False,
        "trusted_signer_verified": False,
        "model_quality_verified": False,
        "training_authorized": False,
        "provider_mutation_performed": False,
    }
