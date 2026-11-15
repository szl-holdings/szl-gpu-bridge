"""Reward-core tests - pure, no torch/trl/GPU required."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rewards import ABSTAIN_MARK, abstention_reward, completion_text, normalize_answerable


def test_decisive_correct():
    assert abstention_reward(["the answer is 42"], answer=["42"], answerable=[True]) == [1.0]


def test_decisive_wrong():
    assert abstention_reward(["the answer is 41"], answer=["42"], answerable=[True]) == [0.0]


def test_honest_abstention():
    r = abstention_reward([f"I cannot tell: {ABSTAIN_MARK}"], answer=[None], answerable=[False])
    assert r == [0.5]


def test_over_abstention():
    r = abstention_reward([f"hmm: {ABSTAIN_MARK}"], answer=["42"], answerable=[True])
    assert r == [0.0]


def test_misaligned_columns_raise():
    with pytest.raises(ValueError):
        abstention_reward(["a", "b"], answer=["42"], answerable=[True])


def test_conversational_completion():
    r = abstention_reward([{"role": "assistant", "content": "so 42"}],
                          answer=["42"], answerable=[True])
    assert r == [1.0]


def test_completion_text_variants():
    assert completion_text("x") == "x"
    assert completion_text({"content": "y"}) == "y"
    assert completion_text([{"role": "user", "content": "z"}]) == "z"
    assert completion_text(None) == ""


def test_normalize_answerable():
    assert normalize_answerable(True) is True
    assert normalize_answerable(False) is False
    assert normalize_answerable("true") is True
    assert normalize_answerable("no") is False
    assert normalize_answerable("") is False
