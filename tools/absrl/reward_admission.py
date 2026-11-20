#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Denial-only ABS-RL training admission while the staged worker uses reward v1.

Issue #294 proved that the v1 substring scorer grants rewards to incorrect
answers. Source-only reward v2 exists but is not qualified against the real
TRL callback, dataset rights, rendered prompt grammar, and checkpoint/stop
contract. This independent gate NEVER authorizes training.

There is deliberately no environment variable, CLI switch, model verdict,
network request, or mutable runtime override. A separately reviewed source
change and all independent owner/compute gates are needed to admit training.
"""

from __future__ import annotations

HOLD_REASON = "ABS_RL_REWARD_V1_UNSAFE"
TRAINING_HOLD_EXIT_CODE = 86


class TrainingAdmissionHold(RuntimeError):
    """The staged ABS-RL reward lane is not eligible to launch training."""


def require_training_admission() -> None:
    """Deny a new GRPO training run before any dataset/GPU/provider actions."""
    raise TrainingAdmissionHold(HOLD_REASON)
