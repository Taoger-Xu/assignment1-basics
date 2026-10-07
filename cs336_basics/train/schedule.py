"""Learning-rate schedules for language-model training."""

from __future__ import annotations

import math


def cosine_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warm-up, cosine decay, then a constant minimum learning rate."""
    if warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("Require 0 <= warmup_iters < cosine_cycle_iters")
    if it < 0:
        raise ValueError("Iteration must be non-negative")

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )
