"""Sensitivity-related helpers ported from legacy routines."""

from __future__ import annotations

import numpy as np

from ..core.base import BooleanFunction

__all__ = [
    "sensitivity_at",
    "sensitivity_profile",
    "total_influence_via_sensitivity",
]


def sensitivity_at(f: BooleanFunction, x: int) -> int:
    """Return the sensitivity of ``f`` at input index ``x``."""

    n = f.n_vars or 0
    base = bool(f.evaluate(int(x)))
    count = 0
    for i in range(n):
        if bool(f.evaluate(int(x) ^ (1 << i))) != base:
            count += 1
    return count


def sensitivity_profile(f: BooleanFunction) -> np.ndarray:
    """Return per-input sensitivities as a NumPy array."""

    n = f.n_vars or 0
    size = 1 << n
    return np.array([sensitivity_at(f, i) for i in range(size)], dtype=int)


def total_influence_via_sensitivity(f: BooleanFunction) -> float:
    """Compute total influence via the average sensitivity definition."""

    profile = sensitivity_profile(f)
    return float(np.mean(profile))


