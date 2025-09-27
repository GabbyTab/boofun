"""Block sensitivity helpers (exact search for small ``n``)."""

from __future__ import annotations

from typing import List

from ..core.base import BooleanFunction

__all__ = ["block_sensitivity_at", "max_block_sensitivity"]


def _sensitive_blocks(f: BooleanFunction, x: int) -> List[int]:
    n = f.n_vars or 0
    base = bool(f.evaluate(int(x)))
    return [m for m in range(1, 1 << n) if bool(f.evaluate(int(x) ^ m)) != base]


def _max_disjoint(blocks: List[int], start: int, used: int) -> int:
    best = 0
    for idx in range(start, len(blocks)):
        b = blocks[idx]
        if used & b:
            continue
        best = max(best, 1 + _max_disjoint(blocks, idx + 1, used | b))
    return best


def block_sensitivity_at(f: BooleanFunction, x: int) -> int:
    """Exact block sensitivity via backtracking search."""

    blocks = _sensitive_blocks(f, x)
    blocks.sort(key=int.bit_count)  # small blocks first
    return _max_disjoint(blocks, 0, 0)


def max_block_sensitivity(f: BooleanFunction) -> int:
    """Maximum block sensitivity across all inputs."""

    n = f.n_vars or 0
    if n == 0:
        return 0
    size = 1 << n
    best = 0
    for x in range(size):
        best = max(best, block_sensitivity_at(f, x))
    return best


