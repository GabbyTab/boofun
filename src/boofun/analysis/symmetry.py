"""Symmetry helpers informed by Krawchouk polynomials."""

from __future__ import annotations

import numpy as np

from ..core.base import BooleanFunction

__all__ = ["symmetrize", "degree_sym", "sens_sym"]


def symmetrize(f: BooleanFunction) -> np.ndarray:
    """Return counts of true outputs grouped by Hamming weight."""

    n = f.n_vars or 0
    counts = np.zeros(n + 1, dtype=int)
    for x in range(1 << n):
        if bool(f.evaluate(x)):
            counts[bin(x).count("1")] += 1
    return counts


def degree_sym(f: BooleanFunction) -> int:
    """Symmetric degree: largest weight with nonzero count."""

    counts = symmetrize(f)
    nz = np.nonzero(counts)[0]
    return int(nz.max()) if nz.size else 0


def sens_sym(f: BooleanFunction) -> float:
    """Crude symmetric sensitivity proxy (mean weight of true inputs)."""

    counts = symmetrize(f)
    total = counts.sum()
    if total == 0:
        return 0.0
    weights = np.arange(len(counts))
    return float(np.dot(weights, counts) / total)


