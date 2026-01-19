"""Certificate utilities (exact search for modest ``n``)."""

from __future__ import annotations

import itertools as it
from typing import List, Tuple

from ..core.base import BooleanFunction

__all__ = ["certificate", "max_certificate_size"]


def certificate(f: BooleanFunction, x: int) -> Tuple[int, List[int]]:
    """Return a minimal certificate size and variables for input ``x``."""

    n = f.n_vars or 0
    target = bool(f.evaluate(int(x)))
    for r in range(0, n + 1):
        for vars_ in it.combinations(range(n), r):
            free = [i for i in range(n) if i not in vars_]
            ok = True
            for mask in range(1 << len(free)):
                y = int(x)
                # enforce variables so that y matches x on vars_
                for v in vars_:
                    bit = (x >> v) & 1
                    if ((y >> v) & 1) != bit:
                        y ^= 1 << v
                # vary free variables by mask
                for idx, v in enumerate(free):
                    bit = (mask >> idx) & 1
                    if ((y >> v) & 1) != bit:
                        y ^= 1 << v
                if bool(f.evaluate(y)) != target:
                    ok = False
                    break
            if ok:
                return r, list(vars_)
    return n, list(range(n))


def max_certificate_size(f: BooleanFunction) -> int:
    """Maximum certificate size across inputs."""

    n = f.n_vars or 0
    if n == 0:
        return 0
    size = 1 << n
    best = 0
    for x in range(size):
        best = max(best, certificate(f, x)[0])
    return best


