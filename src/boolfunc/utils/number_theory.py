"""Number theory helpers with optional SymPy support."""

from __future__ import annotations

from math import gcd as _gcd
from typing import List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import sympy as _sp

    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    _sp = None
    _HAS_SYMPY = False

__all__ = ["gcd", "invmod", "crt", "is_prime", "prime_sieve"]


def gcd(a: int, b: int) -> int:
    """Greatest common divisor via math.gcd."""

    return _gcd(a, b)


def invmod(a: int, m: int) -> int:
    """Modular inverse of *a* modulo *m* (raises ValueError if none)."""

    a %= m
    try:
        return pow(a, -1, m)
    except ValueError:
        t, new_t = 0, 1
        r, new_r = m, a
        while new_r != 0:
            q = r // new_r
            t, new_t = new_t, t - q * new_t
            r, new_r = new_r, r - q * new_r
        if r != 1:
            raise ValueError("inverse does not exist")
        if t < 0:
            t += m
        return t


def crt(moduli: Sequence[int], residues: Sequence[int]) -> Tuple[int, int]:
    """Chinese Remainder Theorem solution (value, modulus)."""

    if len(moduli) != len(residues):
        raise ValueError("moduli and residues must have same length")
    if _HAS_SYMPY:
        x, M = _sp.ntheory.modular.crt(list(moduli), list(residues))
        return int(x % M), int(M)

    x, M = 0, 1
    for m, r in zip(moduli, residues):
        d = _gcd(M, m)
        if (r - x) % d != 0:
            raise ValueError("no solution")
        m1 = m // d
        t = ((r - x) // d) * invmod(M // d, m1) % m1
        x = x + M * t
        M *= m1
        x %= M
    return x, M


def is_prime(n: int) -> bool:
    """Deterministic primality for 64-bit *n* (SymPy-backed if available)."""

    if n < 2:
        return False
    if _HAS_SYMPY:
        return bool(_sp.ntheory.primetest.isprime(n))

    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_sieve(upto: int) -> List[int]:
    """Return primes <= *upto* via a simple Sieve of Eratosthenes."""

    if upto < 2:
        return []
    sieve = bytearray(b"\x01") * (upto + 1)
    sieve[0:2] = b"\x00\x00"
    p = 2
    while p * p <= upto:
        if sieve[p]:
            start = p * p
            step = p
            sieve[start : upto + 1 : step] = b"\x00" * (((upto - start) // step) + 1)
        p += 1
    return [i for i, v in enumerate(sieve) if v]


