"""Number theory helpers with optional SymPy support.

This module provides essential number theory utilities for Boolean function
analysis, including:
- Prime factorization and testing
- Euler's totient function
- Binomial coefficients
- Modular arithmetic (GCD, CRT, modular inverse)
- Prime sieving

Many algorithms in Boolean function analysis require these primitives,
particularly for analyzing function structure and cryptographic properties.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd as _gcd
from typing import Dict, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import sympy as _sp

    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    _sp = None
    _HAS_SYMPY = False

__all__ = [
    # Basic operations
    "gcd",
    "invmod",
    "crt",
    # Prime testing and factorization
    "is_prime",
    "prime_sieve",
    "factor",
    "prime_factorization",
    # Euler's totient
    "euler_phi",
    "totient",
    # Binomial coefficients
    "binomial",
    "binomial_sum",
    # Utility functions
    "lcm",
    "mobius",
]


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
    """Chinese Remainder Theorem solution (value, modulus).

    Raises:
        ValueError: If no solution exists (e.g., incompatible moduli).
    """

    if len(moduli) != len(residues):
        raise ValueError("moduli and residues must have same length")
    if _HAS_SYMPY:
        result = _sp.ntheory.modular.crt(list(moduli), list(residues))
        if result is None:
            raise ValueError("no solution")
        x, M = result
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


def factor(n: int) -> List[int]:
    """
    Return prime factorization of n as a list of prime factors (with multiplicity).

    Args:
        n: Positive integer to factor

    Returns:
        List of prime factors in ascending order

    Example:
        >>> factor(60)
        [2, 2, 3, 5]
        >>> factor(17)
        [17]
    """
    if n < 2:
        return []

    if _HAS_SYMPY:
        pass

        factorization = _sp.ntheory.factorint(n)
        result = []
        for p, exp in sorted(factorization.items()):
            result.extend([p] * exp)
        return result

    factors = []
    tmp = n

    # Handle factor of 2
    while tmp % 2 == 0:
        factors.append(2)
        tmp //= 2

    # Handle odd factors
    p = 3
    while p * p <= tmp:
        while tmp % p == 0:
            factors.append(p)
            tmp //= p
        p += 2

    if tmp > 1:
        factors.append(tmp)

    return factors


def prime_factorization(n: int) -> Dict[int, int]:
    """
    Return prime factorization as a dictionary {prime: exponent}.

    Args:
        n: Positive integer to factor

    Returns:
        Dictionary mapping primes to their exponents

    Example:
        >>> prime_factorization(60)
        {2: 2, 3: 1, 5: 1}
    """
    if n < 2:
        return {}

    if _HAS_SYMPY:
        return dict(_sp.ntheory.factorint(n))

    result: Dict[int, int] = {}
    for p in factor(n):
        result[p] = result.get(p, 0) + 1
    return result


@lru_cache(maxsize=1024)
def euler_phi(n: int) -> int:
    """
    Compute Euler's totient function φ(n).

    φ(n) = count of integers in [1, n] that are coprime to n.

    Uses the formula: φ(n) = n * ∏_{p|n} (1 - 1/p)

    Args:
        n: Positive integer

    Returns:
        Euler's totient of n

    Example:
        >>> euler_phi(12)  # 1, 5, 7, 11 are coprime to 12
        4
    """
    if n < 1:
        return 0
    if n == 1:
        return 1

    if _HAS_SYMPY:
        return int(_sp.ntheory.totient(n))

    # Get unique prime factors
    primes = list(set(factor(n)))
    result = n
    for p in primes:
        result = result * (p - 1) // p
    return result


# Alias for euler_phi
totient = euler_phi


def binomial(n: int, k: int) -> int:
    """
    Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!).

    Uses iterative computation to avoid large intermediate values.

    Args:
        n: Total items
        k: Items to choose

    Returns:
        Number of ways to choose k items from n

    Example:
        >>> binomial(5, 2)
        10
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # Use symmetry: C(n,k) = C(n, n-k)
    if k > n - k:
        k = n - k

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def binomial_sum(n: int, k: int) -> int:
    """
    Compute sum of binomial coefficients: Σ_{i=0}^{k} C(n, i).

    This is useful for counting functions of low degree.

    Args:
        n: Total items
        k: Maximum selection size

    Returns:
        Sum of C(n,0) + C(n,1) + ... + C(n,k)

    Example:
        >>> binomial_sum(5, 2)  # C(5,0) + C(5,1) + C(5,2) = 1 + 5 + 10
        16
    """
    if k < 0:
        return 0
    if k >= n:
        return 2**n

    total = 0
    coeff = 1
    for i in range(k + 1):
        total += coeff
        coeff = coeff * (n - i) // (i + 1)
    return total


def lcm(a: int, b: int) -> int:
    """
    Compute least common multiple of a and b.

    Args:
        a, b: Integers

    Returns:
        LCM(a, b)
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _gcd(a, b)


def mobius(n: int) -> int:
    """
    Compute the Möbius function μ(n).

    μ(n) = 0 if n has a squared prime factor
    μ(n) = (-1)^k if n is a product of k distinct primes
    μ(1) = 1

    The Möbius function is important in combinatorics and number theory,
    particularly for Möbius inversion.

    Args:
        n: Positive integer

    Returns:
        μ(n) ∈ {-1, 0, 1}

    Example:
        >>> mobius(1)
        1
        >>> mobius(6)  # 6 = 2 * 3, two distinct primes
        1
        >>> mobius(12)  # 12 = 2^2 * 3, has squared factor
        0
    """
    if n < 1:
        return 0
    if n == 1:
        return 1

    if _HAS_SYMPY:
        return int(_sp.ntheory.mobius(n))

    factorization = prime_factorization(n)

    # Check for squared prime factor
    for exp in factorization.values():
        if exp > 1:
            return 0

    # Return (-1)^k where k is number of distinct prime factors
    k = len(factorization)
    return (-1) ** k
