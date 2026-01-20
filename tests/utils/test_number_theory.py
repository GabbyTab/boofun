"""
Comprehensive tests for utils/number_theory module.

Tests number-theoretic functions used in Boolean function analysis.
"""

import sys

import pytest

sys.path.insert(0, "src")

from boofun.utils.number_theory import (
    binomial,
    binomial_sum,
    crt,
    euler_phi,
    factor,
    gcd,
    invmod,
    is_prime,
    lcm,
    mobius,
    prime_factorization,
    prime_sieve,
)


class TestGCD:
    """Test greatest common divisor."""

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (12, 8, 4),
            (7, 5, 1),
            (100, 25, 25),
            (0, 5, 5),
            (17, 17, 17),
            (1, 1, 1),
        ],
    )
    def test_gcd_known_values(self, a, b, expected):
        """GCD should compute correct values."""
        assert gcd(a, b) == expected

    def test_gcd_commutative(self):
        """GCD should be commutative."""
        for a, b in [(12, 18), (7, 21), (100, 35)]:
            assert gcd(a, b) == gcd(b, a)

    def test_gcd_with_one(self):
        """GCD(n, 1) = 1."""
        for n in [1, 5, 17, 100]:
            assert gcd(n, 1) == 1


class TestInvMod:
    """Test modular inverse."""

    @pytest.mark.parametrize(
        "a,m,expected",
        [
            (3, 7, 5),  # 3 * 5 = 15 ≡ 1 (mod 7)
            (2, 5, 3),  # 2 * 3 = 6 ≡ 1 (mod 5)
            (7, 11, 8),  # 7 * 8 = 56 ≡ 1 (mod 11)
        ],
    )
    def test_invmod_known_values(self, a, m, expected):
        """invmod should compute correct values."""
        result = invmod(a, m)
        assert (a * result) % m == 1

    def test_invmod_self_inverse(self):
        """invmod(invmod(a, m), m) = a."""
        for a, m in [(3, 7), (2, 5), (4, 11)]:
            inv = invmod(a, m)
            assert invmod(inv, m) == a % m

    def test_invmod_no_inverse(self):
        """invmod should raise when no inverse exists."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            invmod(2, 4)  # gcd(2, 4) = 2 ≠ 1


class TestCRT:
    """Test Chinese Remainder Theorem."""

    def test_crt_basic(self):
        """CRT should solve basic systems."""
        # x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        # Solution: x ≡ 8 (mod 15)
        result, modulus = crt([3, 5], [2, 3])

        assert result % 3 == 2
        assert result % 5 == 3

    def test_crt_coprime_moduli(self):
        """CRT with coprime moduli."""
        # x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        result, modulus = crt([2, 3, 5], [1, 2, 3])

        assert result % 2 == 1
        assert result % 3 == 2
        assert result % 5 == 3
        assert modulus == 30


class TestIsPrime:
    """Test primality testing."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (6, False),
            (7, True),
            (11, True),
            (13, True),
            (17, True),
            (19, True),
            (23, True),
            (25, False),
            (97, True),
            (100, False),
            (1, False),
            (0, False),
        ],
    )
    def test_is_prime_known_values(self, n, expected):
        """is_prime should correctly identify primes."""
        assert is_prime(n) == expected

    def test_large_prime(self):
        """is_prime should work for larger primes."""
        assert is_prime(104729)  # 10000th prime


class TestPrimeSieve:
    """Test prime sieve."""

    def test_sieve_small(self):
        """Sieve should find primes up to 20."""
        primes = prime_sieve(20)
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        assert primes == expected

    def test_sieve_100(self):
        """Sieve should find correct number of primes up to 100."""
        primes = prime_sieve(100)
        assert len(primes) == 25  # 25 primes ≤ 100
        assert all(is_prime(p) for p in primes)

    def test_sieve_empty(self):
        """Sieve of 1 should be empty."""
        primes = prime_sieve(1)
        assert primes == []


class TestFactor:
    """Test integer factorization."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (12, [2, 2, 3]),
            (7, [7]),
            (100, [2, 2, 5, 5]),
            (1, []),
            (2, [2]),
            (8, [2, 2, 2]),
            (30, [2, 3, 5]),
        ],
    )
    def test_factor_known_values(self, n, expected):
        """factor should return correct prime factors."""
        assert factor(n) == expected

    def test_factor_product(self):
        """Product of factors should equal original."""
        for n in [12, 100, 77, 1000]:
            factors = factor(n)
            product = 1
            for f in factors:
                product *= f
            assert product == n or (n == 1 and product == 1)


class TestPrimeFactorization:
    """Test prime factorization with multiplicities."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (12, {2: 2, 3: 1}),
            (100, {2: 2, 5: 2}),
            (7, {7: 1}),
            (8, {2: 3}),
        ],
    )
    def test_prime_factorization_known(self, n, expected):
        """prime_factorization should return correct dict."""
        result = prime_factorization(n)
        assert result == expected

    def test_factorization_reconstruction(self):
        """Product of p^e should equal n."""
        for n in [12, 100, 360, 1001]:
            pf = prime_factorization(n)
            product = 1
            for p, e in pf.items():
                product *= p**e
            assert product == n


class TestEulerPhi:
    """Test Euler's totient function."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 4),
            (6, 2),
            (7, 6),
            (8, 4),
            (9, 6),
            (10, 4),
            (12, 4),
        ],
    )
    def test_euler_phi_known_values(self, n, expected):
        """euler_phi should compute correct values."""
        assert euler_phi(n) == expected

    def test_phi_prime(self):
        """φ(p) = p - 1 for prime p."""
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            assert euler_phi(p) == p - 1

    def test_phi_prime_power(self):
        """φ(p^k) = p^{k-1} * (p - 1)."""
        assert euler_phi(8) == 4  # 2^2 * 1 = 4
        assert euler_phi(9) == 6  # 3^1 * 2 = 6
        assert euler_phi(27) == 18  # 3^2 * 2 = 18


class TestBinomial:
    """Test binomial coefficient."""

    @pytest.mark.parametrize(
        "n,k,expected",
        [
            (5, 0, 1),
            (5, 1, 5),
            (5, 2, 10),
            (5, 3, 10),
            (5, 4, 5),
            (5, 5, 1),
            (10, 5, 252),
            (0, 0, 1),
        ],
    )
    def test_binomial_known_values(self, n, k, expected):
        """binomial should compute correct values."""
        assert binomial(n, k) == expected

    def test_binomial_symmetry(self):
        """C(n, k) = C(n, n-k)."""
        for n in range(10):
            for k in range(n + 1):
                assert binomial(n, k) == binomial(n, n - k)

    def test_binomial_row_sum(self):
        """Sum of row n = 2^n."""
        for n in range(10):
            row_sum = sum(binomial(n, k) for k in range(n + 1))
            assert row_sum == 2**n

    def test_binomial_invalid(self):
        """binomial(n, k) = 0 for k > n."""
        assert binomial(5, 6) == 0
        assert binomial(3, 10) == 0


class TestBinomialSum:
    """Test binomial coefficient sum."""

    def test_binomial_sum_basic(self):
        """binomial_sum should compute sum of first k+1 terms."""
        # Sum C(5, 0) + C(5, 1) + C(5, 2) = 1 + 5 + 10 = 16
        result = binomial_sum(5, 2)
        assert result == 16

    def test_binomial_sum_full(self):
        """binomial_sum(n, n) = 2^n."""
        for n in range(10):
            assert binomial_sum(n, n) == 2**n


class TestLCM:
    """Test least common multiple."""

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (4, 6, 12),
            (3, 5, 15),
            (12, 18, 36),
            (1, 7, 7),
            (7, 7, 7),
        ],
    )
    def test_lcm_known_values(self, a, b, expected):
        """lcm should compute correct values."""
        assert lcm(a, b) == expected

    def test_lcm_gcd_relationship(self):
        """lcm(a, b) * gcd(a, b) = a * b."""
        for a, b in [(12, 18), (7, 5), (100, 35)]:
            assert lcm(a, b) * gcd(a, b) == a * b


class TestMobius:
    """Test Möbius function."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, 1),
            (2, -1),
            (3, -1),
            (4, 0),  # 4 = 2²
            (5, -1),
            (6, 1),  # 6 = 2 * 3
            (7, -1),
            (8, 0),  # 8 = 2³
            (9, 0),  # 9 = 3²
            (10, 1),  # 10 = 2 * 5
            (30, -1),  # 30 = 2 * 3 * 5
        ],
    )
    def test_mobius_known_values(self, n, expected):
        """mobius should compute correct values."""
        assert mobius(n) == expected

    def test_mobius_square_free(self):
        """Möbius is 0 for non-square-free numbers."""
        for n in [4, 8, 9, 12, 16, 18, 20]:
            assert mobius(n) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
