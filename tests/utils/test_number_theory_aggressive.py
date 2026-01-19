"""
Aggressive tests for number_theory module.

These tests are designed to find bugs by testing:
- Edge cases (0, 1, negative numbers)
- Known mathematical identities
- Boundary conditions
- Invalid inputs
"""

import pytest
import numpy as np

from boofun.utils.number_theory import (
    gcd, invmod, crt,
    is_prime, prime_sieve, factor, prime_factorization,
    euler_phi, totient, binomial, binomial_sum,
    lcm, mobius
)


class TestGCD:
    """Tests for gcd function."""
    
    def test_basic(self):
        """Basic GCD computation."""
        assert gcd(12, 8) == 4
        assert gcd(17, 5) == 1  # Coprime
    
    def test_one_zero(self):
        """GCD with zero."""
        assert gcd(5, 0) == 5
        assert gcd(0, 7) == 7
    
    def test_both_zero(self):
        """GCD(0, 0) = 0."""
        assert gcd(0, 0) == 0
    
    def test_commutative(self):
        """GCD is commutative."""
        assert gcd(12, 18) == gcd(18, 12)
    
    def test_negative(self):
        """GCD with negative numbers."""
        assert gcd(-12, 8) == 4
        assert gcd(12, -8) == 4


class TestInvmod:
    """Tests for modular inverse."""
    
    def test_basic(self):
        """Basic modular inverse."""
        # 3 * 7 = 21 ≡ 1 (mod 10)
        inv = invmod(3, 10)
        assert (3 * inv) % 10 == 1
    
    def test_prime_modulus(self):
        """Every non-zero element has inverse mod prime."""
        p = 17
        for a in range(1, p):
            inv = invmod(a, p)
            assert (a * inv) % p == 1
    
    def test_no_inverse_raises(self):
        """Non-coprime elements have no inverse."""
        with pytest.raises(ValueError, match="inverse does not exist"):
            invmod(4, 8)  # gcd(4, 8) = 4 ≠ 1
    
    def test_one(self):
        """invmod(1, m) = 1."""
        assert invmod(1, 7) == 1
        assert invmod(1, 100) == 1


class TestCRT:
    """Tests for Chinese Remainder Theorem."""
    
    def test_basic(self):
        """Basic CRT."""
        # x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        # x = 8 (mod 15)
        x, M = crt([3, 5], [2, 3])
        assert M == 15
        assert x % 3 == 2
        assert x % 5 == 3
    
    def test_single_congruence(self):
        """Single congruence."""
        x, M = crt([7], [3])
        assert M == 7
        assert x == 3
    
    def test_mismatched_lengths_raises(self):
        """Mismatched lengths should raise."""
        with pytest.raises(ValueError, match="same length"):
            crt([3, 5], [2])
    
    def test_no_solution_raises(self):
        """Incompatible congruences should raise."""
        # x ≡ 0 (mod 2), x ≡ 1 (mod 2) - impossible!
        with pytest.raises(ValueError, match="no solution"):
            crt([2, 2], [0, 1])


class TestIsPrime:
    """Tests for primality testing."""
    
    def test_small_primes(self):
        """Small primes are identified correctly."""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in small_primes:
            assert is_prime(p), f"{p} should be prime"
    
    def test_small_composites(self):
        """Small composites are identified correctly."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        for c in composites:
            assert not is_prime(c), f"{c} should be composite"
    
    def test_edge_cases(self):
        """Edge cases."""
        assert not is_prime(0)
        assert not is_prime(1)
        assert not is_prime(-5)
    
    def test_carmichael_numbers(self):
        """Carmichael numbers (pseudoprimes) should be detected."""
        # 561 = 3 * 11 * 17 is the smallest Carmichael number
        assert not is_prime(561)
    
    def test_large_prime(self):
        """Large prime."""
        # Mersenne prime 2^31 - 1
        large_prime = 2147483647
        assert is_prime(large_prime)


class TestPrimeSieve:
    """Tests for prime sieve."""
    
    def test_small_sieve(self):
        """Small sieve."""
        primes = prime_sieve(20)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19]
    
    def test_sieve_zero(self):
        """Sieve up to 0."""
        assert prime_sieve(0) == []
    
    def test_sieve_one(self):
        """Sieve up to 1."""
        assert prime_sieve(1) == []
    
    def test_sieve_two(self):
        """Sieve up to 2."""
        assert prime_sieve(2) == [2]
    
    def test_sieve_count(self):
        """Count of primes up to 100 is 25."""
        primes = prime_sieve(100)
        assert len(primes) == 25


class TestFactor:
    """Tests for factorization."""
    
    def test_prime(self):
        """Prime numbers factor to themselves."""
        assert factor(17) == [17]
        assert factor(2) == [2]
    
    def test_composite(self):
        """Composite numbers factor correctly."""
        assert factor(60) == [2, 2, 3, 5]
        assert factor(100) == [2, 2, 5, 5]
    
    def test_power_of_two(self):
        """Power of 2."""
        assert factor(64) == [2, 2, 2, 2, 2, 2]
    
    def test_one(self):
        """factor(1) = []."""
        assert factor(1) == []
    
    def test_zero(self):
        """factor(0) = []."""
        assert factor(0) == []
    
    def test_product_equals_original(self):
        """Product of factors equals original."""
        n = 1234567
        factors = factor(n)
        product = 1
        for f in factors:
            product *= f
        assert product == n


class TestPrimeFactorization:
    """Tests for prime factorization as dict."""
    
    def test_basic(self):
        """Basic factorization."""
        assert prime_factorization(60) == {2: 2, 3: 1, 5: 1}
    
    def test_prime(self):
        """Prime number."""
        assert prime_factorization(17) == {17: 1}
    
    def test_one(self):
        """Factorization of 1."""
        assert prime_factorization(1) == {}
    
    def test_zero(self):
        """Factorization of 0."""
        assert prime_factorization(0) == {}


class TestEulerPhi:
    """Tests for Euler's totient function."""
    
    def test_basic(self):
        """Basic totient values."""
        assert euler_phi(1) == 1
        assert euler_phi(2) == 1
        assert euler_phi(6) == 2  # 1, 5
        assert euler_phi(12) == 4  # 1, 5, 7, 11
    
    def test_prime(self):
        """phi(p) = p - 1 for prime p."""
        for p in [2, 3, 5, 7, 11, 13]:
            assert euler_phi(p) == p - 1
    
    def test_prime_power(self):
        """phi(p^k) = p^(k-1) * (p-1)."""
        assert euler_phi(8) == 4  # 2^3 -> 2^2 * 1 = 4
        assert euler_phi(9) == 6  # 3^2 -> 3^1 * 2 = 6
    
    def test_zero(self):
        """phi(0) = 0."""
        assert euler_phi(0) == 0
    
    def test_negative(self):
        """phi(negative) = 0."""
        assert euler_phi(-5) == 0
    
    def test_totient_alias(self):
        """totient is alias for euler_phi."""
        assert totient(12) == euler_phi(12)


class TestBinomial:
    """Tests for binomial coefficients."""
    
    def test_basic(self):
        """Basic binomial coefficients."""
        assert binomial(5, 2) == 10
        assert binomial(10, 3) == 120
    
    def test_edges(self):
        """Edge cases."""
        assert binomial(5, 0) == 1
        assert binomial(5, 5) == 1
    
    def test_symmetry(self):
        """C(n, k) = C(n, n-k)."""
        assert binomial(10, 3) == binomial(10, 7)
    
    def test_k_greater_than_n(self):
        """C(n, k) = 0 for k > n."""
        assert binomial(5, 10) == 0
    
    def test_negative_k(self):
        """C(n, k) = 0 for k < 0."""
        assert binomial(5, -1) == 0
    
    def test_pascals_identity(self):
        """C(n, k) = C(n-1, k-1) + C(n-1, k)."""
        n, k = 10, 4
        assert binomial(n, k) == binomial(n-1, k-1) + binomial(n-1, k)
    
    def test_row_sum(self):
        """Sum of row n = 2^n."""
        n = 10
        row_sum = sum(binomial(n, k) for k in range(n + 1))
        assert row_sum == 2**n


class TestBinomialSum:
    """Tests for sum of binomial coefficients."""
    
    def test_basic(self):
        """Basic sums."""
        # C(5,0) + C(5,1) + C(5,2) = 1 + 5 + 10 = 16
        assert binomial_sum(5, 2) == 16
    
    def test_k_zero(self):
        """Sum up to k=0 is C(n,0) = 1."""
        assert binomial_sum(10, 0) == 1
    
    def test_k_negative(self):
        """Sum up to k<0 is 0."""
        assert binomial_sum(10, -1) == 0
    
    def test_k_equals_n(self):
        """Sum up to k=n is 2^n."""
        assert binomial_sum(10, 10) == 2**10
    
    def test_k_greater_than_n(self):
        """Sum up to k>n is 2^n."""
        assert binomial_sum(10, 20) == 2**10


class TestLCM:
    """Tests for least common multiple."""
    
    def test_basic(self):
        """Basic LCM."""
        assert lcm(4, 6) == 12
        assert lcm(3, 5) == 15
    
    def test_coprime(self):
        """LCM of coprime numbers is their product."""
        assert lcm(7, 11) == 77
    
    def test_one_divides_other(self):
        """LCM when one divides the other."""
        assert lcm(3, 12) == 12
        assert lcm(12, 3) == 12
    
    def test_same(self):
        """LCM of same number."""
        assert lcm(5, 5) == 5
    
    def test_with_zero(self):
        """LCM with zero is zero."""
        assert lcm(5, 0) == 0
        assert lcm(0, 5) == 0
        assert lcm(0, 0) == 0
    
    def test_identity(self):
        """gcd * lcm = |a * b|."""
        a, b = 12, 18
        assert gcd(a, b) * lcm(a, b) == abs(a * b)


class TestMobius:
    """Tests for Möbius function."""
    
    def test_one(self):
        """μ(1) = 1."""
        assert mobius(1) == 1
    
    def test_primes(self):
        """μ(p) = -1 for prime p."""
        for p in [2, 3, 5, 7, 11]:
            assert mobius(p) == -1
    
    def test_product_of_distinct_primes(self):
        """μ(p*q) = 1 for distinct primes p, q."""
        assert mobius(6) == 1  # 2 * 3
        assert mobius(10) == 1  # 2 * 5
        assert mobius(15) == 1  # 3 * 5
    
    def test_three_distinct_primes(self):
        """μ(p*q*r) = -1 for three distinct primes."""
        assert mobius(30) == -1  # 2 * 3 * 5
    
    def test_squared_prime(self):
        """μ(n) = 0 if n has squared prime factor."""
        assert mobius(4) == 0  # 2^2
        assert mobius(12) == 0  # 2^2 * 3
        assert mobius(18) == 0  # 2 * 3^2
    
    def test_zero(self):
        """μ(0) = 0."""
        assert mobius(0) == 0
    
    def test_negative(self):
        """μ(negative) = 0."""
        assert mobius(-5) == 0
    
    def test_mobius_sum_property(self):
        """Σ_{d|n} μ(d) = 0 for n > 1, = 1 for n = 1."""
        # This is a fundamental property of Möbius function
        for n in [1, 6, 12, 30]:
            divisors = [d for d in range(1, n + 1) if n % d == 0]
            mobius_sum = sum(mobius(d) for d in divisors)
            expected = 1 if n == 1 else 0
            assert mobius_sum == expected, f"Failed for n={n}"


class TestMathematicalIdentities:
    """Test mathematical identities that should hold."""
    
    def test_euler_product_formula(self):
        """phi(n) = n * Π(1 - 1/p) for primes p dividing n."""
        n = 60  # 2^2 * 3 * 5
        primes = [2, 3, 5]
        expected = n
        for p in primes:
            expected = expected * (p - 1) // p
        assert euler_phi(n) == expected
    
    def test_phi_multiplicative(self):
        """phi is multiplicative: phi(mn) = phi(m)*phi(n) for coprime m,n."""
        m, n = 8, 9  # gcd(8, 9) = 1
        assert gcd(m, n) == 1
        assert euler_phi(m * n) == euler_phi(m) * euler_phi(n)
    
    def test_sum_of_totients(self):
        """Σ_{d|n} phi(d) = n."""
        for n in [1, 6, 12, 24]:
            divisors = [d for d in range(1, n + 1) if n % d == 0]
            phi_sum = sum(euler_phi(d) for d in divisors)
            assert phi_sum == n, f"Failed for n={n}"
