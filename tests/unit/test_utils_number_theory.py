import sys

sys.path.insert(0, "src")
from boofun.utils import number_theory as nt


def test_gcd_and_invmod():
    assert nt.gcd(48, 18) == 6
    assert nt.invmod(3, 11) == 4


def test_crt_round_trip():
    value, modulus = nt.crt([3, 5], [2, 1])
    assert modulus == 15
    assert value % 3 == 2
    assert value % 5 == 1


def test_is_prime_and_sieve():
    primes = nt.prime_sieve(10)
    assert primes == [2, 3, 5, 7]
    assert nt.is_prime(97)
    assert not nt.is_prime(1)
    assert not nt.is_prime(91)
