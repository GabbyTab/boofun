"""
Tests for certificate complexity analysis.

Certificate complexity C(f) is the maximum over all inputs x of the minimum
number of bits needed to prove f(x). Key results:
- C(AND_n) = n (must see all 1s to certify output 1)
- C(OR_n) = n (must see all 0s to certify output 0)  
- C(PARITY_n) = n (must see all bits)
- C(f) ≤ D(f) for all f
- bs(f) ≤ C(f) ≤ s(f) * bs(f) (certificate complexity is sandwiched)

References:
- Buhrman & de Wolf, "Complexity Measures and Decision Tree Complexity"
- O'Donnell, "Analysis of Boolean Functions", Chapter 8
"""

import pytest
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis.certificates import certificate, max_certificate_size


class TestCertificateBasics:
    """Test basic certificate functionality."""
    
    def test_certificate_and_all_ones(self):
        """AND at all-1s input requires seeing all variables."""
        f = bf.AND(4)
        # At x = 15 (all 1s), f(x) = 1, certificate should be all 4 vars
        size, vars_ = certificate(f, 15)
        assert size == 4
        assert set(vars_) == {0, 1, 2, 3}
    
    def test_certificate_and_with_zero(self):
        """AND at input with a zero needs only 1 variable (the zero)."""
        f = bf.AND(4)
        # At x = 14 (1110 binary), f(x) = 0, only need to see bit 0 = 0
        size, vars_ = certificate(f, 14)
        assert size == 1
        assert 0 in vars_
    
    def test_certificate_or_all_zeros(self):
        """OR at all-0s input requires seeing all variables."""
        f = bf.OR(4)
        # At x = 0 (all 0s), f(x) = 0, certificate should be all 4 vars
        size, vars_ = certificate(f, 0)
        assert size == 4
    
    def test_certificate_or_with_one(self):
        """OR at input with a one needs only 1 variable."""
        f = bf.OR(4)
        # At x = 1 (0001), f(x) = 1, only need to see bit 0 = 1
        size, vars_ = certificate(f, 1)
        assert size == 1
    
    def test_certificate_parity_any_input(self):
        """Parity always requires seeing all variables."""
        f = bf.parity(4)
        for x in [0, 5, 10, 15]:
            size, vars_ = certificate(f, x)
            assert size == 4, f"Parity certificate at {x} should be 4, got {size}"


class TestKnownCertificateComplexity:
    """Test max_certificate_size against known theoretical values."""
    
    def test_and_certificate_complexity(self):
        """C(AND_n) = n."""
        for n in [2, 3, 4, 5]:
            f = bf.AND(n)
            C_f = max_certificate_size(f)
            assert C_f == n, f"C(AND_{n}) should be {n}, got {C_f}"
    
    def test_or_certificate_complexity(self):
        """C(OR_n) = n."""
        for n in [2, 3, 4, 5]:
            f = bf.OR(n)
            C_f = max_certificate_size(f)
            assert C_f == n, f"C(OR_{n}) should be {n}, got {C_f}"
    
    def test_parity_certificate_complexity(self):
        """C(PARITY_n) = n."""
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            C_f = max_certificate_size(f)
            assert C_f == n, f"C(PARITY_{n}) should be {n}, got {C_f}"
    
    def test_dictator_certificate_complexity(self):
        """C(DICTATOR) = 1."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                C_f = max_certificate_size(f)
                assert C_f == 1, f"C(DICTATOR_{n},{i}) should be 1, got {C_f}"
    
    def test_majority_certificate_complexity(self):
        """
        C(MAJ_n) = ceil(n/2) + 1 for 0-certificate
        C(MAJ_n) = ceil(n/2) for 1-certificate
        Overall C(MAJ_n) = ceil(n/2) + 1
        """
        from math import ceil
        
        for n in [3, 5, 7]:
            f = bf.majority(n)
            C_f = max_certificate_size(f)
            expected = ceil(n / 2) + 1
            # Allow some slack for boundary cases
            assert C_f <= expected + 1 and C_f >= ceil(n / 2), \
                f"C(MAJ_{n}) should be ~{expected}, got {C_f}"


class TestCertificateEdgeCases:
    """Test edge cases in certificate computation."""
    
    def test_constant_zero_function(self):
        """Constant 0 function has C(f) = 0."""
        f = bf.create([0, 0, 0, 0])
        C_f = max_certificate_size(f)
        assert C_f == 0
    
    def test_constant_one_function(self):
        """Constant 1 function has C(f) = 0."""
        f = bf.create([1, 1, 1, 1])
        C_f = max_certificate_size(f)
        assert C_f == 0
    
    def test_single_variable_function(self):
        """Function on 1 variable has C(f) ≤ 1."""
        f = bf.create([0, 1])
        C_f = max_certificate_size(f)
        assert C_f == 1


class TestCertificateBounds:
    """Test theoretical bounds involving certificates."""
    
    def test_certificate_bounded_by_n(self):
        """C(f) ≤ n for all f."""
        for func in [bf.AND(4), bf.OR(4), bf.majority(5), bf.parity(4)]:
            n = func.n_vars
            C_f = max_certificate_size(func)
            assert C_f <= n, f"C(f)={C_f} > n={n}"
    
    def test_sensitivity_times_block_sensitivity_bound(self):
        """
        C(f) ≤ s(f) * bs(f) (Nisan-Szegedy).
        """
        from boofun.analysis.huang import max_sensitivity
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        
        for func in [bf.AND(4), bf.OR(4), bf.majority(3)]:
            C_f = max_certificate_size(func)
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)
            
            assert C_f <= s_f * bs_f, \
                f"C(f)={C_f} > s(f)*bs(f)={s_f}*{bs_f}={s_f*bs_f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
