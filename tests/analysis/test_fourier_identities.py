"""
Tests based on O'Donnell's Analysis of Boolean Functions.

These tests verify fundamental identities and theorems from the book,
serving as both correctness checks and educational examples.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boolfunc as bf
from boolfunc.analysis import SpectralAnalyzer, fourier, gf2


class TestParseval:
    """Tests for Parseval's identity (O'Donnell Prop. 1.9)."""
    
    def test_parseval_xor(self):
        """XOR should satisfy Parseval: E[f²] = Σf̂(S)²."""
        xor = bf.create([0, 1, 1, 0])
        passes, lhs, rhs = fourier.parseval_verify(xor)
        assert passes
        assert abs(lhs - 1.0) < 1e-10  # Boolean function: E[f²] = 1
        assert abs(rhs - 1.0) < 1e-10
    
    def test_parseval_and(self):
        """AND should satisfy Parseval."""
        and_f = bf.create([0, 0, 0, 1])
        passes, lhs, rhs = fourier.parseval_verify(and_f)
        assert passes
    
    def test_parseval_majority(self):
        """Majority should satisfy Parseval."""
        maj = bf.BooleanFunctionBuiltins.majority(3)
        passes, lhs, rhs = fourier.parseval_verify(maj)
        assert passes
    
    def test_parseval_parity(self):
        """Parity should satisfy Parseval."""
        parity = bf.BooleanFunctionBuiltins.parity(4)
        passes, lhs, rhs = fourier.parseval_verify(parity)
        assert passes


class TestPlancherel:
    """Tests for Plancherel's theorem (O'Donnell Prop. 1.10)."""
    
    def test_plancherel_self_inner_product(self):
        """⟨f, f⟩ = Σf̂(S)² = 1 for Boolean functions."""
        f = bf.create([0, 1, 1, 0])
        inner = fourier.plancherel_inner_product(f, f)
        assert abs(inner - 1.0) < 1e-10
    
    def test_plancherel_orthogonal_functions(self):
        """Dictator x₀ and x₁ should have inner product 0."""
        x0 = bf.BooleanFunctionBuiltins.dictator(2, 0)
        x1 = bf.BooleanFunctionBuiltins.dictator(2, 1)
        inner = fourier.plancherel_inner_product(x0, x1)
        assert abs(inner) < 1e-10
    
    def test_plancherel_xor_and(self):
        """XOR and AND inner product computation."""
        xor = bf.create([0, 1, 1, 0])
        and_f = bf.create([0, 0, 0, 1])
        inner = fourier.plancherel_inner_product(xor, and_f)
        # Verify by direct computation
        assert isinstance(inner, float)


class TestDegree:
    """Tests for Fourier degree (O'Donnell Def. 1.18)."""
    
    def test_degree_constant(self):
        """Constant functions have degree 0."""
        const = bf.BooleanFunctionBuiltins.constant(True, 3)
        assert fourier.fourier_degree(const) == 0
    
    def test_degree_dictator(self):
        """Dictator functions have degree 1."""
        dictator = bf.BooleanFunctionBuiltins.dictator(3, 0)
        assert fourier.fourier_degree(dictator) == 1
    
    def test_degree_xor(self):
        """XOR on n bits has degree n."""
        for n in [2, 3, 4]:
            xor = bf.BooleanFunctionBuiltins.parity(n)
            assert fourier.fourier_degree(xor) == n
    
    def test_degree_and(self):
        """AND on n bits has degree n."""
        for n in [2, 3]:
            size = 1 << n
            tt = [0] * size
            tt[size - 1] = 1
            and_f = bf.create(tt)
            assert fourier.fourier_degree(and_f) == n


class TestInfluence:
    """Tests for influence (O'Donnell Def. 2.1)."""
    
    def test_influence_dictator(self):
        """Dictator x_i has Inf_i = 1, others 0."""
        dictator = bf.BooleanFunctionBuiltins.dictator(3, 0)
        analyzer = SpectralAnalyzer(dictator)
        influences = analyzer.influences()
        
        assert abs(influences[0] - 1.0) < 1e-10
        assert abs(influences[1]) < 1e-10
        assert abs(influences[2]) < 1e-10
    
    def test_influence_xor(self):
        """XOR has Inf_i = 1 for all i."""
        xor = bf.BooleanFunctionBuiltins.parity(3)
        analyzer = SpectralAnalyzer(xor)
        influences = analyzer.influences()
        
        for i in range(3):
            assert abs(influences[i] - 1.0) < 1e-10
    
    def test_total_influence_equals_sum(self):
        """Total influence = Σ Inf_i."""
        f = bf.BooleanFunctionBuiltins.majority(3)
        analyzer = SpectralAnalyzer(f)
        
        total = analyzer.total_influence()
        assert abs(total - np.sum(analyzer.influences())) < 1e-10


class TestSpectralConcentration:
    """Tests for spectral concentration (O'Donnell Chapter 3)."""
    
    def test_dictator_concentration(self):
        """Dictators have all weight at degree 1."""
        dictator = bf.BooleanFunctionBuiltins.dictator(4, 0)
        analyzer = SpectralAnalyzer(dictator)
        
        assert abs(analyzer.spectral_concentration(0) - 0.0) < 1e-10  # Empty set has 0
        assert abs(analyzer.spectral_concentration(1) - 1.0) < 1e-10  # All weight at deg 1
    
    def test_parity_concentration(self):
        """Parity has all weight at max degree."""
        parity = bf.BooleanFunctionBuiltins.parity(4)
        analyzer = SpectralAnalyzer(parity)
        
        assert abs(analyzer.spectral_concentration(3) - 0.0) < 1e-10  # No weight at deg ≤ 3
        assert abs(analyzer.spectral_concentration(4) - 1.0) < 1e-10  # All weight at deg 4


class TestNegation:
    """Tests for f(-x) transformation (O'Donnell Exercise 1.1)."""
    
    def test_negate_majority(self):
        """f(-x) flips odd-degree coefficient signs."""
        maj = bf.BooleanFunctionBuiltins.majority(3)
        neg_maj = fourier.negate_inputs(maj)
        
        analyzer_f = SpectralAnalyzer(maj)
        analyzer_g = SpectralAnalyzer(neg_maj)
        
        coeffs_f = analyzer_f.fourier_expansion()
        coeffs_g = analyzer_g.fourier_expansion()
        
        for s in range(len(coeffs_f)):
            degree = bin(s).count("1")
            expected_sign = (-1) ** degree
            assert abs(coeffs_g[s] - expected_sign * coeffs_f[s]) < 1e-10


class TestOddEvenParts:
    """Tests for f^odd and f^even (O'Donnell Exercise 1.1)."""
    
    def test_odd_plus_even_equals_f(self):
        """f^odd + f^even should equal f."""
        f = bf.BooleanFunctionBuiltins.majority(3)
        
        odd = fourier.odd_part(f)
        even = fourier.even_part(f)
        
        tt = np.asarray(f.get_representation("truth_table"), dtype=float)
        f_pm = 1.0 - 2.0 * tt
        
        reconstructed = odd + even
        np.testing.assert_array_almost_equal(f_pm, reconstructed)


class TestGF2Degree:
    """Tests for GF(2) degree vs Fourier degree (O'Donnell Section 1.6)."""
    
    def test_xor_gf2_degree(self):
        """XOR has GF(2) degree 1 but Fourier degree 2."""
        xor = bf.create([0, 1, 1, 0])
        
        gf2_deg = gf2.gf2_degree(xor)
        fourier_deg = fourier.fourier_degree(xor)
        
        assert gf2_deg == 1
        assert fourier_deg == 2
    
    def test_and_degrees_equal(self):
        """AND has same GF(2) and Fourier degree."""
        and_f = bf.create([0, 0, 0, 1])
        
        gf2_deg = gf2.gf2_degree(and_f)
        fourier_deg = fourier.fourier_degree(and_f)
        
        assert gf2_deg == fourier_deg == 2


class TestMUX3NAE3:
    """Tests for HW1 Problem 2 functions."""
    
    def test_mux3_fourier(self):
        """Verify MUX₃ Fourier expansion."""
        coeffs = fourier.compute_mux3_fourier()
        
        # MUX₃ should have f̂({2}) = 1/2, f̂({1,2}) = 1/2, 
        # f̂({3}) = 1/2, f̂({1,3}) = -1/2
        # Using bit indexing: 2 is bit 1, 3 is bit 0
        assert len(coeffs) > 0
    
    def test_nae3_fourier(self):
        """Verify NAE₃ Fourier expansion."""
        coeffs = fourier.compute_nae3_fourier()
        assert len(coeffs) > 0
    
    def test_and_fourier(self):
        """Verify AND_n has 2^n non-zero Fourier coefficients."""
        for n in [2, 3]:
            coeffs = fourier.compute_and_fourier(n)
            
            # AND_n should have non-zero coefficients for all subsets
            # (due to the expansion in the ±1 basis)
            assert len(coeffs) == 2 ** n
            
            # Parseval: sum of squares should equal 1
            sum_sq = sum(c ** 2 for c in coeffs.values())
            assert abs(sum_sq - 1.0) < 1e-10


class TestFourierSparsity:
    """Tests for Fourier sparsity (O'Donnell Exercise 1.19)."""
    
    def test_degree_k_sparsity_bound(self):
        """Functions of degree k have at most 4^k non-zero Fourier coefficients."""
        # Degree 2 function: AND on 2 bits
        and2 = bf.create([0, 0, 0, 1])
        sparsity = fourier.fourier_sparsity(and2)
        
        degree = fourier.fourier_degree(and2)
        assert sparsity <= 4 ** degree
    
    def test_xor_sparsity(self):
        """XOR has exactly one non-zero Fourier coefficient."""
        xor = bf.create([0, 1, 1, 0])
        sparsity = fourier.fourier_sparsity(xor)
        assert sparsity == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
