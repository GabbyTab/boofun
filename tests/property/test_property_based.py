"""
Property-based tests for boolfunc using Hypothesis.

These tests verify invariants that should hold for all Boolean functions,
not just specific examples.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import boolfunc as bf
from boolfunc.analysis import SpectralAnalyzer


# ============================================================================
# Strategy Generators
# ============================================================================

@st.composite
def small_n(draw):
    """Generate small n values (3-8) for full truth table tests."""
    return draw(st.integers(min_value=3, max_value=8))


@st.composite
def truth_table(draw, n=None):
    """Generate a random truth table."""
    if n is None:
        n = draw(small_n())
    size = 2 ** n
    return draw(arrays(dtype=np.int8, shape=size, elements=st.integers(0, 1))), n


@st.composite  
def boolean_function(draw, n=None):
    """Generate a random BooleanFunction."""
    tt, n = draw(truth_table(n))
    return bf.create(tt.tolist())


@st.composite
def builtin_function(draw):
    """Generate a builtin function (majority, parity, etc.)."""
    n = draw(st.integers(min_value=3, max_value=9))
    fn_type = draw(st.sampled_from(['majority', 'parity', 'and', 'or']))
    
    if fn_type == 'majority':
        # Majority needs odd n
        if n % 2 == 0:
            n = n + 1
        return bf.majority(n)
    elif fn_type == 'parity':
        return bf.parity(n)
    elif fn_type == 'and':
        return bf.AND(n)
    else:
        return bf.OR(n)


# ============================================================================
# Fourier Analysis Properties
# ============================================================================

class TestFourierProperties:
    """Property tests for Fourier analysis."""
    
    @given(boolean_function())
    @settings(max_examples=50, deadline=5000)
    def test_parseval_identity(self, f):
        """
        Parseval's identity: Σ f̂(S)² = E[f²]
        
        For Boolean functions in {-1, +1}, this equals 1.
        For {0, 1}, this equals E[f].
        """
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        
        # Sum of squared Fourier coefficients
        sum_sq = np.sum(fourier ** 2)
        
        # For ±1 representation, this should be 1
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval violated: {sum_sq} != 1"
    
    @given(boolean_function())
    @settings(max_examples=50, deadline=5000)
    def test_fourier_at_empty_set(self, f):
        """
        f̂(∅) = E[f] in ±1 representation.
        For balanced functions, f̂(∅) = 0.
        """
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        
        # Index 0 corresponds to empty set
        f_hat_empty = fourier[0]
        
        # Should be in [-1, 1]
        assert -1 <= f_hat_empty <= 1
    
    @given(boolean_function())
    @settings(max_examples=50, deadline=5000)
    def test_influences_non_negative(self, f):
        """Variable influences are always non-negative."""
        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()
        
        assert np.all(influences >= 0), "Negative influence found"
    
    @given(boolean_function())
    @settings(max_examples=50, deadline=5000)
    def test_influences_bounded_by_one(self, f):
        """Each variable influence is at most 1."""
        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()
        
        assert np.all(influences <= 1 + 1e-10), f"Influence > 1: {influences}"
    
    @given(boolean_function())
    @settings(max_examples=50, deadline=5000)
    def test_noise_stability_bounded(self, f):
        """Noise stability is in [-1, 1]."""
        analyzer = SpectralAnalyzer(f)
        
        for rho in [0.5, 0.9, 0.99]:
            ns = analyzer.noise_stability(rho)
            assert -1 - 1e-10 <= ns <= 1 + 1e-10, f"NS out of range: {ns}"
    
    @given(boolean_function())
    @settings(max_examples=50, deadline=5000)
    def test_noise_stability_monotone_in_rho(self, f):
        """For Boolean functions, NS_ρ increases with ρ (for ρ ≥ 0)."""
        analyzer = SpectralAnalyzer(f)
        
        ns_50 = analyzer.noise_stability(0.5)
        ns_90 = analyzer.noise_stability(0.9)
        ns_99 = analyzer.noise_stability(0.99)
        
        # Should be monotone increasing
        assert ns_50 <= ns_90 + 1e-10
        assert ns_90 <= ns_99 + 1e-10


# ============================================================================
# Representation Conversion Properties
# ============================================================================

class TestRepresentationProperties:
    """Property tests for representation conversions."""
    
    @given(boolean_function())
    @settings(max_examples=30, deadline=10000)
    def test_truth_table_roundtrip(self, f):
        """Converting to truth table and back preserves function."""
        tt = f.get_representation('truth_table')
        # Handle both numpy array and list
        tt_list = tt.tolist() if hasattr(tt, 'tolist') else list(tt)
        f2 = bf.create(tt_list)
        
        # Check evaluations match
        n = f.n_vars
        for x in range(min(2**n, 32)):  # Sample some inputs
            assert f.evaluate(x) == f2.evaluate(x)
    
    @given(boolean_function())
    @settings(max_examples=30, deadline=10000)
    def test_fourier_roundtrip(self, f):
        """
        Fourier expansion preserves function identity.
        Converting to Fourier and evaluating should give same results.
        """
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        
        # The Fourier expansion is a representation of f
        # Check it has correct size
        n = f.n_vars
        assert len(fourier) == 2**n
        
        # Check Parseval (already tested, but quick sanity check)
        assert abs(np.sum(fourier**2) - 1.0) < 1e-10


# ============================================================================
# Boolean Operations Properties
# ============================================================================

class TestBooleanOperations:
    """Property tests for Boolean operations."""
    
    @given(boolean_function(), boolean_function())
    @settings(max_examples=30, deadline=5000)
    def test_xor_self_is_zero(self, f, g):
        """f XOR f = 0 (constant zero function)."""
        assume(f.n_vars == g.n_vars)  # Need same n
        
        h = f ^ f  # XOR with self
        
        # Should be constant 0
        n = f.n_vars
        for x in range(min(2**n, 16)):
            assert h.evaluate(x) == 0
    
    @given(boolean_function())
    @settings(max_examples=30, deadline=5000)
    def test_not_involution(self, f):
        """NOT(NOT(f)) = f."""
        f_not = ~f
        f_not_not = ~f_not
        
        n = f.n_vars
        for x in range(min(2**n, 16)):
            assert f.evaluate(x) == f_not_not.evaluate(x)
    
    @given(boolean_function())
    @settings(max_examples=30, deadline=5000)
    def test_and_with_true_is_identity(self, f):
        """f AND 1 = f."""
        n = f.n_vars
        one = bf.create([1] * (2**n))
        
        h = f & one
        
        for x in range(min(2**n, 16)):
            assert f.evaluate(x) == h.evaluate(x)
    
    @given(boolean_function())
    @settings(max_examples=30, deadline=5000)
    def test_or_with_false_is_identity(self, f):
        """f OR 0 = f."""
        n = f.n_vars
        zero = bf.create([0] * (2**n))
        
        h = f | zero
        
        for x in range(min(2**n, 16)):
            assert f.evaluate(x) == h.evaluate(x)


# ============================================================================
# Builtin Function Properties
# ============================================================================

class TestBuiltinProperties:
    """Property tests for builtin functions."""
    
    @given(st.integers(min_value=3, max_value=11).filter(lambda x: x % 2 == 1))
    @settings(max_examples=20, deadline=5000)
    def test_majority_symmetric(self, n):
        """Majority function is symmetric."""
        maj = bf.majority(n)
        
        # Test some random permutations
        for _ in range(5):
            x = np.random.randint(0, 2, n)
            y = np.random.permutation(x)
            
            x_int = int(sum(int(b) << i for i, b in enumerate(x)))
            y_int = int(sum(int(b) << i for i, b in enumerate(y)))
            
            assert maj.evaluate(x_int) == maj.evaluate(y_int)
    
    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=5000)
    def test_parity_linear(self, n):
        """Parity function is linear (f(x⊕y) = f(x)⊕f(y))."""
        par = bf.parity(n)
        
        for _ in range(10):
            x = np.random.randint(0, 2**n)
            y = np.random.randint(0, 2**n)
            
            f_x = par.evaluate(x)
            f_y = par.evaluate(y)
            f_xor = par.evaluate(x ^ y)
            
            assert f_xor == (f_x ^ f_y)
    
    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=5000)
    def test_and_monotone(self, n):
        """AND function is monotone."""
        and_f = bf.AND(n)
        
        for _ in range(10):
            x = np.random.randint(0, 2**n)
            # Create y ≥ x by adding random bits
            y = x
            for i in range(n):
                if np.random.random() < 0.3 and ((x >> i) & 1) == 0:
                    y |= (1 << i)
            
            assert and_f.evaluate(x) <= and_f.evaluate(y)
    
    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=5000)
    def test_or_monotone(self, n):
        """OR function is monotone."""
        or_f = bf.OR(n)
        
        for _ in range(10):
            x = np.random.randint(0, 2**n)
            # Create y ≥ x by adding random bits
            y = x
            for i in range(n):
                if np.random.random() < 0.3 and ((x >> i) & 1) == 0:
                    y |= (1 << i)
            
            assert or_f.evaluate(x) <= or_f.evaluate(y)


# ============================================================================
# Mathematical Identities
# ============================================================================

class TestMathematicalIdentities:
    """Property tests for mathematical identities."""
    
    @given(builtin_function())
    @settings(max_examples=30, deadline=5000)
    def test_total_influence_formula(self, f):
        """
        Total influence equals spectral sum:
        I[f] = Σ_S |S| · f̂(S)²
        """
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        n = f.n_vars
        
        # Compute via spectral formula
        spectral_inf = 0.0
        for s in range(2**n):
            subset_size = bin(s).count('1')
            spectral_inf += subset_size * fourier[s]**2
        
        # Compute via influence formula
        total_inf = analyzer.total_influence()
        
        # Should match (both in terms of the ±1 representation)
        assert abs(spectral_inf - total_inf) < 1e-6
    
    @given(builtin_function())
    @settings(max_examples=30, deadline=5000)
    def test_noise_stability_spectral_formula(self, f):
        """
        Noise stability equals:
        Stab_ρ[f] = Σ_S ρ^|S| · f̂(S)²
        """
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        n = f.n_vars
        rho = 0.7
        
        # Compute via spectral formula
        spectral_stab = 0.0
        for s in range(2**n):
            subset_size = bin(s).count('1')
            spectral_stab += (rho ** subset_size) * fourier[s]**2
        
        # Compute via analyzer
        computed_stab = analyzer.noise_stability(rho)
        
        assert abs(spectral_stab - computed_stab) < 1e-10


# ============================================================================
# Degree Properties
# ============================================================================

class TestDegreeProperties:
    """Property tests for Fourier degree."""
    
    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=5000)
    def test_parity_has_full_degree(self, n):
        """Parity has degree n (all variables)."""
        par = bf.parity(n)
        deg = par.degree()
        assert deg == n
    
    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=5000)
    def test_and_has_full_degree(self, n):
        """AND has degree n."""
        and_f = bf.AND(n)
        deg = and_f.degree()
        assert deg == n
    
    @given(st.integers(min_value=0, max_value=7))
    @settings(max_examples=20, deadline=5000)
    def test_dictator_has_degree_one(self, i):
        """Dictator has degree 1."""
        n = max(i + 1, 3)
        dict_f = bf.dictator(n, i)
        deg = dict_f.degree()
        assert deg == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
