"""
Fuzz Testing for BooFun API

Uses Hypothesis to generate random inputs and test API robustness.
These tests aim to find edge cases, crashes, and unexpected behavior.

Run with: pytest tests/fuzz/ -v --hypothesis-show-statistics
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis import SpectralAnalyzer, PropertyTester
from boofun.analysis.certificates import certificate, max_certificate_size
from boofun.analysis.block_sensitivity import max_block_sensitivity
from boofun.analysis.query_complexity import QueryComplexityProfile


# =============================================================================
# Strategies for generating Boolean functions
# =============================================================================

@st.composite
def truth_tables(draw, n_vars=None, max_vars=6):
    """Generate random truth tables."""
    if n_vars is None:
        n_vars = draw(st.integers(min_value=1, max_value=max_vars))
    size = 2 ** n_vars
    return draw(st.lists(st.booleans(), min_size=size, max_size=size))


@st.composite
def boolean_functions(draw, max_vars=6):
    """Generate random Boolean functions."""
    tt = draw(truth_tables(max_vars=max_vars))
    return bf.create(tt)


@st.composite
def small_functions(draw):
    """Generate small functions (n <= 4) for expensive tests."""
    return draw(boolean_functions(max_vars=4))


# =============================================================================
# Core API Fuzz Tests
# =============================================================================

class TestCreateFuzz:
    """Fuzz test bf.create() with various inputs."""
    
    @given(truth_tables())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_create_from_list(self, tt):
        """Create from any valid truth table list."""
        f = bf.create(tt)
        assert f.n_vars == int(np.log2(len(tt)))
        
        # Verify evaluation matches
        for i, val in enumerate(tt):
            assert f.evaluate(i) == val
    
    @given(truth_tables())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_create_from_numpy(self, tt):
        """Create from numpy array."""
        arr = np.array(tt, dtype=bool)
        f = bf.create(arr)
        assert f.n_vars == int(np.log2(len(tt)))
    
    @given(st.integers(min_value=1, max_value=7))
    def test_builtin_functions(self, n):
        """Built-in functions should work for any valid n."""
        # Test all builtin function generators
        funcs = [
            bf.AND(n),
            bf.OR(n),
            bf.parity(n),
        ]
        
        if n % 2 == 1:  # Majority requires odd n
            funcs.append(bf.majority(n))
        
        if n > 0:
            funcs.append(bf.dictator(n, 0))
        
        for f in funcs:
            assert f.n_vars == n
            assert len(f.get_representation("truth_table")) == 2**n


# =============================================================================
# Fourier Analysis Fuzz Tests
# =============================================================================

class TestFourierFuzz:
    """Fuzz test Fourier analysis."""
    
    @given(boolean_functions())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_parseval_always_holds(self, f):
        """Parseval identity should hold for any function."""
        fourier = f.fourier()
        sum_sq = np.sum(fourier ** 2)
        assert abs(sum_sq - 1.0) < 1e-9, f"Parseval failed: sum = {sum_sq}"
    
    @given(boolean_functions())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_influences_bounded(self, f):
        """Influences should be in [0, 1]."""
        influences = f.influences()
        assert all(0 <= inf <= 1 + 1e-10 for inf in influences)
    
    @given(boolean_functions())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_total_influence_non_negative(self, f):
        """Total influence should be non-negative."""
        ti = f.total_influence()
        assert ti >= -1e-10
    
    @given(boolean_functions(), st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_noise_stability_bounded(self, f, rho):
        """Noise stability should be in [-1, 1]."""
        stab = f.noise_stability(rho)
        assert -1 - 1e-10 <= stab <= 1 + 1e-10


# =============================================================================
# Query Complexity Fuzz Tests
# =============================================================================

class TestComplexityFuzz:
    """Fuzz test complexity measures."""
    
    @given(small_functions())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_certificate_bounded(self, f):
        """Certificate complexity should be <= n."""
        n = f.n_vars
        C = max_certificate_size(f)
        assert 0 <= C <= n
    
    @given(small_functions())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_block_sensitivity_bounded(self, f):
        """Block sensitivity should be <= n."""
        n = f.n_vars
        bs = max_block_sensitivity(f)
        assert 0 <= bs <= n
    
    @given(small_functions())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_complexity_chain(self, f):
        """s(f) <= bs(f) <= C(f) should hold."""
        from boofun.analysis.huang import max_sensitivity
        
        s = max_sensitivity(f)
        bs = max_block_sensitivity(f)
        C = max_certificate_size(f)
        
        assert s <= bs + 0.01, f"s={s} > bs={bs}"
        assert bs <= C + 0.01, f"bs={bs} > C={C}"


# =============================================================================
# Property Testing Fuzz Tests
# =============================================================================

class TestPropertyTesterFuzz:
    """Fuzz test PropertyTester."""
    
    @given(boolean_functions())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_tester_no_crash(self, f):
        """PropertyTester should not crash on any function."""
        tester = PropertyTester(f, random_seed=42)
        
        # These should all return without crashing
        tester.blr_linearity_test(num_queries=50)
        tester.monotonicity_test(num_queries=50)
        tester.balanced_test()
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_parity_always_linear(self, n):
        """Parity should always pass linearity test."""
        f = bf.parity(n)
        tester = PropertyTester(f, random_seed=42)
        assert tester.blr_linearity_test(num_queries=100)
    
    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_and_always_monotone(self, n):
        """AND should always pass monotonicity test."""
        f = bf.AND(n)
        tester = PropertyTester(f, random_seed=42)
        assert tester.monotonicity_test(num_queries=100)


# =============================================================================
# Representation Fuzz Tests
# =============================================================================

class TestRepresentationFuzz:
    """Fuzz test representation conversions."""
    
    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_truth_table_roundtrip(self, f):
        """Truth table should roundtrip correctly."""
        tt1 = f.get_representation("truth_table")
        f2 = bf.create(list(tt1))
        tt2 = f2.get_representation("truth_table")
        
        assert np.array_equal(tt1, tt2)
    
    @given(boolean_functions(max_vars=4))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_fourier_roundtrip(self, f):
        """Fourier should be consistent with evaluation."""
        fourier = f.fourier()
        
        # Reconstruct function from Fourier and verify
        n = f.n_vars
        for x in range(min(16, 2**n)):  # Sample some inputs
            # Compute f(x) from Fourier expansion
            reconstructed = 0.0
            for S in range(2**n):
                # chi_S(x) = (-1)^(|x ∩ S|)
                chi = 1 if bin(x & S).count('1') % 2 == 0 else -1
                reconstructed += fourier[S] * chi
            
            # Convert from ±1 to Boolean
            # O'Donnell: f(x)=0 → +1, f(x)=1 → -1
            expected_pm = -1 if f.evaluate(x) else 1
            assert abs(reconstructed - expected_pm) < 1e-9


# =============================================================================
# Edge Case Fuzz Tests
# =============================================================================

class TestEdgeCasesFuzz:
    """Fuzz test edge cases."""
    
    @given(st.integers(min_value=1, max_value=7))
    def test_constant_functions(self, n):
        """Constant functions should work correctly."""
        f_zero = bf.create([False] * (2**n))
        f_one = bf.create([True] * (2**n))
        
        # Both should have degree 0
        assert f_zero.degree() == 0
        assert f_one.degree() == 0
        
        # Parseval should hold
        assert abs(np.sum(f_zero.fourier()**2) - 1.0) < 1e-9
        assert abs(np.sum(f_one.fourier()**2) - 1.0) < 1e-9
    
    @given(st.integers(min_value=1, max_value=6), st.integers(min_value=0))
    def test_dictator_any_variable(self, n, var_seed):
        """Dictator on any variable should work."""
        var = var_seed % n
        f = bf.dictator(n, var)
        
        assert f.degree() == 1
        influences = f.influences()
        
        # Only the dictator variable should have influence 1
        for i in range(n):
            if i == var:
                assert abs(influences[i] - 1.0) < 1e-9
            else:
                assert abs(influences[i]) < 1e-9
    
    @given(st.integers(min_value=3, max_value=7).filter(lambda x: x % 2 == 1))
    def test_majority_symmetric(self, n):
        """Majority should have equal influences."""
        f = bf.majority(n)
        influences = f.influences()
        
        # All influences should be equal
        assert np.allclose(influences, influences[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
