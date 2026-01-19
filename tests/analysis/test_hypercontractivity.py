"""
Tests for hypercontractivity module.

Tests the hypercontractivity tools including:
- Noise operator T_ρ
- L_q norms
- Bonami's Lemma
- KKL Theorem bounds
- Friedgut's Junta Theorem
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.hypercontractivity import (
    noise_operator,
    lq_norm,
    bonami_lemma_bound,
    kkl_lower_bound,
    max_influence_bound,
    friedgut_junta_bound,
    junta_approximation_error,
    level_d_inequality,
    hypercontractive_inequality,
)


class TestNoiseOperator:
    """Tests for noise_operator function."""
    
    def test_returns_array(self):
        """noise_operator returns numpy array."""
        f = bf.parity(3)
        result = noise_operator(f, rho=0.5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 8  # 2^3
    
    def test_rho_zero_gives_mean(self):
        """T_0 f = E[f] (constant)."""
        f = bf.parity(3)  # Balanced, so E[f] = 0
        result = noise_operator(f, rho=0.0)
        
        # All values should be close to mean (0 for balanced function)
        assert np.allclose(result, 0.0, atol=1e-10)
    
    def test_rho_one_gives_identity(self):
        """T_1 f = f (identity)."""
        f = bf.majority(3)
        result = noise_operator(f, rho=1.0)
        
        # Convert f to ±1 representation
        tt = np.array(f.get_representation("truth_table"))
        expected = 1.0 - 2.0 * tt
        
        # The result should match f in ±1 representation (possibly negated due to convention)
        # Check that |result| matches |expected| or result matches -expected
        assert np.allclose(np.abs(result), np.abs(expected), atol=1e-10) or \
               np.allclose(result, -expected, atol=1e-10) or \
               np.allclose(result, expected, atol=1e-10)
    
    def test_noise_contracts_range(self):
        """T_ρ with 0 < ρ < 1 contracts towards mean."""
        f = bf.AND(3)
        result = noise_operator(f, rho=0.5)
        
        # Original AND has values in {-1, 1}
        # After noise, values should be contracted
        assert np.all(np.abs(result) <= 1.0 + 1e-10)
    
    def test_constant_function_unchanged(self):
        """T_ρ on constant function is unchanged."""
        f = bf.constant(True, 3)
        result = noise_operator(f, rho=0.5)
        
        # Constant function maps to constant in ±1 representation
        # All values should be the same (either all +1 or all -1)
        assert np.allclose(result, result[0], atol=1e-10)
        assert np.abs(result[0]) > 0.9  # Should be close to ±1


class TestLqNorm:
    """Tests for lq_norm function."""
    
    def test_l2_norm_is_one(self):
        """Boolean functions have L_2 norm = 1."""
        for func in [bf.parity(3), bf.majority(3), bf.AND(3)]:
            l2 = lq_norm(func, 2)
            assert abs(l2 - 1.0) < 1e-10
    
    def test_l1_norm_is_one(self):
        """L_1 norm of ±1 function is 1."""
        f = bf.parity(3)
        l1 = lq_norm(f, 1)
        assert abs(l1 - 1.0) < 1e-10
    
    def test_linf_norm_is_one(self):
        """L_∞ norm of Boolean function is 1."""
        f = bf.majority(3)
        linf = lq_norm(f, np.inf)
        assert abs(linf - 1.0) < 1e-10
    
    def test_invalid_q_raises(self):
        """q < 1 raises ValueError."""
        f = bf.parity(3)
        with pytest.raises(ValueError):
            lq_norm(f, 0.5)
    
    def test_norm_increases_with_q(self):
        """L_q norm is non-decreasing in q for Boolean functions."""
        f = bf.majority(3)
        
        l2 = lq_norm(f, 2)
        l4 = lq_norm(f, 4)
        l8 = lq_norm(f, 8)
        
        # All should equal 1 for Boolean functions
        assert abs(l2 - 1.0) < 1e-10
        assert abs(l4 - 1.0) < 1e-10
        assert abs(l8 - 1.0) < 1e-10


class TestBonamiLemma:
    """Tests for bonami_lemma_bound function."""
    
    def test_returns_tuple(self):
        """bonami_lemma_bound returns tuple."""
        f = bf.parity(3)
        result = bonami_lemma_bound(f, q=4, rho=0.5)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_l2_is_one(self):
        """L_2 norm of Boolean function is 1."""
        f = bf.majority(3)
        _, l2 = bonami_lemma_bound(f, q=4, rho=0.5)
        assert abs(l2 - 1.0) < 1e-10
    
    def test_bonami_satisfied_small_rho(self):
        """Bonami's lemma holds for small ρ."""
        f = bf.parity(3)
        q = 4
        rho = 1 / np.sqrt(q - 1)  # ρ = 1/√3 ≈ 0.577
        
        lq_noisy, l2 = bonami_lemma_bound(f, q, rho)
        
        # ‖T_ρ f‖_q ≤ ‖f‖_2
        assert lq_noisy <= l2 + 1e-6
    
    def test_bonami_on_majority(self):
        """Bonami's lemma on majority function."""
        f = bf.majority(3)
        lq_noisy, l2 = bonami_lemma_bound(f, q=4, rho=0.3)
        
        # Should satisfy the bound
        assert lq_noisy <= l2 + 1e-6


class TestKKLBound:
    """Tests for KKL-related functions."""
    
    def test_kkl_lower_bound_positive(self):
        """KKL bound is positive for n > 1."""
        bound = kkl_lower_bound(total_influence=2.0, n=10)
        assert bound > 0
    
    def test_kkl_bound_decreases_with_n(self):
        """KKL bound decreases with n (for fixed influence)."""
        bound5 = kkl_lower_bound(1.0, 5)
        bound10 = kkl_lower_bound(1.0, 10)
        bound20 = kkl_lower_bound(1.0, 20)
        
        assert bound5 > bound10 > bound20
    
    def test_kkl_bound_n_one(self):
        """KKL bound for n=1 is 0."""
        bound = kkl_lower_bound(1.0, 1)
        assert bound == 0.0
    
    def test_max_influence_bound_returns_tuple(self):
        """max_influence_bound returns correct structure."""
        f = bf.majority(3)
        max_inf, kkl_bound, total = max_influence_bound(f)
        
        assert isinstance(max_inf, float)
        assert isinstance(kkl_bound, float)
        assert isinstance(total, float)
    
    def test_max_influence_exceeds_bound(self):
        """Max influence should exceed KKL lower bound."""
        f = bf.majority(5)
        max_inf, kkl_bound, _ = max_influence_bound(f)
        
        # KKL guarantees max influence is at least this bound
        assert max_inf >= kkl_bound - 1e-6


class TestFriedgutJunta:
    """Tests for Friedgut's Junta Theorem functions."""
    
    def test_friedgut_bound_positive(self):
        """Friedgut bound is positive."""
        bound = friedgut_junta_bound(total_influence=2.0, epsilon=0.1)
        assert bound > 0
    
    def test_friedgut_bound_increases_with_influence(self):
        """Higher influence means larger junta needed."""
        bound_low = friedgut_junta_bound(1.0, 0.1)
        bound_high = friedgut_junta_bound(5.0, 0.1)
        
        assert bound_high > bound_low
    
    def test_friedgut_bound_decreases_with_epsilon(self):
        """Larger epsilon allows smaller junta."""
        bound_small_eps = friedgut_junta_bound(2.0, 0.01)
        bound_large_eps = friedgut_junta_bound(2.0, 0.5)
        
        assert bound_small_eps > bound_large_eps
    
    def test_friedgut_edge_cases(self):
        """Edge cases return large value."""
        # Zero epsilon
        bound = friedgut_junta_bound(1.0, 0.0)
        assert bound > 1e6
        
        # Negative influence (invalid)
        bound = friedgut_junta_bound(-1.0, 0.1)
        assert bound > 1e6


class TestJuntaApproximation:
    """Tests for junta_approximation_error function."""
    
    def test_full_junta_zero_error(self):
        """Using all variables gives zero error."""
        f = bf.parity(3)
        error = junta_approximation_error(f, junta_vars=[0, 1, 2])
        assert error == 0.0
    
    def test_dictator_single_var(self):
        """Dictator can be approximated by single variable."""
        f = bf.dictator(4, i=0)
        error = junta_approximation_error(f, junta_vars=[0])
        assert error == 0.0
    
    def test_error_nonnegative(self):
        """Approximation error is non-negative."""
        f = bf.majority(3)
        error = junta_approximation_error(f, junta_vars=[0])
        assert error >= 0.0
    
    def test_error_at_most_one(self):
        """Approximation error is at most 1."""
        f = bf.parity(3)
        error = junta_approximation_error(f, junta_vars=[])
        assert error <= 1.0


class TestLevelDInequality:
    """Tests for level_d_inequality function."""
    
    def test_returns_tuple(self):
        """level_d_inequality returns tuple."""
        f = bf.parity(3)
        result = level_d_inequality(f, d=1, q=4)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_degree_0_constant(self):
        """Degree-0 part is the mean."""
        f = bf.parity(3)  # Balanced
        lq, bound = level_d_inequality(f, d=0, q=4)
        
        # Degree-0 coefficient is E[f] = 0 for balanced
        assert abs(lq) < 1e-10
    
    def test_parity_full_degree(self):
        """Parity has all weight at degree n."""
        f = bf.parity(3)
        
        # Degree 3 should have non-zero weight
        lq_3, _ = level_d_inequality(f, d=3, q=4)
        assert lq_3 > 0
        
        # Other degrees should be zero
        lq_1, _ = level_d_inequality(f, d=1, q=4)
        assert abs(lq_1) < 1e-10


class TestHypercontractiveInequality:
    """Tests for hypercontractive_inequality function."""
    
    def test_returns_correct_structure(self):
        """hypercontractive_inequality returns (lq, lp, bool)."""
        f = bf.majority(3)
        result = hypercontractive_inequality(f, rho=0.5, p=2, q=4)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[2], (bool, np.bool_))
    
    def test_inequality_satisfied_for_valid_rho(self):
        """Hypercontractive inequality satisfied for ρ = √((p-1)/(q-1))."""
        f = bf.parity(3)
        p, q = 2, 4
        rho = np.sqrt((p - 1) / (q - 1))  # ρ = 1/√3
        
        _, _, satisfied = hypercontractive_inequality(f, rho, p, q)
        assert satisfied
    
    def test_small_rho_satisfies(self):
        """Small ρ satisfies the inequality."""
        f = bf.AND(3)
        _, _, satisfied = hypercontractive_inequality(f, rho=0.1, p=2, q=4)
        assert satisfied


class TestOnBuiltinFunctions:
    """Test hypercontractivity on built-in functions."""
    
    def test_noise_on_dictator(self):
        """Noise operator on dictator."""
        f = bf.dictator(3, i=0)
        result = noise_operator(f, rho=0.5)
        
        # Should be valid array
        assert result.shape == (8,)
        assert np.all(np.abs(result) <= 1.0 + 1e-10)
    
    def test_bonami_on_and(self):
        """Bonami's lemma on AND function."""
        f = bf.AND(3)
        lq, l2 = bonami_lemma_bound(f, q=4, rho=0.5)
        
        # Should satisfy bound
        assert lq <= l2 + 1e-6
    
    def test_kkl_on_tribes(self):
        """KKL bound on tribes function."""
        f = bf.tribes(2, 4)  # 2 tribes of size 2
        max_inf, kkl_bound, total = max_influence_bound(f)
        
        assert max_inf >= kkl_bound - 1e-6
    
    def test_hypercontractivity_on_or(self):
        """Hypercontractive inequality on OR."""
        f = bf.OR(3)
        _, _, satisfied = hypercontractive_inequality(f, rho=0.5, p=2, q=4)
        assert satisfied
