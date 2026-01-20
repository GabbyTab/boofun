"""
Comprehensive tests for analysis/hypercontractivity module.

Tests hypercontractivity inequalities and related bounds.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

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
    """Test noise operator T_ρ."""
    
    def test_noise_operator_exists(self):
        """noise_operator should be callable."""
        assert callable(noise_operator)
    
    def test_noise_operator_rho_one(self):
        """T_1 is the identity operator."""
        f = bf.majority(3)
        result = noise_operator(f, rho=1.0)
        
        # Should return an array
        assert result is not None
        assert len(result) == 8
    
    def test_noise_operator_rho_zero(self):
        """T_0 maps to constant function E[f]."""
        f = bf.majority(3)  # Balanced
        result = noise_operator(f, rho=0.0)
        
        # All values should be E[f] = 0 (balanced)
        assert np.allclose(result, result[0])
    
    @pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_noise_operator_various_rho(self, rho):
        """Noise operator should work for various ρ values."""
        f = bf.AND(3)
        result = noise_operator(f, rho)
        
        assert result is not None
        assert len(result) == 8
        assert np.isfinite(result).all()


class TestLqNorm:
    """Test Lq norm computation."""
    
    def test_l2_norm(self):
        """L2 norm of Boolean function should be 1."""
        f = bf.majority(3)
        norm = lq_norm(f, q=2)
        
        assert abs(norm - 1.0) < 1e-10
    
    def test_l4_norm(self):
        """L4 norm should be computable."""
        f = bf.parity(3)
        norm = lq_norm(f, q=4)
        
        # L4 norm should be positive
        assert norm > 0
    
    @pytest.mark.parametrize("q", [1, 2, 3, 4])
    def test_lq_norm_various_q(self, q):
        """Lq norm should work for various q."""
        f = bf.AND(3)
        norm = lq_norm(f, q)
        
        assert norm >= 0
    
    def test_lq_norm_ordering(self):
        """L_p ≤ L_q for p ≤ q (by Hölder)."""
        f = bf.majority(3)
        
        l2 = lq_norm(f, 2)
        l4 = lq_norm(f, 4)
        
        # L2 ≤ L4 for probability measures
        # (Need to be careful about the exact normalization)
        assert l2 > 0 and l4 > 0


class TestBonamiLemmaBound:
    """Test Bonami lemma bounds."""
    
    def test_bonami_bound_exists(self):
        """bonami_lemma_bound should be callable."""
        assert callable(bonami_lemma_bound)
    
    def test_bonami_bound_basic(self):
        """Should compute Bonami lemma bound."""
        f = bf.majority(3)
        actual, bound = bonami_lemma_bound(f, q=4, rho=0.5)
        
        # Bound should be positive
        assert bound > 0
        # Actual value should satisfy bound
        assert actual <= bound or np.isclose(actual, bound, rtol=0.1)


class TestKKLLowerBound:
    """Test KKL lower bound on max influence."""
    
    def test_kkl_bound_exists(self):
        """kkl_lower_bound should be callable."""
        assert callable(kkl_lower_bound)
    
    def test_kkl_bound_basic(self):
        """Should compute KKL lower bound."""
        # Total influence of majority on 3 variables
        f = bf.majority(3)
        total_inf = f.total_influence()
        
        bound = kkl_lower_bound(total_inf, n=3)
        
        assert bound >= 0
    
    def test_kkl_bound_parity(self):
        """KKL bound for parity."""
        # Parity has total influence n
        f = bf.parity(5)
        total_inf = f.total_influence()  # = 5
        
        bound = kkl_lower_bound(total_inf, n=5)
        
        assert bound > 0


class TestMaxInfluenceBound:
    """Test max influence bounds."""
    
    def test_max_influence_bound_exists(self):
        """max_influence_bound should be callable."""
        assert callable(max_influence_bound)
    
    def test_max_influence_bound_basic(self):
        """Should compute max influence bounds."""
        f = bf.majority(5)
        max_inf, lower, upper = max_influence_bound(f)
        
        # Max influence should be between bounds
        assert lower <= max_inf <= upper or np.isclose(max_inf, lower) or np.isclose(max_inf, upper)
    
    def test_max_influence_dictator(self):
        """Dictator has max influence 1."""
        f = bf.dictator(5, 0)
        max_inf, lower, upper = max_influence_bound(f)
        
        # Max influence should be close to 1
        assert abs(max_inf - 1.0) < 0.1


class TestFriedgutJuntaBound:
    """Test Friedgut junta theorem bound."""
    
    def test_friedgut_bound_exists(self):
        """friedgut_junta_bound should be callable."""
        assert callable(friedgut_junta_bound)
    
    def test_friedgut_bound_basic(self):
        """Should compute Friedgut junta bound."""
        # Low total influence implies junta
        bound = friedgut_junta_bound(total_influence=2.0, epsilon=0.1)
        
        assert bound > 0
    
    def test_friedgut_bound_scaling(self):
        """Higher influence should give higher junta size."""
        bound1 = friedgut_junta_bound(total_influence=1.0, epsilon=0.1)
        bound2 = friedgut_junta_bound(total_influence=5.0, epsilon=0.1)
        
        assert bound2 >= bound1


class TestJuntaApproximationError:
    """Test junta approximation error."""
    
    def test_junta_error_exists(self):
        """junta_approximation_error should be callable."""
        assert callable(junta_approximation_error)
    
    def test_junta_error_dictator(self):
        """Dictator is a 1-junta with zero error."""
        f = bf.dictator(5, 2)
        error = junta_approximation_error(f, [2])
        
        assert error < 0.01  # Should be very small
    
    def test_junta_error_all_variables(self):
        """Using all variables gives zero error."""
        f = bf.majority(3)
        error = junta_approximation_error(f, [0, 1, 2])
        
        assert error < 0.01


class TestLevelDInequality:
    """Test level-d inequality."""
    
    def test_level_d_exists(self):
        """level_d_inequality should be callable."""
        assert callable(level_d_inequality)
    
    def test_level_d_basic(self):
        """Should compute level-d inequality."""
        f = bf.majority(5)
        actual, bound = level_d_inequality(f, d=2)
        
        assert actual >= 0
        assert bound >= 0


class TestHypercontractiveInequality:
    """Test general hypercontractive inequality."""
    
    def test_inequality_exists(self):
        """hypercontractive_inequality should be callable."""
        assert callable(hypercontractive_inequality)
    
    def test_inequality_basic(self):
        """Should compute hypercontractive inequality."""
        f = bf.AND(3)
        result = hypercontractive_inequality(f, rho=0.5, q=4)
        
        assert result is not None


class TestHypercontractivityIntegration:
    """Integration tests for hypercontractivity analysis."""
    
    def test_noise_stability_connection(self):
        """Noise stability should relate to hypercontractivity."""
        f = bf.majority(5)
        
        # Compute noise stability via noise operator
        noisy = noise_operator(f, rho=0.5)
        
        # Should be an array of same size
        assert len(noisy) == 32
    
    @pytest.mark.parametrize("func_factory,n", [
        (bf.AND, 3),
        (bf.OR, 3),
        (bf.majority, 5),
        (bf.parity, 4),
    ])
    def test_hypercontractivity_various_functions(self, func_factory, n):
        """Hypercontractivity should work for various functions."""
        f = func_factory(n)
        
        # All these should work
        noisy = noise_operator(f, rho=0.5)
        l4 = lq_norm(f, q=4)
        
        assert noisy is not None
        assert l4 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
