"""
Tests for families/theoretical module.

Tests for TheoreticalBounds class with known theoretical results.
"""

import pytest
import numpy as np

from boofun.families.theoretical import TheoreticalBounds


class TestTotalInfluenceBounds:
    """Tests for total influence bounds."""
    
    def test_majority_total_influence(self):
        """Majority total influence ≈ √(2/π)·√n."""
        result = TheoreticalBounds.majority_total_influence(9)
        
        expected = np.sqrt(2 / np.pi) * 3
        assert abs(result - expected) < 0.01
    
    def test_majority_scales_with_sqrt_n(self):
        """Majority influence scales as √n."""
        inf_9 = TheoreticalBounds.majority_total_influence(9)
        inf_36 = TheoreticalBounds.majority_total_influence(36)
        
        # √36 / √9 = 6/3 = 2
        assert abs(inf_36 / inf_9 - 2.0) < 0.01
    
    def test_parity_total_influence(self):
        """Parity total influence = n."""
        assert TheoreticalBounds.parity_total_influence(5) == 5.0
        assert TheoreticalBounds.parity_total_influence(10) == 10.0
    
    def test_and_total_influence(self):
        """AND total influence = n·2^{-(n-1)}."""
        result = TheoreticalBounds.and_total_influence(3)
        
        expected = 3 * 2**(-2)  # 3/4 = 0.75
        assert abs(result - expected) < 0.01
    
    def test_tribes_total_influence(self):
        """Tribes total influence ≈ log(n)."""
        result = TheoreticalBounds.tribes_total_influence(16)
        
        assert result > 0


class TestNoiseStabilityBounds:
    """Tests for noise stability bounds."""
    
    def test_majority_noise_stability_sheppard(self):
        """Majority noise stability follows Sheppard's formula."""
        rho = 0.5
        result = TheoreticalBounds.majority_noise_stability(rho)
        
        expected = 0.5 + (1/np.pi) * np.arcsin(0.5)
        assert abs(result - expected) < 0.01
    
    def test_majority_noise_stability_rho_zero(self):
        """At ρ=0, majority stability is 0.5."""
        result = TheoreticalBounds.majority_noise_stability(0.0)
        
        assert abs(result - 0.5) < 0.01
    
    def test_majority_noise_stability_rho_one(self):
        """At ρ=1, majority stability is 1."""
        result = TheoreticalBounds.majority_noise_stability(1.0)
        
        assert abs(result - 1.0) < 0.01
    
    def test_parity_noise_stability(self):
        """Parity noise stability = ρ^n."""
        result = TheoreticalBounds.parity_noise_stability(3, 0.5)
        
        expected = 0.5 ** 3
        assert abs(result - expected) < 0.01
    
    def test_parity_noise_stability_converges_to_zero(self):
        """Parity stability → 0 as n → ∞ for ρ < 1."""
        stab_small = TheoreticalBounds.parity_noise_stability(5, 0.5)
        stab_large = TheoreticalBounds.parity_noise_stability(20, 0.5)
        
        assert stab_large < stab_small
    
    def test_dictator_noise_stability(self):
        """Dictator noise stability = ρ."""
        assert TheoreticalBounds.dictator_noise_stability(0.7) == 0.7
        assert TheoreticalBounds.dictator_noise_stability(0.3) == 0.3
    
    def test_and_noise_stability(self):
        """AND noise stability ≈ ((1+ρ)/2)^n."""
        result = TheoreticalBounds.and_noise_stability(3, 0.5)
        
        expected = (0.75) ** 3
        assert abs(result - expected) < 0.01


class TestInfluenceBounds:
    """Tests for individual influence bounds."""
    
    def test_majority_influence_i(self):
        """Majority influence = √(2/(πn)) for all i."""
        result = TheoreticalBounds.majority_influence_i(9)
        
        expected = np.sqrt(2 / (np.pi * 9))
        assert abs(result - expected) < 0.01
    
    def test_parity_influence_i(self):
        """Parity influence = 1 for all variables."""
        assert TheoreticalBounds.parity_influence_i(5) == 1.0
        assert TheoreticalBounds.parity_influence_i(10, i=5) == 1.0
    
    def test_and_influence_i(self):
        """AND influence = 2^{-(n-1)} for all variables."""
        result = TheoreticalBounds.and_influence_i(3)
        
        expected = 2**(-2)
        assert abs(result - expected) < 0.01


class TestPoincarreAndKKL:
    """Tests for Poincaré and KKL bounds."""
    
    def test_poincare_lower_bound(self):
        """Poincaré: I[f] ≥ Var[f]."""
        result = TheoreticalBounds.poincare_lower_bound(0.5)
        
        assert result == 0.5
    
    def test_kkl_lower_bound(self):
        """KKL: max Inf_i ≥ Var·c·log(n)/n."""
        result = TheoreticalBounds.kkl_lower_bound(100, 0.5)
        
        expected = 0.5 * 1.0 * np.log(100) / 100
        assert abs(result - expected) < 0.01
    
    def test_friedgut_junta_bound(self):
        """Friedgut: f is ε-close to 2^{O(k/ε)}-junta."""
        result = TheoreticalBounds.friedgut_junta_bound(2.0, 0.1)
        
        expected = 2 ** (4 * 2.0 / 0.1)
        assert result == expected


class TestFourierConcentration:
    """Tests for Fourier concentration bounds."""
    
    def test_decision_tree_fourier_support(self):
        """Decision tree has at most 4^d Fourier coefficients."""
        result = TheoreticalBounds.decision_tree_fourier_support(3)
        
        assert result == 4**3
    
    def test_decision_tree_spectral_norm(self):
        """Decision tree of size s has spectral norm ≤ s."""
        result = TheoreticalBounds.decision_tree_spectral_norm(10)
        
        assert result == 10.0
    
    def test_mansour_spectral_concentration(self):
        """Mansour's theorem for DNF."""
        result = TheoreticalBounds.mansour_spectral_concentration(2, 0.1)
        
        expected = int((1/0.1)**2 * 2**(2*2))
        assert result == expected


class TestLTFBounds:
    """Tests for LTF-specific bounds."""
    
    def test_ltf_total_influence_regular(self):
        """Regular LTF influence ≈ √(2/π)·√n."""
        result = TheoreticalBounds.ltf_total_influence(9, regularity=0.0)
        
        expected = np.sqrt(2/np.pi) * 3
        assert abs(result - expected) < 0.01
    
    def test_ltf_total_influence_dictator(self):
        """Dictator (regularity=1) has influence → 1."""
        result = TheoreticalBounds.ltf_total_influence(9, regularity=1.0)
        
        assert abs(result - 1.0) < 0.01
    
    def test_ltf_noise_stability_regular(self):
        """Regular LTF stability follows Sheppard's formula."""
        result = TheoreticalBounds.ltf_noise_stability(0.5, regularity=0.0)
        
        expected = 0.5 + (1/np.pi) * np.arcsin(0.5)
        assert abs(result - expected) < 0.01
    
    def test_ltf_noise_stability_dictator(self):
        """Dictator stability = ρ."""
        result = TheoreticalBounds.ltf_noise_stability(0.7, regularity=1.0)
        
        assert abs(result - 0.7) < 0.01


class TestQueryComplexityBounds:
    """Tests for query complexity bounds."""
    
    def test_sensitivity_vs_block_sensitivity(self):
        """bs(f) ≤ s(f)^4."""
        result = TheoreticalBounds.sensitivity_vs_block_sensitivity(3.0)
        
        assert result == 3.0 ** 4
    
    def test_certificate_vs_sensitivity(self):
        """C(f) ≤ s(f)^5."""
        result = TheoreticalBounds.certificate_vs_sensitivity(2.0)
        
        assert result == 2.0 ** 5
    
    def test_degree_vs_sensitivity(self):
        """deg(f) ≤ s(f)^2."""
        result = TheoreticalBounds.degree_vs_sensitivity(4.0)
        
        assert result == 4.0 ** 2


class TestGetBoundsForFamily:
    """Tests for get_bounds_for_family method."""
    
    def test_majority_bounds(self):
        """Get bounds for majority family."""
        bounds = TheoreticalBounds.get_bounds_for_family("majority")
        
        assert "total_influence" in bounds
        assert "influence_i" in bounds
        assert "noise_stability" in bounds
    
    def test_parity_bounds(self):
        """Get bounds for parity family."""
        bounds = TheoreticalBounds.get_bounds_for_family("parity")
        
        assert "total_influence" in bounds
        assert callable(bounds["total_influence"])
    
    def test_and_bounds(self):
        """Get bounds for AND family."""
        bounds = TheoreticalBounds.get_bounds_for_family("and")
        
        assert "total_influence" in bounds
    
    def test_tribes_bounds(self):
        """Get bounds for tribes family."""
        bounds = TheoreticalBounds.get_bounds_for_family("tribes")
        
        assert "total_influence" in bounds
    
    def test_dictator_bounds(self):
        """Get bounds for dictator family."""
        bounds = TheoreticalBounds.get_bounds_for_family("dictator")
        
        assert "total_influence" in bounds
    
    def test_ltf_bounds(self):
        """Get bounds for LTF family."""
        bounds = TheoreticalBounds.get_bounds_for_family("ltf")
        
        assert "total_influence" in bounds
        assert "noise_stability" in bounds
    
    def test_unknown_family(self):
        """Unknown family returns empty dict."""
        bounds = TheoreticalBounds.get_bounds_for_family("unknown")
        
        assert bounds == {}
    
    def test_bounds_are_callable(self):
        """All returned bounds are callable."""
        bounds = TheoreticalBounds.get_bounds_for_family("majority")
        
        for name, fn in bounds.items():
            assert callable(fn)
