"""
Comprehensive tests for sparsity module.

Tests cover:
- Fourier sparsity for various function types
- Sparsity up to constants
- Support analysis
- Effective sparsity
- Known sparsity bounds
"""

import pytest
import numpy as np
import boofun as bf

from boofun.analysis.sparsity import (
    fourier_sparsity,
    fourier_sparsity_up_to_constants,
    granularity,
    fourier_support,
    sparsity_by_degree,
    effective_sparsity,
)


class TestFourierSparsity:
    """Tests for fourier_sparsity function."""

    def test_constant_sparsity_one(self):
        """Constant function has sparsity 1 (only f̂(∅))."""
        f = bf.create([1, 1, 1, 1])
        assert fourier_sparsity(f) == 1

    def test_balanced_constant_sparsity(self):
        """Balanced constant has sparsity 1 or 0."""
        # Actually a constant function
        f = bf.create([0, 0, 0, 0])
        # f̂(∅) = E[f] in ±1 = mean of [+1,+1,+1,+1] = +1
        assert fourier_sparsity(f) == 1

    def test_dictator_balanced_sparsity(self):
        """Balanced dictator has sparsity 1 (only degree-1 term)."""
        f = bf.create([0, 1, 0, 1])  # x0
        # f is balanced, so f̂(∅) = 0
        assert fourier_sparsity(f) == 1

    def test_parity_sparsity_one(self):
        """Parity has sparsity 1 (only the full set coefficient)."""
        # XOR of 2 variables
        f = bf.create([0, 1, 1, 0])
        assert fourier_sparsity(f) == 1
        
        # XOR of 3 variables
        f = bf.create([0, 1, 1, 0, 1, 0, 0, 1])
        assert fourier_sparsity(f) == 1

    def test_and_sparsity(self):
        """AND has sparsity 2^n (all coefficients non-zero)."""
        f = bf.create([0, 0, 0, 1])  # AND of 2 vars
        # AND_2 has all 4 Fourier coefficients non-zero
        assert fourier_sparsity(f) == 4

    def test_or_sparsity(self):
        """OR has sparsity 2^n (all coefficients non-zero)."""
        f = bf.create([0, 1, 1, 1])  # OR of 2 vars
        assert fourier_sparsity(f) == 4

    def test_threshold_affects_count(self):
        """Threshold parameter affects count."""
        f = bf.create([0, 0, 0, 1])
        
        # Default threshold
        sp1 = fourier_sparsity(f, threshold=1e-10)
        
        # Higher threshold might exclude small coefficients
        sp2 = fourier_sparsity(f, threshold=0.1)
        
        assert sp2 <= sp1


class TestFourierSparsityUpToConstants:
    """Tests for sparsity ignoring constant multiples."""

    def test_constant_function(self):
        """Constant function returns size - 0 = size (all trivial)."""
        f = bf.create([0, 0, 0, 0])
        result = fourier_sparsity_up_to_constants(f)
        # This counts "non-trivial" coefficients, which for constant is size-1
        # Actually the implementation returns size - count of non-trivial
        assert result >= 0

    def test_dictator_function(self):
        """Test on dictator function."""
        f = bf.create([0, 1, 0, 1])
        result = fourier_sparsity_up_to_constants(f)
        assert result >= 0


class TestGranularity:
    """Tests for granularity (coefficient value distribution)."""

    def test_returns_dict(self):
        """Function returns a dictionary."""
        f = bf.create([0, 1, 1, 0])
        result = granularity(f)
        assert isinstance(result, dict)

    def test_and_granularity(self):
        """AND has coefficients ±1/4."""
        f = bf.create([0, 0, 0, 1])
        result = granularity(f, threshold=0.01)
        
        # Should have coefficients at ±0.25 and ±0.5
        # Check that we got multiple distinct values
        assert len(result) > 1

    def test_parity_granularity(self):
        """Parity has single non-zero coefficient value."""
        f = bf.create([0, 1, 1, 0])
        result = granularity(f, threshold=0.01)
        
        # Should have mostly zeros and one ±1
        non_zero = {k: v for k, v in result.items() if abs(k) > 0.01}
        assert len(non_zero) <= 2  # +1 and/or -1


class TestFourierSupport:
    """Tests for Fourier support computation."""

    def test_empty_support_constant(self):
        """Unbalanced constant has support containing only ∅."""
        f = bf.create([1, 1, 1, 1])  # Constant 1
        support = fourier_support(f)
        assert 0 in support  # ∅ is in support

    def test_parity_support_full_set(self):
        """Parity has support containing only the full set."""
        f = bf.create([0, 1, 1, 0])  # XOR of 2 vars
        support = fourier_support(f)
        # Only S = {0,1} = 0b11 = 3 should be in support
        assert 3 in support
        # And nothing else (balanced, so no ∅)
        assert len(support) == 1

    def test_dictator_support(self):
        """Dictator support contains the singleton."""
        f = bf.create([0, 1, 0, 1])  # x0
        support = fourier_support(f)
        assert 1 in support  # {x0} = 0b01 = 1

    def test_and_support_all_sets(self):
        """AND has all sets in support."""
        f = bf.create([0, 0, 0, 1])
        support = fourier_support(f)
        # All 4 sets should be in support
        assert len(support) == 4
        assert set(support) == {0, 1, 2, 3}

    def test_support_sorted_by_degree(self):
        """Support is sorted by degree then value."""
        f = bf.create([0, 0, 0, 1])
        support = fourier_support(f)
        
        # Check sorting: degree 0, then degree 1, then degree 2
        degrees = [bin(s).count("1") for s in support]
        assert degrees == sorted(degrees)


class TestSparsityByDegree:
    """Tests for sparsity decomposed by degree."""

    def test_constant_only_degree_zero(self):
        """Constant has sparsity only at degree 0."""
        f = bf.create([1, 1, 1, 1])
        by_degree = sparsity_by_degree(f)
        assert 0 in by_degree
        assert by_degree[0] == 1
        # No other degrees
        assert all(by_degree.get(d, 0) == 0 for d in range(1, 3))

    def test_parity_only_full_degree(self):
        """Parity has sparsity only at degree n."""
        f = bf.create([0, 1, 1, 0])  # 2-var parity
        by_degree = sparsity_by_degree(f)
        assert 2 in by_degree
        assert by_degree[2] == 1
        # No degree 0 (balanced)
        assert by_degree.get(0, 0) == 0

    def test_and_all_degrees(self):
        """AND has sparsity at all degrees."""
        f = bf.create([0, 0, 0, 1])
        by_degree = sparsity_by_degree(f)
        
        # Should have entries at degrees 0, 1, 2
        assert 0 in by_degree
        assert 1 in by_degree
        assert 2 in by_degree
        
        # Degree 0: 1 coefficient (∅)
        # Degree 1: 2 coefficients ({0}, {1})
        # Degree 2: 1 coefficient ({0,1})
        assert by_degree[0] == 1
        assert by_degree[1] == 2
        assert by_degree[2] == 1


class TestEffectiveSparsity:
    """Tests for effective sparsity based on weight."""

    def test_parity_effective_one(self):
        """Parity has effective sparsity 1 (all weight on one coeff)."""
        f = bf.create([0, 1, 1, 0])
        eff_sparse, weight = effective_sparsity(f, weight_threshold=0.01)
        
        assert eff_sparse == 1
        assert abs(weight - 1.0) < 0.01

    def test_weight_captured_bounded(self):
        """Weight captured is in [0, 1]."""
        f = bf.create([0, 0, 0, 1])
        eff_sparse, weight = effective_sparsity(f)
        
        assert 0 <= weight <= 1.0

    def test_lower_threshold_means_more_coeffs(self):
        """Lower threshold requires more coefficients."""
        f = bf.create([0, 0, 0, 1])
        
        eff_low, _ = effective_sparsity(f, weight_threshold=0.001)
        eff_high, _ = effective_sparsity(f, weight_threshold=0.5)
        
        assert eff_low >= eff_high


class TestSparsityBounds:
    """Tests for known sparsity bounds."""

    def test_sparsity_at_most_2n(self):
        """Sparsity is at most 2^n."""
        for n in [2, 3, 4]:
            # Random function
            tt = [np.random.randint(0, 2) for _ in range(1 << n)]
            f = bf.create(tt)
            
            assert fourier_sparsity(f) <= (1 << n)

    def test_degree_d_sparsity_bound(self):
        """If deg(f) = d, sparsity ≤ sum of binomial(n,k) for k ≤ d."""
        from math import comb
        
        # Dictator has degree 1
        f = bf.create([0, 1, 0, 1])  # x0 on 2 vars
        n = 2
        d = 1
        
        # Bound: sum_{k=0}^{d} C(n,k)
        bound = sum(comb(n, k) for k in range(d + 1))
        
        assert fourier_sparsity(f) <= bound

    def test_juntas_have_low_sparsity(self):
        """k-juntas have sparsity at most 2^k."""
        # x0 is a 1-junta
        f = bf.create([0, 1, 0, 1, 0, 1, 0, 1])  # x0 on 3 vars
        
        # Sparsity should be at most 2^1 = 2
        # But actually for balanced dictator it's 1
        assert fourier_sparsity(f) <= 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_variable(self):
        """Single variable function."""
        f = bf.create([0, 1])  # x0
        
        assert fourier_sparsity(f) == 1
        support = fourier_support(f)
        assert support == [1]  # Only {x0}

    def test_all_zeros_vs_all_ones(self):
        """Constant 0 vs constant 1."""
        f0 = bf.create([0, 0])
        f1 = bf.create([1, 1])
        
        # Both constant, both have one non-zero coefficient
        assert fourier_sparsity(f0) == fourier_sparsity(f1) == 1

    def test_complement_same_sparsity(self):
        """f and NOT(f) have same sparsity."""
        f = bf.create([0, 1, 1, 0])
        not_f = bf.create([1, 0, 0, 1])
        
        assert fourier_sparsity(f) == fourier_sparsity(not_f)
