"""
Tests for functions integrated from Avishay Tal's PhD library.

These tests verify the correctness of:
- Decision tree analysis (decision_trees.py)
- Enhanced p-biased analysis (p_biased.py)
- Enhanced Fourier analysis (fourier.py)
- Enhanced sensitivity analysis (sensitivity.py)
- Fourier sparsity analysis (sparsity.py)
"""

import numpy as np
import pytest

import boofun as bf


class TestDecisionTrees:
    """Tests for decision_trees module."""

    def test_decision_tree_depth_dp_and(self):
        """AND function should have decision tree depth equal to n."""
        and3 = bf.create([0, 0, 0, 0, 0, 0, 0, 1])  # AND of 3 vars
        
        from boofun.analysis.decision_trees import decision_tree_depth_dp
        depth = decision_tree_depth_dp(and3)
        
        # AND requires checking all variables in worst case
        assert depth == 3

    def test_decision_tree_depth_dp_or(self):
        """OR function should have decision tree depth equal to n."""
        or3 = bf.create([0, 1, 1, 1, 1, 1, 1, 1])  # OR of 3 vars
        
        from boofun.analysis.decision_trees import decision_tree_depth_dp
        depth = decision_tree_depth_dp(or3)
        
        # OR requires checking all variables in worst case
        assert depth == 3

    def test_decision_tree_depth_dp_dictator(self):
        """Dictator function should have depth 1."""
        dict_x0 = bf.create([0, 1, 0, 1])  # x0 dictator (2 vars)
        
        from boofun.analysis.decision_trees import decision_tree_depth_dp
        depth = decision_tree_depth_dp(dict_x0)
        
        assert depth == 1

    def test_decision_tree_depth_dp_constant(self):
        """Constant function should have depth 0."""
        const = bf.create([0, 0, 0, 0])
        
        from boofun.analysis.decision_trees import decision_tree_depth_dp
        depth = decision_tree_depth_dp(const)
        
        assert depth == 0

    def test_decision_tree_class(self):
        """Test DecisionTree class operations."""
        from boofun.analysis.decision_trees import DecisionTree
        
        # Build a simple tree: query x0, return x0
        leaf0 = DecisionTree(value=0)
        leaf1 = DecisionTree(value=1)
        tree = DecisionTree(var=0, left=leaf0, right=leaf1)
        
        assert tree.depth() == 1
        assert tree.size() == 2
        assert not tree.is_leaf()
        assert leaf0.is_leaf()

    def test_decision_tree_evaluate(self):
        """Test decision tree evaluation."""
        from boofun.analysis.decision_trees import DecisionTree
        
        # Tree for AND(x0, x1): query x0, if 1 query x1, else return 0
        leaf0 = DecisionTree(value=0)
        leaf1 = DecisionTree(value=1)
        subtree = DecisionTree(var=1, left=leaf0, right=leaf1)  # x1
        tree = DecisionTree(var=0, left=leaf0, right=subtree)   # x0 ? subtree : 0
        
        # Test all inputs for 2 variables
        assert tree.evaluate(0b00, 2) == 0  # x0=0, x1=0 -> 0
        assert tree.evaluate(0b01, 2) == 0  # x0=1, x1=0 -> 0
        assert tree.evaluate(0b10, 2) == 0  # x0=0, x1=1 -> 0
        assert tree.evaluate(0b11, 2) == 1  # x0=1, x1=1 -> 1

    def test_tree_depth_helper(self):
        """Test tree_depth utility function."""
        from boofun.analysis.decision_trees import tree_depth
        
        # Test with list format (Tal's style)
        tree_list = [0, [1, [], []], []]
        assert tree_depth(tree_list) == 2
        
        # Test empty tree
        assert tree_depth([]) == 0


class TestPBiasedEnhancements:
    """Tests for enhanced p-biased module."""

    def test_p_biased_average_sensitivity(self):
        """Test average sensitivity under μ_p."""
        xor2 = bf.create([0, 1, 1, 0])  # XOR of 2 vars
        
        from boofun.analysis.p_biased import p_biased_average_sensitivity
        
        # At p=0.5 (uniform), average sensitivity = total influence = 2 for XOR
        as_p = p_biased_average_sensitivity(xor2, p=0.5)
        assert abs(as_p - 2.0) < 0.01

    def test_p_biased_total_influence_fourier(self):
        """Test total influence via Fourier formula matches average sensitivity."""
        and2 = bf.create([0, 0, 0, 1])  # AND of 2 vars
        
        from boofun.analysis.p_biased import (
            p_biased_average_sensitivity,
            p_biased_total_influence_fourier,
        )
        
        # Test at multiple p values - by Poincaré inequality, these must be equal
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            as_p = p_biased_average_sensitivity(and2, p=p)
            ti_f = p_biased_total_influence_fourier(and2, p=p)
            assert abs(as_p - ti_f) < 1e-10, f"Mismatch at p={p}: {as_p} vs {ti_f}"

    def test_p_biased_fourier_coefficient(self):
        """Test individual p-biased Fourier coefficient."""
        dict_x0 = bf.create([0, 1, 0, 1])  # x0 dictator
        
        from boofun.analysis.p_biased import p_biased_fourier_coefficient
        
        # For dictator at p=0.5, f̂({0}) should be ±1
        coeff = p_biased_fourier_coefficient(dict_x0, 0.5, 0b01)  # S = {x0}
        assert abs(abs(coeff) - 1.0) < 0.01

    def test_parity_biased_coefficient(self):
        """Test parity biased coefficient computation."""
        from boofun.analysis.p_biased import parity_biased_coefficient
        
        # Basic sanity check
        result = parity_biased_coefficient(4, 2, 0)
        assert isinstance(result, float)

    def test_p_biased_analyzer_validate(self):
        """Test PBiasedAnalyzer validation."""
        xor2 = bf.create([0, 1, 1, 0])
        
        from boofun.analysis.p_biased import PBiasedAnalyzer
        
        analyzer = PBiasedAnalyzer(xor2, p=0.5)
        validation = analyzer.validate()
        
        # All validations should pass
        assert all(validation.values())


class TestFourierEnhancements:
    """Tests for enhanced Fourier module."""

    def test_correlation_self(self):
        """Correlation of f with itself should be 1."""
        f = bf.create([0, 1, 1, 0])
        
        from boofun.analysis.fourier import correlation
        
        corr = correlation(f, f)
        assert abs(corr - 1.0) < 0.01

    def test_correlation_negation(self):
        """Correlation of f with NOT(f) should be -1."""
        f = bf.create([0, 1, 1, 0])
        not_f = bf.create([1, 0, 0, 1])
        
        from boofun.analysis.fourier import correlation
        
        corr = correlation(f, not_f)
        assert abs(corr - (-1.0)) < 0.01

    def test_truncate_to_degree_zero(self):
        """Truncating to degree 0 gives constant function."""
        xor2 = bf.create([0, 1, 1, 0])
        
        from boofun.analysis.fourier import truncate_to_degree
        
        truncated = truncate_to_degree(xor2, 0)
        
        # For balanced XOR, degree-0 part is 0 (bias is 0)
        assert np.allclose(truncated, 0, atol=0.01)

    def test_annealed_influence(self):
        """Test annealed influence computation."""
        dict_x0 = bf.create([0, 1, 0, 1])  # x0 dictator
        
        from boofun.analysis.fourier import annealed_influence
        
        # Dictator has influence 1 for x0, 0 for x1
        inf_0 = annealed_influence(dict_x0, 0, rho=1.0)
        inf_1 = annealed_influence(dict_x0, 1, rho=1.0)
        
        assert abs(inf_0 - 1.0) < 0.01
        assert abs(inf_1) < 0.01

    def test_fourier_weight_distribution(self):
        """Test Fourier weight distribution."""
        xor2 = bf.create([0, 1, 1, 0])  # XOR has all weight on degree 2
        
        from boofun.analysis.fourier import fourier_weight_distribution
        
        weights = fourier_weight_distribution(xor2)
        
        # XOR should have all weight on degree 2
        assert weights.get(0, 0) < 0.01  # No degree-0 weight
        assert weights.get(1, 0) < 0.01  # No degree-1 weight
        assert abs(weights.get(2, 0) - 1.0) < 0.01  # All weight on degree 2

    def test_min_fourier_coefficient_size(self):
        """Test minimum Fourier coefficient size."""
        xor2 = bf.create([0, 1, 1, 0])  # XOR
        
        from boofun.analysis.fourier import min_fourier_coefficient_size
        
        # XOR has min non-zero coefficient at degree 2
        min_size = min_fourier_coefficient_size(xor2)
        assert min_size == 2


class TestSensitivityEnhancements:
    """Tests for enhanced sensitivity module."""

    def test_sensitive_coordinates(self):
        """Test sensitive_coordinates function."""
        and2 = bf.create([0, 0, 0, 1])  # AND
        
        from boofun.analysis.sensitivity import sensitive_coordinates
        
        # At input 11 (both 1s), both coordinates are sensitive
        coords = sensitive_coordinates(and2, 0b11)
        assert set(coords) == {0, 1}
        
        # At input 00, no coordinates are sensitive (flipping any bit keeps output 0)
        coords = sensitive_coordinates(and2, 0b00)
        assert coords == []

    def test_average_sensitivity_moment(self):
        """Test t-th moment of sensitivity."""
        xor3 = bf.create([0, 1, 1, 0, 1, 0, 0, 1])  # XOR of 3 vars
        
        from boofun.analysis.sensitivity import (
            average_sensitivity,
            average_sensitivity_moment,
        )
        
        # 1st moment = average sensitivity
        as_1 = average_sensitivity_moment(xor3, 1)
        avg_s = average_sensitivity(xor3)
        assert abs(as_1 - avg_s) < 0.01
        
        # For XOR, all inputs have sensitivity = n, so variance = 0
        as_2 = average_sensitivity_moment(xor3, 2)
        # E[s^2] should equal s^2 when s is constant
        assert abs(as_2 - 9.0) < 0.01  # 3^2 = 9

    def test_sensitivity_histogram(self):
        """Test sensitivity histogram."""
        and2 = bf.create([0, 0, 0, 1])
        
        from boofun.analysis.sensitivity import sensitivity_histogram
        
        hist = sensitivity_histogram(and2)
        
        # AND has 3 inputs with sensitivity 0 or 1, 1 input with sensitivity 2
        assert len(hist) == 3  # Histogram for n=2 has 3 entries (0, 1, 2)
        assert abs(sum(hist) - 1.0) < 0.01  # Should sum to 1

    def test_max_min_sensitivity(self):
        """Test max and min sensitivity."""
        and3 = bf.create([0, 0, 0, 0, 0, 0, 0, 1])
        
        from boofun.analysis.sensitivity import max_sensitivity, min_sensitivity
        
        max_s = max_sensitivity(and3)
        min_s = min_sensitivity(and3)
        
        # AND_3 has max sensitivity 3 (at 111) and min 0 (at 000)
        assert max_s == 3
        assert min_s == 0

    def test_arg_max_sensitivity(self):
        """Test arg_max_sensitivity."""
        and2 = bf.create([0, 0, 0, 1])
        
        from boofun.analysis.sensitivity import arg_max_sensitivity
        
        x, sens = arg_max_sensitivity(and2)
        
        assert sens == 2
        assert x == 0b11  # Input 11 has max sensitivity for AND


class TestSparsity:
    """Tests for sparsity module."""

    def test_fourier_sparsity_dictator(self):
        """Balanced dictator has sparsity 1 (only degree-1 term, since E[f]=0)."""
        dict_x0 = bf.create([0, 1, 0, 1])  # x0 dictator on 2 vars (balanced)
        
        from boofun.analysis.sparsity import fourier_sparsity
        
        sparsity = fourier_sparsity(dict_x0)
        # Balanced dictator: f̂(∅) = E[f] = 0 (in ±1 convention: 50% +1, 50% -1)
        # Only f̂({x0}) = ±1 is non-zero
        assert sparsity == 1

    def test_fourier_sparsity_xor(self):
        """XOR of n variables has sparsity 1 (only degree-n term)."""
        xor3 = bf.create([0, 1, 1, 0, 1, 0, 0, 1])
        
        from boofun.analysis.sparsity import fourier_sparsity
        
        sparsity = fourier_sparsity(xor3)
        assert sparsity == 1

    def test_fourier_support(self):
        """Test Fourier support computation."""
        dict_x0 = bf.create([0, 1, 0, 1])  # x0 on 2 variables (balanced)
        
        from boofun.analysis.sparsity import fourier_support
        
        support = fourier_support(dict_x0)
        
        # Balanced dictator has support {{x0}} only (f̂(∅) = 0 because E[f] = 0)
        assert 1 in support  # {x0} = 0b01
        
        # For unbalanced function, constant term is non-zero
        and2 = bf.create([0, 0, 0, 1])  # AND (unbalanced)
        support_and = fourier_support(and2)
        assert 0 in support_and  # Constant term is non-zero for AND

    def test_sparsity_by_degree(self):
        """Test sparsity by degree."""
        and2 = bf.create([0, 0, 0, 1])
        
        from boofun.analysis.sparsity import sparsity_by_degree
        
        by_degree = sparsity_by_degree(and2)
        
        # AND has non-zero coefficients at all degrees 0, 1, 2
        assert 0 in by_degree
        assert 2 in by_degree

    def test_effective_sparsity(self):
        """Test effective sparsity."""
        xor2 = bf.create([0, 1, 1, 0])
        
        from boofun.analysis.sparsity import effective_sparsity
        
        eff_sparse, weight = effective_sparsity(xor2, weight_threshold=0.01)
        
        # XOR has all weight on one coefficient
        assert eff_sparse == 1
        assert abs(weight - 1.0) < 0.01


class TestCrossValidation:
    """Cross-validation tests ensuring different implementations agree."""

    def test_total_influence_methods_agree(self):
        """Different methods for computing total influence should agree."""
        f = bf.create([0, 1, 1, 0, 1, 0, 0, 1])  # 3-XOR
        
        from boofun.analysis.sensitivity import total_influence_via_sensitivity
        from boofun.analysis import SpectralAnalyzer
        
        # Method 1: Via sensitivity
        ti_sens = total_influence_via_sensitivity(f)
        
        # Method 2: Via SpectralAnalyzer
        analyzer = SpectralAnalyzer(f)
        ti_spectral = analyzer.total_influence()
        
        assert abs(ti_sens - ti_spectral) < 0.01

    def test_p_biased_cross_validation(self):
        """p-biased methods should agree when p=0.5."""
        f = bf.create([0, 0, 0, 1])  # AND
        
        from boofun.analysis.p_biased import (
            p_biased_average_sensitivity,
            p_biased_total_influence,
            p_biased_total_influence_fourier,
        )
        
        as_p = p_biased_average_sensitivity(f, 0.5)
        ti_direct = p_biased_total_influence(f, 0.5)
        ti_fourier = p_biased_total_influence_fourier(f, 0.5)
        
        # All should be approximately equal
        assert abs(as_p - ti_direct) < 0.1
        assert abs(as_p - ti_fourier) < 0.1
