"""
Tests for canalization analysis module.

These tests verify our canalization implementation against known results.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis.canalization import (
    is_canalizing,
    is_k_canalizing,
    is_nested_canalizing,
    get_canalizing_variables,
    get_canalizing_depth,
    get_essential_variables,
    get_input_types,
    get_symmetry_groups,
    input_redundancy,
    CanalizationAnalyzer,
)


class TestIsCanalizing:
    """Test basic canalization detection."""
    
    def test_and_is_canalizing(self):
        """AND is canalizing: x_i = 0 → output = 0."""
        for n in [2, 3, 4, 5]:
            assert is_canalizing(bf.AND(n))
    
    def test_or_is_canalizing(self):
        """OR is canalizing: x_i = 1 → output = 1."""
        for n in [2, 3, 4, 5]:
            assert is_canalizing(bf.OR(n))
    
    def test_parity_is_not_canalizing(self):
        """Parity is NOT canalizing: no single input determines output."""
        for n in [2, 3, 4, 5]:
            assert not is_canalizing(bf.parity(n))
    
    def test_majority_small_not_canalizing(self):
        """Majority on 3+ variables is not canalizing."""
        assert not is_canalizing(bf.majority(3))
        assert not is_canalizing(bf.majority(5))
    
    def test_dictator_is_canalizing(self):
        """Dictator is canalizing."""
        for n in [3, 5]:
            assert is_canalizing(bf.dictator(n, 0))
    
    def test_constant_is_canalizing(self):
        """Constant functions are trivially canalizing."""
        # All zeros
        f = bf.create([0] * 8)
        assert is_canalizing(f)
        
        # All ones  
        f = bf.create([1] * 8)
        assert is_canalizing(f)


class TestCanalizingDepth:
    """Test canalizing depth computation."""
    
    def test_and_depth_equals_n(self):
        """AND has canalizing depth n (fully nested)."""
        for n in [2, 3, 4]:
            depth = get_canalizing_depth(bf.AND(n))
            assert depth == n, f"AND({n}) depth should be {n}, got {depth}"
    
    def test_or_depth_equals_n(self):
        """OR has canalizing depth n (fully nested)."""
        for n in [2, 3, 4]:
            depth = get_canalizing_depth(bf.OR(n))
            assert depth == n, f"OR({n}) depth should be {n}, got {depth}"
    
    def test_parity_depth_zero(self):
        """Parity has canalizing depth 0."""
        for n in [2, 3, 4]:
            depth = get_canalizing_depth(bf.parity(n))
            assert depth == 0, f"Parity({n}) depth should be 0, got {depth}"
    
    def test_majority_depth_zero(self):
        """Majority has canalizing depth 0 (not canalizing)."""
        for n in [3, 5]:
            depth = get_canalizing_depth(bf.majority(n))
            assert depth == 0


class TestKCanalizing:
    """Test k-canalizing detection."""
    
    def test_and_is_n_canalizing(self):
        """AND is n-canalizing."""
        n = 3
        f = bf.AND(n)
        for k in range(n + 1):
            assert is_k_canalizing(f, k)
    
    def test_parity_not_1_canalizing(self):
        """Parity is 0-canalizing but not 1-canalizing."""
        f = bf.parity(3)
        assert is_k_canalizing(f, 0)  # Everything is 0-canalizing
        assert not is_k_canalizing(f, 1)


class TestNestedCanalizing:
    """Test nested canalizing function detection."""
    
    def test_and_is_ncf(self):
        """AND is a nested canalizing function."""
        for n in [2, 3, 4]:
            assert is_nested_canalizing(bf.AND(n))
    
    def test_or_is_ncf(self):
        """OR is a nested canalizing function."""
        for n in [2, 3, 4]:
            assert is_nested_canalizing(bf.OR(n))
    
    def test_parity_not_ncf(self):
        """Parity is not nested canalizing."""
        for n in [2, 3, 4]:
            assert not is_nested_canalizing(bf.parity(n))
    
    def test_majority_not_ncf(self):
        """Majority is not nested canalizing."""
        for n in [3, 5]:
            assert not is_nested_canalizing(bf.majority(n))


class TestCanalizingVariables:
    """Test detection of canalizing variables."""
    
    def test_and_all_variables_canalize_on_zero(self):
        """In AND, every variable canalizes with input 0 → output 0."""
        f = bf.AND(3)
        can_vars = get_canalizing_variables(f)
        
        # Should have entries for all 3 variables
        assert len(can_vars) >= 3
        
        # Each variable should canalize on 0 → 0
        for entry in can_vars:
            if entry['canalizing_input'] == 0:
                assert entry['canalized_output'] == 0
    
    def test_parity_no_canalizing_variables(self):
        """Parity has no canalizing variables."""
        f = bf.parity(3)
        can_vars = get_canalizing_variables(f)
        assert len(can_vars) == 0


class TestEssentialVariables:
    """Test essential variable detection."""
    
    def test_and_all_essential(self):
        """All variables in AND are essential."""
        for n in [2, 3, 4]:
            essential = get_essential_variables(bf.AND(n))
            assert len(essential) == n
    
    def test_dictator_one_essential(self):
        """Dictator has one essential variable."""
        f = bf.dictator(5, 0)
        essential = get_essential_variables(f)
        assert len(essential) == 1
    
    def test_constant_no_essential(self):
        """Constant functions have no essential variables."""
        f = bf.create([0] * 8)
        essential = get_essential_variables(f)
        assert len(essential) == 0


class TestInputTypes:
    """Test input type classification."""
    
    def test_and_all_positive(self):
        """All AND inputs are positive (monotone increasing)."""
        f = bf.AND(3)
        types = get_input_types(f)
        
        for i in range(3):
            assert types[i] == "positive"
    
    def test_constant_all_nonessential(self):
        """Constant function inputs are non-essential."""
        f = bf.create([0] * 8)
        types = get_input_types(f)
        
        for i in range(3):
            assert types[i] == "non-essential"
    
    def test_parity_all_conditional(self):
        """Parity inputs are conditional (non-monotone)."""
        f = bf.parity(3)
        types = get_input_types(f)
        
        for i in range(3):
            assert types[i] == "conditional"


class TestSymmetryGroups:
    """Test symmetry group detection."""
    
    def test_and_one_group(self):
        """AND has one symmetry group (all variables interchangeable)."""
        f = bf.AND(3)
        groups = get_symmetry_groups(f)
        
        # All variables should be in one group
        assert len(groups) == 1
        assert groups[0] == {0, 1, 2}
    
    def test_majority_one_group(self):
        """Majority has one symmetry group."""
        f = bf.majority(3)
        groups = get_symmetry_groups(f)
        
        assert len(groups) == 1
        assert groups[0] == {0, 1, 2}
    
    def test_parity_one_group(self):
        """Parity has one symmetry group."""
        f = bf.parity(3)
        groups = get_symmetry_groups(f)
        
        assert len(groups) == 1


class TestInputRedundancy:
    """Test input redundancy computation."""
    
    def test_constant_redundancy_one(self):
        """Constant functions have redundancy 1."""
        f = bf.create([0] * 8)
        assert input_redundancy(f) == 1.0
    
    def test_parity_redundancy_zero(self):
        """Parity has redundancy 0 (all inputs essential)."""
        f = bf.parity(3)
        assert input_redundancy(f) == 0.0
    
    def test_dictator_high_redundancy(self):
        """Dictator has high redundancy (n-1)/n."""
        n = 5
        f = bf.dictator(n, 0)
        expected = (n - 1) / n
        assert abs(input_redundancy(f) - expected) < 0.01


class TestCanalizationAnalyzer:
    """Test the CanalizationAnalyzer class."""
    
    def test_analyzer_and(self):
        """Analyzer should work correctly for AND."""
        f = bf.AND(3)
        analyzer = CanalizationAnalyzer(f)
        
        summary = analyzer.summary()
        
        assert summary['is_canalizing'] is True
        assert summary['canalizing_depth'] == 3
        assert summary['is_nested_canalizing'] is True
        assert summary['n_essential'] == 3
    
    def test_analyzer_parity(self):
        """Analyzer should work correctly for parity."""
        f = bf.parity(3)
        analyzer = CanalizationAnalyzer(f)
        
        summary = analyzer.summary()
        
        assert summary['is_canalizing'] is False
        assert summary['canalizing_depth'] == 0
        assert summary['is_nested_canalizing'] is False
        assert summary['n_essential'] == 3
        assert summary['input_redundancy'] == 0.0


class TestCrossValidation:
    """Cross-validation with known theoretical results."""
    
    def test_influence_vs_edge_effectiveness(self):
        """Edge effectiveness should equal influence."""
        f = bf.majority(5)
        
        from boofun.analysis.canalization import edge_effectiveness
        
        inf = f.influences()
        eff = edge_effectiveness(f)
        
        assert np.allclose(inf, eff)
    
    def test_total_influence_vs_effective_degree(self):
        """Effective degree should equal total influence."""
        f = bf.majority(5)
        
        from boofun.analysis.canalization import effective_degree
        
        ti = f.total_influence()
        ed = effective_degree(f)
        
        assert abs(ti - ed) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
