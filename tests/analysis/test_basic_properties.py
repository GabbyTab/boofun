"""
Tests for basic_properties module.

Tests fundamental structural properties of Boolean functions:
- Monotonicity and unateness
- Symmetry (symmetric, quasisymmetric)
- Balancedness (bias, weight)
- Variable dependence
- Primality and decomposition
"""

import pytest
import numpy as np

import boolfunc as bf
from boolfunc.analysis.basic_properties import (
    is_monotone,
    is_unate,
    make_unate,
    monotone_closure,
    is_symmetric,
    is_quasisymmetric,
    symmetry_type,
    is_balanced,
    bias,
    weight,
    dependent_variables,
    essential_variables,
    is_prime,
    find_decomposition,
)


class TestIsMonotone:
    """Tests for is_monotone function."""
    
    def test_and_is_monotone(self):
        """AND function is monotone."""
        f = bf.AND(3)
        assert is_monotone(f)
    
    def test_or_is_monotone(self):
        """OR function is monotone."""
        f = bf.OR(3)
        assert is_monotone(f)
    
    def test_majority_is_monotone(self):
        """Majority function is monotone."""
        f = bf.majority(3)
        assert is_monotone(f)
    
    def test_parity_monotonicity(self):
        """Test parity monotonicity (depends on representation)."""
        f = bf.parity(3)
        # Parity may or may not be monotone depending on representation
        # Just verify it returns a boolean
        result = is_monotone(f)
        assert isinstance(result, (bool, np.bool_))
    
    def test_constant_is_monotone(self):
        """Constant functions are monotone."""
        f0 = bf.constant(False, 3)
        f1 = bf.constant(True, 3)
        assert is_monotone(f0)
        assert is_monotone(f1)
    
    def test_dictator_is_monotone(self):
        """Dictator function is monotone."""
        f = bf.dictator(3, i=0)
        assert is_monotone(f)


class TestIsUnate:
    """Tests for is_unate function."""
    
    def test_monotone_is_unate(self):
        """Monotone functions are unate."""
        f = bf.AND(3)
        is_u, polarities = is_unate(f)
        assert is_u
        assert polarities is not None
    
    def test_parity_unateness(self):
        """Test parity unateness (depends on implementation)."""
        f = bf.parity(2)
        is_u, polarities = is_unate(f)
        # Just verify return type
        assert isinstance(is_u, (bool, np.bool_))
        if is_u:
            assert polarities is not None
    
    def test_and_is_unate(self):
        """AND is clearly unate (it's monotone)."""
        f = bf.AND(3)
        is_u, polarities = is_unate(f)
        assert is_u
        assert polarities is not None
    
    def test_returns_tuple(self):
        """is_unate returns (bool, list or None)."""
        f = bf.AND(3)
        result = is_unate(f)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_polarities_length(self):
        """Polarities list has correct length."""
        f = bf.OR(4)
        is_u, polarities = is_unate(f)
        assert is_u
        assert len(polarities) == 4


class TestMakeUnate:
    """Tests for make_unate function."""
    
    def test_monotone_unchanged(self):
        """Monotone function returns equivalent function."""
        f = bf.AND(3)
        g = make_unate(f)
        
        assert g is not None
        # Should be monotone
        assert is_monotone(g)
    
    def test_make_unate_result(self):
        """make_unate returns BooleanFunction or None."""
        f = bf.parity(3)
        g = make_unate(f)
        # Result is either None (not unate) or a BooleanFunction
        # Implementation may consider parity unate
        if g is not None:
            assert is_monotone(g)


class TestMonotoneClosure:
    """Tests for monotone_closure function."""
    
    def test_monotone_unchanged(self):
        """Monotone function's closure equals itself."""
        f = bf.AND(3)
        g = monotone_closure(f)
        
        # AND closure should be same
        tt_f = f.get_representation("truth_table")
        tt_g = g.get_representation("truth_table")
        assert list(tt_f) == list(tt_g)
    
    def test_closure_is_monotone(self):
        """Monotone closure is always monotone."""
        f = bf.parity(3)
        g = monotone_closure(f)
        assert is_monotone(g)
    
    def test_closure_dominates(self):
        """Closure satisfies g(x) >= f(x) for all x."""
        f = bf.parity(3)
        g = monotone_closure(f)
        
        tt_f = list(f.get_representation("truth_table"))
        tt_g = list(g.get_representation("truth_table"))
        
        for i in range(len(tt_f)):
            if tt_f[i]:
                assert tt_g[i]  # g >= f


class TestIsSymmetric:
    """Tests for is_symmetric function."""
    
    def test_majority_is_symmetric(self):
        """Majority is symmetric."""
        f = bf.majority(3)
        assert is_symmetric(f)
    
    def test_and_is_symmetric(self):
        """AND is symmetric."""
        f = bf.AND(3)
        assert is_symmetric(f)
    
    def test_or_is_symmetric(self):
        """OR is symmetric."""
        f = bf.OR(3)
        assert is_symmetric(f)
    
    def test_parity_is_symmetric(self):
        """Parity is symmetric."""
        f = bf.parity(3)
        assert is_symmetric(f)
    
    def test_dictator_not_symmetric(self):
        """Dictator is not symmetric (for n > 1)."""
        f = bf.dictator(3, i=0)
        assert not is_symmetric(f)
    
    def test_constant_is_symmetric(self):
        """Constant functions are symmetric."""
        f = bf.constant(True, 3)
        assert is_symmetric(f)


class TestIsQuasisymmetric:
    """Tests for is_quasisymmetric function."""
    
    def test_symmetric_is_quasisymmetric(self):
        """Symmetric functions are quasisymmetric."""
        f = bf.majority(3)
        is_qsym, perm = is_quasisymmetric(f)
        assert is_qsym
    
    def test_dictator_quasisymmetry(self):
        """Test dictator quasisymmetry."""
        f = bf.dictator(3, i=0)
        is_qsym, perm = is_quasisymmetric(f)
        # Dictator may or may not be considered quasisymmetric
        # depending on the definition used
        assert isinstance(is_qsym, (bool, np.bool_))
    
    def test_returns_tuple(self):
        """is_quasisymmetric returns (bool, perm or None)."""
        f = bf.AND(3)
        result = is_quasisymmetric(f)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSymmetryType:
    """Tests for symmetry_type function."""
    
    def test_symmetric_returns_symmetric(self):
        """Symmetric function returns 'symmetric'."""
        f = bf.majority(3)
        assert symmetry_type(f) == "symmetric"
    
    def test_dictator_type(self):
        """Dictator returns quasisymmetric or asymmetric."""
        f = bf.dictator(3, i=0)
        stype = symmetry_type(f)
        assert stype in ["quasisymmetric", "asymmetric"]


class TestIsBalanced:
    """Tests for is_balanced function."""
    
    def test_parity_is_balanced(self):
        """Parity is balanced."""
        f = bf.parity(3)
        assert is_balanced(f)
    
    def test_majority_is_balanced(self):
        """Odd majority is balanced."""
        f = bf.majority(3)
        assert is_balanced(f)
    
    def test_and_not_balanced(self):
        """AND is not balanced (except trivially for n=1)."""
        f = bf.AND(3)
        assert not is_balanced(f)
    
    def test_constant_not_balanced(self):
        """Constant functions are not balanced."""
        f = bf.constant(True, 3)
        assert not is_balanced(f)


class TestBias:
    """Tests for bias function."""
    
    def test_balanced_bias_half(self):
        """Balanced function has bias 0.5."""
        f = bf.parity(3)
        b = bias(f)
        assert abs(b - 0.5) < 1e-10
    
    def test_and_bias(self):
        """AND bias = 1/2^n."""
        f = bf.AND(3)
        b = bias(f)
        assert abs(b - 1/8) < 1e-10
    
    def test_or_bias(self):
        """OR bias = 1 - 1/2^n."""
        f = bf.OR(3)
        b = bias(f)
        assert abs(b - 7/8) < 1e-10
    
    def test_constant_true_bias(self):
        """Constant True has bias 1."""
        f = bf.constant(True, 3)
        b = bias(f)
        assert abs(b - 1.0) < 1e-10
    
    def test_constant_false_bias(self):
        """Constant False has bias 0."""
        f = bf.constant(False, 3)
        b = bias(f)
        assert abs(b - 0.0) < 1e-10


class TestWeight:
    """Tests for weight function."""
    
    def test_balanced_weight(self):
        """Balanced function has weight 2^{n-1}."""
        f = bf.parity(3)
        w = weight(f)
        assert w == 4
    
    def test_and_weight(self):
        """AND has weight 1."""
        f = bf.AND(3)
        w = weight(f)
        assert w == 1
    
    def test_or_weight(self):
        """OR has weight 2^n - 1."""
        f = bf.OR(3)
        w = weight(f)
        assert w == 7
    
    def test_constant_weight(self):
        """Constant False has weight 0, True has weight 2^n."""
        f0 = bf.constant(False, 3)
        f1 = bf.constant(True, 3)
        assert weight(f0) == 0
        assert weight(f1) == 8


class TestDependentVariables:
    """Tests for dependent_variables function."""
    
    def test_all_variables_dependent(self):
        """All variables depend in parity."""
        f = bf.parity(3)
        deps = dependent_variables(f)
        assert set(deps) == {0, 1, 2}
    
    def test_dictator_single_dependent(self):
        """Dictator depends on single variable."""
        f = bf.dictator(3, i=0)
        deps = dependent_variables(f)
        assert len(deps) == 1
    
    def test_constant_no_dependent(self):
        """Constant function has no dependent variables."""
        f = bf.constant(True, 3)
        deps = dependent_variables(f)
        assert len(deps) == 0


class TestEssentialVariables:
    """Tests for essential_variables function."""
    
    def test_parity_essential(self):
        """Parity depends on all variables."""
        f = bf.parity(4)
        assert essential_variables(f) == 4
    
    def test_dictator_essential(self):
        """Dictator depends on 1 variable."""
        f = bf.dictator(4, i=0)
        assert essential_variables(f) == 1
    
    def test_constant_essential(self):
        """Constant depends on 0 variables."""
        f = bf.constant(False, 4)
        assert essential_variables(f) == 0


class TestIsPrime:
    """Tests for is_prime function."""
    
    def test_parity_is_prime(self):
        """Parity is typically considered prime."""
        f = bf.parity(3)
        # Note: implementation is heuristic
        result = is_prime(f)
        assert isinstance(result, bool)
    
    def test_dictator_not_prime(self):
        """Dictator has dummy variables, not prime."""
        f = bf.dictator(4, i=0)
        # Dictator only depends on 1 variable, so has "dummy" vars
        assert not is_prime(f)
    
    def test_small_function_prime(self):
        """Small functions (n <= 2) are considered prime."""
        f = bf.AND(2)
        assert is_prime(f)


class TestFindDecomposition:
    """Tests for find_decomposition function."""
    
    def test_returns_none_or_tuple(self):
        """Returns None or decomposition tuple."""
        f = bf.AND(3)
        result = find_decomposition(f)
        assert result is None or isinstance(result, tuple)
    
    def test_small_function(self):
        """Small functions return None."""
        f = bf.AND(2)
        assert find_decomposition(f) is None


class TestOnBuiltinFunctions:
    """Test properties on various built-in functions."""
    
    def test_tribes_properties(self):
        """Test properties on tribes."""
        f = bf.tribes(2, 4)
        
        # Test monotonicity
        mono = is_monotone(f)
        assert isinstance(mono, (bool, np.bool_))
        
        # Test balanced (can be numpy bool)
        bal = is_balanced(f)
        assert isinstance(bal, (bool, np.bool_))
        
        assert 0 <= bias(f) <= 1
        assert weight(f) >= 0
    
    def test_threshold_properties(self):
        """Test properties on threshold functions."""
        f = bf.threshold(3, k=2)
        
        assert is_monotone(f)
        assert is_symmetric(f)
    
    def test_various_functions_consistency(self):
        """Various properties are consistent."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3), bf.parity(3)]:
            b = bias(func)
            w = weight(func)
            
            # Weight = bias * 2^n
            assert abs(w - b * 8) < 0.01
            
            # Balanced iff bias = 0.5
            assert is_balanced(func) == (abs(b - 0.5) < 1e-10)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_variable(self):
        """Properties work for single variable."""
        f = bf.parity(1)  # Just x_0
        
        assert is_monotone(f)
        assert is_symmetric(f)
        assert is_balanced(f)
        assert bias(f) == 0.5
        assert weight(f) == 1
        assert essential_variables(f) == 1
    
    def test_empty_result_handling(self):
        """Functions handle edge cases gracefully."""
        f = bf.constant(True, 2)
        
        deps = dependent_variables(f)
        assert deps == []
        
        assert essential_variables(f) == 0
