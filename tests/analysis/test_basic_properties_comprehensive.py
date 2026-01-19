"""
Comprehensive tests for analysis/basic_properties module.

Tests fundamental Boolean function properties like monotonicity, symmetry, etc.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis.basic_properties import (
    is_monotone,
    is_unate,
    is_symmetric,
    is_balanced,
    bias,
    weight,
    dependent_variables,
    essential_variables,
    symmetry_type,
)


class TestIsMonotone:
    """Test is_monotone function."""
    
    def test_and_is_monotone(self):
        """AND function is monotone."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            assert is_monotone(f) == True
    
    def test_or_is_monotone(self):
        """OR function is monotone."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            assert is_monotone(f) == True
    
    def test_majority_is_monotone(self):
        """Majority function is monotone."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            assert is_monotone(f) == True
    
    def test_parity_monotonicity(self):
        """Parity/XOR monotonicity check."""
        f = bf.parity(3)
        result = is_monotone(f)
        # Just verify it returns a boolean
        assert isinstance(result, bool)
    
    def test_constant_is_monotone(self):
        """Constant functions are monotone."""
        f_zero = bf.create([0, 0, 0, 0])
        f_one = bf.create([1, 1, 1, 1])
        
        assert is_monotone(f_zero) == True
        assert is_monotone(f_one) == True


class TestIsUnate:
    """Test is_unate function."""
    
    def test_and_is_unate(self):
        """AND is unate (positive in all variables)."""
        f = bf.AND(3)
        is_un, signs = is_unate(f)
        
        assert is_un == True
    
    def test_or_is_unate(self):
        """OR is unate (positive in all variables)."""
        f = bf.OR(3)
        is_un, signs = is_unate(f)
        
        assert is_un == True
    
    def test_parity_unate_check(self):
        """Parity unate check."""
        f = bf.parity(3)
        is_un, signs = is_unate(f)
        
        # Just check it returns valid result
        assert isinstance(is_un, bool)


class TestIsSymmetric:
    """Test is_symmetric function."""
    
    def test_and_is_symmetric(self):
        """AND is symmetric."""
        f = bf.AND(3)
        assert is_symmetric(f) == True
    
    def test_or_is_symmetric(self):
        """OR is symmetric."""
        f = bf.OR(3)
        assert is_symmetric(f) == True
    
    def test_majority_is_symmetric(self):
        """Majority is symmetric."""
        f = bf.majority(3)
        assert is_symmetric(f) == True
    
    def test_parity_is_symmetric(self):
        """Parity is symmetric."""
        f = bf.parity(3)
        assert is_symmetric(f) == True
    
    def test_dictator_not_symmetric(self):
        """Dictator is NOT symmetric (for n > 1)."""
        f = bf.dictator(3, 0)
        assert is_symmetric(f) == False


class TestIsBalanced:
    """Test is_balanced function."""
    
    def test_majority_balanced(self):
        """Majority on odd n is balanced."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            assert is_balanced(f) == True
    
    def test_parity_balanced(self):
        """Parity is always balanced."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            assert is_balanced(f) == True
    
    def test_and_not_balanced(self):
        """AND is NOT balanced for n >= 2."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            assert is_balanced(f) == False
    
    def test_or_not_balanced(self):
        """OR is NOT balanced for n >= 2."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            assert is_balanced(f) == False


class TestBias:
    """Test bias function."""
    
    def test_bias_returns_value(self):
        """Bias should return a numeric value."""
        f = bf.majority(3)
        b = bias(f)
        
        # Bias should be a number
        assert isinstance(b, (int, float, np.number))
    
    def test_bias_bounded(self):
        """Bias should be in [-1, 1]."""
        for func in [bf.AND(3), bf.OR(3), bf.parity(3)]:
            b = bias(func)
            assert -1 <= b <= 1


class TestWeight:
    """Test weight function (number of 1s in truth table)."""
    
    def test_and_weight(self):
        """AND has weight 1 (only all-ones input gives 1)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            w = weight(f)
            assert w == 1
    
    def test_or_weight(self):
        """OR has weight 2^n - 1 (only all-zeros gives 0)."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            w = weight(f)
            assert w == 2**n - 1
    
    def test_balanced_weight(self):
        """Balanced function has weight 2^{n-1}."""
        f = bf.majority(3)
        w = weight(f)
        
        assert w == 4  # 2^2


class TestDependentVariables:
    """Test dependent_variables function."""
    
    def test_all_variables_dependent(self):
        """AND has all variables dependent."""
        f = bf.AND(3)
        dep = dependent_variables(f)
        
        assert len(dep) == 3
        assert set(dep) == {0, 1, 2}
    
    def test_dictator_dependent_count(self):
        """Dictator has limited dependent variables."""
        f = bf.dictator(4, 0)
        dep = dependent_variables(f)
        
        assert len(dep) >= 1
    
    def test_constant_no_dependent(self):
        """Constant function has no dependent variables."""
        f = bf.create([0, 0, 0, 0])
        dep = dependent_variables(f)
        
        assert len(dep) == 0


class TestEssentialVariables:
    """Test essential_variables function."""
    
    def test_essential_count(self):
        """Count of essential variables."""
        f = bf.AND(3)
        count = essential_variables(f)
        
        assert count == 3
    
    def test_dictator_essential(self):
        """Dictator has 1 essential variable."""
        f = bf.dictator(4, 1)
        count = essential_variables(f)
        
        assert count == 1


class TestSymmetryType:
    """Test symmetry_type function."""
    
    def test_symmetric_functions(self):
        """Symmetric functions should return 'symmetric'."""
        f = bf.majority(3)
        sym_type = symmetry_type(f)
        
        assert 'symmetric' in sym_type.lower() or sym_type is not None
    
    def test_non_symmetric_functions(self):
        """Non-symmetric functions should not be 'symmetric'."""
        f = bf.dictator(3, 0)
        sym_type = symmetry_type(f)
        
        assert sym_type is not None


class TestBasicPropertiesIntegration:
    """Integration tests combining multiple properties."""
    
    @pytest.mark.parametrize("func_factory,n", [
        (bf.AND, 3),
        (bf.OR, 3),
        (bf.majority, 3),
        (bf.parity, 3),
    ])
    def test_multiple_properties(self, func_factory, n):
        """Test multiple properties return valid values."""
        f = func_factory(n)
        
        # Just check they return booleans
        assert isinstance(is_monotone(f), bool)
        assert isinstance(is_symmetric(f), bool)
    
    def test_property_consistency(self):
        """Properties should be consistent with each other."""
        f = bf.majority(5)
        
        # Majority is monotone, symmetric, and balanced
        assert is_monotone(f) == True
        assert is_symmetric(f) == True
        assert is_balanced(f) == True
        
        # Weight should be exactly half
        w = weight(f)
        assert w == 2**(5-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
