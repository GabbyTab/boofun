"""
Tests for families/builtins module.

Tests for built-in function families like Majority, Parity, Tribes, etc.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.families.builtins import (
    MajorityFamily,
    ParityFamily,
    TribesFamily,
    ThresholdFamily,
    ANDFamily,
    ORFamily,
    DictatorFamily,
)


class TestMajorityFamily:
    """Test MajorityFamily class."""
    
    def test_creation(self):
        """MajorityFamily should be creatable."""
        family = MajorityFamily()
        assert family is not None
    
    def test_generate_function(self):
        """MajorityFamily should generate functions."""
        family = MajorityFamily()
        
        for n in [3, 5, 7]:
            f = family.generate(n)
            assert f is not None
            assert f.n_vars == n
    
    def test_function_correctness(self):
        """Generated majority should be correct."""
        family = MajorityFamily()
        f = family.generate(3)
        
        # Majority of 3: output 1 if at least 2 inputs are 1
        expected_outputs = [0, 0, 0, 1, 0, 1, 1, 1]
        for x, expected in enumerate(expected_outputs):
            assert f.evaluate(x) == expected
    
    def test_metadata(self):
        """MajorityFamily should have metadata."""
        family = MajorityFamily()
        assert hasattr(family, 'metadata')


class TestParityFamily:
    """Test ParityFamily class."""
    
    def test_creation(self):
        """ParityFamily should be creatable."""
        family = ParityFamily()
        assert family is not None
    
    def test_generate_function(self):
        """ParityFamily should generate functions."""
        family = ParityFamily()
        
        for n in [2, 3, 4]:
            f = family.generate(n)
            assert f is not None
            assert f.n_vars == n
    
    def test_parity_correctness(self):
        """Generated parity should be correct (XOR of all bits)."""
        family = ParityFamily()
        f = family.generate(3)
        
        for x in range(8):
            expected = bin(x).count('1') % 2
            assert f.evaluate(x) == expected


class TestTribesFamily:
    """Test TribesFamily class."""
    
    def test_creation(self):
        """TribesFamily should be creatable."""
        family = TribesFamily()
        assert family is not None
    
    def test_generate_function(self):
        """TribesFamily should generate functions."""
        family = TribesFamily()
        
        f = family.generate(9)  # Tribes with 3 tribes of 3
        assert f is not None


class TestThresholdFamily:
    """Test ThresholdFamily class."""
    
    def test_creation(self):
        """ThresholdFamily should be creatable."""
        family = ThresholdFamily()
        assert family is not None
    
    def test_generate_function(self):
        """ThresholdFamily should generate functions."""
        family = ThresholdFamily()
        
        f = family.generate(4)
        assert f is not None


class TestANDFamily:
    """Test ANDFamily class."""
    
    def test_creation(self):
        """ANDFamily should be creatable."""
        family = ANDFamily()
        assert family is not None
    
    def test_generate_function(self):
        """ANDFamily should generate AND functions."""
        family = ANDFamily()
        
        for n in [2, 3, 4]:
            f = family.generate(n)
            assert f is not None
    
    def test_and_correctness(self):
        """Generated AND should be correct."""
        family = ANDFamily()
        f = family.generate(3)
        
        # AND is 1 only when all inputs are 1
        for x in range(8):
            expected = 1 if x == 7 else 0
            assert f.evaluate(x) == expected


class TestORFamily:
    """Test ORFamily class."""
    
    def test_creation(self):
        """ORFamily should be creatable."""
        family = ORFamily()
        assert family is not None
    
    def test_generate_function(self):
        """ORFamily should generate OR functions."""
        family = ORFamily()
        
        for n in [2, 3, 4]:
            f = family.generate(n)
            assert f is not None
    
    def test_or_correctness(self):
        """Generated OR should be correct."""
        family = ORFamily()
        f = family.generate(3)
        
        # OR is 0 only when all inputs are 0
        for x in range(8):
            expected = 0 if x == 0 else 1
            assert f.evaluate(x) == expected


class TestDictatorFamily:
    """Test DictatorFamily class."""
    
    def test_creation(self):
        """DictatorFamily should be creatable."""
        family = DictatorFamily()
        assert family is not None
    
    def test_generate_function(self):
        """DictatorFamily should generate dictator functions."""
        family = DictatorFamily()
        
        f = family.generate(4)
        assert f is not None


class TestFamilyProperties:
    """Test general properties of function families."""
    
    def test_families_have_metadata(self):
        """All families should have metadata."""
        families = [
            MajorityFamily(),
            ParityFamily(),
            ANDFamily(),
            ORFamily(),
        ]
        
        for family in families:
            assert hasattr(family, 'metadata')
    
    def test_families_generate_consistent_functions(self):
        """Same n should give same function."""
        family = MajorityFamily()
        
        f1 = family.generate(5)
        f2 = family.generate(5)
        
        # Should produce equivalent functions
        for x in range(32):
            assert f1.evaluate(x) == f2.evaluate(x)


class TestFamilyTheoretical:
    """Test theoretical properties of families."""
    
    def test_majority_balanced_odd(self):
        """Majority on odd n should be balanced."""
        family = MajorityFamily()
        
        for n in [3, 5, 7]:
            f = family.generate(n)
            
            ones = sum(f.evaluate(x) for x in range(2**n))
            zeros = 2**n - ones
            
            assert ones == zeros  # Balanced
    
    def test_parity_balanced(self):
        """Parity should always be balanced."""
        family = ParityFamily()
        
        for n in [2, 3, 4]:
            f = family.generate(n)
            
            ones = sum(f.evaluate(x) for x in range(2**n))
            zeros = 2**n - ones
            
            assert ones == zeros


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
