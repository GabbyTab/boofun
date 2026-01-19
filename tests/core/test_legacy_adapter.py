"""
Tests for core/legacy_adapter module.

Tests legacy API compatibility adapters.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.legacy_adapter import *


class TestLegacyAdapterImports:
    """Test that legacy adapter exports are available."""
    
    def test_module_imports(self):
        """Legacy adapter module should import without errors."""
        from boofun.core import legacy_adapter
        assert legacy_adapter is not None


class TestLegacyBooleanFunctionAPI:
    """Test legacy Boolean function API compatibility."""
    
    def test_create_from_truth_table(self):
        """Legacy create from truth table should work."""
        tt = [0, 0, 0, 1]
        f = bf.create(tt)
        
        # Should work as expected
        assert f.evaluate(0) == 0
        assert f.evaluate(3) == 1
    
    def test_builtin_functions(self):
        """Built-in functions should work."""
        f_and = bf.AND(3)
        f_or = bf.OR(3)
        f_maj = bf.majority(3)
        
        assert f_and is not None
        assert f_or is not None
        assert f_maj is not None
    
    def test_evaluation(self):
        """Evaluation should work with various inputs."""
        f = bf.AND(3)
        
        # Integer input
        assert f.evaluate(0) == 0
        assert f.evaluate(7) == 1
        
        # List/tuple input if supported
        try:
            assert f.evaluate([1, 1, 1]) == 1
        except (TypeError, ValueError):
            pass  # May not be supported


class TestLegacyFourierAPI:
    """Test legacy Fourier API compatibility."""
    
    def test_fourier_method(self):
        """fourier() method should work."""
        f = bf.majority(3)
        fourier = f.fourier()
        
        assert len(fourier) == 8
    
    def test_influences_method(self):
        """influences() method should work."""
        f = bf.majority(3)
        influences = f.influences()
        
        assert len(influences) == 3
    
    def test_total_influence_method(self):
        """total_influence() method should work."""
        f = bf.parity(3)
        total = f.total_influence()
        
        assert abs(total - 3.0) < 1e-10


class TestLegacyPropertyMethods:
    """Test legacy property methods."""
    
    def test_n_vars(self):
        """n_vars property should work."""
        for n in [2, 3, 4, 5]:
            f = bf.AND(n)
            assert f.n_vars == n
    
    def test_is_balanced(self):
        """is_balanced() method should work."""
        f_balanced = bf.majority(3)
        f_unbalanced = bf.AND(3)
        
        assert f_balanced.is_balanced() == True
        assert f_unbalanced.is_balanced() == False


class TestLegacyRepresentationAPI:
    """Test legacy representation API compatibility."""
    
    def test_get_representation_truth_table(self):
        """get_representation('truth_table') should work."""
        f = bf.OR(3)
        tt = f.get_representation("truth_table")
        
        assert tt is not None
    
    def test_representations_property(self):
        """representations property should be accessible."""
        f = bf.AND(3)
        
        # Should have some representations
        assert hasattr(f, 'representations')


class TestBackwardsCompatibility:
    """Test backwards compatibility with older API patterns."""
    
    def test_function_call_style(self):
        """Function creation style should work."""
        # Modern style
        f1 = bf.AND(3)
        
        # Alternative styles if they exist
        assert f1 is not None
    
    def test_analysis_integration(self):
        """Analysis should work with legacy-style functions."""
        f = bf.create([0, 1, 1, 0, 1, 0, 0, 1])  # Parity
        
        # All these should work
        fourier = f.fourier()
        influences = f.influences()
        total_inf = f.total_influence()
        
        assert len(fourier) == 8
        assert len(influences) == 3
        assert abs(total_inf - 3.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
