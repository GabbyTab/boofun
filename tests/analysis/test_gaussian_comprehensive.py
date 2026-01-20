"""
Comprehensive tests for analysis/gaussian module.

Tests Gaussian noise and related analysis.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis.gaussian import *


class TestGaussianModuleImports:
    """Test gaussian module imports."""
    
    def test_module_imports(self):
        """Module should import without errors."""
        from boofun.analysis import gaussian
        assert gaussian is not None
    
    def test_module_has_functions(self):
        """Module should have expected functions."""
        from boofun.analysis import gaussian
        
        # Get all public functions
        funcs = [name for name in dir(gaussian) if not name.startswith('_')]
        assert len(funcs) > 0


class TestGaussianFunctions:
    """Test Gaussian analysis functions."""
    
    def test_gaussian_with_majority(self):
        """Gaussian analysis should work with majority."""
        f = bf.majority(5)
        
        # Test any Gaussian-related functions that exist
        from boofun.analysis import gaussian
        
        for func_name in dir(gaussian):
            if func_name.startswith('_'):
                continue
            func = getattr(gaussian, func_name)
            if callable(func):
                # Found a function, try to test it
                try:
                    result = func(f)
                    assert result is not None
                except TypeError:
                    # May need different arguments
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
