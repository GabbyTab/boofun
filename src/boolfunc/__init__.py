"""
BoolFunc: A comprehensive Boolean function analysis library.

This library provides tools for creating, analyzing, and visualizing Boolean functions
using multiple representations and advanced mathematical techniques.

Key Features:
- Multiple Boolean function representations (truth tables, circuits, BDDs, etc.)
- Spectral analysis and Fourier transforms
- Property testing algorithms
- Visualization tools
- Built-in Boolean function generators

Basic Usage:
    >>> import boolfunc as bf
    >>> # Create XOR function
    >>> xor = bf.create([0, 1, 1, 0])
    >>> # Analyze spectral properties
    >>> analyzer = bf.SpectralAnalyzer(xor)
    >>> influences = analyzer.influences()
"""

from .api import create
from .core import BooleanFunction, Space, ExactErrorModel, Property
from .core.builtins import BooleanFunctionBuiltins
from .analysis import SpectralAnalyzer, PropertyTester

# Version information
__version__ = "0.2.0"
__author__ = "Gabriel Taboada"

# Core exports for typical usage
__all__ = [
    # Core functionality
    "create",
    "BooleanFunction",
    "BooleanFunctionBuiltins",
    
    # Analysis tools
    "SpectralAnalyzer", 
    "PropertyTester",
    
    # Utilities
    "Space",
    "Property",
    "ExactErrorModel",
    
    # Version info
    "__version__"
]
