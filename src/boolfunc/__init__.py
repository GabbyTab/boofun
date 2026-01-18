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
from .core import BooleanFunction, Space, ExactErrorModel, Property, PACErrorModel, NoiseErrorModel
from .core.builtins import BooleanFunctionBuiltins
from .core.adapters import (
    LegacyAdapter, CallableAdapter, SymPyAdapter, NumPyAdapter,
    adapt_legacy_function, adapt_callable, adapt_sympy_expr, adapt_numpy_function
)
from .core.legacy_adapter import from_legacy, to_legacy, LegacyWrapper
from .analysis import SpectralAnalyzer, PropertyTester
from .analysis import sensitivity as analysis_sensitivity
from .analysis import block_sensitivity as analysis_block_sensitivity
from .analysis import certificates as analysis_certificates
from .analysis import symmetry as analysis_symmetry
from .testing import BooleanFunctionValidator, quick_validate, test_representation
from .utils.finite_fields import get_field as get_gf_field, GFField

# Optional imports with graceful fallback
try:
    from .visualization import BooleanFunctionVisualizer
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

try:
    from .quantum import QuantumBooleanFunction, create_quantum_boolean_function
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

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
    "analysis_sensitivity",
    "analysis_block_sensitivity",
    "analysis_certificates",
    "analysis_symmetry",
    
    # Testing and validation
    "BooleanFunctionValidator",
    "quick_validate",
    "test_representation",
    
    # Adapters for external integration
    "LegacyAdapter",
    "CallableAdapter", 
    "SymPyAdapter",
    "NumPyAdapter",
    "adapt_legacy_function",
    "adapt_callable",
    "adapt_sympy_expr", 
    "adapt_numpy_function",
    "from_legacy",
    "to_legacy",
    "LegacyWrapper",
    
    # Core utilities
    "Space",
    "Property",
    "ExactErrorModel",
    "PACErrorModel", 
    "NoiseErrorModel",
    "get_gf_field",
    "GFField",
    
    # Version info
    "__version__"
]

# Add optional exports if available
if HAS_VISUALIZATION:
    __all__.append("BooleanFunctionVisualizer")

if HAS_QUANTUM:
    __all__.extend(["QuantumBooleanFunction", "create_quantum_boolean_function"])
