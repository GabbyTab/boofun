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
    >>> 
    >>> # Create functions from any input
    >>> xor = bf.create([0, 1, 1, 0])     # From truth table
    >>> maj = bf.majority(5)              # Built-in majority
    >>> parity = bf.parity(4)             # Built-in parity (XOR)
    >>> 
    >>> # Natural operations
    >>> g = xor & maj                     # AND
    >>> h = ~xor                          # NOT
    >>> 
    >>> # Spectral analysis
    >>> xor.fourier()                     # Fourier coefficients
    >>> xor.influences()                  # Variable influences
    >>> xor.degree()                      # Fourier degree
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

# =============================================================================
# Top-level shortcuts for common functions (mathematician-friendly API)
# =============================================================================

def majority(n: int) -> BooleanFunction:
    """
    Create majority function on n variables: Maj_n(x) = 1 iff |{i: x_i=1}| > n/2.
    
    Example:
        >>> maj5 = bf.majority(5)
        >>> maj5([1, 1, 1, 0, 0])  # True (3 > 2.5)
    """
    return BooleanFunctionBuiltins.majority(n)


def parity(n: int) -> BooleanFunction:
    """
    Create parity (XOR) function on n variables: ⊕_n(x) = x_1 ⊕ x_2 ⊕ ... ⊕ x_n.
    
    Example:
        >>> xor3 = bf.parity(3)
        >>> xor3([1, 1, 0])  # False (even number of 1s)
    """
    return BooleanFunctionBuiltins.parity(n)


def tribes(k: int, n: int) -> BooleanFunction:
    """
    Create tribes function: AND of ORs on groups of k variables.
    
    Tribes_{k,n}(x) = ⋀_{j=1}^{n/k} ⋁_{i∈T_j} x_i
    
    Example:
        >>> t = bf.tribes(2, 4)  # (x₁ ∨ x₂) ∧ (x₃ ∨ x₄)
    """
    return BooleanFunctionBuiltins.tribes(k, n)


def dictator(i: int, n: int) -> BooleanFunction:
    """
    Create dictator function on variable i: f(x) = x_i.
    
    Example:
        >>> d = bf.dictator(0, 3)  # f(x) = x₀
    """
    return BooleanFunctionBuiltins.dictator(i, n)


def constant(value: bool, n: int) -> BooleanFunction:
    """
    Create constant function: f(x) = value for all x.
    
    Example:
        >>> zero = bf.constant(False, 3)
        >>> one = bf.constant(True, 3)
    """
    return BooleanFunctionBuiltins.constant(value, n)


def AND(n: int) -> BooleanFunction:
    """
    Create AND function on n variables: f(x) = x_1 ∧ x_2 ∧ ... ∧ x_n.
    
    Example:
        >>> and3 = bf.AND(3)
    """
    truth_table = [0] * (2**n)
    truth_table[-1] = 1  # Only all-1s input gives 1
    return create(truth_table)


def OR(n: int) -> BooleanFunction:
    """
    Create OR function on n variables: f(x) = x_1 ∨ x_2 ∨ ... ∨ x_n.
    
    Example:
        >>> or3 = bf.OR(3)
    """
    truth_table = [1] * (2**n)
    truth_table[0] = 0  # Only all-0s input gives 0
    return create(truth_table)

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
    # =====================================================
    # PRIMARY API (mathematician-friendly)
    # =====================================================
    # Creation
    "create",
    "BooleanFunction",
    
    # Built-in functions (short names)
    "majority",
    "parity", 
    "tribes",
    "dictator",
    "constant",
    "AND",
    "OR",
    
    # Analysis (use directly or via function methods)
    "SpectralAnalyzer", 
    "PropertyTester",
    
    # =====================================================
    # SECONDARY API (advanced users)
    # =====================================================
    # Full builtins class
    "BooleanFunctionBuiltins",
    
    # Analysis submodules
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
