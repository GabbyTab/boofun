"""
BooFun: A comprehensive Boolean function analysis library.

This library provides tools for creating, analyzing, and visualizing Boolean functions
using multiple representations and advanced mathematical techniques.

Key Features:
- Multiple Boolean function representations (truth tables, circuits, BDDs, etc.)
- Spectral analysis and Fourier transforms
- Property testing algorithms
- Visualization tools
- Built-in Boolean function generators

Basic Usage:
    >>> import boofun as bf
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
    CallableAdapter, SymPyAdapter, NumPyAdapter,
    adapt_callable, adapt_sympy_expr, adapt_numpy_function
)

# Legacy adapter for migrating from old BooleanFunc class (kept for backwards compatibility)
# Import on demand: from boofun.core.legacy_adapter import from_legacy, to_legacy
try:
    from .core.legacy_adapter import from_legacy, to_legacy, LegacyWrapper
    _HAS_LEGACY = True
except ImportError:
    _HAS_LEGACY = False
from .analysis import SpectralAnalyzer, PropertyTester
from .analysis import sensitivity as analysis_sensitivity
from .analysis import block_sensitivity as analysis_block_sensitivity
from .analysis import certificates as analysis_certificates
from .analysis import symmetry as analysis_symmetry

# Fourier analysis utilities (Chapter 1 O'Donnell)
from .analysis.fourier import (
    parseval_verify,
    plancherel_inner_product,
    convolution,
    negate_inputs,
    odd_part,
    even_part,
    tensor_product,
    restriction,
    fourier_degree,
    spectral_norm,
    fourier_sparsity,
    dominant_coefficients,
)

# GF(2) analysis (Algebraic Normal Form)
from .analysis.gf2 import (
    gf2_fourier_transform,
    gf2_degree,
    gf2_monomials,
    gf2_to_string,
    is_linear_over_gf2,
    correlation_with_parity,
)
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


def dictator(n: int, i: int = 0) -> BooleanFunction:
    """
    Create dictator function on variable i: f(x) = x_i.
    
    Args:
        n: Number of variables
        i: Index of dictating variable (default 0)
    
    Examples:
        >>> d = bf.dictator(5)     # 5-var dictator on x₀
        >>> d = bf.dictator(5, 2)  # 5-var dictator on x₂
    """
    return BooleanFunctionBuiltins.dictator(n, i)


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


def random(n: int, balanced: bool = False, seed: int = None) -> BooleanFunction:
    """
    Create a random Boolean function on n variables.
    
    Args:
        n: Number of variables
        balanced: If True, output has equal 0s and 1s (default False)
        seed: Random seed for reproducibility
        
    Returns:
        Random Boolean function
        
    Example:
        >>> f = bf.random(4)                    # Random 4-variable function
        >>> g = bf.random(4, balanced=True)     # Random balanced function
        >>> h = bf.random(4, seed=42)           # Reproducible random function
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    
    size = 2**n
    if balanced:
        # Balanced: exactly half 0s and half 1s
        truth_table = np.zeros(size, dtype=int)
        ones_positions = np.random.choice(size, size//2, replace=False)
        truth_table[ones_positions] = 1
    else:
        truth_table = np.random.randint(0, 2, size)
    
    return create(truth_table.tolist())


def from_weights(weights, threshold_value=None) -> BooleanFunction:
    """
    Create LTF (Linear Threshold Function) from weight vector.
    
    Alias for weighted_majority() with more intuitive name for LTF creation.
    
    f(x) = 1 iff w₁x₁ + w₂x₂ + ... + wₙxₙ ≥ θ
    
    Args:
        weights: List of integer/float weights for each variable
        threshold_value: Threshold (default: sum(weights)/2)
        
    Returns:
        LTF (Linear Threshold Function)
        
    Example:
        >>> # Electoral college with 3 states: CA(55), TX(38), NY(29)
        >>> electoral = bf.from_weights([55, 38, 29], threshold=61)
    """
    return weighted_majority(weights, threshold_value)


def threshold(n: int, k: int) -> BooleanFunction:
    """
    Create k-threshold function on n variables.
    
    f(x) = 1 if Σxᵢ ≥ k, else 0
    
    Special cases:
    - threshold(n, n) = AND
    - threshold(n, 1) = OR  
    - threshold(n, (n+1)/2) = MAJORITY (for odd n)
    
    Example:
        >>> at_least_2 = bf.threshold(4, 2)  # True if ≥2 inputs are 1
    """
    from .analysis.ltf_analysis import create_threshold_function
    return create_threshold_function(n, k)


def weighted_majority(weights, threshold_value=None) -> BooleanFunction:
    """
    Create a weighted majority (LTF) function.
    
    f(x) = sign(w₁x₁ + ... + wₙxₙ - θ)
    
    LTFs are also called "halfspaces" - they represent hyperplanes
    cutting through the Boolean hypercube.
    
    Example:
        # Nassau County voting system  
        >>> nassau = bf.weighted_majority([31, 31, 28, 21, 2, 2])
        
        # Standard majority (all equal weights)
        >>> maj = bf.weighted_majority([1, 1, 1, 1, 1])
    """
    from .analysis.ltf_analysis import create_weighted_majority
    return create_weighted_majority(weights, threshold_value)


# Function families for growth analysis
try:
    from .families import (
        MajorityFamily, ParityFamily, TribesFamily, ThresholdFamily,
        ANDFamily, ORFamily, DictatorFamily, LTFFamily,
        GrowthTracker, FunctionFamily, InductiveFamily,
    )
    HAS_FAMILIES = True
except ImportError:
    HAS_FAMILIES = False

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
    "threshold",
    "weighted_majority",
    "random",           # Random function generator
    "from_weights",     # Alias for weighted_majority
    
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
    
    # Fourier analysis (Chapter 1 O'Donnell)
    "parseval_verify",
    "plancherel_inner_product",
    "convolution",
    "negate_inputs",
    "odd_part",
    "even_part",
    "tensor_product",
    "restriction",
    "fourier_degree",
    "spectral_norm",
    "fourier_sparsity",
    "dominant_coefficients",
    
    # GF(2) analysis
    "gf2_fourier_transform",
    "gf2_degree",
    "gf2_monomials",
    "gf2_to_string",
    "is_linear_over_gf2",
    "correlation_with_parity",
    
    # Testing and validation
    "BooleanFunctionValidator",
    "quick_validate",
    "test_representation",
    
    # Adapters for external integration
    "CallableAdapter", 
    "SymPyAdapter",
    "NumPyAdapter",
    "adapt_callable",
    "adapt_sympy_expr", 
    "adapt_numpy_function",
    
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
if HAS_FAMILIES:
    __all__.extend([
        "MajorityFamily", "ParityFamily", "TribesFamily", "ThresholdFamily",
        "ANDFamily", "ORFamily", "DictatorFamily", "LTFFamily",
        "GrowthTracker", "FunctionFamily", "InductiveFamily",
    ])

if HAS_VISUALIZATION:
    __all__.append("BooleanFunctionVisualizer")

if HAS_QUANTUM:
    __all__.extend(["QuantumBooleanFunction", "create_quantum_boolean_function"])
