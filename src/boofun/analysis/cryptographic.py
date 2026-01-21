"""
Cryptographic analysis of Boolean functions.

This module provides cryptographic measures commonly used in symmetric cryptography,
block cipher design, and S-box analysis. These measures help assess the resistance
of Boolean functions against linear and differential cryptanalysis.

Key concepts:
- **Nonlinearity**: Distance to the nearest affine function (resistance to linear attacks)
- **Bent functions**: Functions achieving maximum nonlinearity (perfect nonlinearity)
- **Walsh spectrum**: Fourier transform over F_2 (captures linear correlations)
- **Algebraic degree**: Degree of the ANF (resistance to algebraic attacks)
- **Balancedness**: Equal number of 0s and 1s (no bias in output)

Cross-validation:
This module is designed to produce results that can be validated against:
- thomasarmel/boolean_function (Rust): nonlinearity, is_bent, algebraic_degree
- SageMath: walsh_hadamard_transform, nonlinearity
- BooLSPLG: S-box analysis tools

References:
- Carlet, "Boolean Functions for Cryptography and Coding Theory"
- O'Donnell, "Analysis of Boolean Functions" (Fourier/Walsh connection)
- Crama & Hammer, "Boolean Functions: Theory, Algorithms, and Applications"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Core cryptographic measures
    "walsh_transform",
    "walsh_spectrum",
    "nonlinearity",
    "is_bent",
    "is_balanced",
    # Algebraic properties
    "algebraic_degree",
    "algebraic_normal_form",
    "anf_monomials",
    # Additional measures
    "correlation_immunity",
    "resiliency",
    "propagation_criterion",
    "strict_avalanche_criterion",
    # Analysis class
    "CryptographicAnalyzer",
]


def walsh_transform(f: "BooleanFunction") -> np.ndarray:
    """
    Compute the Walsh-Hadamard transform of a Boolean function.
    
    The Walsh transform W_f(a) measures the correlation between f and the
    linear function <a, x>:
    
        W_f(a) = Σ_{x ∈ F_2^n} (-1)^{f(x) ⊕ <a,x>}
    
    This is related to the Fourier transform by: W_f(a) = 2^n · f̂(a)
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Array of Walsh coefficients W_f(a) for a = 0, 1, ..., 2^n - 1
        
    Note:
        - For balanced functions, W_f(0) = 0
        - For bent functions, |W_f(a)| = 2^{n/2} for all a
        
    Cross-validation:
        thomasarmel/boolean_function uses the same definition.
    """
    n = f.n_vars or 0
    if n == 0:
        val = 1 - 2 * int(f.evaluate(0))
        return np.array([val])
    
    size = 1 << n
    
    # Get function values as ±1
    f_vals = np.array([1 - 2 * int(f.evaluate(x)) for x in range(size)])
    
    # Walsh-Hadamard transform (unnormalized)
    from ..core.optimizations import fast_walsh_hadamard
    
    # fast_walsh_hadamard normalizes by 2^n, so we need to undo that
    coeffs = fast_walsh_hadamard(f_vals.astype(float), normalize=True)
    
    # Convert from normalized Fourier to Walsh: W_f(a) = 2^n · f̂(a)
    return (coeffs * size).astype(int)


def walsh_spectrum(f: "BooleanFunction") -> Dict[int, int]:
    """
    Compute the Walsh spectrum (distribution of Walsh coefficients).
    
    The spectrum is a histogram showing how many times each Walsh value appears.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Dictionary {walsh_value: count}
        
    Example:
        >>> f = bf.parity(3)
        >>> walsh_spectrum(f)
        {-8: 1, 0: 7}  # XOR is bent-like, one large coefficient
    """
    W = walsh_transform(f)
    unique, counts = np.unique(W, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def nonlinearity(f: "BooleanFunction") -> int:
    """
    Compute the nonlinearity of a Boolean function.
    
    Nonlinearity is the minimum Hamming distance to any affine function:
    
        NL(f) = 2^{n-1} - (1/2) · max_a |W_f(a)|
    
    Higher nonlinearity means better resistance to linear cryptanalysis.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Nonlinearity (non-negative integer)
        
    Bounds:
        - NL(f) ≤ 2^{n-1} - 2^{n/2 - 1} (bent bound, achieved iff bent)
        - NL(f) = 0 for affine functions
        
    Cross-validation:
        thomasarmel/boolean_function: f.nonlinearity() should match
        
    Example:
        >>> f = bf.AND(3)
        >>> nonlinearity(f)
        2
    """
    n = f.n_vars or 0
    if n == 0:
        return 0
    
    W = walsh_transform(f)
    max_walsh = np.max(np.abs(W))
    
    return int((1 << (n - 1)) - max_walsh // 2)


def is_bent(f: "BooleanFunction") -> bool:
    """
    Check if a Boolean function is bent.
    
    A function is bent if it achieves maximum nonlinearity:
        |W_f(a)| = 2^{n/2} for all a
    
    Bent functions exist only for even n and are never balanced.
    
    Args:
        f: BooleanFunction to check
        
    Returns:
        True if f is bent
        
    Cross-validation:
        thomasarmel/boolean_function: f.is_bent() should match
        
    Example:
        >>> # The function 0x0113077C165E76A8 (6 vars) is bent
        >>> f = bf.from_hex("0113077C165E76A8", n_vars=6)
        >>> is_bent(f)
        True
    """
    n = f.n_vars or 0
    
    # Bent functions require even n
    if n % 2 != 0:
        return False
    
    if n == 0:
        return False
    
    W = walsh_transform(f)
    bent_value = 1 << (n // 2)
    
    # All Walsh coefficients must have absolute value 2^{n/2}
    return bool(np.all(np.abs(W) == bent_value))


def is_balanced(f: "BooleanFunction") -> bool:
    """
    Check if a Boolean function is balanced.
    
    A function is balanced if it has equal number of 0s and 1s in its truth table,
    equivalently W_f(0) = 0.
    
    Args:
        f: BooleanFunction to check
        
    Returns:
        True if f is balanced
        
    Cross-validation:
        thomasarmel/boolean_function: f.is_balanced() should match
    """
    n = f.n_vars or 0
    if n == 0:
        return False  # Single value can't be balanced
    
    size = 1 << n
    ones_count = sum(1 for x in range(size) if f.evaluate(x))
    
    return ones_count == size // 2


def algebraic_degree(f: "BooleanFunction") -> int:
    """
    Compute the algebraic degree (ANF degree) of a Boolean function.
    
    The algebraic degree is the maximum number of variables in any monomial
    of the Algebraic Normal Form (ANF).
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Algebraic degree (0 for constant, n for full degree)
        
    Note:
        This is computed via GF(2) analysis, not Fourier degree.
        For ±1-valued functions, ANF degree equals Fourier degree.
        
    Cross-validation:
        thomasarmel/boolean_function: f.algebraic_normal_form().degree()
    """
    from .gf2 import gf2_degree
    return gf2_degree(f)


def algebraic_normal_form(f: "BooleanFunction") -> np.ndarray:
    """
    Compute the Algebraic Normal Form (ANF) coefficients.
    
    The ANF represents f as a XOR of AND monomials over GF(2):
        f(x) = ⊕_{S ⊆ [n]} a_S · ∏_{i ∈ S} x_i
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Array of ANF coefficients a_S (0 or 1) indexed by subset mask
        
    Cross-validation:
        thomasarmel/boolean_function: f.algebraic_normal_form()
    """
    from .gf2 import gf2_fourier_transform
    return gf2_fourier_transform(f)


def anf_monomials(f: "BooleanFunction") -> List[Tuple[int, ...]]:
    """
    Return the monomials present in the ANF.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        List of tuples, each tuple contains variable indices in the monomial
        Empty tuple () represents the constant term 1
        
    Example:
        >>> f = bf.AND(2)  # x0 AND x1
        >>> anf_monomials(f)
        [(0, 1)]  # Just the monomial x0·x1
    """
    n = f.n_vars or 0
    anf = algebraic_normal_form(f)
    
    monomials = []
    for s in range(len(anf)):
        if anf[s]:
            # Convert bitmask to tuple of variable indices
            vars_in_monomial = tuple(i for i in range(n) if (s >> i) & 1)
            monomials.append(vars_in_monomial)
    
    return monomials


def correlation_immunity(f: "BooleanFunction") -> int:
    """
    Compute the correlation immunity order of a Boolean function.
    
    A function is t-th order correlation immune if it is statistically
    independent of any t input variables.
    
    Equivalently, W_f(a) = 0 for all a with 1 ≤ |a| ≤ t.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Maximum t such that f is t-th order correlation immune (0 if not CI)
    """
    n = f.n_vars or 0
    if n == 0:
        return 0
    
    W = walsh_transform(f)
    
    # Find the smallest weight where some Walsh coefficient is nonzero
    for t in range(1, n + 1):
        for a in range(1 << n):
            if bin(a).count("1") == t and W[a] != 0:
                return t - 1
    
    return n  # All non-constant Walsh coefficients are zero


def resiliency(f: "BooleanFunction") -> int:
    """
    Compute the resiliency order of a Boolean function.
    
    A function is t-resilient if it is balanced and t-th order correlation immune.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Resiliency order (-1 if not balanced, 0+ otherwise)
    """
    if not is_balanced(f):
        return -1
    
    return correlation_immunity(f)


def propagation_criterion(f: "BooleanFunction", order: int = 1) -> bool:
    """
    Check if f satisfies the Propagation Criterion of order k.
    
    PC(k) means: for all a with 1 ≤ |a| ≤ k, the function f(x) ⊕ f(x ⊕ a)
    is balanced.
    
    Args:
        f: BooleanFunction to check
        order: Order k to check (default 1)
        
    Returns:
        True if f satisfies PC(k)
    """
    import boofun as bf
    
    n = f.n_vars or 0
    if n == 0:
        return False
    
    size = 1 << n
    
    for k in range(1, order + 1):
        for a in range(1, 1 << n):
            if bin(a).count("1") != k:
                continue
            
            # Compute f(x) ⊕ f(x ⊕ a) for all x
            derivative = []
            for x in range(size):
                fx = int(f.evaluate(x))
                fxa = int(f.evaluate(x ^ a))
                derivative.append(fx ^ fxa)
            
            # Check if balanced
            if sum(derivative) != size // 2:
                return False
    
    return True


def strict_avalanche_criterion(f: "BooleanFunction") -> bool:
    """
    Check if f satisfies the Strict Avalanche Criterion (SAC).
    
    SAC means: flipping any single input bit changes the output with
    probability exactly 1/2. This is equivalent to PC(1).
    
    Args:
        f: BooleanFunction to check
        
    Returns:
        True if f satisfies SAC
    """
    return propagation_criterion(f, order=1)


class CryptographicAnalyzer:
    """
    Comprehensive cryptographic analysis of a Boolean function.
    
    This class computes and caches various cryptographic measures.
    
    Example:
        >>> f = bf.create([0, 1, 1, 0, 1, 0, 0, 1])  # 3-var bent-like
        >>> analyzer = CryptographicAnalyzer(f)
        >>> print(analyzer.summary())
    """
    
    def __init__(self, f: "BooleanFunction"):
        """
        Initialize analyzer with a Boolean function.
        
        Args:
            f: BooleanFunction to analyze
        """
        self.function = f
        self._walsh: Optional[np.ndarray] = None
        self._anf: Optional[np.ndarray] = None
    
    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.function.n_vars or 0
    
    @property
    def walsh(self) -> np.ndarray:
        """Walsh transform (cached)."""
        if self._walsh is None:
            self._walsh = walsh_transform(self.function)
        return self._walsh
    
    @property
    def anf(self) -> np.ndarray:
        """ANF coefficients (cached)."""
        if self._anf is None:
            self._anf = algebraic_normal_form(self.function)
        return self._anf
    
    def nonlinearity(self) -> int:
        """Compute nonlinearity."""
        max_walsh = np.max(np.abs(self.walsh))
        return int((1 << (self.n_vars - 1)) - max_walsh // 2)
    
    def is_bent(self) -> bool:
        """Check if bent."""
        if self.n_vars % 2 != 0 or self.n_vars == 0:
            return False
        bent_value = 1 << (self.n_vars // 2)
        return bool(np.all(np.abs(self.walsh) == bent_value))
    
    def is_balanced(self) -> bool:
        """Check if balanced."""
        return bool(self.walsh[0] == 0)
    
    def algebraic_degree(self) -> int:
        """Compute algebraic degree."""
        max_deg = 0
        for s in range(len(self.anf)):
            if self.anf[s]:
                deg = bin(s).count("1")
                max_deg = max(max_deg, deg)
        return max_deg
    
    def correlation_immunity(self) -> int:
        """Compute correlation immunity order."""
        return correlation_immunity(self.function)
    
    def resiliency(self) -> int:
        """Compute resiliency order."""
        if not self.is_balanced():
            return -1
        return self.correlation_immunity()
    
    def satisfies_sac(self) -> bool:
        """Check if satisfies SAC."""
        return strict_avalanche_criterion(self.function)
    
    def walsh_spectrum(self) -> Dict[int, int]:
        """Get Walsh spectrum."""
        unique, counts = np.unique(self.walsh, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def summary(self) -> str:
        """Return human-readable summary."""
        n = self.n_vars
        nl = self.nonlinearity()
        
        # Compute maximum possible nonlinearity
        if n % 2 == 0 and n > 0:
            max_nl = (1 << (n - 1)) - (1 << (n // 2 - 1))
        else:
            max_nl = (1 << (n - 1)) - (1 << ((n - 1) // 2))
        
        lines = [
            f"CryptographicAnalyzer (n={n})",
            f"  Nonlinearity: {nl} / {max_nl} (max possible)",
            f"  Balanced: {self.is_balanced()}",
            f"  Bent: {self.is_bent()}",
            f"  Algebraic degree: {self.algebraic_degree()}",
            f"  Correlation immunity: {self.correlation_immunity()}",
            f"  Resiliency: {self.resiliency()}",
            f"  Satisfies SAC: {self.satisfies_sac()}",
            "",
            f"  Walsh spectrum: {self.walsh_spectrum()}",
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Export all measures as dictionary (for cross-validation)."""
        return {
            "n_vars": self.n_vars,
            "nonlinearity": self.nonlinearity(),
            "is_balanced": self.is_balanced(),
            "is_bent": self.is_bent(),
            "algebraic_degree": self.algebraic_degree(),
            "correlation_immunity": self.correlation_immunity(),
            "resiliency": self.resiliency(),
            "satisfies_sac": self.satisfies_sac(),
            "walsh_spectrum": self.walsh_spectrum(),
        }
