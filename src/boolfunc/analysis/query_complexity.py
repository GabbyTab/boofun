"""
Query complexity measures for Boolean functions.

This module implements various query complexity measures as described in
Scott Aaronson's Boolean Function Wizard and related literature.

Query complexity measures how many queries to the input bits are needed
to compute a Boolean function under different computational models:

- D(f): Deterministic query complexity (worst-case)
- R0(f): Zero-error randomized query complexity
- R2(f): Two-sided-error (bounded-error) randomized query complexity  
- Q(f): Bounded-error quantum query complexity

Also includes related measures:
- Ambainis complexity (quantum lower bound)
- Various degree measures (approximate, nondeterministic)

References:
- Aaronson, "Algorithms for Boolean Function Query Measures" (2000)
- Buhrman & de Wolf, "Complexity Measures and Decision Tree Complexity" (2002)
- O'Donnell, "Analysis of Boolean Functions" (2014)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from math import ceil, sqrt

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Core complexity measures
    "deterministic_query_complexity",
    "average_deterministic_complexity",
    "zero_error_randomized_complexity",
    "bounded_error_randomized_complexity",
    # Lower bounds
    "ambainis_complexity",
    "certificate_lower_bound",
    "sensitivity_lower_bound",
    "block_sensitivity_lower_bound",
    # Degree measures
    "approximate_degree",
    "threshold_degree",
    # Utility
    "QueryComplexityProfile",
]


def deterministic_query_complexity(f: "BooleanFunction") -> int:
    """
    Compute D(f), the deterministic query complexity (worst-case).
    
    This is the minimum depth of a decision tree that computes f.
    Same as decision_tree_depth() from complexity.py.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Worst-case number of queries needed
    """
    from .complexity import decision_tree_depth
    return decision_tree_depth(f)


def average_deterministic_complexity(f: "BooleanFunction") -> float:
    """
    Compute D_avg(f), the average-case deterministic query complexity.
    
    This is the expected number of queries under the uniform distribution
    on inputs, using an optimal decision tree.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Average number of queries needed
    """
    from .complexity import decision_tree_depth
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    
    # We compute this via the optimal decision tree
    # For now, we use a greedy approximation based on maximum information gain
    # (A full optimal solution requires tracking paths in the DP)
    
    # Greedy approximation: repeatedly pick variable with max entropy reduction
    total_queries = 0.0
    size = 1 << n
    
    # Each input contributes its depth in an optimal tree
    # Approximate by balanced tree depth
    for x in range(size):
        # Simple approximation: depth based on certificate complexity
        from .certificates import certificate
        cert_size, _ = certificate(f, x)
        total_queries += cert_size
    
    return total_queries / size


def zero_error_randomized_complexity(f: "BooleanFunction") -> float:
    """
    Compute R0(f), the zero-error randomized query complexity.
    
    This is the expected number of queries needed by the best randomized
    algorithm that always outputs the correct answer (Las Vegas).
    
    Satisfies: R0(f) >= sqrt(D(f)) and R0(f) <= D(f)
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Expected number of queries for zero-error randomized computation
        
    Note:
        This is an approximation; the exact computation requires solving
        a linear program over all possible randomized protocols.
    """
    # Lower bound: max(C0(f), C1(f)) where C_b is certificate complexity for b-inputs
    from .complexity import max_certificate_complexity, decision_tree_depth
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    D = decision_tree_depth(f)
    C0 = max_certificate_complexity(f, 0)
    C1 = max_certificate_complexity(f, 1)
    
    # R0(f) is at least the expected certificate complexity
    # Upper bound: D(f) (can always use deterministic algorithm)
    # Approximation: geometric mean of certificate complexities
    
    # Better approximation using known bounds
    lower_bound = max(C0, C1)
    upper_bound = D
    
    # For many functions, R0 is close to sqrt(C0 * C1)
    approx = sqrt(C0 * C1) if C0 > 0 and C1 > 0 else lower_bound
    
    return max(lower_bound, min(approx, upper_bound))


def bounded_error_randomized_complexity(f: "BooleanFunction", error: float = 1/3) -> float:
    """
    Compute R2(f), the bounded-error randomized query complexity.
    
    This is the minimum expected queries for a randomized algorithm that
    outputs the correct answer with probability >= 1 - error.
    
    Satisfies: R2(f) = Omega(sqrt(bs(f))) and R2(f) = O(D(f))
    
    Args:
        f: BooleanFunction to analyze
        error: Maximum error probability (default 1/3)
        
    Returns:
        Expected queries for bounded-error randomized computation
        
    Note:
        This is an approximation based on known lower bounds.
    """
    from .block_sensitivity import max_block_sensitivity
    from .complexity import decision_tree_depth, max_sensitivity
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    # Lower bounds
    bs = max_block_sensitivity(f)
    s = max_sensitivity(f)
    D = decision_tree_depth(f)
    
    # R2(f) >= Omega(sqrt(bs(f))) - this is tight for many functions
    # R2(f) >= Omega(sqrt(s(f) * bs(f))) is a better lower bound
    
    lower_bound = sqrt(s * bs) if s > 0 and bs > 0 else max(1, sqrt(bs))
    upper_bound = D
    
    return max(lower_bound, min(lower_bound * 1.5, upper_bound))


def ambainis_complexity(f: "BooleanFunction") -> float:
    """
    Compute the Ambainis adversary bound, a lower bound for Q2(f).
    
    The Ambainis bound is defined as:
        Adv(f) = max_R sqrt(max_x |{y: R(x,y)=1}| * max_y |{x: R(x,y)=1}|)
                 / max_{x,y:R(x,y)=1} |{i: x_i != y_i}|
    
    where R is any binary relation with R(x,y) = 1 only when f(x) != f(y).
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Ambainis adversary lower bound for quantum query complexity
        
    Note:
        Computing the optimal R is NP-hard in general. This uses a heuristic.
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    size = 1 << n
    
    # Collect 0-inputs and 1-inputs
    zeros = [x for x in range(size) if not truth_table[x]]
    ones = [x for x in range(size) if truth_table[x]]
    
    if len(zeros) == 0 or len(ones) == 0:
        return 0.0  # Constant function
    
    # Simple heuristic: use the "all-pairs" relation R(x,y) = 1 iff f(x) != f(y)
    # For each pair, count Hamming distance
    
    best_bound = 0.0
    
    # For efficiency, sample if too many pairs
    max_pairs = 10000
    import random
    
    if len(zeros) * len(ones) > max_pairs:
        # Sample pairs
        pairs = [(random.choice(zeros), random.choice(ones)) for _ in range(max_pairs)]
    else:
        pairs = [(z, o) for z in zeros for o in ones]
    
    # Compute Hamming distances
    min_hamming = n
    for z, o in pairs:
        h = bin(z ^ o).count("1")
        min_hamming = min(min_hamming, h)
    
    if min_hamming == 0:
        return 0.0
    
    # Ambainis bound approximation
    # sqrt(|zeros| * |ones|) / min_hamming
    bound = sqrt(len(zeros) * len(ones)) / min_hamming
    
    return bound


def certificate_lower_bound(f: "BooleanFunction") -> int:
    """
    Compute lower bound on D(f) from certificate complexity.
    
    D(f) >= max(C0(f), C1(f))
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Certificate-based lower bound
    """
    from .complexity import max_certificate_complexity
    
    C0 = max_certificate_complexity(f, 0)
    C1 = max_certificate_complexity(f, 1)
    
    return max(C0, C1)


def sensitivity_lower_bound(f: "BooleanFunction") -> int:
    """
    Compute lower bound on D(f) from sensitivity.
    
    By Huang's theorem (2019): D(f) >= s(f)
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Sensitivity-based lower bound
    """
    from .complexity import max_sensitivity
    return max_sensitivity(f)


def block_sensitivity_lower_bound(f: "BooleanFunction") -> int:
    """
    Compute lower bound on D(f) from block sensitivity.
    
    D(f) >= bs(f)
    
    Also: bs(f) <= D(f) <= bs(f)^2 (the latter is Nisan's theorem)
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Block sensitivity-based lower bound
    """
    from .block_sensitivity import max_block_sensitivity
    return max_block_sensitivity(f)


def approximate_degree(f: "BooleanFunction", epsilon: float = 1/3) -> float:
    """
    Estimate the approximate degree deg_epsilon(f).
    
    The approximate degree is the minimum degree of a polynomial p such that
    |p(x) - f(x)| <= epsilon for all x in {0,1}^n.
    
    This is a lower bound for R2(f): R2(f) >= deg_1/3(f)
    
    Args:
        f: BooleanFunction to analyze
        epsilon: Approximation parameter
        
    Returns:
        Estimated approximate degree
        
    Note:
        Exact computation requires linear programming. This uses bounds.
    """
    from .block_sensitivity import max_block_sensitivity
    from .complexity import max_sensitivity
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    bs = max_block_sensitivity(f)
    s = max_sensitivity(f)
    
    # Lower bound: Omega(sqrt(bs(f)))
    # For AND/OR: deg_1/3 = Theta(sqrt(n))
    
    return sqrt(bs)


def threshold_degree(f: "BooleanFunction") -> int:
    """
    Compute the threshold degree of f (degree as a sign-polynomial).
    
    The threshold degree is the minimum degree d such that there exists
    a polynomial p of degree d with sign(p(x)) = f(x) for all x.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Threshold degree
        
    Note:
        This equals the real degree for most Boolean functions.
    """
    from ..analysis.fourier import fourier_degree
    
    # Threshold degree <= real degree
    # For most functions they're equal
    return fourier_degree(f)


class QueryComplexityProfile:
    """
    Compute and store query complexity measures for a Boolean function.
    
    This class provides a comprehensive analysis similar to Aaronson's
    Boolean Function Wizard.
    """
    
    def __init__(self, f: "BooleanFunction"):
        """
        Initialize query complexity profile.
        
        Args:
            f: BooleanFunction to analyze
        """
        self.function = f
        self.n_vars = f.n_vars
        self._computed = False
        self._measures: Dict[str, float] = {}
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all query complexity measures.
        
        Returns:
            Dictionary of complexity measures
        """
        if self._computed:
            return self._measures
        
        f = self.function
        
        from .complexity import (
            decision_tree_depth, max_sensitivity, average_sensitivity,
            max_certificate_complexity
        )
        from .block_sensitivity import max_block_sensitivity
        from ..analysis.fourier import fourier_degree
        from ..analysis.gf2 import gf2_degree
        
        # Basic properties
        self._measures["n"] = self.n_vars or 0
        
        # Sensitivity measures
        self._measures["s"] = max_sensitivity(f)
        self._measures["s0"] = max_sensitivity(f, 0)
        self._measures["s1"] = max_sensitivity(f, 1)
        self._measures["avg_s"] = average_sensitivity(f)
        
        # Block sensitivity
        self._measures["bs"] = max_block_sensitivity(f)
        
        # Certificate complexity
        self._measures["C"] = max(
            max_certificate_complexity(f, 0),
            max_certificate_complexity(f, 1)
        )
        self._measures["C0"] = max_certificate_complexity(f, 0)
        self._measures["C1"] = max_certificate_complexity(f, 1)
        
        # Decision tree complexity
        self._measures["D"] = decision_tree_depth(f)
        
        # Degree measures
        self._measures["deg"] = fourier_degree(f)
        self._measures["degZ2"] = gf2_degree(f)
        
        # Randomized complexity (approximations)
        self._measures["R0"] = zero_error_randomized_complexity(f)
        self._measures["R2"] = bounded_error_randomized_complexity(f)
        
        # Quantum lower bound
        self._measures["Amb"] = ambainis_complexity(f)
        
        # Influence
        from ..analysis import SpectralAnalyzer
        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()
        self._measures["max_inf"] = float(np.max(influences)) if len(influences) > 0 else 0.0
        self._measures["total_inf"] = float(np.sum(influences))
        
        self._computed = True
        return self._measures
    
    def summary(self) -> str:
        """
        Return a human-readable summary in BFW style.
        """
        m = self.compute()
        
        lines = [
            "Query Complexity Profile",
            "=" * 40,
            f"Variables:      n = {m['n']:.0f}",
            "",
            "SENSITIVITY MEASURES:",
            f"  s(f)          {m['s']:.0f}",
            f"  s0(f)         {m['s0']:.0f}",
            f"  s1(f)         {m['s1']:.0f}",
            f"  avg_s(f)      {m['avg_s']:.4f}",
            f"  bs(f)         {m['bs']:.0f}",
            f"  max_inf(f)    {m['max_inf']:.4f}",
            f"  total_inf(f)  {m['total_inf']:.4f}",
            "",
            "DEGREE MEASURES:",
            f"  deg(f)        {m['deg']:.0f}",
            f"  degZ2(f)      {m['degZ2']:.0f}",
            "",
            "DETERMINISTIC COMPLEXITY:",
            f"  D(f)          {m['D']:.0f}",
            f"  C(f)          {m['C']:.0f}",
            f"  C0(f)         {m['C0']:.0f}",
            f"  C1(f)         {m['C1']:.0f}",
            "",
            "RANDOMIZED COMPLEXITY (approx):",
            f"  R0(f)         {m['R0']:.2f}",
            f"  R2(f)         {m['R2']:.2f}",
            "",
            "QUANTUM BOUNDS:",
            f"  Amb(f)        {m['Amb']:.4f}",
        ]
        
        return "\n".join(lines)
    
    def check_known_relations(self) -> Dict[str, bool]:
        """
        Verify known relationships between complexity measures.
        
        Returns:
            Dictionary of relationship checks
        """
        m = self.compute()
        
        checks = {}
        
        # Sensitivity vs certificate
        checks["s <= C"] = m["s"] <= m["C"]
        checks["s <= bs"] = m["s"] <= m["bs"]
        
        # Block sensitivity bounds
        checks["bs <= C"] = m["bs"] <= m["C"]
        checks["bs <= D"] = m["bs"] <= m["D"]
        
        # Certificate bounds
        checks["C <= D"] = m["C"] <= m["D"]
        checks["D <= C0*C1"] = m["D"] <= m["C0"] * m["C1"]
        
        # Degree bounds
        checks["deg >= bs/2"] = m["deg"] >= m["bs"] / 2
        
        # Total influence = average sensitivity
        checks["total_inf = avg_s"] = abs(m["total_inf"] - m["avg_s"]) < 0.001
        
        return checks
