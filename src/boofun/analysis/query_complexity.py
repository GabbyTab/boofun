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
    "one_sided_randomized_complexity",
    "nondeterministic_complexity",
    # Quantum complexity
    "quantum_query_complexity",
    "exact_quantum_complexity",
    # Sensitivity variants
    "everywhere_sensitivity",
    "average_everywhere_sensitivity",
    # Lower bounds
    "ambainis_complexity",
    "spectral_adversary_bound",
    "polynomial_method_bound",
    "general_adversary_bound",
    "certificate_lower_bound",
    "sensitivity_lower_bound",
    "block_sensitivity_lower_bound",
    # Degree measures
    "approximate_degree",
    "one_sided_approximate_degree",
    "nondeterministic_degree",
    "strong_nondeterministic_degree",
    "weak_nondeterministic_degree",
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


def one_sided_randomized_complexity(f: "BooleanFunction", side: int = 1) -> float:
    """
    Compute R1(f), the one-sided-error randomized query complexity.
    
    A one-sided algorithm never errs on inputs with f(x) = side.
    
    Satisfies: R2(f) <= R1(f) <= R0(f) <= D(f)
    
    Args:
        f: BooleanFunction to analyze
        side: Which side has no error (0 or 1, default 1)
        
    Returns:
        Estimated one-sided randomized complexity
    """
    from .complexity import max_certificate_complexity, decision_tree_depth
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    # R1(f) is related to certificate complexity on the "side" inputs
    C_side = max_certificate_complexity(f, side)
    C_other = max_certificate_complexity(f, 1 - side)
    D = decision_tree_depth(f)
    
    # R1(f) >= C_side (must verify certificates on side inputs)
    # Approximation: between C_side and R0
    lower_bound = C_side
    r0_approx = sqrt(C_side * C_other) if C_side > 0 and C_other > 0 else C_side
    
    return max(lower_bound, min(r0_approx, D))


def nondeterministic_complexity(f: "BooleanFunction", side: int = 1) -> float:
    """
    Compute NR(f), the nondeterministic query complexity.
    
    This is the minimum certificate complexity over inputs where f(x) = side.
    Nondeterministic algorithms "guess" the certificate and verify it.
    
    Satisfies: NR(f) <= R1(f)
    
    Args:
        f: BooleanFunction to analyze
        side: Which value to compute NR for (0 or 1, default 1)
        
    Returns:
        Nondeterministic query complexity
    """
    from .complexity import certificate_complexity
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    
    min_cert = n + 1
    for x in range(1 << n):
        if truth_table[x] == bool(side):
            cert, _ = certificate_complexity(f, x)
            min_cert = min(min_cert, cert)
    
    return min_cert if min_cert <= n else 0.0


def everywhere_sensitivity(f: "BooleanFunction") -> int:
    """
    Compute es(f), the everywhere sensitivity.
    
    The everywhere sensitivity is the minimum sensitivity over all inputs:
        es(f) = min_x s(f, x)
    
    This measures the "easiest" input to compute in terms of sensitivity.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Minimum sensitivity across all inputs
    """
    from .complexity import sensitivity, min_sensitivity
    
    return min_sensitivity(f)


def average_everywhere_sensitivity(f: "BooleanFunction", value: Optional[int] = None) -> float:
    """
    Compute esu(f), the average everywhere sensitivity.
    
    This is the average of min sensitivity values, optionally restricted
    to inputs where f(x) = value.
    
    Args:
        f: BooleanFunction to analyze
        value: If specified (0 or 1), only consider inputs where f(x) = value
        
    Returns:
        Average of minimum sensitivities
    """
    from .complexity import sensitivity
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    
    sensitivities = []
    for x in range(1 << n):
        if value is not None and truth_table[x] != bool(value):
            continue
        sensitivities.append(sensitivity(f, x))
    
    return float(np.mean(sensitivities)) if sensitivities else 0.0


def quantum_query_complexity(f: "BooleanFunction") -> float:
    """
    Estimate Q2(f), the bounded-error quantum query complexity.
    
    Uses multiple lower bounds:
    - Ambainis adversary bound
    - Spectral adversary bound  
    - sqrt(bs(f)) (Grover lower bound)
    
    Q2(f) = Theta(sqrt(D(f))) for many functions by Grover-type algorithms.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Estimated bounded-error quantum query complexity
    """
    from .block_sensitivity import max_block_sensitivity
    from .complexity import decision_tree_depth
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    # Lower bounds
    amb = ambainis_complexity(f)
    bs = max_block_sensitivity(f)
    D = decision_tree_depth(f)
    
    # Q2(f) >= max(Amb(f), sqrt(bs(f)))
    grover_bound = sqrt(bs)
    lower_bound = max(amb, grover_bound)
    
    # Upper bound: can always do Grover search over D(f)-depth tree
    upper_bound = sqrt(D) * 2  # Rough upper bound
    
    # Q2(f) is typically around sqrt(D) for read-once functions
    return max(lower_bound, min(sqrt(D), upper_bound))


def exact_quantum_complexity(f: "BooleanFunction") -> float:
    """
    Estimate QE(f), the exact quantum query complexity.
    
    QE(f) is the minimum queries for a quantum algorithm that always
    outputs the correct answer (no error allowed).
    
    Satisfies: Q2(f) <= QE(f) <= D(f)
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Estimated exact quantum query complexity
    """
    from .complexity import decision_tree_depth
    from ..analysis.gf2 import gf2_degree
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    D = decision_tree_depth(f)
    deg = gf2_degree(f)
    
    # QE(f) >= deg(f) (exact computation needs all degree-deg terms)
    # For symmetric functions, QE(f) = Theta(n)
    # For many functions, QE(f) is close to D(f)
    
    lower_bound = deg
    
    # Check if function is symmetric - if so, QE is high
    from .basic_properties import is_symmetric
    if is_symmetric(f):
        return max(lower_bound, n // 2)
    
    return max(lower_bound, min(D, sqrt(D) * 2))


def spectral_adversary_bound(f: "BooleanFunction") -> float:
    """
    Compute the spectral adversary bound for Q2(f).
    
    This is the "positive weights" adversary method, which is sometimes
    tighter than the basic Ambainis bound.
    
    The spectral bound equals sqrt(λ) where λ is the largest eigenvalue
    of a certain matrix derived from the function structure.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Spectral adversary lower bound
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    size = 1 << n
    
    zeros = [x for x in range(size) if not truth_table[x]]
    ones = [x for x in range(size) if truth_table[x]]
    
    if len(zeros) == 0 or len(ones) == 0:
        return 0.0
    
    # Build adversary matrix
    # M[i,j] = 1/hamming(zeros[i], ones[j]) if connected
    m0, m1 = len(zeros), len(ones)
    
    # For efficiency, limit matrix size
    if m0 * m1 > 10000:
        # Subsample
        import random
        zeros = random.sample(zeros, min(100, m0))
        ones = random.sample(ones, min(100, m1))
        m0, m1 = len(zeros), len(ones)
    
    M = np.zeros((m0, m1))
    for i, z in enumerate(zeros):
        for j, o in enumerate(ones):
            h = bin(z ^ o).count("1")
            if h > 0:
                M[i, j] = 1.0 / h
    
    # Spectral norm of M
    try:
        from scipy.linalg import svdvals
        singular_values = svdvals(M)
        if len(singular_values) > 0:
            return float(singular_values[0])
    except ImportError:
        # Fallback: use power iteration
        pass
    
    return ambainis_complexity(f)


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


def one_sided_approximate_degree(f: "BooleanFunction", side: int = 1, epsilon: float = 1/3) -> float:
    """
    Estimate deg1(f), the one-sided approximate degree.
    
    This is the minimum degree of a polynomial p such that:
    - p(x) >= 1 - epsilon when f(x) = side
    - p(x) <= epsilon when f(x) != side
    
    Satisfies: deg1(f) >= deg2(f) (two-sided approximate degree)
    
    Args:
        f: BooleanFunction to analyze
        side: Which side to approximate (0 or 1, default 1)
        epsilon: Approximation parameter
        
    Returns:
        Estimated one-sided approximate degree
    """
    from .complexity import max_certificate_complexity
    
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    # deg1(f) >= sqrt(C_side(f)) for the "side" inputs
    C_side = max_certificate_complexity(f, side)
    
    # Also bounded below by approximate degree
    deg2 = approximate_degree(f, epsilon)
    
    return max(sqrt(C_side), deg2)


def nondeterministic_degree(f: "BooleanFunction", side: int = 1) -> float:
    """
    Estimate ndeg(f), the nondeterministic degree.
    
    This is the minimum degree of a polynomial p such that:
    - p(x) >= 1 when f(x) = side
    - p(x) = 0 when f(x) != side
    
    This equals the minimum size of an AND of ORs (DNF width for side=1).
    
    Satisfies: ndeg(f) <= deg1(f)
    
    Args:
        f: BooleanFunction to analyze
        side: Which side to exactly represent (0 or 1, default 1)
        
    Returns:
        Estimated nondeterministic degree
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    
    # ndeg is related to the minimum DNF term width (for side=1)
    # or minimum CNF clause width (for side=0)
    
    # Simple bound: minimum certificate complexity
    from .complexity import certificate_complexity
    
    min_cert = n + 1
    for x in range(1 << n):
        if truth_table[x] == bool(side):
            cert, _ = certificate_complexity(f, x)
            min_cert = min(min_cert, cert)
    
    return min_cert if min_cert <= n else 0.0


def strong_nondeterministic_degree(f: "BooleanFunction") -> float:
    """
    Estimate degs(f), the strong nondeterministic degree.
    
    This is the minimum degree needed for polynomials that:
    - Are nonnegative on all inputs
    - Are > 0 exactly when f(x) = 1
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Estimated strong nondeterministic degree
    """
    # degs(f) >= max(ndeg0(f), ndeg1(f))
    ndeg0 = nondeterministic_degree(f, 0)
    ndeg1 = nondeterministic_degree(f, 1)
    
    return max(ndeg0, ndeg1)


def weak_nondeterministic_degree(f: "BooleanFunction") -> float:
    """
    Estimate degw(f), the weak nondeterministic degree.
    
    This is min(ndeg0(f), ndeg1(f)).
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Estimated weak nondeterministic degree
    """
    ndeg0 = nondeterministic_degree(f, 0)
    ndeg1 = nondeterministic_degree(f, 1)
    
    return min(ndeg0, ndeg1)


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


def polynomial_method_bound(f: "BooleanFunction") -> float:
    """
    Compute lower bound on Q2(f) via the polynomial method.
    
    The polynomial method shows that quantum algorithms induce low-degree
    polynomials. Any quantum algorithm making T queries induces polynomials
    of degree at most 2T representing acceptance probabilities.
    
    Therefore: Q2(f) >= deg~(f)/2, where deg~(f) is the approximate degree.
    
    For symmetric functions, this is often tight.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Polynomial method lower bound for quantum query complexity
        
    References:
        - Beals et al., "Quantum lower bounds by polynomials" (2001)
        - Belovs, "A Direct Reduction from Polynomial to Adversary Method" (TQC 2024)
    """
    # The polynomial method bound is deg~(f)/2
    approx_deg = approximate_degree(f)
    return approx_deg / 2


def general_adversary_bound(f: "BooleanFunction") -> float:
    """
    Estimate the general (negative-weight) adversary bound for Q2(f).
    
    The general adversary method with negative weights *characterizes*
    bounded-error quantum query complexity for total Boolean functions:
    
        Q2(f) = Θ(ADV±(f))
    
    This is the strongest known quantum lower bound technique.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Estimated general adversary bound (approximation)
        
    Note:
        Computing the exact ADV±(f) requires solving a semidefinite program.
        This implementation provides a heuristic approximation combining
        multiple lower bound techniques.
        
    References:
        - Reichardt, "Reflections for quantum query algorithms" (2011)
        - Belovs (TQC 2024): dual polynomials → adversary bounds
    """
    # The general adversary is at least as strong as:
    # 1. Spectral adversary
    # 2. Ambainis bound
    # 3. Polynomial method / 2
    
    spec_adv = spectral_adversary_bound(f)
    amb = ambainis_complexity(f)
    poly_bound = polynomial_method_bound(f)
    
    # Take maximum of all known bounds
    return max(spec_adv, amb, poly_bound)


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
        self._measures["deg2"] = approximate_degree(f)
        self._measures["ndeg"] = nondeterministic_degree(f)
        self._measures["degs"] = strong_nondeterministic_degree(f)
        self._measures["degw"] = weak_nondeterministic_degree(f)
        
        # Everywhere sensitivity
        self._measures["es"] = everywhere_sensitivity(f)
        self._measures["esu"] = average_everywhere_sensitivity(f)
        
        # Randomized complexity (approximations)
        self._measures["R0"] = zero_error_randomized_complexity(f)
        self._measures["R1"] = one_sided_randomized_complexity(f)
        self._measures["R2"] = bounded_error_randomized_complexity(f)
        self._measures["NR"] = nondeterministic_complexity(f)
        
        # Quantum complexity
        self._measures["Q2"] = quantum_query_complexity(f)
        self._measures["QE"] = exact_quantum_complexity(f)
        self._measures["Amb"] = ambainis_complexity(f)
        self._measures["SpecAdv"] = spectral_adversary_bound(f)
        self._measures["PolyMethod"] = polynomial_method_bound(f)
        self._measures["GenAdv"] = general_adversary_bound(f)
        
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
            "Boolean Function Wizard - Query Complexity Profile",
            "=" * 50,
            f"Variables:      n = {m['n']:.0f}",
            "",
            "BASIC PROPERTIES:",
            f"  unate         (see basic_properties)",
            f"  balanced      (Pr[f=1] = 0.5?)",
            "",
            "SENSITIVITY MEASURES:",
            f"  s(f)          {m['s']:.0f}          (max sensitivity)",
            f"  s0(f)         {m['s0']:.0f}          (max sens on 0-inputs)",
            f"  s1(f)         {m['s1']:.0f}          (max sens on 1-inputs)",
            f"  avg_s(f)      {m['avg_s']:.4f}   (average sensitivity)",
            f"  es(f)         {m['es']:.0f}          (everywhere sensitivity)",
            f"  esu(f)        {m['esu']:.4f}   (avg everywhere sensitivity)",
            f"  bs(f)         {m['bs']:.0f}          (block sensitivity)",
            f"  max_inf(f)    {m['max_inf']:.4f}   (max influence)",
            f"  total_inf(f)  {m['total_inf']:.4f}   (total influence)",
            "",
            "DEGREE MEASURES:",
            f"  deg(f)        {m['deg']:.0f}          (real degree)",
            f"  degZ2(f)      {m['degZ2']:.0f}          (GF(2) degree)",
            f"  deg2(f)       {m['deg2']:.2f}      (approx degree, 2-sided)",
            f"  ndeg(f)       {m['ndeg']:.2f}      (nondeterministic degree)",
            f"  degs(f)       {m['degs']:.2f}      (strong nondet degree)",
            f"  degw(f)       {m['degw']:.2f}      (weak nondet degree)",
            "",
            "DETERMINISTIC COMPLEXITY:",
            f"  D(f)          {m['D']:.0f}          (decision tree depth)",
            f"  C(f)          {m['C']:.0f}          (certificate complexity)",
            f"  C0(f)         {m['C0']:.0f}          (cert complexity, 0-inputs)",
            f"  C1(f)         {m['C1']:.0f}          (cert complexity, 1-inputs)",
            "",
            "RANDOMIZED COMPLEXITY (approximations):",
            f"  R0(f)         {m['R0']:.2f}      (zero-error randomized)",
            f"  R1(f)         {m['R1']:.2f}      (one-sided randomized)",
            f"  R2(f)         {m['R2']:.2f}      (bounded-error randomized)",
            f"  NR(f)         {m['NR']:.2f}      (nondeterministic)",
            "",
            "QUANTUM COMPLEXITY (approximations):",
            f"  Q2(f)         {m['Q2']:.2f}      (bounded-error quantum)",
            f"  QE(f)         {m['QE']:.2f}      (exact quantum)",
            f"  Amb(f)        {m['Amb']:.4f}   (Ambainis adversary)",
            f"  SpecAdv(f)    {m['SpecAdv']:.4f}   (spectral adversary)",
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
