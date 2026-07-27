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

from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Utility
    "QueryComplexityProfile",
    # Lower bounds
    "ambainis_complexity",
    # Degree measures
    "approximate_degree",
    "average_deterministic_complexity",
    "average_everywhere_sensitivity",
    "block_sensitivity_lower_bound",
    "bounded_error_randomized_complexity",
    "certificate_lower_bound",
    # Core complexity measures
    "deterministic_query_complexity",
    # Sensitivity variants
    "everywhere_sensitivity",
    "exact_quantum_complexity",
    "general_adversary_bound",
    "nondeterministic_complexity",
    "nondeterministic_degree",
    "one_sided_approximate_degree",
    "one_sided_randomized_complexity",
    "polynomial_method_bound",
    # Quantum complexity
    "quantum_query_complexity",
    "sensitivity_lower_bound",
    "spectral_adversary_bound",
    "strong_nondeterministic_degree",
    "threshold_degree",
    "weak_nondeterministic_degree",
    "zero_error_randomized_complexity",
]


def deterministic_query_complexity(f: BooleanFunction) -> int:
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


def average_deterministic_complexity(f: BooleanFunction) -> float:
    """
    Compute D_avg(f), the average-case deterministic query complexity.

    This is the expected number of queries under the uniform distribution
    on inputs, using an optimal decision tree.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Average number of queries needed
    """

    n = f.n_vars
    if n is None or n == 0:
        return 0.0

    # Get truth table
    np.asarray(f.get_representation("truth_table"), dtype=float)

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


def zero_error_randomized_complexity(f: BooleanFunction) -> float:
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
    from .complexity import decision_tree_depth, max_certificate_complexity

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


def bounded_error_randomized_complexity(f: BooleanFunction, error: float = 1 / 3) -> float:  # noqa: ARG001
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


def one_sided_randomized_complexity(f: BooleanFunction, side: int = 1) -> float:
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
    from .complexity import decision_tree_depth, max_certificate_complexity

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


def nondeterministic_complexity(f: BooleanFunction, side: int = 1) -> float:
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


def everywhere_sensitivity(f: BooleanFunction) -> int:
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
    from .complexity import min_sensitivity

    return min_sensitivity(f)


def average_everywhere_sensitivity(f: BooleanFunction, value: int | None = None) -> float:
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


def quantum_query_complexity(f: BooleanFunction) -> float:
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


def exact_quantum_complexity(f: BooleanFunction) -> float:
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
    from ..analysis.gf2 import gf2_degree
    from .complexity import decision_tree_depth

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


def _sensitive_edge_degrees(truth_table: np.ndarray, n: int) -> np.ndarray:
    """
    For each input x, count the sensitive directions: the number of i
    with f(x) != f(x ^ e_i). Vectorized over the whole cube.
    """
    degrees = np.zeros(1 << n, dtype=np.int64)
    indices = np.arange(1 << n)
    for i in range(n):
        flipped = truth_table[indices ^ (1 << i)] != truth_table
        degrees += flipped
    return degrees


def spectral_adversary_bound(f: BooleanFunction) -> float:
    """
    Compute a certified spectral adversary lower bound on ADV(f).

    Uses the spectral formulation of the positive-weight adversary method
    (Barnum-Saks-Szegedy): for any nonnegative symmetric matrix Gamma
    supported on pairs with f(x) != f(y),

        ADV(f) >= ||Gamma|| / max_i ||Gamma o D_i||

    where D_i[x, y] = 1 iff x_i != y_i. This implementation evaluates that
    ratio exactly for one canonical witness: Gamma = the adjacency matrix
    of the bipartite sensitivity graph (pairs at Hamming distance 1 with
    different f-values). For that Gamma each Gamma o D_i is a partial
    matching, so ||Gamma o D_i|| = 1 and the bound is exactly the largest
    singular value of the bipartite sensitivity matrix.

    Status: certified lower bound on ADV(f) (and hence on ADV+-(f)); not
    the optimal spectral adversary, which requires an SDP. Deterministic.
    Since Q2(f) = Theta(ADV+-(f)) with constants below 1, this value is
    NOT claimed to be numerically <= Q2(f); see the module docstring.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Exact value of the canonical spectral-adversary witness

    Raises:
        ValueError: if n exceeds the dense-matrix size cap (n <= 12).

    References:
        - Barnum, Saks, Szegedy, "Quantum query complexity and semi-definite
          programming" (2003)
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0
    if n > _LP_MAX_VARS:
        raise ValueError(
            f"spectral_adversary_bound builds a dense 2^(n-1) x 2^(n-1) matrix and "
            f"supports n <= {_LP_MAX_VARS}; got n = {n}"
        )

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    zeros = np.flatnonzero(~truth_table)
    ones = np.flatnonzero(truth_table)
    if len(zeros) == 0 or len(ones) == 0:
        return 0.0  # Constant function: no adversary pairs

    # Bipartite sensitivity matrix: M[i, j] = 1 iff zeros[i] and ones[j]
    # differ in exactly one bit.
    xor = zeros[:, None] ^ ones[None, :]
    hamming_one = (xor & (xor - 1)) == 0  # xor is a power of two
    M = hamming_one.astype(float)

    if not M.any():
        # No sensitive edges is impossible for non-constant f, but guard anyway.
        return 0.0

    # Largest singular value of M (||Gamma o D_i|| = 1 for every direction i
    # that contains a sensitive edge, since each restriction is a matching).
    if min(M.shape) <= 2:
        return float(np.linalg.norm(M, 2))
    from scipy.sparse.linalg import svds

    # Fixed starting vector keeps the Lanczos iteration bit-for-bit
    # deterministic across calls.
    v0 = np.ones(min(M.shape))
    return float(svds(M, k=1, v0=v0, return_singular_vectors=False)[0])


def ambainis_complexity(f: BooleanFunction) -> float:
    """
    Compute a certified Ambainis adversary lower bound on ADV(f).

    Ambainis's theorem: choose X subset f^-1(0), Y subset f^-1(1) and a
    relation R subset X x Y such that every x in X is related to at least
    m elements of Y and every y in Y to at least m' elements of X. With
    l = max_{x,i} |{y : (x,y) in R, x_i != y_i}| and l' defined dually,

        ADV(f) >= sqrt(m * m' / (l * l'))

    This implementation evaluates that bound exactly for the canonical
    sensitive-edge relation: R = pairs at Hamming distance 1 with
    different f-values, with X and Y restricted to inputs that have at
    least one sensitive neighbor. For that relation l = l' = 1, so the
    bound is sqrt(m * m') with m, m' the minimum sensitive-edge counts on
    each side. Examples: AND_n gives sqrt(n); PARITY_n gives n.

    Status: certified lower bound on ADV(f) (and hence on ADV+-(f)) via an
    explicit feasible relation; not the optimum over all relations, which
    is hard in general. Deterministic (the previous implementation sampled
    pairs with an unseeded RNG). Since Q2(f) = Theta(ADV+-(f)) with
    constants below 1, this value is NOT claimed to be numerically
    <= Q2(f); see the module docstring.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Exact value of the sensitive-edge Ambainis bound

    References:
        - Ambainis, "Quantum lower bounds by quantum arguments" (2002)
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    degrees = _sensitive_edge_degrees(truth_table, n)

    zero_degrees = degrees[~truth_table]
    one_degrees = degrees[truth_table]
    if len(zero_degrees) == 0 or len(one_degrees) == 0:
        return 0.0  # Constant function

    m = int(zero_degrees[zero_degrees > 0].min())
    m_prime = int(one_degrees[one_degrees > 0].min())
    return sqrt(m * m_prime)


def certificate_lower_bound(f: BooleanFunction) -> int:
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


def sensitivity_lower_bound(f: BooleanFunction) -> int:
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


def block_sensitivity_lower_bound(f: BooleanFunction) -> int:
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


# Exact LP-based degree computations enumerate all 2^n inputs and up to
# 2^n monomials; above this cap the LPs stop being interactive.
_LP_MAX_VARS = 12


def _character_matrix(n: int, degree: int) -> np.ndarray:
    """
    Build the matrix A with A[x, j] = chi_{S_j}(x) = (-1)^{|x & S_j|}.

    Columns enumerate all subsets S of [n] with |S| <= degree, so a real
    polynomial of degree <= degree is exactly p(x) = A @ c for some
    coefficient vector c (Fourier/character basis over {0,1}^n inputs).
    """
    from itertools import combinations

    size = 1 << n
    xs = np.arange(size)
    columns = []
    for d in range(degree + 1):
        for subset in combinations(range(n), d):
            parity = np.zeros(size, dtype=np.int64)
            for i in subset:
                parity ^= (xs >> i) & 1
            columns.append(1.0 - 2.0 * parity)
    return np.column_stack(columns)


def _min_linf_error(f_values: np.ndarray, n: int, degree: int) -> float:
    """
    Exact minimum over degree-``degree`` real polynomials of
    max_x |p(x) - f(x)|, via linear programming (variables: Fourier
    coefficients c and the error t; minimize t).
    """
    from scipy.optimize import linprog

    A = _character_matrix(n, degree)
    size, k = A.shape
    # Variables: [c_1..c_k, t]; minimize t subject to |A c - f| <= t.
    objective = np.zeros(k + 1)
    objective[-1] = 1.0
    ones = np.ones((size, 1))
    A_ub = np.block([[A, -ones], [-A, -ones]])
    b_ub = np.concatenate([f_values, -f_values])
    bounds = [(None, None)] * k + [(0.0, None)]
    result = linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"approximate-degree LP failed at degree {degree}: {result.message}")
    return float(result.fun)


def approximate_degree(f: BooleanFunction, epsilon: float = 1 / 3) -> int:
    """
    Compute deg_epsilon(f), the approximate degree (exact, via LP).

    The approximate degree is the minimum degree of a real polynomial p
    such that |p(x) - f(x)| <= epsilon for all x in {0,1}^n, with f
    valued in {0, 1}. Computed exactly by solving one Chebyshev-style
    linear program per candidate degree (scipy ``linprog``/HiGHS).

    Status: exact (for n <= 12; raises ValueError above).

    Args:
        f: BooleanFunction to analyze
        epsilon: Approximation parameter (default 1/3)

    Returns:
        The exact approximate degree (an integer)

    Raises:
        ValueError: if n exceeds the LP size cap.
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0
    if n > _LP_MAX_VARS:
        raise ValueError(
            f"approximate_degree is computed exactly by LP and supports n <= {_LP_MAX_VARS}; "
            f"got n = {n}"
        )

    f_values = np.asarray(f.get_representation("truth_table"), dtype=float)
    # LP optima are exact up to solver tolerance; the feasibility margin
    # only needs to absorb that tolerance.
    solver_slack = 1e-9
    for degree in range(n + 1):
        if _min_linf_error(f_values, n, degree) <= epsilon + solver_slack:
            return degree
    return n  # degree n always represents f exactly (error 0)


def one_sided_approximate_degree(
    f: BooleanFunction, side: int = 1, epsilon: float = 1 / 3
) -> float:
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


def nondeterministic_degree(f: BooleanFunction, side: int = 1) -> float:
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


def strong_nondeterministic_degree(f: BooleanFunction) -> float:
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


def weak_nondeterministic_degree(f: BooleanFunction) -> float:
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


def _sign_representable(sign_values: np.ndarray, n: int, degree: int) -> bool:
    """
    Exact LP feasibility: does a degree-``degree`` real polynomial p exist
    with sign(p(x)) = sign_values[x] for all x? Strict sign conditions are
    scale-invariant, so feasibility of s(x) * p(x) >= 1 is equivalent.
    """
    from scipy.optimize import linprog

    A = _character_matrix(n, degree)
    _, k = A.shape
    # Constraint: -s(x) * p(x) <= -1 for every x.
    A_ub = -sign_values[:, None] * A
    b_ub = -np.ones(A.shape[0])
    result = linprog(np.zeros(k), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * k, method="highs")
    return bool(result.success)


def threshold_degree(f: BooleanFunction) -> int:
    """
    Compute the threshold degree of f (exact, via LP).

    The threshold degree is the minimum degree d of a real polynomial p
    with sign(p(x)) = (-1)^{f(x)} for all x (equivalently, p sign-
    represents f). For example, every linear threshold function (AND, OR,
    majority) has threshold degree 1, while parity on n variables has
    threshold degree n.

    Status: exact (for n <= 12; raises ValueError above).

    Args:
        f: BooleanFunction to analyze

    Returns:
        The exact threshold degree (an integer)

    Raises:
        ValueError: if n exceeds the LP size cap.
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0
    if n > _LP_MAX_VARS:
        raise ValueError(
            f"threshold_degree is computed exactly by LP and supports n <= {_LP_MAX_VARS}; "
            f"got n = {n}"
        )

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    # O'Donnell sign convention: f(x) = 1 maps to -1, f(x) = 0 maps to +1.
    sign_values = 1.0 - 2.0 * truth_table
    for degree in range(n + 1):
        if _sign_representable(sign_values, n, degree):
            return degree
    return n  # the exact multilinear representation sign-represents f


def polynomial_method_bound(f: BooleanFunction) -> float:
    """
    Compute a lower bound on Q2(f) via the polynomial method.

    A quantum algorithm making T queries induces acceptance-probability
    polynomials of degree at most 2T, so Q2(f) >= deg_{1/3}(f) / 2
    (Beals et al. 2001). Since ``approximate_degree`` is computed exactly
    by LP, this is a certified lower bound, not an estimate.

    Status: certified lower bound on Q2(f) (for n <= 12).

    Args:
        f: BooleanFunction to analyze

    Returns:
        Polynomial-method lower bound for bounded-error quantum query
        complexity

    References:
        - Beals et al., "Quantum lower bounds by polynomials" (2001)
        - Belovs, "A Direct Reduction from Polynomial to Adversary Method" (TQC 2024)
    """
    return approximate_degree(f) / 2


def general_adversary_bound(f: BooleanFunction) -> float:
    """
    Compute a certified lower bound on the general adversary bound ADV+-(f).

    The general (negative-weight) adversary bound characterizes
    bounded-error quantum query complexity for total Boolean functions:
    Q2(f) = Theta(ADV+-(f)) (Hoyer-Lee-Spalek 2007; Reichardt 2011).
    Computing ADV+-(f) exactly requires a semidefinite program, which this
    library deliberately does not ship; for exact values see the pinned
    quantum-query-optimizer fixtures in tests/cross_validation/.

    This function returns the best certified positive-weight witness we
    evaluate exactly: max(spectral_adversary_bound, ambainis_complexity).
    Both are feasible-solution values for ADV(f) <= ADV+-(f), so the
    result is a true lower bound on ADV+-(f), never an overestimate.

    Status: certified lower bound on ADV+-(f). Deterministic.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Certified lower bound on ADV+-(f)

    Raises:
        ValueError: if n exceeds the dense-matrix size cap (n <= 12).

    References:
        - Hoyer, Lee, Spalek, "Negative weights make adversaries stronger" (2007)
        - Reichardt, "Reflections for quantum query algorithms" (2011)
    """
    return max(spectral_adversary_bound(f), ambainis_complexity(f))


class QueryComplexityProfile:
    """
    Compute and store query complexity measures for a Boolean function.

    This class provides a comprehensive analysis similar to Aaronson's
    Boolean Function Wizard.
    """

    def __init__(self, f: BooleanFunction) -> None:
        """
        Initialize query complexity profile.

        Args:
            f: BooleanFunction to analyze
        """
        self.function = f
        self.n_vars = f.n_vars
        self._computed = False
        self._measures: dict[str, float] = {}

    def compute(self) -> dict[str, float]:
        """
        Compute all query complexity measures.

        Returns:
            Dictionary of complexity measures
        """
        if self._computed:
            return self._measures

        f = self.function

        from ..analysis.fourier import fourier_degree
        from ..analysis.gf2 import gf2_degree
        from .block_sensitivity import max_block_sensitivity
        from .complexity import (
            average_sensitivity,
            decision_tree_depth,
            max_certificate_complexity,
            max_sensitivity,
        )

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
            max_certificate_complexity(f, 0), max_certificate_complexity(f, 1)
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
            "  unate         (see basic_properties)",
            "  balanced      (Pr[f=1] = 0.5?)",
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

    def check_known_relations(self) -> dict[str, bool]:
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
