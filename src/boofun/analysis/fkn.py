"""
FKN Theorem and related results about functions close to dictators.

The FKN (Friedgut-Kalai-Naor) Theorem states that Boolean functions with
small total influence are close to dictator functions or constants.

References:
- Friedgut, Kalai, Naor: "Boolean Functions whose Fourier Transform is
  Concentrated on the First Two Levels" (2002)
- O'Donnell: "Analysis of Boolean Functions" Chapter 2
"""

from typing import TYPE_CHECKING, Any, Dict, Tuple

from ..utils.exceptions import InvalidInputError

if TYPE_CHECKING:
    from ..core.base import BooleanFunction


def _require_n_vars(f: "BooleanFunction") -> int:
    """Return n_vars or raise if unknown."""
    n = f.n_vars
    if n is None:
        raise InvalidInputError("BooleanFunction must have a known n_vars")
    return n


def distance_to_dictator(f: "BooleanFunction", i: int) -> float:
    """
    Compute distance of f to the i-th dictator function.

    In the ±1 domain, the i-th dictator is chi_i(x) = (-1)^{x_i}.
    The distance (fraction of disagreeing inputs) is:

        Pr[f(x) ≠ x_i] = (1 - f̂({i})) / 2

    where f̂({i}) is the Fourier coefficient on the singleton {i}.

    Args:
        f: Boolean function
        i: Variable index

    Returns:
        Fraction of inputs where f differs from dictator on x_i
    """
    n = _require_n_vars(f)
    fourier = f.fourier()

    subset_i = 1 << i
    coeff_i = float(fourier[subset_i])

    # Analytical formula: Pr[f(x) ≠ x_i] = (1 - f̂({i})) / 2
    return (1.0 - coeff_i) / 2.0


def distance_to_negated_dictator(f: "BooleanFunction", i: int) -> float:
    """
    Compute distance of f to the negated i-th dictator: 1 - x_i.

    In the ±1 domain, the negated dictator is -chi_i(x) = -(-1)^{x_i},
    with Fourier coefficient f̂({i}) = -1 on the singleton {i}.
    The distance is:

        Pr[f(x) ≠ NOT(x_i)] = (1 + f̂({i})) / 2

    Args:
        f: Boolean function
        i: Variable index

    Returns:
        Fraction of inputs where f differs from NOT(x_i)
    """
    n = _require_n_vars(f)
    fourier = f.fourier()

    subset_i = 1 << i
    coeff_i = float(fourier[subset_i])

    # Analytical formula: Pr[f(x) ≠ NOT(x_i)] = (1 + f̂({i})) / 2
    return (1.0 + coeff_i) / 2.0


def closest_dictator(f: "BooleanFunction") -> Tuple[int, float, bool]:
    """
    Find the dictator (or negated dictator) closest to f.

    Returns:
        Tuple of (variable_index, distance, is_negated)

    Example:
        >>> f = bf.majority(3)
        >>> idx, dist, neg = closest_dictator(f)
        >>> print(f"Closest to {'NOT ' if neg else ''}x_{idx}, distance={dist}")
    """
    n = _require_n_vars(f)
    best_idx = 0
    best_dist = float("inf")
    best_negated = False

    for i in range(n):
        # Check dictator
        dist = distance_to_dictator(f, i)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_negated = False

        # Check negated dictator
        neg_dist = distance_to_negated_dictator(f, i)
        if neg_dist < best_dist:
            best_dist = neg_dist
            best_idx = i
            best_negated = True

    return best_idx, best_dist, best_negated


def fkn_theorem_bound(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Apply the FKN Theorem to bound distance to dictators.

    FKN Theorem: Let f: {-1,1}^n → {-1,1} with E[f] = 0 (balanced).
    Then:
        dist(f, dictators) ≤ O(I[f])

    More precisely, if W^{≤1}[f] = 1 - ε (most weight on degree ≤1),
    then f is O(ε)-close to a dictator or constant.

    Returns:
        Dict with:
        - 'total_influence': I[f]
        - 'degree_1_weight': W^{=1}[f] (weight on degree 1)
        - 'low_degree_weight': W^{≤1}[f]
        - 'fkn_bound': Upper bound on distance to dictator
        - 'closest_dictator': (var, distance, is_negated)
        - 'is_close_to_dictator': Boolean
    """
    n = _require_n_vars(f)
    fourier = f.fourier()

    # Compute spectral weights
    w0: float = float(fourier[0] ** 2)  # Constant term
    w1: float = float(sum(fourier[1 << i] ** 2 for i in range(n)))  # Degree 1
    total_weight = sum(c**2 for c in fourier)  # Should be 1 for ±1 valued

    # Total influence
    total_inf = f.total_influence()

    # FKN bound: distance ≤ C * (1 - W^{≤1}[f])
    # More refined: distance ≤ C * I[f] for low-influence functions
    low_degree_weight: float = w0 + w1
    high_degree_weight: float = 1 - low_degree_weight

    # FKN constant (approximate)
    fkn_bound: float = min(total_inf, 2 * high_degree_weight)

    # Find actual closest dictator
    idx, dist, neg = closest_dictator(f)

    return {
        "total_influence": total_inf,
        "degree_0_weight": w0,
        "degree_1_weight": w1,
        "low_degree_weight": low_degree_weight,
        "high_degree_weight": high_degree_weight,
        "fkn_bound": fkn_bound,
        "closest_dictator": (idx, dist, neg),
        "actual_distance": dist,
        "is_close_to_dictator": dist < 0.1,
    }


def is_close_to_dictator(f: "BooleanFunction", epsilon: float = 0.1) -> bool:
    """
    Check if f is ε-close to some dictator function.

    Args:
        f: Boolean function
        epsilon: Distance threshold

    Returns:
        True if f is within epsilon of some dictator
    """
    _, dist, _ = closest_dictator(f)
    return dist <= epsilon


def spectral_gap(f: "BooleanFunction") -> float:
    """
    Compute the spectral gap: 1 - max_i |f̂({i})|.

    Small spectral gap → close to dictator.
    Large spectral gap → far from all dictators.

    Returns:
        Spectral gap value in [0, 1]
    """
    n = _require_n_vars(f)
    fourier = f.fourier()

    max_degree1: float = float(max(abs(fourier[1 << i]) for i in range(n)))
    return 1 - max_degree1


def analyze_dictator_proximity(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Comprehensive analysis of how close f is to dictator functions.

    Returns:
        Dict with detailed proximity analysis including FKN bounds,
        spectral information, and recommendations.
    """
    result = fkn_theorem_bound(f)
    result["spectral_gap"] = spectral_gap(f)
    result["n_vars"] = f.n_vars
    result["is_balanced"] = f.is_balanced()

    # Interpretation
    if result["actual_distance"] < 0.01:
        result["interpretation"] = "Very close to dictator (essentially a dictator)"
    elif result["actual_distance"] < 0.1:
        result["interpretation"] = "Close to dictator (FKN regime)"
    elif result["actual_distance"] < 0.25:
        result["interpretation"] = "Somewhat close to dictator"
    else:
        result["interpretation"] = "Far from all dictators"

    return result
