"""
Query Model for Boolean Functions.

This module formalizes the distinction between:
1. EXPLICIT functions - we have the full truth table in memory
2. QUERY-ACCESS functions - we can only evaluate f(x) on demand

This is CRITICAL for production safety. A user should be able to:
    f = bf.create(massive_neural_network, n=1000)
    f.is_linear(num_tests=100)  # SAFE: only 300 queries

Without the library trying to compute 2^1000 entries.

Design Philosophy:
    "Never assume we can enumerate the domain"

    The library should work correctly even if the function represents:
    - A billion-parameter neural network
    - A database lookup
    - An external API call
    - A physical measurement device

    For such functions, only QUERY-BASED algorithms are valid.
"""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .base import BooleanFunction

__all__ = [
    "QueryModel",
    "AccessType",
    "get_access_type",
    "check_query_safety",
    "QuerySafetyWarning",
    "ExplicitEnumerationError",
    "QUERY_COMPLEXITY",
]


class AccessType(Enum):
    """How can we access the function's values?"""

    EXPLICIT = auto()  # Full truth table in memory - can enumerate
    QUERY = auto()  # Can only evaluate f(x) - cannot enumerate safely
    STREAMING = auto()  # Can iterate once but not random access
    SYMBOLIC = auto()  # Have formula but may be expensive to evaluate all


class QuerySafetyWarning(UserWarning):
    """Warning when an operation may be unsafe for query-access functions."""


class ExplicitEnumerationError(RuntimeError):
    """
    Raised when trying to enumerate a query-access function.

    This protects users from accidentally trying to compute 2^n
    evaluations on a huge function.
    """


# Query complexity for each operation
# Format: (base_queries, scaling)
# Total queries ≈ base_queries * scaling(n)
QUERY_COMPLEXITY: Dict[str, Dict[str, Any]] = {
    # SAFE operations - O(k) queries where k is user-specified
    "is_linear": {"safe": True, "queries": lambda n, k: 3 * k, "description": "BLR test"},
    "is_monotone": {"safe": True, "queries": lambda n, k: 2 * k, "description": "Edge test"},
    "is_symmetric": {
        "safe": True,
        "queries": lambda n, k: 2 * k,
        "description": "Permutation test",
    },
    "is_balanced_approx": {"safe": True, "queries": lambda n, k: k, "description": "Sample mean"},
    "evaluate": {"safe": True, "queries": lambda n, k: 1, "description": "Single query"},
    "estimate_fourier": {
        "safe": True,
        "queries": lambda n, k: k,
        "description": "Sample estimator",
    },
    "goldreich_levin": {"safe": True, "queries": lambda n, k: k, "description": "GL algorithm"},
    # UNSAFE operations - O(2^n) or worse
    "fourier": {"safe": False, "queries": lambda n, k: 2**n, "description": "Full WHT"},
    "influences": {
        "safe": False,
        "queries": lambda n, k: n * 2**n,
        "description": "All inputs, all flips",
    },
    "degree": {"safe": False, "queries": lambda n, k: 2**n, "description": "Uses fourier"},
    "total_influence": {
        "safe": False,
        "queries": lambda n, k: n * 2**n,
        "description": "Uses influences",
    },
    "W": {"safe": False, "queries": lambda n, k: 2**n, "description": "Uses fourier"},
    "W_leq": {"safe": False, "queries": lambda n, k: 2**n, "description": "Uses fourier"},
    "sparsity": {"safe": False, "queries": lambda n, k: 2**n, "description": "Uses fourier"},
    "is_balanced": {"safe": False, "queries": lambda n, k: 2**n, "description": "Count all"},
    "is_junta": {"safe": False, "queries": lambda n, k: n * 2**n, "description": "Uses influences"},
    "fix": {"safe": False, "queries": lambda n, k: 2 ** (n - 1), "description": "Builds new table"},
    "derivative": {"safe": False, "queries": lambda n, k: 2**n, "description": "Builds new table"},
    "constant_test": {
        "safe": False,
        "queries": lambda n, k: 2**n,
        "description": "Exhaustive check",
    },
    "decision_tree_depth": {
        "safe": False,
        "queries": lambda n, k: 3**n,
        "description": "DP over subcubes",
    },
    "get_representation:truth_table": {
        "safe": False,
        "queries": lambda n, k: 2**n,
        "description": "Materialize",
    },
}


def get_access_type(f: "BooleanFunction") -> AccessType:
    """
    Determine how the function's values can be accessed.

    Args:
        f: BooleanFunction to check

    Returns:
        AccessType indicating safest access pattern
    """
    if f is None or f.n_vars is None:
        return AccessType.QUERY  # Assume most restrictive

    # Check what representations we have
    reps = set(f.representations.keys())

    # If we have truth table, we're explicit
    if "truth_table" in reps:
        return AccessType.EXPLICIT

    # If we have symbolic formula, depends on complexity
    if "symbolic" in reps or "anf" in reps:
        return AccessType.SYMBOLIC

    # If we only have a function callable, it's query-access
    if "function" in reps:
        return AccessType.QUERY

    # If we have Fourier coefficients, we can reconstruct
    if "fourier_expansion" in reps:
        return AccessType.EXPLICIT

    return AccessType.QUERY


def check_query_safety(
    f: "BooleanFunction",
    operation: str,
    max_safe_n: int = 20,
    num_queries: int = 100,
    strict: bool = False,
) -> bool:
    """
    Check if an operation is safe to perform on this function.

    Args:
        f: BooleanFunction to check
        operation: Name of operation (e.g., "fourier", "is_linear")
        max_safe_n: Maximum n for which we allow unsafe operations
        num_queries: Number of queries for safe operations
        strict: If True, raise error instead of warning

    Returns:
        True if operation is safe to proceed

    Raises:
        ExplicitEnumerationError: If strict=True and operation is unsafe

    Example:
        >>> f = bf.create(huge_function, n=100)
        >>> check_query_safety(f, "fourier")  # Returns False, warns
        >>> check_query_safety(f, "is_linear")  # Returns True
    """
    n = f.n_vars or 0
    access_type = get_access_type(f)

    # Get operation info
    op_info = QUERY_COMPLEXITY.get(operation, {"safe": False, "queries": lambda n, k: 2**n})
    is_safe = op_info["safe"]
    query_count = op_info["queries"](n, num_queries)

    # If operation is safe (query-based), always allow
    if is_safe:
        return True

    # If we have explicit representation, allow up to max_safe_n
    if access_type == AccessType.EXPLICIT:
        if n <= max_safe_n:
            return True
        else:
            msg = (
                f"Operation '{operation}' requires {query_count:,} queries for n={n}. "
                f"This may be slow. Use query-based alternatives or reduce n."
            )
            if strict:
                raise ExplicitEnumerationError(msg)
            warnings.warn(msg, QuerySafetyWarning)
            return True  # Still allow but warn

    # Query-access function with unsafe operation
    if access_type == AccessType.QUERY:
        if n > max_safe_n:
            msg = (
                f"UNSAFE: Operation '{operation}' would require ~{query_count:,} queries "
                f"on a query-access function with n={n}. "
                f"This is likely impossible (2^{n} evaluations). "
                f"Use query-based alternatives like estimate_fourier() or is_linear()."
            )
            if strict:
                raise ExplicitEnumerationError(msg)
            warnings.warn(msg, QuerySafetyWarning)
            return False  # Don't allow
        else:
            # Small n, we can convert to truth table
            return True

    return True


class QueryModel:
    """
    Manages query complexity and safety for Boolean function operations.

    This class helps users understand and control the computational
    cost of operations on their functions.

    Example:
        >>> f = bf.create(my_function, n=30)
        >>> qm = QueryModel(f)
        >>> qm.can_compute("fourier")
        False
        >>> qm.can_compute("is_linear", num_queries=100)
        True
        >>> qm.estimate_cost("influences")
        {'queries': 32212254720, 'feasible': False}
    """

    def __init__(self, f: "BooleanFunction", max_queries: int = 10_000_000):
        """
        Initialize query model for a function.

        Args:
            f: BooleanFunction to analyze
            max_queries: Maximum acceptable query count
        """
        self.f = f
        self.n = f.n_vars or 0
        self.max_queries = max_queries
        self.access_type = get_access_type(f)

    def can_compute(self, operation: str, **kwargs) -> bool:
        """Check if operation is computationally feasible."""
        cost = self.estimate_cost(operation, **kwargs)
        return cost["feasible"]

    def estimate_cost(self, operation: str, num_queries: int = 100) -> Dict[str, Any]:
        """
        Estimate computational cost of an operation.

        Returns:
            Dict with keys:
                - queries: Estimated number of function evaluations
                - feasible: Whether this is computationally reasonable
                - time_estimate: Rough time estimate (assuming 1µs per query)
                - description: Human-readable description
        """
        op_info = QUERY_COMPLEXITY.get(
            operation, {"safe": False, "queries": lambda n, k: 2**self.n, "description": "Unknown"}
        )

        queries = op_info["queries"](self.n, num_queries)
        feasible = queries <= self.max_queries

        # Rough time estimate at 1µs per query
        time_us = queries
        if time_us < 1000:
            time_str = f"{time_us}µs"
        elif time_us < 1_000_000:
            time_str = f"{time_us/1000:.1f}ms"
        elif time_us < 1_000_000_000:
            time_str = f"{time_us/1_000_000:.1f}s"
        else:
            time_str = f"{time_us/1_000_000_000:.1f}ks (hours+)"

        return {
            "queries": queries,
            "feasible": feasible,
            "safe": op_info["safe"],
            "time_estimate": time_str,
            "description": op_info["description"],
            "access_type": self.access_type.name,
        }

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Get cost summary for all operations."""
        return {op: self.estimate_cost(op) for op in QUERY_COMPLEXITY}

    def print_summary(self):
        """Print human-readable cost summary."""
        print(f"Query Model Summary for n={self.n}")
        print(f"Access type: {self.access_type.name}")
        print(f"Max queries: {self.max_queries:,}")
        print("-" * 60)

        safe_ops = []
        unsafe_ops = []

        for op, info in QUERY_COMPLEXITY.items():
            cost = self.estimate_cost(op)
            if info["safe"]:
                safe_ops.append((op, cost))
            else:
                unsafe_ops.append((op, cost))

        print("\n✓ SAFE operations (query-based):")
        for op, cost in safe_ops:
            print(f"  {op}: ~{cost['queries']} queries ({cost['time_estimate']})")

        print("\n⚠ POTENTIALLY UNSAFE operations (may enumerate):")
        for op, cost in unsafe_ops:
            status = "✓" if cost["feasible"] else "✗"
            print(f"  {status} {op}: ~{cost['queries']:,} queries ({cost['time_estimate']})")


def safe_alternatives(operation: str) -> Optional[str]:
    """
    Suggest query-safe alternative for an unsafe operation.

    Args:
        operation: Unsafe operation name

    Returns:
        Name of safe alternative, or None if none exists
    """
    alternatives = {
        "fourier": "estimate_fourier (sample-based)",
        "influences": "estimate_influence (sample-based)",
        "is_balanced": "is_balanced_approx (sample-based)",
        "total_influence": "estimate_total_influence",
        "is_junta": "detect_influential_vars (sample-based)",
        "constant_test": "is_constant_approx (sample-based)",
    }
    return alternatives.get(operation)
