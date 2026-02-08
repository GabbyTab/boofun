"""
Huang's Sensitivity Theorem and related results.

Huang's Theorem (2019): For any Boolean function f: {0,1}^n → {0,1},
    s(f) ≥ √deg(f)

where s(f) is the sensitivity and deg(f) is the Fourier degree.

This resolved the long-standing Sensitivity Conjecture, proving that
sensitivity is polynomially related to all other complexity measures.

Key relationships established:
- bs(f) ≤ s(f)^2 (block sensitivity)
- D(f) ≤ s(f)^4 (deterministic query complexity)
- C(f) ≤ s(f)^2 (certificate complexity)
- deg(f) ≤ s(f)^2 (Fourier degree)

References:
- Huang, "Induced subgraphs of hypercubes and a proof of the
  Sensitivity Conjecture" (2019)
- O'Donnell, "Analysis of Boolean Functions" Chapter 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "sensitivity",
    "sensitivity_at",
    "max_sensitivity",
    "average_sensitivity",
    "block_sensitivity",
    "verify_huang_theorem",
    "sensitivity_vs_degree",
    "HuangAnalysis",
]


def sensitivity_at(f: "BooleanFunction", x: int) -> int:
    """
    Compute sensitivity of f at input x.

    s(f, x) = |{i : f(x) ≠ f(x ⊕ eᵢ)}|

    The number of bits that, if flipped, change the output.

    Args:
        f: Boolean function
        x: Input (as integer)

    Returns:
        Number of sensitive coordinates at x
    """
    n = f.n_vars
    assert n is not None
    f_x = f.evaluate(x)
    count = 0

    for i in range(n):
        neighbor = x ^ (1 << i)  # Flip bit i
        if f.evaluate(neighbor) != f_x:
            count += 1

    return count


def max_sensitivity(f: "BooleanFunction") -> int:
    """
    Compute maximum sensitivity: s(f) = max_x s(f, x).

    Also called just "sensitivity" of f.

    Args:
        f: Boolean function

    Returns:
        Maximum sensitivity over all inputs
    """
    n = f.n_vars
    assert n is not None
    return max(sensitivity_at(f, x) for x in range(2**n))


def sensitivity(f: "BooleanFunction") -> int:
    """Alias for max_sensitivity."""
    return max_sensitivity(f)


def average_sensitivity(f: "BooleanFunction") -> float:
    """
    Compute average sensitivity: E_x[s(f, x)].

    This equals the total influence I[f].

    Args:
        f: Boolean function

    Returns:
        Average sensitivity
    """
    n = f.n_vars
    assert n is not None
    total = sum(sensitivity_at(f, x) for x in range(2**n))
    return total / (2**n)


def block_sensitivity(f: "BooleanFunction") -> int:
    """
    Compute block sensitivity: bs(f) = max_x bs(f, x).

    bs(f, x) is the maximum number of disjoint blocks B₁, ..., Bₖ
    such that flipping all bits in any Bᵢ changes f(x).

    Huang showed: bs(f) ≤ s(f)²

    This delegates to the exact algorithm in block_sensitivity module,
    which uses minimal sensitive blocks + backtracking to find the
    optimal disjoint packing.

    Args:
        f: Boolean function

    Returns:
        Block sensitivity
    """
    from .block_sensitivity import max_block_sensitivity

    return max_block_sensitivity(f)


def verify_huang_theorem(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Verify Huang's Sensitivity Theorem for function f.

    Checks: s(f) ≥ √deg(f)

    Returns:
        Dict with sensitivity, degree, and verification result
    """
    s = sensitivity(f)
    deg = f.degree()

    # Huang's bound
    huang_bound = np.sqrt(deg)
    satisfied = s >= huang_bound - 1e-10  # Allow small numerical error

    # Also check block sensitivity relationship
    bs = block_sensitivity(f)
    bs_bound_satisfied = bs <= s**2

    return {
        "sensitivity": s,
        "fourier_degree": deg,
        "block_sensitivity": bs,
        "huang_bound": huang_bound,
        "huang_satisfied": satisfied,
        "bs_bound_satisfied": bs_bound_satisfied,
        "gap": s - huang_bound,
        "tightness": s / huang_bound if huang_bound > 0 else float("inf"),
    }


def sensitivity_vs_degree(f: "BooleanFunction") -> Tuple[int, int, float]:
    """
    Compare sensitivity and degree for Huang analysis.

    Returns:
        (sensitivity, degree, ratio s/√deg)
    """
    s = sensitivity(f)
    deg = f.degree()
    ratio = s / np.sqrt(deg) if deg > 0 else float("inf")
    return (s, deg, ratio)


class HuangAnalysis:
    """
    Comprehensive sensitivity analysis with Huang's theorem.

    Computes various sensitivity-related measures and verifies
    the polynomial relationships established by Huang.
    """

    def __init__(self, f: "BooleanFunction"):
        """Initialize analyzer with a Boolean function."""
        self.f = f
        n_vars = f.n_vars
        assert n_vars is not None
        self.n: int = n_vars
        self._cache: Dict[str, Any] = {}

    def sensitivity(self) -> int:
        """Get maximum sensitivity."""
        if "sensitivity" not in self._cache:
            self._cache["sensitivity"] = max_sensitivity(self.f)
        return self._cache["sensitivity"]

    def block_sensitivity(self) -> int:
        """Get block sensitivity."""
        if "block_sensitivity" not in self._cache:
            self._cache["block_sensitivity"] = block_sensitivity(self.f)
        return self._cache["block_sensitivity"]

    def degree(self) -> int:
        """Get Fourier degree."""
        if "degree" not in self._cache:
            self._cache["degree"] = self.f.degree()
        return self._cache["degree"]

    def sensitivity_profile(self) -> np.ndarray:
        """Get sensitivity at each input."""
        return np.array([sensitivity_at(self.f, x) for x in range(2**self.n)])

    def verify_all_bounds(self) -> Dict[str, Any]:
        """
        Verify all Huang-related polynomial bounds.

        Checks:
        - s(f) ≥ √deg(f)  [Huang]
        - bs(f) ≤ s(f)²
        - C(f) ≤ s(f)²
        - D(f) ≤ s(f)⁴
        """
        s = self.sensitivity()
        bs = self.block_sensitivity()
        deg = self.degree()

        # Certificate complexity (max over all inputs)
        from .complexity import certificate_complexity

        cert = max(certificate_complexity(self.f, x)[0] for x in range(2**self.n))

        # Decision tree depth (deterministic query complexity)
        from .complexity import decision_tree_depth

        D = decision_tree_depth(self.f)

        results: Dict[str, Any] = {
            "sensitivity": s,
            "block_sensitivity": bs,
            "certificate_complexity": cert,
            "deterministic_complexity": D,
            "fourier_degree": deg,
            "bounds": {
                "huang": {
                    "inequality": "s(f) >= √deg(f)",
                    "lhs": s,
                    "rhs": np.sqrt(deg),
                    "satisfied": s >= np.sqrt(deg) - 1e-10,
                },
                "block_sensitivity": {
                    "inequality": "bs(f) <= s(f)²",
                    "lhs": bs,
                    "rhs": s**2,
                    "satisfied": bs <= s**2,
                },
                "certificate": {
                    "inequality": "C(f) <= s(f)²",
                    "lhs": cert,
                    "rhs": s**2,
                    "satisfied": cert <= s**2,
                },
                "deterministic": {
                    "inequality": "D(f) <= s(f)⁴",
                    "lhs": D,
                    "rhs": s**4,
                    "satisfied": D <= s**4,
                },
            },
            "all_satisfied": True,  # Will update below
        }

        # Check if all bounds are satisfied
        results["all_satisfied"] = all(b["satisfied"] for b in results["bounds"].values())

        return results

    def summary(self) -> str:
        """Get a text summary of the analysis."""
        verification = self.verify_all_bounds()

        lines = [
            f"=== Huang's Sensitivity Analysis ===",
            f"n = {self.n} variables",
            f"",
            f"Measures:",
            f"  s(f)  = {verification['sensitivity']} (sensitivity)",
            f"  bs(f) = {verification['block_sensitivity']} (block sensitivity)",
            f"  C(f)  = {verification['certificate_complexity']} (certificate complexity)",
            f"  D(f)  = {verification['deterministic_complexity']} (deterministic complexity)",
            f"  deg(f)= {verification['fourier_degree']} (Fourier degree)",
            f"",
            f"Polynomial Bounds:",
        ]

        for name, bound in verification["bounds"].items():
            status = "✓" if bound["satisfied"] else "✗"
            lines.append(f"  {status} {bound['inequality']}: {bound['lhs']} vs {bound['rhs']:.2f}")

        lines.append("")
        status = (
            "All bounds satisfied! ✓"
            if verification["all_satisfied"]
            else "Some bounds violated! ✗"
        )
        lines.append(status)

        return "\n".join(lines)
