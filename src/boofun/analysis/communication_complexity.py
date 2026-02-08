"""
Communication Complexity Analysis.

This module provides tools for analyzing the communication complexity
of Boolean functions, including:
- Deterministic communication complexity D(f)
- Randomized communication complexity R(f)
- Partition number and fooling sets
- Log-rank bounds
- Information complexity

References:
- Kushilevitz & Nisan, "Communication Complexity"
- O'Donnell, "Analysis of Boolean Functions", Chapter 6
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svdvals

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "deterministic_cc",
    "log_rank_bound",
    "fooling_set_bound",
    "rectangle_partition_bound",
    "discrepancy",
    "CommunicationMatrix",
    "CommunicationComplexityProfile",
]


def get_communication_matrix(f: "BooleanFunction") -> np.ndarray:
    """
    Build the communication matrix M_f for a Boolean function.

    For f: {0,1}^n → {0,1}, we split into Alice's variables x ∈ {0,1}^{n/2}
    and Bob's variables y ∈ {0,1}^{n/2}.

    M_f[x][y] = f(x, y)

    Args:
        f: Boolean function

    Returns:
        2^{n/2} × 2^{n/2} matrix
    """
    n = f.n_vars
    assert n is not None
    n_alice = n // 2
    n_bob = n - n_alice

    rows = 2**n_alice
    cols = 2**n_bob

    M = np.zeros((rows, cols), dtype=int)

    for x in range(rows):
        for y in range(cols):
            # Combine x and y into single input
            combined = (x << n_bob) | y
            M[x, y] = int(f.evaluate(combined))

    return M


def log_rank_bound(f: "BooleanFunction") -> Tuple[int, float]:
    """
    Compute the log-rank lower bound on communication complexity.

    D(f) ≥ log₂(rank(M_f))

    This is a fundamental lower bound: any deterministic protocol
    must exchange at least log₂(rank(M_f)) bits.

    Args:
        f: Boolean function

    Returns:
        Tuple of (rank, log_rank_lower_bound)
    """
    M = get_communication_matrix(f)

    # Compute rank
    rank = np.linalg.matrix_rank(M)

    # Lower bound is log2 of rank
    lower_bound = np.log2(max(1, rank))

    return rank, lower_bound


def fooling_set_bound(f: "BooleanFunction") -> Tuple[int, float]:
    """
    Compute fooling set lower bound on communication complexity.

    A fooling set is a set of input pairs {(x₁,y₁), ..., (x_k,y_k)} such that:
    - All f(x_i, y_i) = b for some fixed b
    - For i ≠ j: f(x_i, y_j) ≠ b or f(x_j, y_i) ≠ b

    D(f) ≥ log₂|fooling set|

    Args:
        f: Boolean function

    Returns:
        Tuple of (fooling_set_size, lower_bound)
    """
    M = get_communication_matrix(f)
    rows, cols = M.shape

    # Find fooling set for 1-entries (can also do for 0-entries)
    best_size = 0

    for target in [0, 1]:
        # Find all entries with target value
        positions = [(i, j) for i in range(rows) for j in range(cols) if M[i, j] == target]

        if not positions:
            continue

        # Greedy construction of fooling set
        fooling_set: List[Tuple[int, int]] = []

        for x, y in positions:
            # Check if (x, y) can be added to fooling set
            can_add = True
            for x2, y2 in fooling_set:
                # Must have f(x,y2) ≠ target or f(x2,y) ≠ target
                if M[x, y2] == target and M[x2, y] == target:
                    can_add = False
                    break

            if can_add:
                fooling_set.append((x, y))

        best_size = max(best_size, len(fooling_set))

    lower_bound = np.log2(max(1, best_size))

    return best_size, lower_bound


def rectangle_partition_bound(
    f: "BooleanFunction", max_iterations: int = 1000
) -> Tuple[int, float]:
    """
    Estimate partition number (number of monochromatic rectangles needed).

    The partition number C(f) = min number of monochromatic rectangles
    needed to partition the input space.

    D(f) ≤ log₂(C(f)) + 1

    This provides an upper bound on communication complexity.

    Args:
        f: Boolean function
        max_iterations: Maximum iterations for greedy partition

    Returns:
        Tuple of (partition_size, upper_bound)
    """
    M = get_communication_matrix(f)
    rows, cols = M.shape

    # Greedy rectangle covering
    remaining = np.ones((rows, cols), dtype=bool)
    rectangles = 0

    while np.any(remaining) and rectangles < max_iterations:
        # Find largest monochromatic rectangle
        best_area = 0
        best_rect = None

        # Try random starting points
        for _ in range(min(100, rows * cols)):
            i = np.random.randint(rows)
            j = np.random.randint(cols)

            if not remaining[i, j]:
                continue

            target = M[i, j]

            # Grow rectangle greedily
            row_set = {i}
            col_set = {j}

            # Add rows
            for r in range(rows):
                if r in row_set:
                    continue
                # Check if row is compatible
                compatible = all((not remaining[r, c] or M[r, c] == target) for c in col_set)
                if compatible and any(remaining[r, c] for c in col_set):
                    row_set.add(r)

            # Add columns
            for c in range(cols):
                if c in col_set:
                    continue
                compatible = all((not remaining[r, c] or M[r, c] == target) for r in row_set)
                if compatible and any(remaining[r, c] for r in row_set):
                    col_set.add(c)

            area = sum(remaining[r, c] for r in row_set for c in col_set)
            if area > best_area:
                best_area = area
                best_rect = (row_set, col_set)

        if best_rect is None:
            break

        # Mark rectangle as covered
        row_set, col_set = best_rect
        for r in row_set:
            for c in col_set:
                remaining[r, c] = False

        rectangles += 1

    upper_bound = np.log2(rectangles) + 1 if rectangles > 0 else 0

    return rectangles, upper_bound


def discrepancy(f: "BooleanFunction") -> float:
    """
    Compute the discrepancy of a Boolean function.

    Discrepancy measures how "balanced" the function is over rectangles.
    Low discrepancy implies high randomized communication complexity.

    disc(f) = max over rectangles R of |Pr[f=1 on R] - 1/2|

    Args:
        f: Boolean function

    Returns:
        Discrepancy value
    """
    M = get_communication_matrix(f)
    rows, cols = M.shape

    # Convert to ±1 matrix
    M_pm: np.ndarray = 2 * M - 1

    # Discrepancy is related to spectral norm of M_pm divided by sqrt(N)
    # disc(f) ≥ ||M_pm||_2 / N

    N = rows * cols
    spectral_norm: float = float(np.max(svdvals(M_pm.astype(float))))

    # Approximate discrepancy from spectral norm
    disc = spectral_norm / N

    return disc


def deterministic_cc(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Analyze deterministic communication complexity.

    Computes multiple bounds:
    - Log-rank lower bound
    - Fooling set lower bound
    - Rectangle partition upper bound

    Args:
        f: Boolean function

    Returns:
        Dictionary with bounds and analysis
    """
    n = f.n_vars
    assert n is not None

    # Compute bounds
    rank, log_rank = log_rank_bound(f)
    fool_size, fool_bound = fooling_set_bound(f)
    partition, partition_bound = rectangle_partition_bound(f)

    return {
        "n_vars": n,
        "n_alice": n // 2,
        "n_bob": n - n // 2,
        "matrix_size": (2 ** (n // 2), 2 ** (n - n // 2)),
        "rank": rank,
        "log_rank_lower_bound": log_rank,
        "fooling_set_size": fool_size,
        "fooling_set_lower_bound": fool_bound,
        "partition_size": partition,
        "partition_upper_bound": partition_bound,
        "lower_bound": max(log_rank, fool_bound),
        "upper_bound": partition_bound,
    }


class CommunicationMatrix:
    """
    Represents and analyzes the communication matrix of a Boolean function.
    """

    def __init__(self, f: "BooleanFunction"):
        """
        Initialize with Boolean function.

        Args:
            f: Boolean function
        """
        self.function = f
        n_vars = f.n_vars
        assert n_vars is not None
        self.n: int = n_vars
        self.n_alice = self.n // 2
        self.n_bob = self.n - self.n_alice
        self._matrix: Optional[np.ndarray] = None
        self._rank: Optional[int] = None
        self._svd: Optional[np.ndarray] = None

    @property
    def matrix(self) -> np.ndarray:
        """Get the communication matrix (computed lazily)."""
        if self._matrix is None:
            self._matrix = get_communication_matrix(self.function)
        return self._matrix

    @property
    def rank(self) -> int:
        """Get matrix rank."""
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(self.matrix)
        return self._rank

    @property
    def singular_values(self) -> np.ndarray:
        """Get singular values."""
        if self._svd is None:
            self._svd = svdvals(self.matrix.astype(float))
        return self._svd

    def density(self) -> float:
        """Fraction of 1s in matrix."""
        return float(np.mean(self.matrix))

    def spectral_norm(self) -> float:
        """Get largest singular value."""
        return float(np.max(self.singular_values))

    def nuclear_norm(self) -> float:
        """Get sum of singular values (nuclear norm)."""
        return float(np.sum(self.singular_values))

    def visualize(self) -> str:
        """Get ASCII visualization of matrix."""
        M = self.matrix
        rows, cols = M.shape

        if rows > 32 or cols > 32:
            return f"Matrix too large to visualize ({rows}×{cols})"

        lines = []
        for i in range(rows):
            row_str = "".join("█" if M[i, j] else "·" for j in range(cols))
            lines.append(row_str)

        return "\n".join(lines)

    def summary(self) -> str:
        """Get summary of matrix properties."""
        return (
            f"Communication Matrix ({self.matrix.shape[0]}×{self.matrix.shape[1]})\n"
            f"  Rank: {self.rank}\n"
            f"  Density: {self.density():.2%}\n"
            f"  Spectral norm: {self.spectral_norm():.3f}\n"
            f"  Nuclear norm: {self.nuclear_norm():.3f}\n"
            f"  Log-rank bound: {np.log2(max(1, self.rank)):.2f}"
        )


class CommunicationComplexityProfile:
    """
    Complete communication complexity analysis.
    """

    def __init__(self, f: "BooleanFunction"):
        """
        Initialize profile.

        Args:
            f: Boolean function
        """
        self.function = f
        self.matrix = CommunicationMatrix(f)
        self._analysis: Optional[Dict[str, Any]] = None

    def compute(self) -> Dict[str, Any]:
        """Compute full analysis."""
        if self._analysis is None:
            self._analysis = deterministic_cc(self.function)
            self._analysis["discrepancy"] = discrepancy(self.function)
            self._analysis["density"] = self.matrix.density()
            self._analysis["spectral_norm"] = self.matrix.spectral_norm()
        return self._analysis

    def summary(self) -> str:
        """Get summary string."""
        a = self.compute()
        return (
            f"Communication Complexity Profile (n={a['n_vars']})\n"
            f"{'='*50}\n"
            f"Alice: {a['n_alice']} vars, Bob: {a['n_bob']} vars\n"
            f"\nLower Bounds:\n"
            f"  Log-rank: {a['log_rank_lower_bound']:.2f} (rank={a['rank']})\n"
            f"  Fooling set: {a['fooling_set_lower_bound']:.2f} (size={a['fooling_set_size']})\n"
            f"  Best lower: {a['lower_bound']:.2f}\n"
            f"\nUpper Bounds:\n"
            f"  Partition: {a['partition_upper_bound']:.2f} (rectangles={a['partition_size']})\n"
            f"\nOther Properties:\n"
            f"  Discrepancy: {a['discrepancy']:.4f}\n"
            f"  Density: {a['density']:.2%}\n"
        )


# Standard function communication complexity
def cc_inner_product(n: int) -> Dict[str, Any]:
    """
    Communication complexity of inner product function.

    IP_n(x,y) = Σ x_i·y_i mod 2

    Known: D(IP) = n, R(IP) = Ω(n)

    Args:
        n: Number of variables (per party)

    Returns:
        Analysis dictionary
    """
    return {
        "function": "Inner Product",
        "n_per_party": n,
        "deterministic_cc": n,
        "randomized_cc": f"Ω({n})",
        "log_rank": n,
        "is_tight": True,
        "notes": "Optimal example for log-rank bound",
    }


def cc_equality(n: int) -> Dict[str, Any]:
    """
    Communication complexity of equality function.

    EQ_n(x,y) = 1 iff x = y

    Known: D(EQ) = n+1, R(EQ) = Θ(1) with shared randomness

    Args:
        n: Number of bits per party

    Returns:
        Analysis dictionary
    """
    return {
        "function": "Equality",
        "n_per_party": n,
        "deterministic_cc": n + 1,
        "randomized_cc": "O(1) with shared randomness",
        "log_rank": 1,
        "notes": "Huge gap between deterministic and randomized",
    }


def cc_disjointness(n: int) -> Dict[str, Any]:
    """
    Communication complexity of disjointness function.

    DISJ_n(x,y) = 1 iff x ∩ y = ∅ (as sets)

    Known: D(DISJ) = n+1, R(DISJ) = Ω(n)

    Args:
        n: Set size

    Returns:
        Analysis dictionary
    """
    return {
        "function": "Disjointness",
        "n_per_party": n,
        "deterministic_cc": n + 1,
        "randomized_cc": f"Ω({n})",
        "notes": "Hard function for randomized protocols",
    }
