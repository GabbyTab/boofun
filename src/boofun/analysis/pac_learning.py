"""
PAC (Probably Approximately Correct) Learning for Boolean functions.

This module implements PAC learning algorithms for various classes of
Boolean functions as covered in O'Donnell Chapter 3 and Lecture 6.

Key algorithms:
- Low-degree learning: Learn functions with spectral concentration
- Junta learning: Learn k-juntas with query complexity O(2^k)
- LMN algorithm: Learn decision trees from uniform samples
- Sparse Fourier learning: Learn functions with few Fourier coefficients

PAC Learning Framework:
- Given: Sample access to f (can draw (x, f(x)) for random x)
- Goal: Output hypothesis h such that Pr[h(x) ≠ f(x)] ≤ ε
- With probability at least 1 - δ

References:
- Linial, Mansour, Nisan (1993): "Constant depth circuits, Fourier transform"
- O'Donnell Chapter 3: Learning
- Mossel, O'Donnell, Servedio (2004): "Learning juntas"
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "pac_learn_low_degree",
    "pac_learn_junta",
    "pac_learn_sparse_fourier",
    "pac_learn_decision_tree",
    "pac_learn_monotone",
    "lmn_algorithm",
    "PACLearner",
    "sample_function",
]


def sample_function(
    f: "BooleanFunction", num_samples: int, rng: Optional[np.random.Generator] = None
) -> List[Tuple[int, int]]:
    """
    Draw random labeled samples from a Boolean function.

    Args:
        f: Boolean function to sample
        num_samples: Number of samples to draw
        rng: Random number generator

    Returns:
        List of (input, output) pairs
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars
    samples = []

    for _ in range(num_samples):
        x = int(rng.integers(0, 2**n))
        y = int(f.evaluate(x))
        samples.append((x, y))

    return samples


def pac_learn_low_degree(
    f: "BooleanFunction",
    max_degree: int,
    epsilon: float = 0.1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    PAC learn a function assuming it has degree at most max_degree.

    Uses the "Low-Degree Algorithm" from O'Donnell Chapter 3:
    1. Estimate all Fourier coefficients of degree ≤ max_degree
    2. Threshold small coefficients
    3. Return the truncated polynomial

    Sample complexity: O(n^d / ε²) where d = max_degree

    Args:
        f: Target function (used for sampling)
        max_degree: Maximum degree to learn
        epsilon: Target error rate
        delta: Failure probability
        rng: Random number generator

    Returns:
        Dictionary mapping subset masks to estimated Fourier coefficients
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars

    # Number of samples needed
    from math import comb

    num_coeffs = sum(comb(n, k) for k in range(max_degree + 1))
    num_samples = int(4 * num_coeffs / epsilon**2 * np.log(2 * num_coeffs / delta))
    num_samples = max(num_samples, 100)

    # Draw samples
    samples = sample_function(f, num_samples, rng)

    # Estimate Fourier coefficients for low-degree monomials
    estimated_coeffs = {}

    for S in range(2**n):
        # Only consider low-degree
        if bin(S).count("1") > max_degree:
            continue

        # Estimate f̂(S) = E[f(x) χ_S(x)]
        total = 0.0
        for x, y in samples:
            # Convert to ±1
            f_x = 1 - 2 * y
            # χ_S(x) = (-1)^{|x ∩ S|}
            chi_S_x = 1 - 2 * (bin(x & S).count("1") % 2)
            total += f_x * chi_S_x

        estimate = total / num_samples

        # Threshold: keep only significant coefficients
        if abs(estimate) >= epsilon / (2 * np.sqrt(num_coeffs)):
            estimated_coeffs[S] = estimate

    return estimated_coeffs


def pac_learn_junta(
    f: "BooleanFunction",
    k: int,
    epsilon: float = 0.1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], Dict[int, float]]:
    """
    PAC learn a k-junta (function depending on at most k variables).

    Uses influence-based junta finding followed by exhaustive learning
    on the relevant variables.

    Args:
        f: Target function
        k: Maximum number of relevant variables
        epsilon: Target error rate
        delta: Failure probability
        rng: Random number generator

    Returns:
        Tuple of (relevant_variables, learned_function_on_those_vars)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars

    # Step 1: Find influential variables using sampling
    # Estimate influence of each variable
    num_samples = int(4 * n / epsilon**2 * np.log(2 * n / delta))

    influence_estimates = np.zeros(n)

    for _ in range(num_samples):
        x = int(rng.integers(0, 2**n))
        f_x = f.evaluate(x)

        for i in range(n):
            # Check if flipping bit i changes output
            x_flipped = x ^ (1 << i)
            if f.evaluate(x_flipped) != f_x:
                influence_estimates[i] += 1

    influence_estimates /= num_samples

    # Step 2: Select top k variables by influence
    relevant_vars = list(np.argsort(influence_estimates)[-k:])
    relevant_vars.sort()

    # Step 3: Learn the function restricted to these variables
    # Build truth table on the k variables
    learned_function = {}

    num_verify_samples = int(2**k * np.log(2**k / delta) / epsilon**2)
    num_verify_samples = max(num_verify_samples, 2 ** (k + 2))

    # Sample and majority vote for each setting of relevant variables
    vote_counts = defaultdict(lambda: [0, 0])

    for _ in range(num_verify_samples):
        x = int(rng.integers(0, 2**n))
        y = int(f.evaluate(x))

        # Extract relevant variable values
        rel_val = 0
        for j, var in enumerate(relevant_vars):
            if (x >> var) & 1:
                rel_val |= 1 << j

        vote_counts[rel_val][y] += 1

    # Majority vote
    for rel_val in range(2**k):
        counts = vote_counts[rel_val]
        learned_function[rel_val] = 1 if counts[1] > counts[0] else 0

    return relevant_vars, learned_function


def lmn_algorithm(
    f: "BooleanFunction",
    epsilon: float = 0.1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    Learn a function using the Linial-Mansour-Nisan algorithm.

    The LMN algorithm learns any function that is well-approximated
    by a polynomial of degree O(log(1/ε)).

    Particularly effective for:
    - Functions computed by small depth circuits (AC⁰)
    - Decision trees
    - DNFs with few terms

    Args:
        f: Target function
        epsilon: Target error rate
        delta: Failure probability
        rng: Random number generator

    Returns:
        Dictionary of learned Fourier coefficients
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars

    # LMN degree: O(log(1/ε)/log(log(1/ε)))
    # Simplified: use log(1/ε)
    lmn_degree = max(1, int(np.log(1 / epsilon) / np.log(2)))
    lmn_degree = min(lmn_degree, n)  # Can't exceed n

    # Use low-degree learning
    return pac_learn_low_degree(f, lmn_degree, epsilon, delta, rng)


def pac_learn_sparse_fourier(
    f: "BooleanFunction",
    sparsity: int,
    epsilon: float = 0.1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    Learn a function assuming it has at most 'sparsity' non-zero Fourier coefficients.

    Uses heavy coefficient detection followed by precise estimation.

    Args:
        f: Target function
        sparsity: Maximum number of non-zero coefficients
        epsilon: Target error rate
        delta: Failure probability
        rng: Random number generator

    Returns:
        Dictionary of learned Fourier coefficients
    """
    if rng is None:
        rng = np.random.default_rng()

    from .learning import goldreich_levin

    # Threshold: significant coefficients have magnitude at least ε/√s
    threshold = epsilon / np.sqrt(sparsity)

    # Find heavy coefficients using Goldreich-Levin
    heavy = goldreich_levin(f, threshold, 1 - delta / 2, rng)

    # Refine estimates
    num_samples = int(4 * sparsity / epsilon**2 * np.log(2 * sparsity / delta))
    samples = sample_function(f, num_samples, rng)

    refined = {}
    for S, _ in heavy:
        total = 0.0
        for x, y in samples:
            f_x = 1 - 2 * y
            chi_S_x = 1 - 2 * (bin(x & S).count("1") % 2)
            total += f_x * chi_S_x
        refined[S] = total / num_samples

    return refined


def pac_learn_decision_tree(
    f: "BooleanFunction",
    max_depth: int,
    epsilon: float = 0.1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    PAC learn a function computed by a depth-d decision tree.

    Decision trees of depth d have Fourier concentration on degrees 0 to d.
    Uses this fact to apply low-degree learning.

    Args:
        f: Target function
        max_depth: Maximum tree depth
        epsilon: Target error rate
        delta: Failure probability
        rng: Random number generator

    Returns:
        Dictionary of learned Fourier coefficients
    """
    return pac_learn_low_degree(f, max_depth, epsilon, delta, rng)


def pac_learn_monotone(
    f: "BooleanFunction",
    epsilon: float = 0.1,
    delta: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    PAC learn a monotone Boolean function.

    Monotone functions have special structure: all non-zero Fourier
    coefficients f̂(S) ≥ 0 for non-empty S.

    Args:
        f: Target function (assumed monotone)
        epsilon: Target error rate
        delta: Failure probability
        rng: Random number generator

    Returns:
        Dictionary of learned Fourier coefficients
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars

    # Monotone functions are O(√n)-concentrated
    # Use degree about √n
    degree = max(1, int(np.sqrt(n)))

    coeffs = pac_learn_low_degree(f, degree, epsilon, delta, rng)

    # Enforce monotonicity: non-negative coefficients for non-empty S
    for S in list(coeffs.keys()):
        if S != 0 and coeffs[S] < 0:
            coeffs[S] = 0

    return coeffs


class PACLearner:
    """
    PAC learning framework for Boolean functions.

    Provides a unified interface for various PAC learning algorithms
    with sample complexity tracking.
    """

    def __init__(
        self,
        f: "BooleanFunction",
        epsilon: float = 0.1,
        delta: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize PAC learner.

        Args:
            f: Target function
            epsilon: Target error rate
            delta: Failure probability
            rng: Random number generator
        """
        self.f = f
        self.n = f.n_vars
        self.epsilon = epsilon
        self.delta = delta
        self.rng = rng or np.random.default_rng()

        self.sample_count = 0
        self._samples_cache = []

    def sample(self, num_samples: int) -> List[Tuple[int, int]]:
        """Draw samples and track count."""
        samples = sample_function(self.f, num_samples, self.rng)
        self.sample_count += num_samples
        self._samples_cache.extend(samples)
        return samples

    def learn_low_degree(self, max_degree: int) -> Dict[int, float]:
        """Learn assuming low degree."""
        return pac_learn_low_degree(self.f, max_degree, self.epsilon, self.delta, self.rng)

    def learn_junta(self, k: int) -> Tuple[List[int], Dict[int, float]]:
        """Learn assuming k-junta."""
        return pac_learn_junta(self.f, k, self.epsilon, self.delta, self.rng)

    def learn_sparse(self, sparsity: int) -> Dict[int, float]:
        """Learn assuming sparse Fourier spectrum."""
        return pac_learn_sparse_fourier(self.f, sparsity, self.epsilon, self.delta, self.rng)

    def learn_decision_tree(self, max_depth: int) -> Dict[int, float]:
        """Learn assuming decision tree structure."""
        return pac_learn_decision_tree(self.f, max_depth, self.epsilon, self.delta, self.rng)

    def learn_monotone(self) -> Dict[int, float]:
        """Learn assuming monotone function."""
        return pac_learn_monotone(self.f, self.epsilon, self.delta, self.rng)

    def learn_adaptive(self) -> Dict[str, Any]:
        """
        Adaptively choose learning algorithm based on function properties.

        Returns:
            Dict with learned coefficients and algorithm used
        """
        # Check if likely a junta
        influences = self.f.influences()
        num_significant = sum(1 for inf in influences if inf > 0.01)

        if num_significant <= 5:
            # Likely a small junta
            vars_, func = self.learn_junta(num_significant + 1)
            return {
                "algorithm": "junta",
                "relevant_variables": vars_,
                "function": func,
            }

        # Check if monotone
        if self.f.is_monotone(100):
            coeffs = self.learn_monotone()
            return {
                "algorithm": "monotone",
                "coefficients": coeffs,
            }

        # Check degree
        deg = self.f.degree()
        if deg <= int(np.log(1 / self.epsilon) * 2):
            coeffs = self.learn_low_degree(deg)
            return {
                "algorithm": "low_degree",
                "degree": deg,
                "coefficients": coeffs,
            }

        # Default: LMN
        coeffs = lmn_algorithm(self.f, self.epsilon, self.delta, self.rng)
        return {
            "algorithm": "lmn",
            "coefficients": coeffs,
        }

    def evaluate_hypothesis(self, coefficients: Dict[int, float], x: int) -> int:
        """
        Evaluate learned hypothesis on input x.

        h(x) = sign(Σ f̂(S) χ_S(x))
        """
        total = 0.0
        for S, coeff in coefficients.items():
            chi_S_x = 1 - 2 * (bin(x & S).count("1") % 2)
            total += coeff * chi_S_x

        return 1 if total <= 0 else 0  # Convert from ±1 to {0,1}

    def test_accuracy(self, coefficients: Dict[int, float], num_tests: int = 1000) -> float:
        """Test accuracy of learned hypothesis."""
        correct = 0
        for _ in range(num_tests):
            x = int(self.rng.integers(0, 2**self.n))
            if self.evaluate_hypothesis(coefficients, x) == self.f.evaluate(x):
                correct += 1
        return correct / num_tests

    def summary(self) -> str:
        """Get summary of learning progress."""
        return f"PACLearner: ε={self.epsilon}, δ={self.delta}, {self.sample_count} samples drawn"
