"""
Algebraic Normal Form (ANF) representation for Boolean functions.

ANF represents Boolean functions as multivariate polynomials over GF(2):
f(x) = ⊕ c_S * ∏_{i∈S} x_i

where c_S ∈ {0,1} are coefficients and ⊕ denotes XOR.
This representation enables sparse storage when few high-order monomials appear.

Mathematical Background:
    Every Boolean function can be uniquely represented as a polynomial over GF(2).
    The ANF is obtained by applying the Möbius transform to the truth table.

    Key properties:
    - Degree of ANF equals the algebraic degree of the function
    - Linear functions have degree ≤ 1
    - Quadratic functions have degree ≤ 2
    - XOR function: x₀ ⊕ x₁ has ANF {∅: 0, {0}: 1, {1}: 1, {0,1}: 0}

Performance Characteristics:
    - Space: O(2^n) worst case, often much smaller for sparse functions
    - Evaluation: O(number of monomials)
    - Conversion from truth table: O(n * 2^n) via Möbius transform

Examples:
    >>> # XOR function: f(x0, x1) = x0 ⊕ x1
    >>> anf_data = {frozenset(): 0, frozenset({0}): 1, frozenset({1}): 1, frozenset({0,1}): 0}
    >>>
    >>> # Majority function has higher degree terms
    >>> # f(x0, x1, x2) = x0*x1 + x0*x2 + x1*x2 + x0*x1*x2
    >>> maj_anf = {frozenset({0,1}): 1, frozenset({0,2}): 1, frozenset({1,2}): 1, frozenset({0,1,2}): 1}
    >>>
    >>> # Constant functions
    >>> zero_anf = {frozenset(): 0}  # Always 0
    >>> one_anf = {frozenset(): 1}   # Always 1
"""

from typing import Any, Dict, FrozenSet, List, Optional, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


@register_strategy("anf")
class ANFRepresentation(BooleanFunctionRepresentation[Dict[FrozenSet[int], int]]):
    """
    Algebraic Normal Form representation of Boolean functions.

    Stores Boolean functions as sparse polynomials over GF(2) where:
    - Keys are frozensets representing variable subsets (monomials)
    - Values are coefficients in {0, 1}

    Example: f(x0, x1) = x0 ⊕ x1 ⊕ x0*x1
    Representation: {frozenset(): 0, frozenset({0}): 1, frozenset({1}): 1, frozenset({0,1}): 1}
    """

    def evaluate(
        self, inputs: np.ndarray, data: Dict[FrozenSet[int], int], space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate ANF polynomial at given inputs.

        Args:
            inputs: Input values - can be integer indices or binary vectors
            data: ANF coefficients as {monomial: coefficient} dict
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        # Handle different input formats
        if inputs.ndim == 0:
            # Single integer index
            return self._evaluate_single_index(int(inputs), data, n_vars, space)
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                return self._evaluate_single_vector(inputs, data, space)
            else:
                # Batch of integer indices
                return np.array(
                    [self._evaluate_single_index(int(x), data, n_vars, space) for x in inputs],
                    dtype=bool,
                )
        elif inputs.ndim == 2:
            # Batch of binary vectors
            return np.array(
                [self._evaluate_single_vector(x, data, space) for x in inputs], dtype=bool
            )
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _evaluate_single_index(
        self, x: int, data: Dict[FrozenSet[int], int], n_vars: int, space: Space
    ) -> bool:
        """Evaluate ANF at single integer input."""
        # Convert integer to binary vector
        binary_vec = np.array([(x >> i) & 1 for i in range(n_vars)], dtype=int)
        return self._evaluate_single_vector(binary_vec, data, space)

    def _evaluate_single_vector(
        self, x: np.ndarray, data: Dict[FrozenSet[int], int], space: Space
    ) -> bool:
        """Evaluate ANF at single binary vector."""
        # Convert space if needed
        if space == Space.PLUS_MINUS_CUBE:
            x = Space.translate(x, Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE)

        result = 0
        for monomial, coeff in data.items():
            if coeff == 0:
                continue

            # Compute monomial value: ∏_{i∈monomial} x_i
            monomial_value = 1
            for var_idx in monomial:
                if var_idx < len(x):
                    monomial_value *= x[var_idx]
                else:
                    monomial_value = 0  # Variable not present
                    break

            # XOR with current result (addition in GF(2))
            result ^= coeff * monomial_value

        return bool(result)

    def dump(self, data: Dict[FrozenSet[int], int], space: Space, **kwargs) -> Dict[str, Any]:
        """Export ANF in serializable format."""
        # Convert frozensets to lists for JSON serialization
        serializable_data = {}
        for monomial, coeff in data.items():
            key = sorted(list(monomial)) if monomial else []
            serializable_data[str(key)] = coeff

        return {
            "type": "anf",
            "coefficients": serializable_data,
            "space": space.name,
            "degree": self._get_degree(data),
            "num_terms": len([c for c in data.values() if c != 0]),
        }

    def convert_from(
        self,
        source_repr: "BooleanFunctionRepresentation",
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> Dict[FrozenSet[int], int]:
        """Convert from another representation to ANF."""
        from .truth_table import TruthTableRepresentation

        # If source is truth table, use direct ANF computation
        if isinstance(source_repr, TruthTableRepresentation):
            return self._truth_table_to_anf(source_data, n_vars)

        # For other representations, convert via truth table first
        tt_repr = TruthTableRepresentation()
        truth_table = source_repr.convert_to(tt_repr, source_data, space, n_vars, **kwargs)
        return self._truth_table_to_anf(truth_table, n_vars)

    def convert_to(
        self,
        target_repr: "BooleanFunctionRepresentation",
        source_data: Dict[FrozenSet[int], int],
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> Any:
        """Convert from ANF to another representation."""
        # Use the target representation's convert_from method
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> Dict[FrozenSet[int], int]:
        """Create empty ANF (zero function)."""
        return {frozenset(): 0}  # Constant zero function

    def is_complete(self, data: Dict[FrozenSet[int], int]) -> bool:
        """Check if ANF representation is complete."""
        return len(data) > 0

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity estimates for ANF operations."""
        return {
            "evaluation": 2**n_vars,  # Worst case: all monomials present
            "creation_from_tt": n_vars * (2**n_vars),  # Möbius transform
            "storage": 2**n_vars,  # Maximum number of monomials
            "conversion": 2**n_vars,
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return memory requirements for ANF representation."""
        max_monomials = 2**n_vars
        # Each monomial: frozenset overhead + coefficient
        # Rough estimate: 64 bytes per monomial (frozenset + dict overhead)
        bytes_per_monomial = 64

        return {
            "max_monomials": max_monomials,
            "bytes_per_monomial": bytes_per_monomial,
            "max_bytes": max_monomials * bytes_per_monomial,
            "sparse_advantage": True,
            "human_readable": f"Up to {max_monomials * bytes_per_monomial / 1024:.1f} KB (sparse)",
        }

    def _truth_table_to_anf(
        self, truth_table: np.ndarray, n_vars: int
    ) -> Dict[FrozenSet[int], int]:
        """
        Convert truth table to ANF using Möbius transform.

        The Möbius transform computes ANF coefficients as:
        c_S = ⊕_{T⊇S} f(T)
        """
        size = len(truth_table)
        if size != 2**n_vars:
            raise ValueError(f"Truth table size {size} doesn't match 2^{n_vars}")

        # Copy truth table for in-place Möbius transform
        coeffs = np.array(truth_table, dtype=int)

        # Möbius transform (inverse of Walsh-Hadamard for Boolean functions)
        for i in range(n_vars):
            step = 1 << i  # 2^i
            for mask in range(size):
                if mask & step:
                    coeffs[mask] ^= coeffs[mask ^ step]

        # Convert to sparse representation
        anf_dict = {}
        for mask in range(size):
            if coeffs[mask] != 0:
                # Convert mask to variable set
                monomial = frozenset(i for i in range(n_vars) if mask & (1 << i))
                anf_dict[monomial] = int(coeffs[mask])

        return anf_dict

    def _get_degree(self, data: Dict[FrozenSet[int], int]) -> int:
        """Get the degree of the ANF polynomial."""
        if not data:
            return 0
        return max(len(monomial) for monomial, coeff in data.items() if coeff != 0)

    def get_monomials(self, data: Dict[FrozenSet[int], int]) -> List[FrozenSet[int]]:
        """Get all monomials with non-zero coefficients."""
        return [monomial for monomial, coeff in data.items() if coeff != 0]

    def get_degree_k_terms(
        self, data: Dict[FrozenSet[int], int], k: int
    ) -> Dict[FrozenSet[int], int]:
        """Get all terms of exactly degree k."""
        return {
            monomial: coeff for monomial, coeff in data.items() if len(monomial) == k and coeff != 0
        }

    def is_linear(self, data: Dict[FrozenSet[int], int]) -> bool:
        """Check if the function is linear (degree ≤ 1)."""
        return self._get_degree(data) <= 1

    def is_quadratic(self, data: Dict[FrozenSet[int], int]) -> bool:
        """Check if the function is quadratic (degree ≤ 2)."""
        return self._get_degree(data) <= 2

    def __str__(self) -> str:
        return "ANFRepresentation()"

    def __repr__(self) -> str:
        return "ANFRepresentation()"


def create_anf_from_monomials(monomials: List[List[int]], n_vars: int) -> Dict[FrozenSet[int], int]:
    """
    Create ANF representation from list of monomials.

    Args:
        monomials: List of variable lists, e.g., [[0], [1], [0,1]] for x0 ⊕ x1 ⊕ x0*x1
        n_vars: Number of variables

    Returns:
        ANF dictionary representation
    """
    anf_dict = {}
    for monomial_list in monomials:
        monomial = frozenset(monomial_list)
        anf_dict[monomial] = anf_dict.get(monomial, 0) ^ 1  # XOR for GF(2)

    return anf_dict


def anf_to_string(
    data: Dict[FrozenSet[int], int], variable_names: Optional[List[str]] = None
) -> str:
    """
    Convert ANF to human-readable string representation.

    Args:
        data: ANF coefficients
        variable_names: Optional variable names (default: x0, x1, ...)

    Returns:
        String representation like "x0 ⊕ x1 ⊕ x0*x1"
    """
    if not data:
        return "0"

    terms = []
    for monomial, coeff in sorted(data.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
        if coeff == 0:
            continue

        if not monomial:  # Constant term
            terms.append("1")
        else:
            if variable_names:
                var_strs = [variable_names[i] for i in sorted(monomial)]
            else:
                var_strs = [f"x{i}" for i in sorted(monomial)]
            terms.append("*".join(var_strs))

    if not terms:
        return "0"

    return " ⊕ ".join(terms)
