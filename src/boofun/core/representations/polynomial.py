"""
Polynomial (ANF) representation for Boolean functions.

This module implements the Algebraic Normal Form (ANF) representation,
where Boolean functions are represented as multivariate polynomials over GF(2).
"""

from typing import Any, Dict, List, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


@register_strategy("polynomial")
class PolynomialRepresentation(BooleanFunctionRepresentation[Dict[frozenset, int]]):
    """
    Polynomial (ANF) representation using coefficient dictionaries.

    Represents Boolean functions as polynomials over GF(2):
    f(x₁,...,xₙ) = ⊕ᵢ aᵢ ∏ⱼ∈Sᵢ xⱼ

    Data format: Dict[frozenset, int] mapping subsets to coefficients
    """

    def evaluate(
        self, inputs: np.ndarray, data: Dict[frozenset, int], space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate the polynomial at given inputs.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: Polynomial coefficients as {subset: coefficient}
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)  # type: ignore[unreachable]

        if inputs.ndim == 0:
            # Single input
            return self._evaluate_single(int(inputs), data, n_vars)
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                index = self._binary_to_index(inputs)
                return self._evaluate_single(index, data, n_vars)
            else:
                # Array of indices
                return np.array([self._evaluate_single(int(x), data, n_vars) for x in inputs])
        elif inputs.ndim == 2:
            # Batch of binary vectors
            indices = [self._binary_to_index(row) for row in inputs]
            return np.array([self._evaluate_single(idx, data, n_vars) for idx in indices])
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _evaluate_single(self, x: int, coeffs: Dict[frozenset, int], n_vars: int) -> bool:
        """Evaluate polynomial at single input x (as integer)."""
        result = 0

        # Convert x to binary representation
        x_bits = [(x >> i) & 1 for i in range(n_vars)]

        # Sum over all monomials
        for subset, coeff in coeffs.items():
            if coeff % 2 == 0:  # Skip zero coefficients in GF(2)
                continue

            # Compute monomial value: ∏ⱼ∈subset xⱼ
            monomial_val = 1
            for var_idx in subset:
                if var_idx < n_vars:
                    monomial_val *= x_bits[var_idx]

            result ^= monomial_val  # XOR in GF(2)

        return bool(result)

    def _binary_to_index(self, binary_vector: np.ndarray) -> int:
        """Convert binary vector to integer index using LSB=x₀ convention."""
        # LSB-first: binary_vector[i] corresponds to x_i, so index = Σ x_i * 2^i
        return int(np.dot(binary_vector, 2 ** np.arange(len(binary_vector))))

    def dump(self, data: Dict[frozenset, int], space=None, **kwargs) -> Dict[str, Any]:
        """
        Export polynomial in serializable format.

        Returns:
            Dictionary with monomials and coefficients
        """
        monomials = []
        for subset, coeff in data.items():
            if coeff % 2 == 1:  # Only non-zero coefficients in GF(2)
                monomials.append({"variables": sorted(list(subset)), "coefficient": coeff % 2})

        return {"type": "polynomial", "monomials": monomials, "field": "GF(2)"}

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> Dict[frozenset, int]:
        """
        Convert from any representation to polynomial using truth table method.

        Uses the Möbius transform to compute ANF coefficients.
        """
        # First get truth table
        size = 1 << n_vars
        truth_table = np.zeros(size, dtype=int)

        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)  # type: ignore[arg-type]
            truth_table[i] = int(bool(val))

        # Apply Möbius transform to get ANF coefficients
        return self._truth_table_to_anf(truth_table, n_vars)

    def _truth_table_to_anf(self, truth_table: np.ndarray, n_vars: int) -> Dict[frozenset, int]:
        """
        Convert truth table to ANF using Möbius transform.

        The ANF coefficient for subset S is:
        a_S = ⊕_{T⊇S} f(T)
        """
        coeffs = {}
        size = len(truth_table)

        # Iterate over all possible subsets (monomials)
        for s in range(size):
            # Convert subset index to frozenset of variable indices
            subset = frozenset(i for i in range(n_vars) if (s >> i) & 1)

            # Compute ANF coefficient using Möbius transform
            coeff = 0
            for t in range(size):
                # Check if t ⊇ s (all bits in s are also in t)
                if (t & s) == s:
                    coeff ^= truth_table[t]

            if coeff != 0:
                coeffs[subset] = coeff

        return coeffs

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert from polynomial to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> Dict[frozenset, int]:
        """Create empty polynomial (constant 0)."""
        return {}

    def is_complete(self, data: Dict[frozenset, int]) -> bool:
        """Check if polynomial has any non-zero coefficients."""
        return any(coeff % 2 == 1 for coeff in data.values())

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for polynomial operations."""
        return {
            "evaluation": n_vars,  # O(2^n) worst case (all monomials)
            "construction": n_vars,  # O(2^n) Möbius transform
            "conversion_from": n_vars,  # O(2^n) via truth table
            "space_complexity": n_vars,  # O(2^n) worst case
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, Any]:
        """Return storage requirements for polynomial representation."""
        max_monomials = 2**n_vars
        return {
            "max_monomials": max_monomials,
            "bytes_per_monomial": 8 + 4 * n_vars,  # frozenset + coefficient
            "max_bytes": max_monomials * (8 + 4 * n_vars),
            "space_complexity": "O(2^n)",
        }

    def get_degree(self, data: Dict[frozenset, int]) -> int:
        """Get the degree of the polynomial (size of largest monomial)."""
        if not data:
            return 0
        return max(len(subset) for subset, coeff in data.items() if coeff % 2 == 1)

    def get_monomials(self, data: Dict[frozenset, int]) -> List[frozenset]:
        """Get all monomials with non-zero coefficients."""
        return [subset for subset, coeff in data.items() if coeff % 2 == 1]

    def add_polynomials(
        self, poly1: Dict[frozenset, int], poly2: Dict[frozenset, int]
    ) -> Dict[frozenset, int]:
        """Add two polynomials in GF(2) (XOR operation)."""
        result = poly1.copy()

        for subset, coeff in poly2.items():
            if subset in result:
                result[subset] = (result[subset] + coeff) % 2
                if result[subset] == 0:
                    del result[subset]
            else:
                result[subset] = coeff % 2

        return result

    def multiply_polynomials(
        self, poly1: Dict[frozenset, int], poly2: Dict[frozenset, int]
    ) -> Dict[frozenset, int]:
        """Multiply two polynomials in GF(2)."""
        result: Dict[frozenset, int] = {}

        for subset1, coeff1 in poly1.items():
            for subset2, coeff2 in poly2.items():
                # Product of monomials is union of variable sets
                product_subset = subset1 | subset2
                product_coeff = (coeff1 * coeff2) % 2

                if product_coeff != 0:
                    if product_subset in result:
                        result[product_subset] = (result[product_subset] + product_coeff) % 2
                        if result[product_subset] == 0:
                            del result[product_subset]
                    else:
                        result[product_subset] = product_coeff

        return result


# Utility functions for polynomial operations
def create_monomial(variables: List[int]) -> Dict[frozenset, int]:
    """Create a single monomial from list of variable indices."""
    return {frozenset(variables): 1}


def create_constant(value: bool) -> Dict[frozenset, int]:
    """Create constant polynomial."""
    if value:
        return {frozenset(): 1}  # Constant 1
    else:
        return {}  # Constant 0


def create_variable(var_index: int) -> Dict[frozenset, int]:
    """Create polynomial for single variable."""
    return {frozenset([var_index]): 1}
