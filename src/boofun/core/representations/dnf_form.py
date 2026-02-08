"""
Disjunctive Normal Form (DNF) representation for Boolean functions.

DNF represents a Boolean function as a disjunction (OR) of conjunctions (AND)
of literals (variables or their negations).

Example: f(x₁,x₂,x₃) = (x₁ ∧ ¬x₂) ∨ (¬x₁ ∧ x₂ ∧ x₃)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy

if TYPE_CHECKING:
    from .cnf_form import CNFFormula


@dataclass
class DNFTerm:
    """
    A single term (minterm) in DNF form.

    Attributes:
        positive_vars: Set of variables that appear positively
        negative_vars: Set of variables that appear negatively
    """

    positive_vars: Set[int]
    negative_vars: Set[int]

    def __post_init__(self):
        """Validate that positive and negative variables don't overlap."""
        if self.positive_vars & self.negative_vars:
            raise ValueError("Variable cannot be both positive and negative in same term")

    def evaluate(self, x: Union[List[int], np.ndarray]) -> bool:
        """
        Evaluate this DNF term.

        Args:
            x: Binary input vector

        Returns:
            Boolean result of this term
        """
        # All positive variables must be 1
        for var in self.positive_vars:
            if var < len(x) and not x[var]:
                return False

        # All negative variables must be 0
        for var in self.negative_vars:
            if var < len(x) and x[var]:
                return False

        return True

    def get_variables(self) -> Set[int]:
        """Get all variables in this term."""
        return self.positive_vars | self.negative_vars

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """
        Convert term to string representation.

        Args:
            var_names: Optional variable names, defaults to x₀, x₁, ...

        Returns:
            String representation of the term
        """
        if var_names is None:
            var_names = [f"x{i}" for i in range(max(self.get_variables()) + 1)]

        literals = []

        # Add positive literals
        for var in sorted(self.positive_vars):
            if var < len(var_names):
                literals.append(var_names[var])

        # Add negative literals
        for var in sorted(self.negative_vars):
            if var < len(var_names):
                literals.append(f"¬{var_names[var]}")

        if not literals:
            return "⊤"  # Empty term is always true

        return " ∧ ".join(literals)

    def to_dict(self) -> Dict[str, Any]:
        """Export term to dictionary."""
        return {
            "positive_vars": list(self.positive_vars),
            "negative_vars": list(self.negative_vars),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNFTerm":
        """Create term from dictionary."""
        return cls(
            positive_vars=set(data["positive_vars"]), negative_vars=set(data["negative_vars"])
        )


@dataclass
class DNFFormula:
    """
    DNF formula as a list of terms.

    Attributes:
        terms: List of DNF terms
        n_vars: Number of variables
    """

    terms: List[DNFTerm]
    n_vars: int

    def evaluate(self, x: Union[List[int], np.ndarray]) -> bool:
        """
        Evaluate DNF formula.

        Args:
            x: Binary input vector

        Returns:
            Boolean result (OR of all terms)
        """
        if len(x) != self.n_vars:
            raise ValueError(f"Input length {len(x)} doesn't match n_vars {self.n_vars}")

        # Empty DNF is False
        if not self.terms:
            return False

        # OR of all terms
        for term in self.terms:
            if term.evaluate(x):
                return True

        return False

    def add_term(self, term: DNFTerm) -> None:
        """Add a term to the DNF formula."""
        self.terms.append(term)

    def simplify(self) -> "DNFFormula":
        """
        Simplify DNF by removing redundant terms.

        Returns:
            Simplified DNF formula
        """
        simplified_terms = []

        for i, term1 in enumerate(self.terms):
            is_redundant = False

            # Check if this term is subsumed by any other term
            for j, term2 in enumerate(self.terms):
                if i != j and self._subsumes(term2, term1):
                    is_redundant = True
                    break

            if not is_redundant:
                simplified_terms.append(term1)

        return DNFFormula(simplified_terms, self.n_vars)

    def _subsumes(self, term1: DNFTerm, term2: DNFTerm) -> bool:
        """
        Check if term1 subsumes term2.

        Term1 subsumes term2 if term1 is more general (fewer literals).
        """
        return (
            term1.positive_vars <= term2.positive_vars
            and term1.negative_vars <= term2.negative_vars
        )

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """
        Convert DNF to string representation.

        Args:
            var_names: Optional variable names

        Returns:
            String representation of DNF
        """
        if not self.terms:
            return "⊥"  # Empty DNF is False

        term_strings = [term.to_string(var_names) for term in self.terms]
        return " ∨ ".join(term_strings)

    def to_dict(self) -> Dict[str, Any]:
        """Export DNF to dictionary."""
        return {"terms": [term.to_dict() for term in self.terms], "n_vars": self.n_vars}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNFFormula":
        """Create DNF from dictionary."""
        terms = [DNFTerm.from_dict(term_data) for term_data in data["terms"]]
        return cls(terms=terms, n_vars=data["n_vars"])


@register_strategy("dnf")
class DNFRepresentation(BooleanFunctionRepresentation[DNFFormula]):
    """DNF representation for Boolean functions."""

    def evaluate(
        self, inputs: np.ndarray, data: DNFFormula, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate DNF representation.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: DNF formula
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)  # type: ignore[unreachable]

        if inputs.ndim == 0:
            # Single integer index
            binary_input = self._index_to_binary(int(inputs), n_vars)
            return data.evaluate(binary_input)
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                binary_input = inputs.astype(int)  # type: ignore[assignment]
                return data.evaluate(binary_input)
            else:
                # Array of integer indices
                results = []
                for idx in inputs:
                    binary_input = self._index_to_binary(int(idx), n_vars)
                    results.append(data.evaluate(binary_input))
                return np.array(results, dtype=bool)
        elif inputs.ndim == 2:
            # Batch of binary vectors
            results = []
            for row in inputs:
                binary_input = row.astype(int)
                results.append(data.evaluate(binary_input))
            return np.array(results, dtype=bool)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _index_to_binary(self, index: int, n_vars: int) -> List[int]:
        """Convert integer index to binary vector using LSB=x₀ convention."""
        # LSB-first: result[i] = x_i = (index >> i) & 1
        return [(index >> i) & 1 for i in range(n_vars)]

    def dump(self, data: DNFFormula, space=None, **kwargs) -> Dict[str, Any]:
        """Export DNF representation."""
        result = data.to_dict()
        result["type"] = "dnf"
        return result

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> DNFFormula:
        """
        Convert from another representation to DNF.

        Uses truth table method to generate minterms.
        """
        # Get truth table
        size = 1 << n_vars
        truth_table = []

        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)  # type: ignore[arg-type]
            truth_table.append(bool(val))

        # Generate DNF from truth table
        return self._truth_table_to_dnf(truth_table, n_vars)

    def _truth_table_to_dnf(self, truth_table: List[bool], n_vars: int) -> DNFFormula:
        """
        Convert truth table to DNF using minterms.

        Args:
            truth_table: Boolean truth table
            n_vars: Number of variables

        Returns:
            DNF formula
        """
        terms = []

        for i, output in enumerate(truth_table):
            if output:  # Create minterm for each True entry
                # Convert index to binary
                binary = [(i >> j) & 1 for j in range(n_vars - 1, -1, -1)]

                # Create term
                positive_vars = set()
                negative_vars = set()

                for j, bit in enumerate(binary):
                    if bit:
                        positive_vars.add(j)
                    else:
                        negative_vars.add(j)

                term = DNFTerm(positive_vars, negative_vars)
                terms.append(term)

        return DNFFormula(terms, n_vars)

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert DNF to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> DNFFormula:
        """Create empty DNF (constant False)."""
        return DNFFormula(terms=[], n_vars=n_vars)

    def is_complete(self, data: DNFFormula) -> bool:
        """Check if DNF is complete."""
        return data.terms is not None and data.n_vars is not None

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for DNF operations."""
        return {
            "evaluation": 0,  # O(terms * literals_per_term)
            "construction": n_vars,  # O(2^n) - via truth table
            "conversion_from": n_vars,  # O(2^n) - via truth table
            "space_complexity": n_vars,  # O(2^n) worst case for number of terms
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, Any]:
        """Return storage requirements for DNF representation."""
        # Worst case: all 2^n minterms
        max_terms = 2**n_vars
        max_literals_per_term = n_vars

        return {
            "max_terms": max_terms,
            "max_literals_per_term": max_literals_per_term,
            "max_total_literals": max_terms * max_literals_per_term,
            "space_complexity": "O(2^n) worst case",
        }


# Utility functions for DNF manipulation
def create_dnf_from_minterms(minterms: List[int], n_vars: int) -> DNFFormula:
    """
    Create DNF from list of minterm indices.

    Args:
        minterms: List of minterm indices where function is True
        n_vars: Number of variables

    Returns:
        DNF formula
    """
    terms = []

    for minterm in minterms:
        # Convert minterm to binary
        binary = [(minterm >> i) & 1 for i in range(n_vars - 1, -1, -1)]

        positive_vars = set()
        negative_vars = set()

        for i, bit in enumerate(binary):
            if bit:
                positive_vars.add(i)
            else:
                negative_vars.add(i)

        term = DNFTerm(positive_vars, negative_vars)
        terms.append(term)

    return DNFFormula(terms, n_vars)


def dnf_to_cnf(dnf: DNFFormula) -> "CNFFormula":
    """
    Convert DNF to CNF using distribution laws.

    Note: This can result in exponential blowup.
    """
    # This would require CNF implementation
    raise NotImplementedError("CNF conversion requires CNF representation")


def minimize_dnf(dnf: DNFFormula) -> DNFFormula:
    """
    Minimize DNF using Quine-McCluskey algorithm (simplified).

    Args:
        dnf: DNF formula to minimize

    Returns:
        Minimized DNF formula
    """
    # This is a simplified version - full Quine-McCluskey is complex
    return dnf.simplify()


# Export main classes and functions
__all__ = ["DNFTerm", "DNFFormula", "DNFRepresentation", "create_dnf_from_minterms", "minimize_dnf"]
