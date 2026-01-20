"""
Conjunctive Normal Form (CNF) representation for Boolean functions.

CNF represents a Boolean function as a conjunction (AND) of disjunctions (OR)
of literals (variables or their negations).

Example: f(x₁,x₂,x₃) = (x₁ ∨ ¬x₂) ∧ (¬x₁ ∨ x₂ ∨ x₃)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy

if TYPE_CHECKING:
    from .dnf_form import DNFFormula


@dataclass
class CNFClause:
    """
    A single clause in CNF form.

    Attributes:
        positive_vars: Set of variables that appear positively
        negative_vars: Set of variables that appear negatively
    """

    positive_vars: Set[int]
    negative_vars: Set[int]

    def __post_init__(self):
        """Validate clause."""
        # Note: Unlike DNF terms, CNF clauses can have the same variable positive and negative
        # This would make the clause always true (tautology)

    def evaluate(self, x: Union[List[int], np.ndarray]) -> bool:
        """
        Evaluate this CNF clause.

        Args:
            x: Binary input vector

        Returns:
            Boolean result of this clause (OR of all literals)
        """
        # At least one positive variable must be 1
        for var in self.positive_vars:
            if var < len(x) and x[var]:
                return True

        # At least one negative variable must be 0
        for var in self.negative_vars:
            if var < len(x) and not x[var]:
                return True

        # If no literals are satisfied, clause is False
        return len(self.positive_vars) == 0 and len(self.negative_vars) == 0

    def get_variables(self) -> Set[int]:
        """Get all variables in this clause."""
        return self.positive_vars | self.negative_vars

    def is_tautology(self) -> bool:
        """Check if clause is a tautology (always true)."""
        return bool(self.positive_vars & self.negative_vars)

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """
        Convert clause to string representation.

        Args:
            var_names: Optional variable names, defaults to x₀, x₁, ...

        Returns:
            String representation of the clause
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
            return "⊥"  # Empty clause is always false

        return " ∨ ".join(literals)

    def to_dict(self) -> Dict[str, Any]:
        """Export clause to dictionary."""
        return {
            "positive_vars": list(self.positive_vars),
            "negative_vars": list(self.negative_vars),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CNFClause":
        """Create clause from dictionary."""
        return cls(
            positive_vars=set(data["positive_vars"]), negative_vars=set(data["negative_vars"])
        )


@dataclass
class CNFFormula:
    """
    CNF formula as a list of clauses.

    Attributes:
        clauses: List of CNF clauses
        n_vars: Number of variables
    """

    clauses: List[CNFClause]
    n_vars: int

    def evaluate(self, x: Union[List[int], np.ndarray]) -> bool:
        """
        Evaluate CNF formula.

        Args:
            x: Binary input vector

        Returns:
            Boolean result (AND of all clauses)
        """
        if len(x) != self.n_vars:
            raise ValueError(f"Input length {len(x)} doesn't match n_vars {self.n_vars}")

        # Empty CNF is True
        if not self.clauses:
            return True

        # AND of all clauses
        for clause in self.clauses:
            if not clause.evaluate(x):
                return False

        return True

    def add_clause(self, clause: CNFClause) -> None:
        """Add a clause to the CNF formula."""
        self.clauses.append(clause)

    def simplify(self) -> "CNFFormula":
        """
        Simplify CNF by removing tautologies and redundant clauses.

        Returns:
            Simplified CNF formula
        """
        simplified_clauses = []

        for clause in self.clauses:
            # Remove tautologies
            if clause.is_tautology():
                continue

            # Check if this clause is subsumed by any existing clause
            is_redundant = False
            for existing_clause in simplified_clauses:
                if self._subsumes(existing_clause, clause):
                    is_redundant = True
                    break

            if not is_redundant:
                # Remove any existing clauses that this clause subsumes
                simplified_clauses = [
                    c for c in simplified_clauses if not self._subsumes(clause, c)
                ]
                simplified_clauses.append(clause)

        return CNFFormula(simplified_clauses, self.n_vars)

    def _subsumes(self, clause1: CNFClause, clause2: CNFClause) -> bool:
        """
        Check if clause1 subsumes clause2.

        Clause1 subsumes clause2 if clause1 is more general (fewer literals).
        """
        return (
            clause1.positive_vars <= clause2.positive_vars
            and clause1.negative_vars <= clause2.negative_vars
        )

    def to_string(self, var_names: Optional[List[str]] = None) -> str:
        """
        Convert CNF to string representation.

        Args:
            var_names: Optional variable names

        Returns:
            String representation of CNF
        """
        if not self.clauses:
            return "⊤"  # Empty CNF is True

        clause_strings = [f"({clause.to_string(var_names)})" for clause in self.clauses]
        return " ∧ ".join(clause_strings)

    def to_dict(self) -> Dict[str, Any]:
        """Export CNF to dictionary."""
        return {"clauses": [clause.to_dict() for clause in self.clauses], "n_vars": self.n_vars}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CNFFormula":
        """Create CNF from dictionary."""
        clauses = [CNFClause.from_dict(clause_data) for clause_data in data["clauses"]]
        return cls(clauses=clauses, n_vars=data["n_vars"])


@register_strategy("cnf")
class CNFRepresentation(BooleanFunctionRepresentation[CNFFormula]):
    """CNF representation for Boolean functions."""

    def evaluate(
        self, inputs: np.ndarray, data: CNFFormula, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate CNF representation.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: CNF formula
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        if inputs.ndim == 0:
            # Single integer index
            binary_input = self._index_to_binary(int(inputs), n_vars)
            return data.evaluate(binary_input)
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                binary_input = inputs.astype(int)
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
        """Convert integer index to binary vector."""
        return [(index >> i) & 1 for i in range(n_vars - 1, -1, -1)]

    def dump(self, data: CNFFormula, **kwargs) -> Dict[str, Any]:
        """Export CNF representation."""
        result = data.to_dict()
        result["type"] = "cnf"
        return result

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> CNFFormula:
        """
        Convert from another representation to CNF.

        Uses truth table method to generate maxterms.
        """
        # Get truth table
        size = 1 << n_vars
        truth_table = []

        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)
            truth_table.append(bool(val))

        # Generate CNF from truth table
        return self._truth_table_to_cnf(truth_table, n_vars)

    def _truth_table_to_cnf(self, truth_table: List[bool], n_vars: int) -> CNFFormula:
        """
        Convert truth table to CNF using maxterms.

        Args:
            truth_table: Boolean truth table
            n_vars: Number of variables

        Returns:
            CNF formula
        """
        clauses = []

        for i, output in enumerate(truth_table):
            if not output:  # Create maxterm for each False entry
                # Convert index to binary
                binary = [(i >> j) & 1 for j in range(n_vars - 1, -1, -1)]

                # Create clause (negate the minterm)
                positive_vars = set()
                negative_vars = set()

                for j, bit in enumerate(binary):
                    if bit:
                        negative_vars.add(j)  # If bit is 1, we need ¬xⱼ
                    else:
                        positive_vars.add(j)  # If bit is 0, we need xⱼ

                clause = CNFClause(positive_vars, negative_vars)
                clauses.append(clause)

        return CNFFormula(clauses, n_vars)

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert CNF to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> CNFFormula:
        """Create empty CNF (constant True)."""
        return CNFFormula(clauses=[], n_vars=n_vars)

    def is_complete(self, data: CNFFormula) -> bool:
        """Check if CNF is complete."""
        return data.clauses is not None and data.n_vars is not None

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for CNF operations."""
        return {
            "evaluation": 0,  # O(clauses * literals_per_clause)
            "construction": n_vars,  # O(2^n) - via truth table
            "conversion_from": n_vars,  # O(2^n) - via truth table
            "space_complexity": n_vars,  # O(2^n) worst case for number of clauses
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return storage requirements for CNF representation."""
        # Worst case: all 2^n maxterms
        max_clauses = 2**n_vars
        max_literals_per_clause = n_vars

        return {
            "max_clauses": max_clauses,
            "max_literals_per_clause": max_literals_per_clause,
            "max_total_literals": max_clauses * max_literals_per_clause,
            "space_complexity": "O(2^n) worst case",
        }


# Utility functions for CNF manipulation
def create_cnf_from_maxterms(maxterms: List[int], n_vars: int) -> CNFFormula:
    """
    Create CNF from list of maxterm indices.

    Args:
        maxterms: List of maxterm indices where function is False
        n_vars: Number of variables

    Returns:
        CNF formula
    """
    clauses = []

    for maxterm in maxterms:
        # Convert maxterm to binary
        binary = [(maxterm >> i) & 1 for i in range(n_vars - 1, -1, -1)]

        positive_vars = set()
        negative_vars = set()

        for i, bit in enumerate(binary):
            if bit:
                negative_vars.add(i)  # If bit is 1, we need ¬xᵢ
            else:
                positive_vars.add(i)  # If bit is 0, we need xᵢ

        clause = CNFClause(positive_vars, negative_vars)
        clauses.append(clause)

    return CNFFormula(clauses, n_vars)


def cnf_to_dnf(cnf: CNFFormula) -> "DNFFormula":
    """
    Convert CNF to DNF using distribution laws.

    Note: This can result in exponential blowup.
    """
    # This would require DNF implementation

    raise NotImplementedError("DNF conversion requires full implementation")


def is_satisfiable(cnf: CNFFormula) -> bool:
    """
    Check if CNF formula is satisfiable (SAT problem).

    This is a simplified check - full SAT solving is NP-complete.
    """
    # Simple check: if any clause is empty, formula is unsatisfiable
    for clause in cnf.clauses:
        if not clause.positive_vars and not clause.negative_vars:
            return False

    # For small formulas, try all assignments
    if cnf.n_vars <= 10:
        for i in range(2**cnf.n_vars):
            assignment = [(i >> j) & 1 for j in range(cnf.n_vars - 1, -1, -1)]
            if cnf.evaluate(assignment):
                return True
        return False

    # For larger formulas, return True (would need proper SAT solver)
    return True


# Export main classes and functions
__all__ = [
    "CNFClause",
    "CNFFormula",
    "CNFRepresentation",
    "create_cnf_from_maxterms",
    "is_satisfiable",
]
