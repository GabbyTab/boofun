# src/boofun/core/representations/__init__.py
"""
Boolean function representations.

This module provides various representations for Boolean functions:
- TruthTableRepresentation: Dense truth table (standard, fast for n ≤ 14)
- PackedTruthTableRepresentation: Memory-efficient (1 bit/entry, good for n > 14)
- SparseTruthTableRepresentation: Stores only exceptions (good for skewed functions)
- AdaptiveTruthTableRepresentation: Auto-selects between dense/sparse
- FourierExpansionRepresentation: Fourier/Walsh coefficients
- ANFRepresentation: Algebraic Normal Form
- PolynomialRepresentation: GF(2) polynomial
- DNFRepresentation: Disjunctive Normal Form
- CNFRepresentation: Conjunctive Normal Form
- SymbolicRepresentation: Symbolic expression
- BDDRepresentation: Binary Decision Diagram
- CircuitRepresentation: Boolean circuit
- LTFRepresentation: Linear Threshold Function
- DistributionRepresentation: Probabilistic representation
"""

from .anf_form import ANFRepresentation
from .base import BooleanFunctionRepresentation
from .bdd import BDDRepresentation
from .circuit import CircuitRepresentation
from .cnf_form import CNFRepresentation
from .distribution import DistributionRepresentation
from .dnf_form import DNFRepresentation
from .fourier_expansion import FourierExpansionRepresentation
from .ltf import LTFRepresentation
from .packed_truth_table import PackedTruthTableRepresentation
from .polynomial import PolynomialRepresentation
from .sparse_truth_table import AdaptiveTruthTableRepresentation, SparseTruthTableRepresentation
from .symbolic import SymbolicRepresentation
from .truth_table import TruthTableRepresentation

__all__ = [
    "BooleanFunctionRepresentation",
    # Truth table variants
    "TruthTableRepresentation",
    "PackedTruthTableRepresentation",
    "SparseTruthTableRepresentation",
    "AdaptiveTruthTableRepresentation",
    # Spectral/algebraic
    "FourierExpansionRepresentation",
    "ANFRepresentation",
    "PolynomialRepresentation",
    # Normal forms
    "DNFRepresentation",
    "CNFRepresentation",
    # Symbolic/structural
    "SymbolicRepresentation",
    "BDDRepresentation",
    "CircuitRepresentation",
    # Special forms
    "LTFRepresentation",
    "DistributionRepresentation",
]
