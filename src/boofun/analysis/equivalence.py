"""
Equivalence testing and canonical forms for Boolean functions.

This module provides tools for:
- Computing canonical representatives under various symmetry groups
- Testing equivalence of Boolean functions
- Finding automorphisms (self-symmetries)
- Applying variable permutations and input/output transformations

Two functions are considered equivalent if one can be transformed into
the other via some combination of:
- Variable permutation: f(x_π(1), x_π(2), ..., x_π(n))
- Input negation: f(x ⊕ s) for some shift s
- Output negation: ¬f(x)

The canonical form is the lexicographically smallest representative
under these transformations, enabling fast equivalence checking.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Set, Iterator, TYPE_CHECKING
from itertools import permutations
import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "canonical_form",
    "are_equivalent",
    "apply_permutation",
    "automorphisms",
    "equivalence_class_size",
    "PermutationEquivalence",
    "AffineEquivalence",
]


def _generate_permutations(n: int) -> Iterator[Tuple[int, ...]]:
    """Generate all permutations of [0, 1, ..., n-1]."""
    return permutations(range(n))


def _apply_perm_to_index(x: int, perm: Tuple[int, ...], n: int) -> int:
    """
    Apply permutation to input index x.
    
    If perm[i] = j, then bit i of x goes to position j in the result.
    """
    result = 0
    for i in range(n):
        if (x >> i) & 1:
            result |= (1 << perm[i])
    return result


def apply_permutation(f: "BooleanFunction", perm: Tuple[int, ...]) -> "BooleanFunction":
    """
    Apply a variable permutation to a Boolean function.
    
    If perm[i] = j, variable x_i in the original function becomes x_j
    in the result.
    
    Args:
        f: BooleanFunction to permute
        perm: Permutation as a tuple where perm[i] is the new position of var i
        
    Returns:
        New BooleanFunction with permuted variables
        
    Example:
        >>> f = bf.create([0, 0, 0, 1])  # AND(x0, x1)
        >>> g = apply_permutation(f, (1, 0))  # Swap x0 and x1
        >>> # g is still AND(x0, x1) since AND is symmetric
    """
    from ..core.factory import BooleanFunctionFactory
    
    n = f.n_vars or 0
    if len(perm) != n:
        raise ValueError(f"Permutation length {len(perm)} doesn't match n_vars {n}")
    
    # Verify it's a valid permutation
    if sorted(perm) != list(range(n)):
        raise ValueError(f"Invalid permutation: {perm}")
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    size = 1 << n
    
    # Build new truth table: new_f(y) = f(π^{-1}(y))
    # where y = (x_{perm[0]}, x_{perm[1]}, ..., x_{perm[n-1]})
    new_tt = np.zeros(size, dtype=bool)
    
    for x in range(size):
        # Apply permutation to index
        new_x = _apply_perm_to_index(x, perm, n)
        new_tt[new_x] = truth_table[x]
    
    return BooleanFunctionFactory.from_truth_table(
        type(f), new_tt, n=n
    )


def canonical_form(
    f: "BooleanFunction",
    include_shifts: bool = True,
    include_negation: bool = True
) -> Tuple[Tuple[int, ...], Optional[Tuple]]:
    """
    Compute the canonical form of a Boolean function.
    
    The canonical form is the lexicographically smallest truth table achievable
    by applying allowed transformations:
    1. Variable permutations (always applied)
    2. Input shifts: f(x ⊕ s) (if include_shifts=True)
    3. Output negation: ¬f (if include_negation=True)
    
    Two functions have the same canonical form if and only if they are
    equivalent under the specified group of transformations.
    
    Args:
        f: BooleanFunction to canonicalize
        include_shifts: Whether to consider input shifts (affine equivalence)
        include_negation: Whether to consider output negation
        
    Returns:
        Tuple of (canonical_truth_table, transformation) where transformation
        describes how to get from original to canonical form.
        
    Example:
        >>> f = bf.create([0, 1, 1, 0])  # XOR
        >>> g = bf.create([1, 0, 0, 1])  # XNOR
        >>> canonical_form(f)[0] == canonical_form(g)[0]
        True  # Same under negation
    """
    n = f.n_vars or 0
    if n == 0:
        tt = tuple(f.get_representation("truth_table"))
        return (tt, None)
    
    truth_table = list(f.get_representation("truth_table"))
    size = 1 << n
    
    best = tuple(truth_table)
    best_transform = None
    
    for perm in _generate_permutations(n):
        # Apply permutation
        perm_f = [0] * size
        for x in range(size):
            new_x = _apply_perm_to_index(x, perm, n)
            perm_f[new_x] = truth_table[x]
        
        # Try shifts
        shifts_to_try = range(size) if include_shifts else [0]
        
        for shift in shifts_to_try:
            shifted_f = [perm_f[x ^ shift] for x in range(size)]
            
            # Try negation
            negations_to_try = [False, True] if include_negation else [False]
            
            for negate in negations_to_try:
                if negate:
                    final_f = tuple(1 - v for v in shifted_f)
                else:
                    final_f = tuple(shifted_f)
                
                if final_f < best:
                    best = final_f
                    best_transform = (perm, shift, negate)
    
    return (best, best_transform)


def are_equivalent(
    f: "BooleanFunction",
    g: "BooleanFunction",
    include_shifts: bool = True,
    include_negation: bool = True
) -> bool:
    """
    Test if two Boolean functions are equivalent.
    
    Two functions are equivalent if one can be transformed into the other
    via variable permutation, input shift, and/or output negation.
    
    Args:
        f, g: Functions to compare
        include_shifts: Consider input shifts (affine equivalence)
        include_negation: Consider output negation
        
    Returns:
        True if f and g are equivalent
    """
    if f.n_vars != g.n_vars:
        return False
    
    cf = canonical_form(f, include_shifts, include_negation)[0]
    cg = canonical_form(g, include_shifts, include_negation)[0]
    
    return cf == cg


def automorphisms(
    f: "BooleanFunction",
    include_shifts: bool = False,
    include_negation: bool = False
) -> List[Tuple[int, ...]]:
    """
    Find all automorphisms (self-symmetries) of a Boolean function.
    
    An automorphism is a transformation that maps f to itself.
    By default, only variable permutations are considered.
    
    Args:
        f: BooleanFunction to analyze
        include_shifts: Consider input shifts
        include_negation: Consider output negation
        
    Returns:
        List of transformations (permutations) that are automorphisms
        
    Example:
        >>> xor = bf.create([0, 1, 1, 0])
        >>> autos = automorphisms(xor)
        >>> len(autos)
        2  # Identity and swap
    """
    n = f.n_vars or 0
    if n == 0:
        return [()]
    
    truth_table = list(f.get_representation("truth_table"))
    size = 1 << n
    
    autos = []
    
    for perm in _generate_permutations(n):
        # Apply permutation
        perm_f = [0] * size
        for x in range(size):
            new_x = _apply_perm_to_index(x, perm, n)
            perm_f[new_x] = truth_table[x]
        
        # Check if it's an automorphism
        if perm_f == truth_table:
            autos.append(perm)
        
        # Check with shifts
        if include_shifts:
            for shift in range(1, size):  # Skip 0, already checked
                shifted_f = [perm_f[x ^ shift] for x in range(size)]
                if shifted_f == truth_table:
                    autos.append((perm, shift))
        
        # Check with negation
        if include_negation:
            neg_f = [1 - v for v in perm_f]
            if neg_f == truth_table:
                autos.append((perm, 'neg'))
    
    return autos


def equivalence_class_size(f: "BooleanFunction") -> int:
    """
    Compute the size of the equivalence class of f.
    
    This is the number of distinct functions equivalent to f under
    variable permutation (excluding shifts and negation).
    
    By the orbit-stabilizer theorem:
        |orbit| = n! / |automorphisms|
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Size of the equivalence class
    """
    n = f.n_vars or 0
    if n == 0:
        return 1
    
    # Count automorphisms (only permutations)
    num_autos = len(automorphisms(f))
    
    # n! / |Aut(f)|
    from math import factorial
    return factorial(n) // num_autos


class PermutationEquivalence:
    """
    Utility class for permutation equivalence testing.
    
    Provides caching and optimizations for repeated equivalence checks.
    """
    
    def __init__(self):
        self._cache = {}
    
    def canonical(self, f: "BooleanFunction") -> Tuple[int, ...]:
        """Get cached canonical form."""
        key = tuple(f.get_representation("truth_table"))
        if key not in self._cache:
            self._cache[key] = canonical_form(f, include_shifts=False, include_negation=False)[0]
        return self._cache[key]
    
    def equivalent(self, f: "BooleanFunction", g: "BooleanFunction") -> bool:
        """Test permutation equivalence using cached canonical forms."""
        return self.canonical(f) == self.canonical(g)
    
    def clear_cache(self):
        """Clear the canonical form cache."""
        self._cache.clear()


class AffineEquivalence:
    """
    Utility class for affine equivalence testing.
    
    Affine equivalence considers variable permutation, input shifts,
    and output negation.
    """
    
    def __init__(self, include_negation: bool = True):
        self._cache = {}
        self.include_negation = include_negation
    
    def canonical(self, f: "BooleanFunction") -> Tuple[int, ...]:
        """Get cached affine canonical form."""
        key = tuple(f.get_representation("truth_table"))
        if key not in self._cache:
            self._cache[key] = canonical_form(
                f, 
                include_shifts=True, 
                include_negation=self.include_negation
            )[0]
        return self._cache[key]
    
    def equivalent(self, f: "BooleanFunction", g: "BooleanFunction") -> bool:
        """Test affine equivalence using cached canonical forms."""
        return self.canonical(f) == self.canonical(g)
    
    def clear_cache(self):
        """Clear the canonical form cache."""
        self._cache.clear()


def is_symmetric(f: "BooleanFunction") -> bool:
    """
    Check if a function is symmetric (invariant under all variable permutations).
    
    A symmetric function's output depends only on the Hamming weight of the input,
    not on which specific variables are set.
    
    Args:
        f: BooleanFunction to check
        
    Returns:
        True if f is symmetric
    """
    n = f.n_vars or 0
    if n <= 1:
        return True
    
    # Check that all inputs of the same Hamming weight give the same output
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)
    
    weight_outputs = {}
    for x in range(1 << n):
        weight = bin(x).count("1")
        output = int(truth_table[x])
        
        if weight in weight_outputs:
            if weight_outputs[weight] != output:
                return False
        else:
            weight_outputs[weight] = output
    
    return True
