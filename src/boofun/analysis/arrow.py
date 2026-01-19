"""
Arrow's Impossibility Theorem and Social Choice Theory.

Arrow's Theorem (1951): Any social welfare function satisfying:
1. Unrestricted Domain (UD)
2. Pareto Efficiency (PE) 
3. Independence of Irrelevant Alternatives (IIA)
4. Non-Dictatorship (ND)

...does not exist for 3+ alternatives.

For Boolean functions (2 alternatives), the theorem states:
Any Boolean function f: {±1}^n → {±1} that is:
- Unanimous (Pareto): If all voters agree, output matches
- Independent: Output depends only on individual preferences
- Non-dictatorial: Not controlled by a single voter

...must be dictatorial! (Only exception is constant functions)

Connection to Fourier Analysis:
- A function is "close to dictator" if its degree-1 Fourier weight is high
- The FKN theorem formalizes this

This module provides tools to:
1. Check if a function satisfies Arrow's conditions
2. Quantify "distance to dictator"
3. Analyze voting functions in the social choice framework

References:
- Arrow, "Social Choice and Individual Values" (1951)
- O'Donnell, "Analysis of Boolean Functions" Chapter 2
- Kalai, "Social Indeterminacy" (2002)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "is_unanimous",
    "is_dictatorial",
    "is_non_dictatorial",
    "find_dictator",
    "arrow_analysis",
    "social_welfare_properties",
    "voting_power_analysis",
    "ArrowAnalyzer",
]


def is_unanimous(f: "BooleanFunction") -> bool:
    """
    Check if f satisfies unanimity (Pareto efficiency).
    
    Unanimity: f(1,1,...,1) = 1 and f(0,0,...,0) = 0
    (or in ±1: f(+1,...,+1) = +1 and f(-1,...,-1) = -1)
    
    Args:
        f: Boolean function (voting rule)
        
    Returns:
        True if function satisfies unanimity
    """
    n = f.n_vars
    
    # All ones input
    all_ones = (1 << n) - 1  # 111...1
    # All zeros input
    all_zeros = 0  # 000...0
    
    f_all_ones = f.evaluate(all_ones)
    f_all_zeros = f.evaluate(all_zeros)
    
    return f_all_ones == 1 and f_all_zeros == 0


def is_dictatorial(f: "BooleanFunction") -> bool:
    """
    Check if f is a dictator function.
    
    A dictator function depends on exactly one variable:
    f(x) = x_i for some i (or f(x) = 1 - x_i)
    
    Args:
        f: Boolean function
        
    Returns:
        True if f is a dictator
    """
    dictator_idx = find_dictator(f)
    return dictator_idx is not None


def is_non_dictatorial(f: "BooleanFunction") -> bool:
    """
    Check if f is non-dictatorial.
    
    Args:
        f: Boolean function
        
    Returns:
        True if f is NOT a dictator
    """
    return not is_dictatorial(f)


def find_dictator(f: "BooleanFunction") -> Optional[Tuple[int, bool]]:
    """
    Find which variable is the dictator, if f is dictatorial.
    
    Returns:
        Tuple (variable_index, is_negated) if f is a dictator,
        None otherwise.
        
        is_negated=False means f(x) = x_i
        is_negated=True means f(x) = 1 - x_i
    """
    n = f.n_vars
    
    for i in range(n):
        # Check if f equals x_i
        is_dictator_i = True
        is_neg_dictator_i = True
        
        for x in range(2**n):
            bit_i = (x >> i) & 1
            f_x = f.evaluate(x)
            
            if f_x != bit_i:
                is_dictator_i = False
            if f_x != (1 - bit_i):
                is_neg_dictator_i = False
            
            if not is_dictator_i and not is_neg_dictator_i:
                break
        
        if is_dictator_i:
            return (i, False)
        if is_neg_dictator_i:
            return (i, True)
    
    return None


def arrow_analysis(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Analyze a voting function through the lens of Arrow's theorem.
    
    For 2 alternatives (Boolean), Arrow says:
    Unanimous + IIA + Non-dictator = Impossible
    
    Returns:
        Dict with Arrow analysis results
    """
    n = f.n_vars
    
    # Check properties
    unanimous = is_unanimous(f)
    dictator_info = find_dictator(f)
    is_dictator = dictator_info is not None
    
    # IIA is automatic for Boolean functions (output depends only on inputs)
    iia = True
    
    # Compute influences (related to voting power)
    influences = f.influences()
    total_influence = sum(influences)
    max_influence = max(influences)
    
    # Distance to dictator (from FKN analysis)
    from .fkn import closest_dictator, fkn_theorem_bound
    closest_idx, closest_dist, closest_neg = closest_dictator(f)
    
    # Arrow's characterization
    if is_dictator:
        arrow_type = "dictator"
        arrow_explanation = f"This is dictator on variable {dictator_info[0]}"
        if dictator_info[1]:
            arrow_explanation += " (negated)"
    elif not unanimous:
        arrow_type = "non-unanimous"
        arrow_explanation = "Violates unanimity (Pareto), so Arrow doesn't apply"
    else:
        arrow_type = "impossible"
        arrow_explanation = (
            "Unanimous + IIA + Non-dictator should be impossible by Arrow's theorem! "
            "This function must be very close to a dictator."
        )
    
    return {
        'n_voters': n,
        'is_unanimous': unanimous,
        'is_iia': iia,  # Always true for Boolean
        'is_dictator': is_dictator,
        'dictator_info': dictator_info,
        'arrow_type': arrow_type,
        'arrow_explanation': arrow_explanation,
        'influences': influences.tolist(),
        'total_influence': total_influence,
        'max_influence': max_influence,
        'distance_to_dictator': closest_dist,
        'closest_dictator_var': closest_idx,
    }


def social_welfare_properties(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Analyze social welfare properties of a voting function.
    
    Returns dict with:
    - Symmetry: Are all voters equally powerful?
    - Monotonicity: Does voting "yes" never hurt outcome?
    - Neutrality: Is the function balanced?
    """
    n = f.n_vars
    
    # Symmetry: all influences equal
    influences = f.influences()
    is_symmetric = np.allclose(influences, influences[0], rtol=0.01)
    
    # Monotonicity: f is monotone
    is_monotone = f.is_monotone(100)
    
    # Neutrality/Balance
    is_balanced = f.is_balanced()
    
    # Anonymity (same as symmetry for Boolean)
    is_anonymous = is_symmetric
    
    return {
        'is_symmetric': is_symmetric,
        'is_anonymous': is_anonymous,
        'is_monotone': is_monotone,
        'is_balanced': is_balanced,
        'influence_variance': np.var(influences),
        'influence_ratio': max(influences) / min(influences) if min(influences) > 0 else float('inf'),
    }


def voting_power_analysis(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Analyze voting power using different power indices.
    
    Computes:
    - Banzhaf power index (= influence)
    - Shapley-Shubik power index (for simple games)
    - Pivotal voter analysis
    """
    from math import factorial
    
    n = f.n_vars
    
    # Banzhaf power = influence
    banzhaf = f.influences()
    
    # Shapley-Shubik: average marginal contribution over orderings
    # For Boolean functions: SS_i = (1/n!) Σ_{π} [f(π,i) - f(π,-i)]
    # where sum is over orderings π
    
    shapley = np.zeros(n)
    
    # For small n, compute exactly
    if n <= 10:
        from itertools import permutations
        
        for perm in permutations(range(n)):
            for i in range(n):
                # Coalition of voters before i in the ordering
                coalition = 0
                for j in range(n):
                    if perm.index(j) < perm.index(i):
                        coalition |= (1 << j)
                
                # Compute marginal contribution
                f_without_i = int(f.evaluate(coalition))
                f_with_i = int(f.evaluate(coalition | (1 << i)))
                
                shapley[i] += (f_with_i - f_without_i)
        
        shapley /= factorial(n)
    else:
        # Approximate with sampling
        num_samples = 10000
        rng = np.random.default_rng(42)
        
        for _ in range(num_samples):
            perm = rng.permutation(n)
            for pos, i in enumerate(perm):
                coalition = sum(1 << perm[j] for j in range(pos))
                f_without_i = int(f.evaluate(coalition))
                f_with_i = int(f.evaluate(coalition | (1 << i)))
                shapley[i] += (f_with_i - f_without_i)
        
        shapley /= num_samples
    
    # Find pivotal voters (high influence)
    threshold = 1 / n  # Above average
    pivotal = [i for i in range(n) if banzhaf[i] > threshold]
    
    # Dummy voters (zero influence)
    dummies = [i for i in range(n) if banzhaf[i] < 1e-10]
    
    return {
        'banzhaf_index': banzhaf.tolist(),
        'shapley_shubik_index': shapley.tolist(),
        'pivotal_voters': pivotal,
        'dummy_voters': dummies,
        'most_powerful': int(np.argmax(banzhaf)),
        'power_concentration': float(max(banzhaf) / sum(banzhaf)) if sum(banzhaf) > 0 else 0,
    }


class ArrowAnalyzer:
    """
    Comprehensive Arrow's Theorem analysis for voting functions.
    
    Combines all social choice analysis tools.
    """
    
    def __init__(self, f: "BooleanFunction"):
        """
        Initialize with a voting function.
        
        Args:
            f: Boolean function representing the voting rule
        """
        self.f = f
        self.n = f.n_vars
        self._cache = {}
    
    def arrow_properties(self) -> Dict[str, Any]:
        """Get Arrow's theorem analysis."""
        if 'arrow' not in self._cache:
            self._cache['arrow'] = arrow_analysis(self.f)
        return self._cache['arrow']
    
    def welfare_properties(self) -> Dict[str, Any]:
        """Get social welfare properties."""
        if 'welfare' not in self._cache:
            self._cache['welfare'] = social_welfare_properties(self.f)
        return self._cache['welfare']
    
    def voting_power(self) -> Dict[str, Any]:
        """Get voting power analysis."""
        if 'power' not in self._cache:
            self._cache['power'] = voting_power_analysis(self.f)
        return self._cache['power']
    
    def full_analysis(self) -> Dict[str, Any]:
        """Get complete social choice analysis."""
        return {
            'arrow': self.arrow_properties(),
            'welfare': self.welfare_properties(),
            'power': self.voting_power(),
        }
    
    def summary(self) -> str:
        """Get text summary of the analysis."""
        arrow = self.arrow_properties()
        welfare = self.welfare_properties()
        power = self.voting_power()
        
        lines = [
            "=" * 50,
            "SOCIAL CHOICE ANALYSIS",
            "=" * 50,
            f"Number of voters: {self.n}",
            "",
            "--- Arrow's Theorem ---",
            f"Unanimous: {arrow['is_unanimous']}",
            f"IIA (automatic): {arrow['is_iia']}",
            f"Dictator: {arrow['is_dictator']}",
            f"Type: {arrow['arrow_type']}",
            f"Explanation: {arrow['arrow_explanation']}",
            "",
            "--- Social Welfare Properties ---",
            f"Symmetric/Anonymous: {welfare['is_symmetric']}",
            f"Monotone: {welfare['is_monotone']}",
            f"Balanced/Neutral: {welfare['is_balanced']}",
            "",
            "--- Voting Power ---",
            f"Most powerful voter: {power['most_powerful']}",
            f"Power concentration: {power['power_concentration']:.2%}",
            f"Dummy voters: {power['dummy_voters']}",
            "",
            "Banzhaf Power Index:",
        ]
        
        for i, b in enumerate(power['banzhaf_index']):
            lines.append(f"  Voter {i}: {b:.4f}")
        
        return "\n".join(lines)
