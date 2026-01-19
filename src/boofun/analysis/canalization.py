"""
Canalization Analysis for Boolean Functions.

Canalization is a concept from systems biology where certain input values
"dominate" and determine the output regardless of other inputs.

A Boolean function f is canalizing if there exists:
- A variable x_i
- A canalizing input a ∈ {0, 1}
- A canalized output b ∈ {0, 1}

Such that: f(x_1, ..., x_i=a, ..., x_n) = b for all other inputs.

References:
- Kauffman, S.A. (1969). Metabolic stability and epigenesis in randomly 
  constructed genetic nets. Journal of Theoretical Biology.
- Kadelka, C. et al. (2023). Collectively canalizing Boolean functions.
  Advances in Applied Mathematics.
"""

import numpy as np
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Set

if TYPE_CHECKING:
    from ..core.base import BooleanFunction


def is_canalizing(f: "BooleanFunction") -> bool:
    """
    Determine if a Boolean function is canalizing.
    
    A function is canalizing if there exists at least one variable x_i
    and a value a ∈ {0,1} such that fixing x_i = a forces the output
    to a constant value, regardless of other inputs.
    
    Args:
        f: Boolean function to test
        
    Returns:
        True if function is canalizing, False otherwise
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).is_canalizing()  # AND canalizes on 0
        True
        >>> bf.parity(3).is_canalizing()  # Parity is not canalizing
        False
    """
    n = f.n_vars
    if n == 0:
        return True  # Constant functions are trivially canalizing
    
    tt = list(f.get_representation("truth_table"))
    
    for i in range(n):
        for a in [0, 1]:
            # Check if fixing x_i = a gives constant output
            outputs_when_fixed = []
            
            for x in range(2**n):
                bit_i = (x >> i) & 1
                if bit_i == a:
                    outputs_when_fixed.append(tt[x])
            
            # If all outputs are the same when x_i = a, it's canalizing
            if len(set(outputs_when_fixed)) == 1:
                return True
    
    return False


def get_canalizing_variables(f: "BooleanFunction") -> List[Dict]:
    """
    Find all canalizing variables and their canalizing inputs/outputs.
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        List of dicts, each containing:
        - 'variable': index of canalizing variable
        - 'canalizing_input': the input value (0 or 1) that canalizes
        - 'canalized_output': the forced output value (0 or 1)
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).get_canalizing_variables()
        [{'variable': 0, 'canalizing_input': 0, 'canalized_output': 0}, ...]
    """
    n = f.n_vars
    if n == 0:
        return []
    
    tt = list(f.get_representation("truth_table"))
    results = []
    
    for i in range(n):
        for a in [0, 1]:
            outputs_when_fixed = []
            
            for x in range(2**n):
                bit_i = (x >> i) & 1
                if bit_i == a:
                    outputs_when_fixed.append(int(tt[x]))
            
            if len(set(outputs_when_fixed)) == 1:
                results.append({
                    'variable': i,
                    'canalizing_input': a,
                    'canalized_output': outputs_when_fixed[0]
                })
    
    return results


def get_canalizing_depth(f: "BooleanFunction") -> int:
    """
    Compute the canalizing depth of a Boolean function.
    
    The canalizing depth is the maximum k such that the function
    is k-canalizing (has k layers of nested canalization).
    
    A function is k-canalizing if:
    1. It has a canalizing variable x_i1 with input a1 → output b1
    2. The subfunction f|_{x_i1 ≠ a1} is (k-1)-canalizing
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        Canalizing depth (0 for non-canalizing functions)
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).get_canalizing_depth()  # AND is fully nested canalizing
        3
        >>> bf.parity(3).get_canalizing_depth()  # Parity has depth 0
        0
    """
    return _compute_canalizing_depth_simple(f)


def _compute_canalizing_depth_recursive(
    f: "BooleanFunction", 
    fixed_vars: Set[int]
) -> int:
    """Recursive helper to compute canalizing depth."""
    n = f.n_vars
    tt = list(f.get_representation("truth_table"))
    
    # Find available variables (not yet fixed)
    available = [i for i in range(n) if i not in fixed_vars]
    
    if not available:
        return 0
    
    # Check if any available variable is canalizing
    for i in available:
        for a in [0, 1]:
            outputs_when_fixed = []
            indices_when_not_fixed = []
            
            for x in range(2**n):
                # Check if this input respects all fixed variables
                valid = all(
                    ((x >> j) & 1) != fixed_vars_values.get(j, -1)
                    for j in fixed_vars
                ) if fixed_vars else True
                
                if not valid:
                    continue
                
                bit_i = (x >> i) & 1
                if bit_i == a:
                    outputs_when_fixed.append(int(tt[x]))
                else:
                    indices_when_not_fixed.append(x)
            
            # If fixing x_i = a gives constant output
            if outputs_when_fixed and len(set(outputs_when_fixed)) == 1:
                # Recurse on subfunction with x_i ≠ a
                new_fixed = fixed_vars | {i}
                return 1 + _compute_canalizing_depth_recursive(f, new_fixed)
    
    return 0


# Simplified version that doesn't track fixed values
def _compute_canalizing_depth_simple(f: "BooleanFunction") -> int:
    """Simpler canalizing depth computation."""
    n = f.n_vars
    if n == 0:
        return 0
    
    # Check if constant
    tt = list(f.get_representation("truth_table"))
    if len(set(tt)) == 1:
        return 0  # Constant functions have depth 0
    
    # Try each variable as canalizing
    for i in range(n):
        for a in [0, 1]:
            # Collect outputs when x_i = a
            outputs_a = []
            outputs_not_a = []
            
            for x in range(2**n):
                bit_i = (x >> i) & 1
                if bit_i == a:
                    outputs_a.append(int(tt[x]))
                else:
                    outputs_not_a.append((x, int(tt[x])))
            
            # If x_i = a gives constant output
            if len(set(outputs_a)) == 1:
                # Create subfunction on remaining variables
                if not outputs_not_a:
                    return 1
                
                # Build truth table for subfunction (n-1 variables)
                sub_tt = _build_subfunction_tt(tt, n, i, 1 - a)
                
                # Import here to avoid circular imports
                from ..core.factory import BooleanFunctionFactory
                from ..core.base import BooleanFunction
                
                sub_f = BooleanFunctionFactory.from_truth_table(
                    BooleanFunction, sub_tt, n=n-1
                )
                
                return 1 + _compute_canalizing_depth_simple(sub_f)
    
    return 0


def _build_subfunction_tt(tt: list, n: int, var: int, keep_val: int) -> list:
    """Build truth table for subfunction after fixing one variable."""
    sub_tt = []
    
    for x in range(2**n):
        bit_var = (x >> var) & 1
        if bit_var == keep_val:
            sub_tt.append(tt[x])
    
    return sub_tt


def is_k_canalizing(f: "BooleanFunction", k: int) -> bool:
    """
    Determine if a Boolean function is k-canalizing.
    
    A function is k-canalizing if it has at least k layers of
    nested canalization.
    
    Args:
        f: Boolean function to test
        k: Required canalizing depth (0 ≤ k ≤ n)
        
    Returns:
        True if function is at least k-canalizing
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).is_k_canalizing(3)  # AND is fully nested
        True
        >>> bf.parity(3).is_k_canalizing(1)  # Parity is not even 1-canalizing
        False
    """
    return _compute_canalizing_depth_simple(f) >= k


def is_nested_canalizing(f: "BooleanFunction") -> bool:
    """
    Determine if a Boolean function is nested canalizing (NCF).
    
    A function on n variables is nested canalizing if it has
    canalizing depth n (every variable is canalizing in some layer).
    
    NCFs are important in biological modeling.
    
    Args:
        f: Boolean function to test
        
    Returns:
        True if function is nested canalizing
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).is_nested_canalizing()  # AND is NCF
        True
        >>> bf.majority(3).is_nested_canalizing()  # Majority is not NCF
        False
    """
    return _compute_canalizing_depth_simple(f) == f.n_vars


def get_essential_variables(f: "BooleanFunction") -> List[int]:
    """
    Find all essential (non-degenerate) variables.
    
    A variable is essential if there exists some input where
    flipping that variable changes the output.
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        List of indices of essential variables
        
    Example:
        >>> import boofun as bf
        >>> len(bf.AND(3).get_essential_variables())
        3
    """
    n = f.n_vars
    tt = list(f.get_representation("truth_table"))
    essential = []
    
    for i in range(n):
        is_essential = False
        
        for x in range(2**n):
            # Flip bit i
            x_flipped = x ^ (1 << i)
            
            if tt[x] != tt[x_flipped]:
                is_essential = True
                break
        
        if is_essential:
            essential.append(i)
    
    return essential


def get_input_types(f: "BooleanFunction") -> Dict[int, str]:
    """
    Classify each input variable by its type.
    
    Types:
    - "positive": f is monotone increasing in this variable
    - "negative": f is monotone decreasing in this variable
    - "conditional": f depends on this variable non-monotonically
    - "non-essential": f does not depend on this variable
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        Dict mapping variable index to type string
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).get_input_types()
        {0: 'positive', 1: 'positive', 2: 'positive'}
    """
    n = f.n_vars
    tt = list(f.get_representation("truth_table"))
    types = {}
    
    for i in range(n):
        increases = 0
        decreases = 0
        
        for x in range(2**n):
            bit_i = (x >> i) & 1
            if bit_i == 0:
                x_with_1 = x | (1 << i)
                if tt[x_with_1] > tt[x]:
                    increases += 1
                elif tt[x_with_1] < tt[x]:
                    decreases += 1
        
        if increases == 0 and decreases == 0:
            types[i] = "non-essential"
        elif increases > 0 and decreases == 0:
            types[i] = "positive"
        elif decreases > 0 and increases == 0:
            types[i] = "negative"
        else:
            types[i] = "conditional"
    
    return types


def get_symmetry_groups(f: "BooleanFunction") -> List[Set[int]]:
    """
    Find groups of symmetric (interchangeable) variables.
    
    Two variables are symmetric if swapping their values in any input
    does not change the function's output.
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        List of sets, where each set contains indices of
        variables that are mutually symmetric.
        
    Example:
        >>> import boofun as bf
        >>> bf.AND(3).get_symmetry_groups()  # All variables symmetric
        [{0, 1, 2}]
        >>> bf.dictator(3, 0).get_symmetry_groups()
        [{1, 2}, {0}]  # Only non-dictator vars symmetric
    """
    n = f.n_vars
    tt = list(f.get_representation("truth_table"))
    
    # Build adjacency: which pairs of variables are symmetric
    symmetric_pairs = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if swapping i and j always preserves output
            is_symmetric = True
            
            for x in range(2**n):
                # Swap bits i and j
                bit_i = (x >> i) & 1
                bit_j = (x >> j) & 1
                
                if bit_i != bit_j:
                    # Build swapped input
                    x_swapped = x
                    if bit_i:
                        x_swapped &= ~(1 << i)  # Clear bit i
                    else:
                        x_swapped |= (1 << i)   # Set bit i
                    if bit_j:
                        x_swapped &= ~(1 << j)  # Clear bit j
                    else:
                        x_swapped |= (1 << j)   # Set bit j
                    
                    if tt[x] != tt[x_swapped]:
                        is_symmetric = False
                        break
            
            if is_symmetric:
                symmetric_pairs.add((i, j))
    
    # Build groups via union-find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j in symmetric_pairs:
        union(i, j)
    
    # Collect groups
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = set()
        groups_dict[root].add(i)
    
    return list(groups_dict.values())


def input_redundancy(f: "BooleanFunction") -> float:
    """
    Compute input redundancy: fraction of inputs that are redundant.
    
    Input redundancy quantifies how many inputs are not needed to
    determine the output on average. Constant functions have
    redundancy 1, parity has redundancy 0.
    
    This is computed as: k_r / n where k_r is the average number
    of redundant inputs across all input combinations.
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        Input redundancy in [0, 1]
        
    Note:
        This is a simplified version. For full CANA-style redundancy,
        install the CANA package.
    """
    n = f.n_vars
    if n == 0:
        return 1.0
    
    # Count non-essential variables
    essential = get_essential_variables(f)
    n_essential = len(essential)
    
    return 1.0 - (n_essential / n)


def edge_effectiveness(f: "BooleanFunction") -> np.ndarray:
    """
    Compute edge effectiveness for each variable.
    
    Edge effectiveness measures how much flipping a variable
    influences the output. This is related to the influence
    but normalized differently.
    
    e_i = Pr[f(x) ≠ f(x ⊕ e_i)]
    
    This is equivalent to the influence Inf_i[f].
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        Array of effectiveness values in [0, 1]
    """
    return f.influences()


def effective_degree(f: "BooleanFunction") -> float:
    """
    Compute effective degree: sum of edge effectiveness.
    
    This is equivalent to the total influence I[f].
    
    Args:
        f: Boolean function to analyze
        
    Returns:
        Effective degree (= total influence)
    """
    return f.total_influence()


class CanalizationAnalyzer:
    """
    High-level analyzer for canalization properties.
    
    Example:
        >>> import boofun as bf
        >>> analyzer = CanalizationAnalyzer(bf.AND(4))
        >>> analyzer.summary()
    """
    
    def __init__(self, f: "BooleanFunction"):
        """
        Initialize analyzer.
        
        Args:
            f: Boolean function to analyze
        """
        self.f = f
        self._cache = {}
    
    def is_canalizing(self) -> bool:
        """Check if function is canalizing."""
        if 'is_canalizing' not in self._cache:
            self._cache['is_canalizing'] = is_canalizing(self.f)
        return self._cache['is_canalizing']
    
    def canalizing_depth(self) -> int:
        """Get canalizing depth."""
        if 'depth' not in self._cache:
            self._cache['depth'] = _compute_canalizing_depth_simple(self.f)
        return self._cache['depth']
    
    def canalizing_variables(self) -> List[Dict]:
        """Get all canalizing variables."""
        if 'can_vars' not in self._cache:
            self._cache['can_vars'] = get_canalizing_variables(self.f)
        return self._cache['can_vars']
    
    def is_nested_canalizing(self) -> bool:
        """Check if function is nested canalizing."""
        return self.canalizing_depth() == self.f.n_vars
    
    def essential_variables(self) -> List[int]:
        """Get essential variables."""
        if 'essential' not in self._cache:
            self._cache['essential'] = get_essential_variables(self.f)
        return self._cache['essential']
    
    def input_types(self) -> Dict[int, str]:
        """Get input type classification."""
        if 'types' not in self._cache:
            self._cache['types'] = get_input_types(self.f)
        return self._cache['types']
    
    def symmetry_groups(self) -> List[Set[int]]:
        """Get symmetry groups."""
        if 'symmetry' not in self._cache:
            self._cache['symmetry'] = get_symmetry_groups(self.f)
        return self._cache['symmetry']
    
    def input_redundancy(self) -> float:
        """Get input redundancy."""
        return input_redundancy(self.f)
    
    def summary(self) -> Dict:
        """
        Get comprehensive canalization summary.
        
        Returns:
            Dictionary with all canalization metrics
        """
        return {
            'n_vars': self.f.n_vars,
            'is_canalizing': self.is_canalizing(),
            'canalizing_depth': self.canalizing_depth(),
            'is_nested_canalizing': self.is_nested_canalizing(),
            'canalizing_variables': self.canalizing_variables(),
            'essential_variables': self.essential_variables(),
            'n_essential': len(self.essential_variables()),
            'input_types': self.input_types(),
            'symmetry_groups': self.symmetry_groups(),
            'input_redundancy': self.input_redundancy(),
            'effective_degree': effective_degree(self.f),
        }
