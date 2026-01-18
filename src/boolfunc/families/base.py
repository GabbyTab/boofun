"""
Base classes for Boolean function families.

A FunctionFamily represents a parameterized collection of Boolean functions
indexed by the number of variables n (and possibly other parameters).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction


@dataclass
class FamilyMetadata:
    """Metadata about a function family."""
    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    
    # Theoretical asymptotic behavior (formulas as strings or callables)
    asymptotics: Dict[str, Any] = field(default_factory=dict)
    
    # Known properties that hold for all members
    universal_properties: List[str] = field(default_factory=list)
    
    # Constraints on n
    n_constraints: Optional[Callable[[int], bool]] = None
    n_constraint_description: Optional[str] = None


class FunctionFamily(ABC):
    """
    Abstract base class for Boolean function families.
    
    A function family is a parameterized collection f_n indexed by n.
    Examples: Majority_n, Parity_n, Tribes_{k,n}
    
    Subclasses must implement:
    - generate(n) -> BooleanFunction
    - metadata property
    
    Optional overrides:
    - theoretical_value(property_name, n) -> theoretical prediction
    """
    
    @abstractmethod
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        """
        Generate the function for n variables.
        
        Args:
            n: Number of variables
            **kwargs: Additional family-specific parameters
            
        Returns:
            BooleanFunction instance
        """
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> FamilyMetadata:
        """Return metadata about this family."""
        pass
    
    def __call__(self, n: int, **kwargs) -> "BooleanFunction":
        """Shorthand for generate(n)."""
        return self.generate(n, **kwargs)
    
    def validate_n(self, n: int) -> bool:
        """Check if n is valid for this family."""
        if self.metadata.n_constraints is not None:
            return self.metadata.n_constraints(n)
        return n >= 1
    
    def theoretical_value(self, property_name: str, n: int, **kwargs) -> Optional[float]:
        """
        Get theoretical/asymptotic value for a property.
        
        Args:
            property_name: Name of the property (e.g., "total_influence")
            n: Number of variables
            **kwargs: Additional parameters (e.g., rho for noise stability)
            
        Returns:
            Theoretical value if known, None otherwise
        """
        if property_name not in self.metadata.asymptotics:
            return None
        
        formula = self.metadata.asymptotics[property_name]
        
        if callable(formula):
            return formula(n, **kwargs)
        elif isinstance(formula, str):
            # Could parse string formulas if needed
            return None
        else:
            return formula
    
    def generate_range(
        self, 
        n_values: List[int], 
        **kwargs
    ) -> Dict[int, "BooleanFunction"]:
        """
        Generate functions for a range of n values.
        
        Args:
            n_values: List of n values to generate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping n -> BooleanFunction
        """
        return {n: self.generate(n, **kwargs) for n in n_values if self.validate_n(n)}


class InductiveFamily(FunctionFamily):
    """
    A function family defined inductively/recursively.
    
    The user provides:
    - base_case(n) -> BooleanFunction for small n
    - step(f_prev, n) -> how to extend from n-1 to n variables
    
    This is useful for functions defined by their structure:
    - "Add a new variable with weight w_n"
    - "Apply a recursive construction"
    
    Example:
        # Define Majority inductively
        class InductiveMajority(InductiveFamily):
            def base_case(self, n):
                if n == 1:
                    return bf.dictator(1, 0)  # 1 variable, dictator on var 0
                return None
            
            def step(self, f_prev, n, n_prev):
                # Majority_n from Majority_{n-1}
                # Requires knowing the recursive structure
                ...
    """
    
    def __init__(
        self,
        name: str = "InductiveFamily",
        base_cases: Optional[Dict[int, "BooleanFunction"]] = None,
        step_function: Optional[Callable] = None,
        step_size: int = 1,  # How many variables added per step
    ):
        """
        Initialize an inductive family.
        
        Args:
            name: Family name
            base_cases: Dictionary of {n: BooleanFunction} for base cases
            step_function: Function (f_prev, n) -> f_n
            step_size: Number of variables added per step (default 1)
        """
        self._name = name
        self._base_cases = base_cases or {}
        self._step_function = step_function
        self._step_size = step_size
        self._cache: Dict[int, "BooleanFunction"] = {}
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name=self._name,
            description=f"Inductively defined family (step size {self._step_size})",
            parameters={"step_size": str(self._step_size)},
        )
    
    def base_case(self, n: int) -> Optional["BooleanFunction"]:
        """
        Get base case for n, if it exists.
        
        Override this in subclasses or provide base_cases dict.
        """
        return self._base_cases.get(n)
    
    def step(
        self, 
        f_prev: "BooleanFunction", 
        n: int, 
        n_prev: int
    ) -> "BooleanFunction":
        """
        Generate f_n from f_{n_prev}.
        
        Override this in subclasses or provide step_function.
        
        Args:
            f_prev: Function at previous step
            n: Target number of variables
            n_prev: Previous number of variables
            
        Returns:
            Function with n variables
        """
        if self._step_function is not None:
            return self._step_function(f_prev, n, n_prev)
        
        raise NotImplementedError(
            "Must either override step() or provide step_function"
        )
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        """Generate function for n variables using induction."""
        # Check cache
        if n in self._cache:
            return self._cache[n]
        
        # Check base case
        base = self.base_case(n)
        if base is not None:
            self._cache[n] = base
            return base
        
        # Find largest valid n_prev < n
        n_prev = n - self._step_size
        while n_prev > 0:
            if n_prev in self._cache or self.base_case(n_prev) is not None:
                break
            n_prev -= self._step_size
        
        if n_prev <= 0:
            raise ValueError(
                f"Cannot generate n={n}: no base case found. "
                f"Searched step sizes: {self._step_size}"
            )
        
        # Build up from base
        f_prev = self.generate(n_prev)
        current_n = n_prev
        
        while current_n < n:
            next_n = min(current_n + self._step_size, n)
            f_next = self.step(f_prev, next_n, current_n)
            self._cache[next_n] = f_next
            f_prev = f_next
            current_n = next_n
        
        return f_prev
    
    def clear_cache(self):
        """Clear the cached functions."""
        self._cache.clear()


class WeightPatternFamily(FunctionFamily):
    """
    LTF family with weights following a pattern.
    
    The user provides a weight function w(i, n) that determines
    the weight of variable i in the n-variable function.
    
    Examples:
        # Uniform weights (Majority)
        uniform = WeightPatternFamily(lambda i, n: 1)
        
        # Geometric weights (more dictator-like)  
        geometric = WeightPatternFamily(lambda i, n: 2**(-i))
        
        # Harmonic weights
        harmonic = WeightPatternFamily(lambda i, n: 1/(i+1))
    """
    
    def __init__(
        self,
        weight_function: Callable[[int, int], float],
        threshold_function: Optional[Callable[[int], float]] = None,
        name: str = "WeightPatternLTF",
    ):
        """
        Initialize weight pattern family.
        
        Args:
            weight_function: Function (i, n) -> weight of variable i
            threshold_function: Function n -> threshold (default: 0)
            name: Family name
        """
        self._weight_fn = weight_function
        self._threshold_fn = threshold_function or (lambda n: 0)
        self._name = name
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name=self._name,
            description="LTF with parameterized weight pattern",
            parameters={"weight_function": str(self._weight_fn)},
        )
    
    def get_weights(self, n: int) -> np.ndarray:
        """Get weight vector for n variables."""
        return np.array([self._weight_fn(i, n) for i in range(n)])
    
    def get_threshold(self, n: int) -> float:
        """Get threshold for n variables."""
        return self._threshold_fn(n)
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        """Generate LTF with pattern weights."""
        from ..analysis.ltf_analysis import create_weighted_majority
        
        weights = self.get_weights(n)
        threshold = self.get_threshold(n)
        
        return create_weighted_majority(weights.tolist(), threshold)


# Convenience constructors for common patterns
def uniform_ltf_family(name: str = "UniformLTF") -> WeightPatternFamily:
    """Create LTF family with uniform weights (equivalent to Majority)."""
    return WeightPatternFamily(lambda i, n: 1.0, name=name)


def geometric_ltf_family(ratio: float = 0.5, name: str = "GeometricLTF") -> WeightPatternFamily:
    """Create LTF family with geometrically decaying weights."""
    return WeightPatternFamily(lambda i, n: ratio**i, name=name)


def harmonic_ltf_family(name: str = "HarmonicLTF") -> WeightPatternFamily:
    """Create LTF family with harmonic weights 1/(i+1)."""
    return WeightPatternFamily(lambda i, n: 1.0/(i+1), name=name)


def power_ltf_family(power: float = 2.0, name: str = "PowerLTF") -> WeightPatternFamily:
    """Create LTF family with power-law weights (n-i)^power."""
    return WeightPatternFamily(lambda i, n: (n-i)**power, name=name)


__all__ = [
    "FamilyMetadata",
    "FunctionFamily",
    "InductiveFamily", 
    "WeightPatternFamily",
    "uniform_ltf_family",
    "geometric_ltf_family",
    "harmonic_ltf_family",
    "power_ltf_family",
]
