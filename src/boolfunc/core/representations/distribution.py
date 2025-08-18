"""
Distribution representation for Boolean functions.

This module treats Boolean functions as random variables over the Boolean cube,
enabling probabilistic analysis, sampling, and statistical properties.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import warnings
from .registry import register_strategy
from .base import BooleanFunctionRepresentation
from ..spaces import Space

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available - some distribution features limited")


@dataclass
class BooleanDistribution:
    """
    Distribution representation of a Boolean function.
    
    Treats the Boolean function as a random variable f: {0,1}ⁿ → {0,1}
    with various probability distributions over the input domain.
    
    Attributes:
        truth_table: Truth table values
        input_distribution: Probability distribution over inputs
        n_vars: Number of variables
        domain_size: Size of input domain (2^n_vars)
    """
    truth_table: np.ndarray
    input_distribution: np.ndarray  # Probability mass function over inputs
    n_vars: int
    domain_size: int
    
    def __post_init__(self):
        """Validate distribution parameters."""
        if len(self.truth_table) != self.domain_size:
            raise ValueError(f"Truth table length {len(self.truth_table)} doesn't match domain size {self.domain_size}")
        
        if len(self.input_distribution) != self.domain_size:
            raise ValueError(f"Input distribution length {len(self.input_distribution)} doesn't match domain size {self.domain_size}")
        
        # Normalize input distribution
        if not np.allclose(np.sum(self.input_distribution), 1.0):
            self.input_distribution = self.input_distribution / np.sum(self.input_distribution)
        
        if np.any(self.input_distribution < 0):
            raise ValueError("Input distribution must be non-negative")
    
    def evaluate(self, x: Union[int, List[int], np.ndarray]) -> bool:
        """
        Evaluate function at given input.
        
        Args:
            x: Input (index or binary vector)
            
        Returns:
            Boolean function value
        """
        if isinstance(x, (list, np.ndarray)):
            if len(x) != self.n_vars:
                raise ValueError(f"Input length {len(x)} doesn't match n_vars {self.n_vars}")
            # Convert binary vector to index
            index = sum(bit * (2**(self.n_vars - 1 - i)) for i, bit in enumerate(x))
        else:
            index = int(x)
        
        if index < 0 or index >= self.domain_size:
            raise ValueError(f"Input index {index} out of range [0, {self.domain_size})")
        
        return bool(self.truth_table[index])
    
    def sample_input(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample inputs according to input distribution.
        
        Args:
            size: Number of samples
            random_state: Random seed
            
        Returns:
            Array of sampled input indices
        """
        rng = np.random.RandomState(random_state)
        return rng.choice(self.domain_size, size=size, p=self.input_distribution)
    
    def sample_outputs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample function outputs according to induced distribution.
        
        Args:
            size: Number of samples
            random_state: Random seed
            
        Returns:
            Array of sampled outputs
        """
        input_samples = self.sample_input(size, random_state)
        return np.array([self.truth_table[idx] for idx in input_samples])
    
    def output_probability(self, value: bool) -> float:
        """
        Compute probability that function outputs given value.
        
        Args:
            value: Output value (True or False)
            
        Returns:
            Probability P(f(X) = value)
        """
        mask = self.truth_table == value
        return np.sum(self.input_distribution[mask])
    
    def conditional_probability(self, output: bool, input_condition: Callable[[int], bool]) -> float:
        """
        Compute conditional probability P(f(X) = output | condition(X)).
        
        Args:
            output: Desired output value
            input_condition: Function that returns True for inputs satisfying condition
            
        Returns:
            Conditional probability
        """
        # Find inputs satisfying condition
        condition_mask = np.array([input_condition(i) for i in range(self.domain_size)])
        condition_prob = np.sum(self.input_distribution[condition_mask])
        
        if condition_prob == 0:
            return 0.0  # Undefined, return 0
        
        # Find inputs satisfying both condition and output
        output_mask = self.truth_table == output
        joint_mask = condition_mask & output_mask
        joint_prob = np.sum(self.input_distribution[joint_mask])
        
        return joint_prob / condition_prob
    
    def entropy(self) -> float:
        """
        Compute entropy of function output distribution.
        
        Returns:
            Entropy H(f(X)) in bits
        """
        p_true = self.output_probability(True)
        p_false = self.output_probability(False)
        
        entropy = 0.0
        if p_true > 0:
            entropy -= p_true * np.log2(p_true)
        if p_false > 0:
            entropy -= p_false * np.log2(p_false)
        
        return entropy
    
    def mutual_information(self, variable_idx: int) -> float:
        """
        Compute mutual information between function output and a specific variable.
        
        Args:
            variable_idx: Index of variable
            
        Returns:
            Mutual information I(f(X); X_i)
        """
        if variable_idx < 0 or variable_idx >= self.n_vars:
            raise ValueError(f"Variable index {variable_idx} out of range")
        
        # Compute marginal probabilities
        p_var_0 = 0.0  # P(X_i = 0)
        p_var_1 = 0.0  # P(X_i = 1)
        
        for i in range(self.domain_size):
            bit = (i >> (self.n_vars - 1 - variable_idx)) & 1
            if bit == 0:
                p_var_0 += self.input_distribution[i]
            else:
                p_var_1 += self.input_distribution[i]
        
        # Compute joint probabilities
        p_f0_var0 = 0.0  # P(f(X) = 0, X_i = 0)
        p_f0_var1 = 0.0  # P(f(X) = 0, X_i = 1)
        p_f1_var0 = 0.0  # P(f(X) = 1, X_i = 0)
        p_f1_var1 = 0.0  # P(f(X) = 1, X_i = 1)
        
        for i in range(self.domain_size):
            bit = (i >> (self.n_vars - 1 - variable_idx)) & 1
            f_val = self.truth_table[i]
            prob = self.input_distribution[i]
            
            if f_val == 0 and bit == 0:
                p_f0_var0 += prob
            elif f_val == 0 and bit == 1:
                p_f0_var1 += prob
            elif f_val == 1 and bit == 0:
                p_f1_var0 += prob
            elif f_val == 1 and bit == 1:
                p_f1_var1 += prob
        
        # Compute mutual information
        mi = 0.0
        joint_probs = [p_f0_var0, p_f0_var1, p_f1_var0, p_f1_var1]
        marginal_f = [self.output_probability(False), self.output_probability(True)]
        marginal_var = [p_var_0, p_var_1]
        
        for i, (f_val, var_val) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            joint_prob = joint_probs[i]
            if joint_prob > 0:
                marginal_prod = marginal_f[f_val] * marginal_var[var_val]
                if marginal_prod > 0:
                    mi += joint_prob * np.log2(joint_prob / marginal_prod)
        
        return mi
    
    def correlation(self, variable_idx: int) -> float:
        """
        Compute correlation between function output and a variable.
        
        Args:
            variable_idx: Index of variable
            
        Returns:
            Correlation coefficient
        """
        if variable_idx < 0 or variable_idx >= self.n_vars:
            raise ValueError(f"Variable index {variable_idx} out of range")
        
        # Convert to ±1 representation
        f_vals = 2 * self.truth_table.astype(int) - 1  # {0,1} -> {-1,1}
        
        # Extract variable values
        var_vals = np.zeros(self.domain_size)
        for i in range(self.domain_size):
            bit = (i >> (self.n_vars - 1 - variable_idx)) & 1
            var_vals[i] = 2 * bit - 1  # {0,1} -> {-1,1}
        
        # Compute weighted correlation
        mean_f = np.sum(f_vals * self.input_distribution)
        mean_var = np.sum(var_vals * self.input_distribution)
        
        cov = np.sum((f_vals - mean_f) * (var_vals - mean_var) * self.input_distribution)
        
        var_f = np.sum((f_vals - mean_f)**2 * self.input_distribution)
        var_var = np.sum((var_vals - mean_var)**2 * self.input_distribution)
        
        if var_f == 0 or var_var == 0:
            return 0.0
        
        return cov / np.sqrt(var_f * var_var)
    
    def moments(self, order: int = 4) -> List[float]:
        """
        Compute moments of the output distribution.
        
        Args:
            order: Maximum moment order
            
        Returns:
            List of moments [mean, variance, skewness, kurtosis, ...]
        """
        # Convert outputs to numeric values
        outputs = self.truth_table.astype(float)
        
        moments = []
        
        # Mean (1st moment)
        mean = np.sum(outputs * self.input_distribution)
        moments.append(mean)
        
        if order >= 2:
            # Variance (2nd central moment)
            variance = np.sum((outputs - mean)**2 * self.input_distribution)
            moments.append(variance)
        
        if order >= 3:
            # Skewness (3rd standardized moment)
            if variance > 0:
                skewness = np.sum((outputs - mean)**3 * self.input_distribution) / (variance**1.5)
            else:
                skewness = 0.0
            moments.append(skewness)
        
        if order >= 4:
            # Kurtosis (4th standardized moment)
            if variance > 0:
                kurtosis = np.sum((outputs - mean)**4 * self.input_distribution) / (variance**2)
            else:
                kurtosis = 0.0
            moments.append(kurtosis)
        
        return moments
    
    def to_dict(self) -> Dict[str, Any]:
        """Export distribution to dictionary."""
        return {
            "truth_table": self.truth_table.tolist(),
            "input_distribution": self.input_distribution.tolist(),
            "n_vars": self.n_vars,
            "domain_size": self.domain_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BooleanDistribution':
        """Create distribution from dictionary."""
        return cls(
            truth_table=np.array(data["truth_table"]),
            input_distribution=np.array(data["input_distribution"]),
            n_vars=data["n_vars"],
            domain_size=data["domain_size"]
        )
    
    @classmethod
    def uniform(cls, truth_table: np.ndarray, n_vars: int) -> 'BooleanDistribution':
        """
        Create distribution with uniform input distribution.
        
        Args:
            truth_table: Boolean function truth table
            n_vars: Number of variables
            
        Returns:
            BooleanDistribution with uniform input distribution
        """
        domain_size = 2**n_vars
        input_distribution = np.ones(domain_size) / domain_size
        return cls(truth_table, input_distribution, n_vars, domain_size)
    
    @classmethod
    def biased(cls, truth_table: np.ndarray, n_vars: int, bias: float = 0.5) -> 'BooleanDistribution':
        """
        Create distribution with biased coin flips for each variable.
        
        Args:
            truth_table: Boolean function truth table
            n_vars: Number of variables
            bias: Probability that each variable is 1
            
        Returns:
            BooleanDistribution with product distribution
        """
        domain_size = 2**n_vars
        input_distribution = np.zeros(domain_size)
        
        for i in range(domain_size):
            prob = 1.0
            for j in range(n_vars):
                bit = (i >> (n_vars - 1 - j)) & 1
                if bit:
                    prob *= bias
                else:
                    prob *= (1 - bias)
            input_distribution[i] = prob
        
        return cls(truth_table, input_distribution, n_vars, domain_size)


@register_strategy("distribution")
class DistributionRepresentation(BooleanFunctionRepresentation[BooleanDistribution]):
    """Distribution representation for Boolean functions."""
    
    def evaluate(
        self, inputs: np.ndarray, data: BooleanDistribution, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate distribution representation.
        
        Args:
            inputs: Input values (integer indices or binary vectors)
            data: Boolean distribution
            space: Evaluation space
            n_vars: Number of variables
            
        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        
        if inputs.ndim == 0:
            # Single integer index
            return data.evaluate(int(inputs))
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                return data.evaluate(inputs.tolist())
            else:
                # Array of integer indices
                results = []
                for idx in inputs:
                    results.append(data.evaluate(int(idx)))
                return np.array(results, dtype=bool)
        elif inputs.ndim == 2:
            # Batch of binary vectors
            results = []
            for row in inputs:
                results.append(data.evaluate(row.tolist()))
            return np.array(results, dtype=bool)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")
    
    def dump(self, data: BooleanDistribution, **kwargs) -> Dict[str, Any]:
        """Export distribution representation."""
        result = data.to_dict()
        result['type'] = 'distribution'
        return result
    
    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> BooleanDistribution:
        """
        Convert from another representation to distribution.
        
        Args:
            source_repr: Source representation
            source_data: Source data
            space: Evaluation space
            n_vars: Number of variables
            **kwargs: Additional arguments (distribution_type, bias)
            
        Returns:
            Boolean distribution
        """
        # Get truth table
        size = 1 << n_vars
        truth_table = np.zeros(size, dtype=bool)
        
        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)
            truth_table[i] = bool(val)
        
        # Create distribution based on type
        distribution_type = kwargs.get("distribution_type", "uniform")
        
        if distribution_type == "uniform":
            return BooleanDistribution.uniform(truth_table, n_vars)
        elif distribution_type == "biased":
            bias = kwargs.get("bias", 0.5)
            return BooleanDistribution.biased(truth_table, n_vars, bias)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert distribution to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)
    
    def create_empty(self, n_vars: int, **kwargs) -> BooleanDistribution:
        """Create empty distribution (constant False with uniform input)."""
        domain_size = 2**n_vars
        truth_table = np.zeros(domain_size, dtype=bool)
        return BooleanDistribution.uniform(truth_table, n_vars)
    
    def is_complete(self, data: BooleanDistribution) -> bool:
        """Check if distribution is complete."""
        return (data.truth_table is not None and 
                data.input_distribution is not None and
                len(data.truth_table) == data.domain_size)
    
    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for distribution operations."""
        return {
            "evaluation": 0,  # O(1) - direct lookup
            "construction": n_vars,  # O(2^n) - create truth table
            "conversion_from": n_vars,  # O(2^n) - via truth table
            "space_complexity": n_vars,  # O(2^n) - stores full truth table and distribution
        }
    
    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return storage requirements for distribution representation."""
        domain_size = 2**n_vars
        
        return {
            "truth_table_entries": domain_size,
            "distribution_entries": domain_size,
            "total_entries": 2 * domain_size,
            "bytes": 2 * domain_size * 8,  # Assuming 64-bit floats
            "space_complexity": "O(2^n)",
        }


# Export main classes and functions
__all__ = [
    'BooleanDistribution',
    'DistributionRepresentation'
]
