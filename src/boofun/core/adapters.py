"""
Adapters for integrating legacy Boolean function implementations and external libraries.

This module provides adapters to make external Boolean function implementations
compatible with the BooFun library's representation system.
"""

import numpy as np
from typing import Any, Callable, Dict, Optional, Union, Protocol
from abc import ABC, abstractmethod
import warnings

from .base import BooleanFunction
from .representations.base import BooleanFunctionRepresentation
from .factory import BooleanFunctionFactory


class LegacyBooleanFunction(Protocol):
    """Protocol for legacy Boolean function implementations."""
    
    def evaluate(self, inputs: Any) -> Any:
        """Legacy evaluation method."""
        ...


class ExternalLibraryFunction(Protocol):
    """Protocol for external library Boolean functions."""
    
    def __call__(self, *args) -> Any:
        """External function call interface."""
        ...


class BooleanFunctionAdapter(ABC):
    """Abstract base class for Boolean function adapters."""
    
    @abstractmethod
    def adapt(self, external_function: Any) -> BooleanFunction:
        """
        Adapt external function to BooFun interface.
        
        Args:
            external_function: External Boolean function implementation
            
        Returns:
            BooleanFunction compatible with BooFun system
        """
        pass


class LegacyAdapter(BooleanFunctionAdapter):
    """
    Adapter for legacy Boolean function implementations.
    
    Wraps legacy functions that may have different interfaces to work
    with the modern BooFun system.
    """

    def __init__(self, 
                 evaluation_method: str = "evaluate",
                 input_format: str = "auto",
                 output_format: str = "auto",
                 n_vars: Optional[int] = None):
        """
        Initialize legacy adapter.
        
        Args:
            evaluation_method: Name of evaluation method in legacy function
            input_format: Expected input format ("binary", "integer", "auto")
            output_format: Expected output format ("boolean", "integer", "auto")
            n_vars: Number of variables if known
        """
        self.evaluation_method = evaluation_method
        self.input_format = input_format
        self.output_format = output_format
        self.n_vars = n_vars

    def adapt(self, legacy_function: Any) -> BooleanFunction:
        """Adapt legacy function to BooFun interface."""
        
        # Check if legacy function has the expected evaluation method
        if not hasattr(legacy_function, self.evaluation_method):
            raise AttributeError(
                f"Legacy function must have '{self.evaluation_method}' method"
            )
        
        eval_method = getattr(legacy_function, self.evaluation_method)
        
        # Create wrapper function
        def wrapper_function(inputs):
            return self._adapt_evaluation(eval_method, inputs)
        
        # Create BooleanFunction with the wrapper
        return BooleanFunctionFactory.from_function(
            BooleanFunction, 
            wrapper_function,
            n=self.n_vars
        )

    def _adapt_evaluation(self, eval_method: Callable, inputs: np.ndarray) -> Any:
        """Adapt evaluation call between different interfaces."""
        
        # Convert inputs to expected format
        adapted_inputs = self._adapt_inputs(inputs)
        
        # Call legacy evaluation method
        result = eval_method(adapted_inputs)
        
        # Convert output to expected format
        return self._adapt_output(result)

    def _adapt_inputs(self, inputs: np.ndarray) -> Any:
        """Convert inputs to legacy format."""
        if self.input_format == "auto":
            # Try to detect format
            if inputs.ndim == 0:
                return int(inputs)  # Single integer
            elif inputs.ndim == 1:
                return inputs.tolist()  # Binary vector as list
            else:
                return inputs  # Keep as array
        
        elif self.input_format == "integer":
            if inputs.ndim == 0:
                return int(inputs)
            elif inputs.ndim == 1 and len(inputs) == self.n_vars:
                # Convert binary vector to integer
                return sum(int(inputs[i]) * (2**i) for i in range(len(inputs)))
            else:
                return inputs
        
        elif self.input_format == "binary":
            if inputs.ndim == 0:
                # Convert integer to binary vector
                x = int(inputs)
                return [(x >> i) & 1 for i in range(self.n_vars or 8)]
            else:
                return inputs.tolist()
        
        return inputs

    def _adapt_output(self, result: Any) -> bool:
        """Convert output from legacy format."""
        if self.output_format == "auto":
            # Try to detect and convert
            if isinstance(result, bool):
                return result
            elif isinstance(result, (int, np.integer)):
                return bool(result)
            elif isinstance(result, (float, np.floating)):
                return result > 0.5
            else:
                return bool(result)
        
        elif self.output_format == "boolean":
            return bool(result)
        
        elif self.output_format == "integer":
            return bool(int(result))
        
        return bool(result)


class CallableAdapter(BooleanFunctionAdapter):
    """
    Adapter for simple callable functions.
    
    Wraps Python functions or lambdas to work with BooFun system.
    """

    def __init__(self, n_vars: Optional[int] = None, 
                 input_type: str = "binary_vector"):
        """
        Initialize callable adapter.
        
        Args:
            n_vars: Number of variables
            input_type: How to pass inputs ("binary_vector", "individual_args", "integer")
        """
        self.n_vars = n_vars
        self.input_type = input_type

    def adapt(self, callable_function: Callable) -> BooleanFunction:
        """Adapt callable to BooleanFunction."""
        
        def wrapper_function(inputs):
            return self._call_with_adapted_inputs(callable_function, inputs)
        
        return BooleanFunctionFactory.from_function(
            BooleanFunction,
            wrapper_function, 
            n=self.n_vars
        )

    def _call_with_adapted_inputs(self, func: Callable, inputs: np.ndarray) -> bool:
        """Call function with properly formatted inputs."""
        
        if self.input_type == "binary_vector":
            # Pass as single binary vector argument
            if inputs.ndim == 0:
                # Convert integer to binary vector
                x = int(inputs)
                binary_vec = [(x >> i) & 1 for i in range(self.n_vars or 8)]
                return bool(func(binary_vec))
            else:
                return bool(func(inputs))
        
        elif self.input_type == "individual_args":
            # Pass each bit as separate argument
            if inputs.ndim == 0:
                x = int(inputs)
                args = [(x >> i) & 1 for i in range(self.n_vars or 8)]
                return bool(func(*args))
            else:
                return bool(func(*inputs))
        
        elif self.input_type == "integer":
            # Pass as single integer
            if inputs.ndim == 0:
                return bool(func(int(inputs)))
            else:
                # Convert binary vector to integer
                x = sum(int(inputs[i]) * (2**i) for i in range(len(inputs)))
                return bool(func(x))
        
        return bool(func(inputs))


class SymPyAdapter(BooleanFunctionAdapter):
    """
    Adapter for SymPy Boolean expressions.
    
    Integrates SymPy symbolic Boolean functions with BooFun.
    """

    def __init__(self):
        """Initialize SymPy adapter."""
        try:
            import sympy as sp
            self.sp = sp
            self.available = True
        except ImportError:
            warnings.warn("SymPy not available - SymPyAdapter disabled")
            self.available = False

    def adapt(self, sympy_expr, variables: Optional[list] = None) -> BooleanFunction:
        """
        Adapt SymPy Boolean expression to BooleanFunction.
        
        Args:
            sympy_expr: SymPy Boolean expression
            variables: List of variable symbols (auto-detected if None)
            
        Returns:
            BooleanFunction representing the SymPy expression
        """
        if not self.available:
            raise ImportError("SymPy not available")
        
        # Auto-detect variables if not provided
        if variables is None:
            variables = sorted(sympy_expr.free_symbols, key=str)
        
        n_vars = len(variables)
        
        def evaluation_function(inputs):
            if isinstance(inputs, np.ndarray) and inputs.ndim == 0:
                # Convert integer to binary
                x = int(inputs)
                values = [(x >> i) & 1 for i in range(n_vars)]
            else:
                values = inputs
            
            # Create substitution dictionary
            subs_dict = {var: bool(values[i]) for i, var in enumerate(variables)}
            
            # Evaluate SymPy expression
            result = sympy_expr.subs(subs_dict)
            return bool(result)
        
        return BooleanFunctionFactory.from_function(
            BooleanFunction,
            evaluation_function,
            n=n_vars
        )


class NumPyAdapter(BooleanFunctionAdapter):
    """
    Adapter for NumPy-based Boolean function implementations.
    
    Handles vectorized NumPy functions and makes them compatible with BooFun.
    """

    def __init__(self, vectorized: bool = True):
        """
        Initialize NumPy adapter.
        
        Args:
            vectorized: Whether the function supports batch evaluation
        """
        self.vectorized = vectorized

    def adapt(self, numpy_function: Callable, n_vars: int) -> BooleanFunction:
        """
        Adapt NumPy function to BooleanFunction.
        
        Args:
            numpy_function: NumPy-based Boolean function
            n_vars: Number of variables
            
        Returns:
            BooleanFunction wrapper
        """
        
        if self.vectorized:
            # Function supports batch evaluation
            def wrapper_function(inputs):
                return numpy_function(inputs).astype(bool)
        else:
            # Function needs individual evaluation
            def wrapper_function(inputs):
                if isinstance(inputs, np.ndarray) and inputs.ndim > 1:
                    # Batch evaluation
                    return np.array([
                        bool(numpy_function(x)) for x in inputs
                    ])
                else:
                    return bool(numpy_function(inputs))
        
        return BooleanFunctionFactory.from_function(
            BooleanFunction,
            wrapper_function,
            n=n_vars
        )


# Factory function for creating appropriate adapters
def create_adapter(adapter_type: str, **kwargs) -> BooleanFunctionAdapter:
    """
    Factory function for creating adapters.
    
    Args:
        adapter_type: Type of adapter ("legacy", "callable", "sympy", "numpy")
        **kwargs: Adapter-specific parameters
        
    Returns:
        Appropriate adapter instance
    """
    if adapter_type.lower() == "legacy":
        return LegacyAdapter(**kwargs)
    elif adapter_type.lower() == "callable":
        return CallableAdapter(**kwargs)
    elif adapter_type.lower() == "sympy":
        return SymPyAdapter(**kwargs)
    elif adapter_type.lower() == "numpy":
        return NumPyAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# Convenience functions for common use cases
def adapt_legacy_function(legacy_func, **kwargs) -> BooleanFunction:
    """Adapt legacy Boolean function."""
    adapter = LegacyAdapter(**kwargs)
    return adapter.adapt(legacy_func)


def adapt_callable(func: Callable, n_vars: int, **kwargs) -> BooleanFunction:
    """Adapt simple callable to BooleanFunction."""
    adapter = CallableAdapter(n_vars=n_vars, **kwargs)
    return adapter.adapt(func)


def adapt_sympy_expr(expr, variables: Optional[list] = None) -> BooleanFunction:
    """Adapt SymPy Boolean expression."""
    adapter = SymPyAdapter()
    return adapter.adapt(expr, variables)


def adapt_numpy_function(func: Callable, n_vars: int, vectorized: bool = True) -> BooleanFunction:
    """Adapt NumPy-based Boolean function."""
    adapter = NumPyAdapter(vectorized=vectorized)
    return adapter.adapt(func, n_vars)
