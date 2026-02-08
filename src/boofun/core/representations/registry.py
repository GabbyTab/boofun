# representations/registry.py
from typing import Callable, Dict, Optional, Type

from .base import BooleanFunctionRepresentation

STRATEGY_REGISTRY: Dict[str, Type[BooleanFunctionRepresentation]] = {}


def get_strategy(rep_key: str) -> BooleanFunctionRepresentation:
    """
    Retrieve and instantiate the strategy class for the given representation key.

    Args:
        rep_key: Representation key (e.g., 'truth_table')

    Returns:
        Instance of the strategy class

    Raises:
        KeyError: If no strategy is registered for the key
    """
    if rep_key not in STRATEGY_REGISTRY:
        raise KeyError(f"No strategy registered for '{rep_key}'")
    strategy_cls = STRATEGY_REGISTRY[rep_key]
    return strategy_cls()


def register_strategy(key: str):
    """Decorator to register representation classes"""

    def decorator(cls: Type[BooleanFunctionRepresentation]):
        STRATEGY_REGISTRY[key] = cls
        return cls

    return decorator


def register_partial_strategy(
    key: str,
    *,
    evaluate: Callable,
    dump: Optional[Callable] = None,
    convert_from: Optional[Callable] = None,
    convert_to: Optional[Callable] = None,
    create_empty: Optional[Callable] = None,
    is_complete: Optional[Callable] = None,
    get_storage_requirements: Optional[Callable] = None,
    time_complexity_rank: Optional[Callable] = None,
):
    """
    Register a strategy by supplying only the key methods.
    Missing methods raise NotImplementedError by default.
    """
    # Dynamically build subclass
    methods = {
        "evaluate": evaluate,
        "dump": dump or (lambda self, data, **kw: {"data": data}),
        "convert_from": convert_from or (lambda self, src, data, **kw: NotImplementedError()),
        "convert_to": convert_to or (lambda self, tgt, data, **kw: NotImplementedError()),
        "create_empty": create_empty or (lambda self, n, **kw: NotImplementedError()),
        "is_complete": is_complete or (lambda self, data: True),
        "get_storage_requirements": get_storage_requirements or (lambda self, n: {}),
        "time_complexity_rank": time_complexity_rank or (lambda self, n: {}),
    }
    # Create new class
    NewStrategy = type(f"{key.title()}Strategy", (BooleanFunctionRepresentation,), methods)
    # Register it
    STRATEGY_REGISTRY[key] = NewStrategy


# Register a simple function representation for adapted external functions
def _function_evaluate(self, inputs, data, space, n_vars):
    """
    Evaluate the function directly.

    This handles the conversion between different input formats:
    - Integer index (e.g., 5) -> convert to binary array [1, 0, 1, 0, 0]
    - Array with single integer (e.g., [5]) -> extract and convert to binary
    - Array matching n_vars (e.g., [1, 0, 1, 0, 0]) -> pass directly

    User lambdas typically expect array inputs like `lambda x: x[0] ^ x[1]`,
    but truth table conversion passes integer indices. This function bridges
    both calling conventions.
    """
    import numpy as np

    def index_to_binary(idx, n):
        """Convert integer index to binary array (LSB convention: bit i at position i)."""
        return [(idx >> i) & 1 for i in range(n)]

    # Normalize inputs to array format for user functions
    # Handle n_vars being None - in this case, don't convert integers to binary
    if n_vars is None:
        # No conversion possible, pass input as-is
        binary_inputs = (
            inputs
            if not isinstance(inputs, np.ndarray)
            else inputs.tolist() if inputs.ndim > 0 else int(inputs)
        )
    elif isinstance(inputs, (int, np.integer)):
        # Integer index - convert to binary array for user lambdas
        binary_inputs = index_to_binary(int(inputs), n_vars)
    elif isinstance(inputs, np.ndarray) and inputs.ndim == 0:
        # 0-dimensional numpy array (scalar)
        binary_inputs = index_to_binary(int(inputs), n_vars)
    elif isinstance(inputs, np.ndarray) and inputs.ndim == 1 and len(inputs) == 1:
        # Array with single element (wrapped integer from evaluate()) - extract and convert
        binary_inputs = index_to_binary(int(inputs[0]), n_vars)
    elif hasattr(inputs, "__len__"):
        # Array-like with multiple elements
        if len(inputs) < n_vars:
            # Input is shorter than expected - might be an integer stored as array
            # Check if it looks like a single integer
            if len(inputs) == 1:
                binary_inputs = index_to_binary(int(inputs[0]), n_vars)
            else:
                # Pad with zeros
                binary_inputs = list(inputs) + [0] * (n_vars - len(inputs))
        else:
            # Already correct length
            binary_inputs = list(inputs)
    else:
        # Unknown format - try as-is
        binary_inputs = inputs

    # Try calling the function with array input first (most common for user lambdas)
    try:
        result = data(binary_inputs)
        # Ensure we return a boolean scalar, not an array
        if isinstance(result, (list, np.ndarray)):
            if len(result) == 1:
                return bool(result[0])
            else:
                return bool(result[0])
        else:
            return bool(result)
    except (IndexError, TypeError, ValueError) as e:
        # If array call fails, try with original inputs (integer)
        # This supports functions that expect integer indices
        if isinstance(inputs, (int, np.integer)):
            try:
                result = data(inputs)
                return (
                    bool(result) if not isinstance(result, (list, np.ndarray)) else bool(result[0])
                )
            except Exception:
                pass  # Fall through to error
        raise ValueError(f"Function evaluation failed: {e}")


def _function_convert_from(self, source_repr, source_data, space, n_vars, **kwargs):
    """Convert from another representation to function."""

    def func(inputs):
        return source_repr.evaluate(inputs, source_data, space, n_vars)

    return func


def _function_convert_to(self, target_repr, source_data, space, n_vars, **kwargs):
    """Convert function to another representation."""
    return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)


def _function_create_empty(self, n_vars, **kwargs):
    """Create empty function representation."""
    return lambda x: False


def _function_is_complete(self, data):
    """Check if function representation is complete."""
    return callable(data)


def _function_time_complexity_rank(self, n_vars):
    """Time complexity for function operations."""
    return {
        "evaluation": 1,  # O(1) function call
        "conversion": 2**n_vars,  # Need to evaluate all inputs for conversion
    }


def _function_get_storage_requirements(self, n_vars):
    """Storage requirements for function representation."""
    return {
        "memory_bytes": 64,  # Just a function reference
        "disk_bytes": 0,  # Functions can't be serialized easily
    }


register_partial_strategy(
    "function",
    evaluate=_function_evaluate,
    convert_from=_function_convert_from,
    convert_to=_function_convert_to,
    create_empty=_function_create_empty,
    is_complete=_function_is_complete,
    get_storage_requirements=_function_get_storage_requirements,
    time_complexity_rank=_function_time_complexity_rank,
)
