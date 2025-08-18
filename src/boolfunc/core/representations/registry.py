# representations/registry.py
from typing import Callable, Dict, Type, Any
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
    dump: Callable = None,
    convert_from: Callable = None,
    convert_to: Callable = None,
    create_empty: Callable = None,
    is_complete: Callable = None,
    get_storage_requirements: Callable = None,
    time_complexity_rank: Callable = None,
):
    """
    Register a strategy by supplying only the key methods.
    Missing methods raise NotImplementedError by default.
    """
    # Dynamically build subclass
    methods = {
        "evaluate": evaluate,
        "dump": dump or (lambda self, data, **kw: {"data": data}),
        "convert_from": convert_from
        or (lambda self, src, data, **kw: NotImplementedError()),
        "convert_to": convert_to
        or (lambda self, tgt, data, **kw: NotImplementedError()),
        "create_empty": create_empty or (lambda self, n, **kw: NotImplementedError()),
        "is_complete": is_complete or (lambda self, data: True),
        "get_storage_requirements": get_storage_requirements or (lambda self, n: {}),
        "time_complexity_rank": time_complexity_rank or (lambda self, n: {}),
    }
    # Create new class
    NewStrategy = type(
        f"{key.title()}Strategy", (BooleanFunctionRepresentation,), methods
    )
    # Register it
    STRATEGY_REGISTRY[key] = NewStrategy


# Register a simple function representation for adapted external functions
def _function_evaluate(self, inputs, data, space, n_vars):
    """Evaluate the function directly."""
    import numpy as np
    
    # The function representation receives the raw inputs and calls the adapted function
    # For single evaluation, inputs should be passed directly to the function
    try:
        result = data(inputs)
        # Ensure we return a boolean scalar, not an array
        if isinstance(result, (list, np.ndarray)):
            # If result is an array, take the first element or check if it's all the same
            if len(result) == 1:
                return bool(result[0])
            else:
                # This shouldn't happen for single evaluation
                return bool(result[0])  # Take first element as fallback
        else:
            return bool(result)
    except Exception as e:
        # If direct call fails, the function might expect different input format
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
