"""
API module providing user-friendly entry points for creating Boolean functions.

This module provides the main `create` function that serves as the primary
interface for users to create Boolean function objects from various data sources.

Also provides:
- `partial()`: Create partial Boolean functions for streaming/incremental specification
- `from_hex()`: Create from hex string truth table (thomasarmel-compatible)
- Storage hints: `storage='packed'`, `storage='sparse'`, `storage='lazy'`
"""

from typing import Callable, Dict, Optional, Union

import numpy as np

from boofun.core import BooleanFunction
from boofun.core.factory import BooleanFunctionFactory
from boofun.core.partial import PartialBooleanFunction
from boofun.core.partial import partial as _partial_func

# Valid storage hint values
STORAGE_HINTS = frozenset({"auto", "dense", "packed", "sparse", "lazy"})


def create(data=None, storage: str = "auto", **kwargs):
    """
    Create a Boolean function from various data sources.

    This is the main entry point for creating Boolean functions. It accepts
    truth tables, functions, distributions, and other representations.

    Args:
        data: Input data for the Boolean function. Can be:
            - List/array of boolean values (truth table)
            - Callable function
            - Dict (polynomial coefficients)
            - None (creates empty function)
        storage: Storage strategy hint:
            - 'auto': Automatically select best representation (default)
            - 'dense': Use dense truth table (fast for n <= 14)
            - 'packed': Use 1-bit per entry (good for n > 14)
            - 'sparse': Store only exceptions (good for skewed functions)
            - 'lazy': Never materialize, compute on demand (for oracles)
        **kwargs: Additional arguments:
            - n: Number of variables (auto-detected if not provided)
            - space: Mathematical space (default: BOOLEAN_CUBE)
            - rep_type: Representation type override

    Returns:
        BooleanFunction: A Boolean function object

    Examples:
        >>> # Create XOR function from truth table
        >>> xor = create([0, 1, 1, 0])

        >>> # Create majority function
        >>> maj = create(lambda x: sum(x) > len(x)//2, n=3)

        >>> # Create with storage hint for large function
        >>> f = create(truth_table, storage='packed')

        >>> # Create lazy function that computes on demand
        >>> f = create(oracle_func, n=20, storage='lazy')

        >>> # Create from polynomial coefficients
        >>> poly = create({frozenset([0]): 1, frozenset([1]): 1}, rep_type='polynomial')
    """
    if storage not in STORAGE_HINTS:
        raise ValueError(
            f"Invalid storage hint '{storage}'. "
            f"Valid options: {', '.join(sorted(STORAGE_HINTS))}"
        )

    # Handle lazy storage for callable functions
    if storage == "lazy" and callable(data):
        return _create_lazy_function(data, **kwargs)

    # Handle storage hints by setting representation type
    # Only apply to array-like data, not callables (which use query access)
    is_array_like = hasattr(data, "__len__") and not callable(data)

    if storage == "packed" and is_array_like:
        kwargs.setdefault("rep_type", "packed_truth_table")
    elif storage == "sparse" and is_array_like:
        kwargs.setdefault("rep_type", "sparse_truth_table")
    elif storage == "auto" and is_array_like:
        # Auto-select based on size for array-like data
        n = kwargs.get("n")
        if n is None:
            try:
                data_len = len(data)
                if data_len > 0:
                    n = int(np.log2(data_len))
            except (TypeError, ValueError):
                pass
        if n is not None and n > 14:
            kwargs.setdefault("rep_type", "packed_truth_table")

    return BooleanFunctionFactory.create(BooleanFunction, data, **kwargs)


def _create_lazy_function(oracle: Callable, **kwargs) -> BooleanFunction:
    """
    Create a lazy Boolean function that computes values on demand.

    The truth table is never fully materialized; instead, the oracle
    is called whenever a value is needed.
    """
    n = kwargs.get("n")
    if n is None:
        raise ValueError("Must specify n (number of variables) for lazy functions")

    # Create function using the callable adapter
    return BooleanFunctionFactory.create(
        BooleanFunction,
        oracle,
        n=n,
        **{k: v for k, v in kwargs.items() if k != "n"},
    )


def partial(
    n: int,
    known_values: Optional[Dict[int, bool]] = None,
    name: Optional[str] = None,
) -> PartialBooleanFunction:
    """
    Create a partial Boolean function with incremental/streaming support.

    A partial function allows you to specify only some outputs, useful for:
    - Streaming data: adding function values incrementally
    - Sampling: knowing only a subset of outputs
    - Large functions: working with sections without full materialization

    Args:
        n: Number of input variables (determines domain size 2^n)
        known_values: Optional dictionary mapping input indices to output values
        name: Optional name for the function

    Returns:
        PartialBooleanFunction instance

    Examples:
        >>> import boofun as bf
        >>>
        >>> # Create empty partial function
        >>> p = bf.partial(n=20)
        >>>
        >>> # With initial values
        >>> p = bf.partial(n=20, known_values={0: True, 1: False, 7: True})
        >>>
        >>> # Add more values incrementally
        >>> p.add(5, False)
        >>> p.add_batch({10: True, 11: True, 12: False})
        >>>
        >>> # Check status
        >>> p.completeness  # Fraction known
        >>> p.num_known  # Count of known values
        >>>
        >>> # Evaluate
        >>> p.evaluate(0)  # True (known)
        >>> p.evaluate(100)  # None (unknown)
        >>> p[0]  # Indexing syntax also works
        >>>
        >>> # Estimate unknown values
        >>> val, conf = p.evaluate_with_confidence(100)
        >>>
        >>> # Convert to full function when ready
        >>> f = p.to_function(fill_unknown=False)
    """
    return _partial_func(n=n, known_values=known_values, name=name)


def from_hex(
    hex_str: str,
    n: int,
    *,
    storage: str = "auto",
    **kwargs,
) -> BooleanFunction:
    """
    Create a Boolean function from a hexadecimal truth table string.

    This is compatible with thomasarmel/boolean_function's format where
    the truth table is represented as a hex string.

    Args:
        hex_str: Hexadecimal string (with or without '0x' prefix)
        n: Number of input variables
        storage: Storage strategy hint ('auto', 'dense', 'packed', 'sparse')
        **kwargs: Additional arguments passed to create()

    Returns:
        BooleanFunction instance

    Examples:
        >>> import boofun as bf
        >>>
        >>> # From thomasarmel example - 4-bit bent function
        >>> f = bf.from_hex("0xac90", n=4)
        >>> f.is_bent()  # True
        >>>
        >>> # 6-bit function
        >>> f = bf.from_hex("0113077C165E76A8", n=6)
        >>>
        >>> # Export back to hex
        >>> f.to_hex()

    Cross-validation:
        This format is compatible with thomasarmel/boolean_function:
        - `BooleanFunction::from_hex_string_truth_table("0113077C165E76A8")`
        - `SmallBooleanFunction::from_truth_table(0xac90, 4)`
    """
    # Remove '0x' prefix if present
    hex_str = hex_str.lower().replace("0x", "").replace(" ", "")

    # Convert hex to integer
    tt_int = int(hex_str, 16)

    # Expected size
    size = 1 << n

    # Convert to binary truth table
    # The hex string represents the truth table in big-endian bit order
    # We need to extract bits from LSB to MSB for index 0, 1, 2, ...
    truth_table = np.array([(tt_int >> i) & 1 for i in range(size)], dtype=bool)

    return create(truth_table, n=n, storage=storage, **kwargs)


def to_hex(f: BooleanFunction) -> str:
    """
    Export a Boolean function to a hexadecimal truth table string.

    This produces output compatible with thomasarmel/boolean_function format.

    Args:
        f: BooleanFunction to export

    Returns:
        Hexadecimal string representation of the truth table

    Examples:
        >>> import boofun as bf
        >>>
        >>> f = bf.from_hex("ac90", n=4)
        >>> bf.to_hex(f)
        'ac90'
    """
    tt = f.get_representation("truth_table")
    n = f.n_vars or int(np.log2(len(tt)))

    # Convert truth table to integer (bit 0 is tt[0], bit 1 is tt[1], etc.)
    tt_int = sum(int(tt[i]) << i for i in range(len(tt)))

    # Convert to hex, padding to correct length
    hex_len = (1 << n) // 4  # Number of hex digits
    if hex_len == 0:
        hex_len = 1
    return format(tt_int, f"0{hex_len}x")
