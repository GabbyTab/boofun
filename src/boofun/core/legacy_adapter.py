"""
Legacy adapter for converting old BooleanFunc objects to new BooleanFunction.

This module provides adapters to use the legacy BooleanFunc class from
BooleanFunc.py with the modern boofun library, enabling:

1. Import old BooleanFunc objects as new BooleanFunction
2. Export new BooleanFunction to legacy format
3. Wrap legacy objects for analysis with new tools
4. Migrate codebases incrementally

Example usage:
    >>> from boofun.core.legacy_adapter import from_legacy, to_legacy
    >>>
    >>> # Convert legacy object to new
    >>> legacy_func = BooleanFunc([0, 1, 1, 0])
    >>> new_func = from_legacy(legacy_func)
    >>>
    >>> # Use new analysis tools
    >>> from boofun.analysis import SpectralAnalyzer
    >>> analyzer = SpectralAnalyzer(new_func)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .base import BooleanFunction

__all__ = [
    "from_legacy",
    "to_legacy",
    "LegacyWrapper",
    "is_legacy_object",
]


def is_legacy_object(obj: Any) -> bool:
    """
    Check if an object is a legacy BooleanFunc instance.

    Args:
        obj: Object to check

    Returns:
        True if obj appears to be a legacy BooleanFunc
    """
    # Check for characteristic attributes of legacy BooleanFunc
    return (
        hasattr(obj, "f")
        and hasattr(obj, "k")
        and hasattr(obj, "fix_single")
        and hasattr(obj, "FourierTransform")
    )


def from_legacy(legacy_func: Any, import_fourier: bool = False) -> "BooleanFunction":
    """
    Convert a legacy BooleanFunc object to a modern BooleanFunction.

    Args:
        legacy_func: Legacy BooleanFunc object with .f (truth table) and .k (n_vars)
        import_fourier: If True and legacy has Fourier data, import it

    Returns:
        New BooleanFunction with truth table representation

    Raises:
        ValueError: If the legacy object doesn't have required attributes

    Example:
        >>> # Assuming BooleanFunc is the legacy class
        >>> legacy = BooleanFunc([0, 1, 1, 0])
        >>> modern = from_legacy(legacy)
        >>> modern.evaluate(0)  # Works with new API
    """
    from .base import BooleanFunction
    from .factory import BooleanFunctionFactory

    # Validate legacy object
    if not hasattr(legacy_func, "f"):
        raise ValueError("Legacy object must have 'f' attribute (truth table)")
    if not hasattr(legacy_func, "k"):
        raise ValueError("Legacy object must have 'k' attribute (number of variables)")

    # Extract truth table
    truth_table = np.array(legacy_func.f, dtype=bool)
    n_vars = legacy_func.k

    # Create new BooleanFunction
    new_func = BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n_vars)

    # Optionally import Fourier coefficients if available
    if import_fourier and hasattr(legacy_func, "FourierCoef"):
        try:
            fourier_data = legacy_func.FourierCoef
            if fourier_data is not None:
                new_func.add_representation(fourier_data, "fourier")
        except Exception:
            pass  # Ignore if Fourier import fails

    return new_func


def to_legacy(func: "BooleanFunction") -> Any:
    """
    Convert a modern BooleanFunction to legacy format (list-based).

    Note: This returns the truth table as a list, which can be passed
    to the legacy BooleanFunc constructor. It does NOT create a BooleanFunc
    object directly (to avoid importing the legacy module).

    Args:
        func: Modern BooleanFunction

    Returns:
        Tuple of (truth_table_list, n_vars) for legacy BooleanFunc constructor

    Example:
        >>> modern = bf.create([0, 1, 1, 0])
        >>> tt, k = to_legacy(modern)
        >>> legacy = BooleanFunc(tt)  # In code that has legacy import
    """
    truth_table = func.get_representation("truth_table")
    tt_list = [int(x) for x in truth_table]
    n_vars = func.n_vars

    return (tt_list, n_vars)


class LegacyWrapper:
    """
    Wrapper that provides legacy BooleanFunc-like interface for modern functions.

    This allows using modern BooleanFunction objects with code that expects
    the legacy API, without modifying the modern object.

    Example:
        >>> modern = bf.create([0, 1, 1, 0])
        >>> wrapped = LegacyWrapper(modern)
        >>> wrapped.k  # Legacy attribute access
        2
        >>> wrapped.f  # Truth table as list
        [0, 1, 1, 0]
        >>> wrapped.fix(0, 1)  # Legacy method, returns wrapped result
    """

    def __init__(self, func: "BooleanFunction"):
        """
        Wrap a modern BooleanFunction with legacy-compatible interface.

        Args:
            func: Modern BooleanFunction to wrap
        """
        self._func = func
        self._f: Optional[List[int]] = None  # Cached truth table list

    @property
    def f(self) -> List[int]:
        """Truth table as list (legacy format)."""
        if self._f is None:
            tt = self._func.get_representation("truth_table")
            self._f = [int(x) for x in tt]
        assert self._f is not None
        return self._f

    @property
    def k(self) -> int:
        """Number of variables (legacy naming)."""
        return self._func.n_vars or 0

    def __getitem__(self, x: int) -> int:
        """Allow f[x] syntax for evaluation."""
        return int(self._func.evaluate(x))

    def __len__(self) -> int:
        """Length of truth table."""
        return 1 << self.k

    def fix(self, var: int, val: int) -> "LegacyWrapper":
        """
        Fix a variable (legacy method).

        Returns a new LegacyWrapper around the restricted function.
        """
        new_func = self._func.fix(var, val)
        return LegacyWrapper(new_func)

    def fix_single(self, var: int, val: int) -> "LegacyWrapper":
        """Alias for fix() - legacy naming."""
        return self.fix(var, val)

    def sensitivity(self, x: int) -> int:
        """Compute sensitivity at input x."""
        from ..analysis.complexity import sensitivity

        return sensitivity(self._func, x)

    def influence(self, i: int) -> float:
        """Compute influence of variable i."""
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self._func)
        return analyzer.influences()[i]

    def total_influence(self) -> float:
        """Compute total influence."""
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self._func)
        return analyzer.total_influence()

    def bias(self) -> float:
        """Compute bias of the function."""
        return self._func.bias()

    @property
    def modern(self) -> "BooleanFunction":
        """Get the underlying modern BooleanFunction."""
        return self._func

    def __str__(self) -> str:
        return f"LegacyWrapper(k={self.k}, f={self.f[:min(8, len(self.f))]}...)"

    def __repr__(self) -> str:
        return f"LegacyWrapper({self._func!r})"


def convert_legacy_function(
    legacy_class: type, modern_class: type, legacy_func: Any
) -> "BooleanFunction":
    """
    Generic conversion function for custom legacy classes.

    Args:
        legacy_class: The legacy class type (for validation)
        modern_class: The modern BooleanFunction class to create
        legacy_func: Instance of legacy_class to convert

    Returns:
        New instance of modern_class
    """
    from .factory import BooleanFunctionFactory

    if not isinstance(legacy_func, legacy_class):
        raise TypeError(f"Expected {legacy_class.__name__}, got {type(legacy_func).__name__}")

    # Extract truth table - try common attribute names
    truth_table = None
    for attr in ["f", "truth_table", "tt", "_truth_table"]:
        if hasattr(legacy_func, attr):
            truth_table = getattr(legacy_func, attr)
            break

    if truth_table is None:
        raise ValueError("Could not find truth table attribute in legacy object")

    # Extract n_vars - try common attribute names
    n_vars = None
    for attr in ["k", "n_vars", "n", "_n_vars"]:
        if hasattr(legacy_func, attr):
            n_vars = getattr(legacy_func, attr)
            break

    if n_vars is None:
        # Infer from truth table size
        n_vars = int(np.log2(len(truth_table)))

    return BooleanFunctionFactory.from_truth_table(
        modern_class, np.array(truth_table, dtype=bool), n=n_vars
    )
