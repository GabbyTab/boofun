"""
API module providing user-friendly entry points for creating Boolean functions.

This module provides the main `create` function that serves as the primary
interface for users to create Boolean function objects from various data sources.
"""

from boofun.core import BooleanFunction
from boofun.core.factory import BooleanFunctionFactory


def create(data=None, **kwargs):
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

        >>> # Create from polynomial coefficients
        >>> poly = create({frozenset([0]): 1, frozenset([1]): 1}, rep_type='polynomial')
    """
    return BooleanFunctionFactory.create(BooleanFunction, data, **kwargs)
