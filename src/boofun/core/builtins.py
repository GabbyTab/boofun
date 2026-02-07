# src/boofun/core/builtins.py
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import BooleanFunction


class BooleanFunctionBuiltins:
    """Collection of standard Boolean functions used in research and testing."""

    @classmethod
    def majority(cls, n: int) -> "BooleanFunction":
        """
        Create majority function on n variables.

        Returns 1 if more than half of the inputs are 1, 0 otherwise.
        For even n, ties are broken by returning 0.

        Args:
            n: Number of input variables (must be positive)

        Returns:
            BooleanFunction implementing the majority function
        """
        if n <= 0:
            raise ValueError("Number of variables must be positive")

        from .base import BooleanFunction
        from .factory import BooleanFunctionFactory

        # Generate truth table for majority function
        size = 1 << n  # 2^n
        truth_table = np.zeros(size, dtype=bool)

        for i in range(size):
            # Convert index to binary representation
            binary_repr = [(i >> j) & 1 for j in range(n)]
            # Count number of 1s
            ones_count = sum(binary_repr)
            # Majority: more than half are 1
            truth_table[i] = ones_count > n // 2

        return BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n)

    @classmethod
    def dictator(cls, n: int, i: int = 0) -> "BooleanFunction":
        """
        Create dictator function (output equals i-th input).

        The function returns the value of the i-th input variable,
        ignoring all other inputs: f(x) = x_i.

        Args:
            n: Total number of input variables
            i: Index of the dictating variable (0-indexed, default 0)

        Returns:
            BooleanFunction that outputs x_i

        Examples:
            >>> bf.dictator(5)     # 5-var dictator on x₀
            >>> bf.dictator(5, 2)  # 5-var dictator on x₂
        """
        if n <= 0:
            raise ValueError("Number of variables must be positive")
        if i < 0 or i >= n:
            raise ValueError(f"Dictator index {i} must be in range [0, {n-1}]")

        from .base import BooleanFunction
        from .factory import BooleanFunctionFactory

        # Generate truth table for dictator function
        size = 1 << n  # 2^n
        truth_table = np.zeros(size, dtype=bool)

        for idx in range(size):
            # Extract the i-th bit (LSB=x₀ convention: x_i is bit i from the right)
            truth_table[idx] = bool((idx >> i) & 1)

        return BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n)

    @classmethod
    def tribes(cls, k: int, n: int) -> "BooleanFunction":
        """
        Generate tribes function (AND of ⌈n/k⌉ ORs, each of size k).

        The tribes function divides n variables into groups of k,
        computes OR within each group, then AND across groups:

            Tribes_{k,n}(x) = ⋀_{j=1}^{⌈n/k⌉} ⋁_{i∈T_j} x_i

        This is the **dual tribes** convention (AND-of-ORs). The textbook tribes
        (O'Donnell Ch. 4) uses OR-of-ANDs; the two are related by negation.

        Args:
            k: Size of each tribe (number of variables per group)
            n: Total number of variables

        Returns:
            BooleanFunction implementing the tribes function

        Note:
            If n is not divisible by k, the last group will have fewer variables.
        """
        if k <= 0 or n <= 0:
            raise ValueError("Both k and n must be positive")
        if k > n:
            raise ValueError("Tribe size k cannot exceed total variables n")

        from .base import BooleanFunction
        from .factory import BooleanFunctionFactory

        # Calculate number of complete tribes
        num_tribes = n // k
        remainder = n % k

        # Generate truth table
        size = 1 << n  # 2^n
        truth_table = np.zeros(size, dtype=bool)

        for idx in range(size):
            # Convert to binary representation
            bits = [(idx >> j) & 1 for j in range(n)]

            # Compute OR within each tribe, then AND across tribes
            result = True

            # Process complete tribes
            for tribe_idx in range(num_tribes):
                start = tribe_idx * k
                end = start + k
                tribe_or = any(bits[start:end])
                result = result and tribe_or

            # Process remaining variables if any
            if remainder > 0:
                start = num_tribes * k
                tribe_or = any(bits[start:])
                result = result and tribe_or

            truth_table[idx] = result

        return BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n)

    @classmethod
    def parity(cls, n: int) -> "BooleanFunction":
        """
        Create parity function on n variables.

        Returns 1 if an odd number of inputs are 1, 0 otherwise.

        Args:
            n: Number of input variables

        Returns:
            BooleanFunction implementing the parity function
        """
        if n <= 0:
            raise ValueError("Number of variables must be positive")

        from .base import BooleanFunction
        from .factory import BooleanFunctionFactory

        # Generate truth table for parity function
        size = 1 << n  # 2^n
        truth_table = np.zeros(size, dtype=bool)

        for i in range(size):
            # Count number of 1s in binary representation
            ones_count = bin(i).count("1")
            # Parity: odd number of 1s
            truth_table[i] = ones_count % 2 == 1

        return BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n)

    @classmethod
    def constant(cls, value: bool, n: int) -> "BooleanFunction":
        """
        Create constant function.

        Args:
            value: Constant value to return (True or False)
            n: Number of input variables (for compatibility)

        Returns:
            BooleanFunction that always returns the constant value
        """
        if n <= 0:
            raise ValueError("Number of variables must be positive")

        from .base import BooleanFunction
        from .factory import BooleanFunctionFactory

        # Generate truth table with constant value
        size = 1 << n  # 2^n
        truth_table = np.full(size, value, dtype=bool)

        return BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n)
