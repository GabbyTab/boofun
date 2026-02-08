"""
Error models for Boolean function analysis.

This module provides various error models for handling uncertainty,
noise, and approximation in Boolean function computations.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

try:
    import uncertainties.unumpy as unp
    from uncertainties import ufloat

    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False


class ErrorModel(ABC):
    """Abstract base class for all error models."""

    @abstractmethod
    def apply_error(self, result: Any) -> Any:
        """
        Apply error model to a computation result.

        Args:
            result: The original computation result

        Returns:
            Result with error model applied
        """

    @abstractmethod
    def get_confidence(self, result: Any) -> float:
        """
        Get confidence level for a result.

        Args:
            result: Computation result

        Returns:
            Confidence level between 0 and 1
        """

    @abstractmethod
    def is_reliable(self, result: Any) -> bool:
        """
        Check if result meets reliability threshold.

        Args:
            result: Computation result

        Returns:
            True if result is considered reliable
        """


class ExactErrorModel(ErrorModel):
    """
    Exact error model - no errors or uncertainty.

    This model assumes perfect computation with no noise or approximation.
    """

    def apply_error(self, result: Any) -> Any:
        """Return result unchanged."""
        return result

    def get_confidence(self, result: Any) -> float:
        """Always return maximum confidence."""
        return 1.0

    def is_reliable(self, result: Any) -> bool:
        """Always reliable for exact computations."""
        return True

    def __repr__(self) -> str:
        return "ExactErrorModel()"


class PACErrorModel(ErrorModel):
    """
    PAC (Probably Approximately Correct) error model.

    Provides probabilistic guarantees with specified error and confidence bounds.
    """

    def __init__(self, epsilon: float = 0.1, delta: float = 0.1):
        """
        Initialize PAC error model.

        Args:
            epsilon: Approximation error bound (0 < epsilon < 1)
            delta: Confidence failure probability (0 < delta < 1)
        """
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be in (0, 1)")
        if not (0 < delta < 1):
            raise ValueError("Delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.confidence = 1 - delta

    def apply_error(self, result: Any) -> Dict[str, Any]:
        """
        Apply PAC bounds to result.

        Returns:
            Dictionary with result and PAC bounds
        """
        if isinstance(result, (int, float, bool)):
            return {
                "value": result,
                "lower_bound": max(0, result - self.epsilon),
                "upper_bound": min(1, result + self.epsilon),
                "confidence": self.confidence,
                "epsilon": self.epsilon,
                "delta": self.delta,
            }
        else:
            # For complex results, wrap with metadata
            return {
                "value": result,
                "confidence": self.confidence,
                "epsilon": self.epsilon,
                "delta": self.delta,
            }

    def get_confidence(self, result: Any) -> float:
        """Return PAC confidence level."""
        return self.confidence

    def is_reliable(self, result: Any) -> bool:
        """Check if confidence meets threshold (>= 0.9)."""
        return self.confidence >= 0.9

    def combine_pac_bounds(
        self, error1: "PACErrorModel", error2: "PACErrorModel", operation: str
    ) -> "PACErrorModel":
        """
        Combine PAC learning error bounds for two results.

        Args:
            error1, error2: PAC error models to combine
            operation: Type of operation ('addition', 'multiplication', etc.)

        Returns:
            Combined PAC error model
        """
        # Union bound for probability
        combined_delta = min(1.0, error1.delta + error2.delta)

        # Error combination depends on operation
        if operation == "addition":
            combined_epsilon = error1.epsilon + error2.epsilon
        elif operation == "multiplication":
            combined_epsilon = error1.epsilon * error2.epsilon + error1.epsilon + error2.epsilon
        else:
            # Conservative bound
            combined_epsilon = min(0.5, error1.epsilon + error2.epsilon)

        return PACErrorModel(combined_epsilon, combined_delta)

    def __repr__(self) -> str:
        return f"PACErrorModel(epsilon={self.epsilon}, delta={self.delta})"


class NoiseErrorModel(ErrorModel):
    """
    Noise error model for handling bit-flip and measurement errors.

    Simulates realistic noise in Boolean function evaluation and analysis.
    """

    def __init__(self, noise_rate: float = 0.01, random_seed: Optional[int] = None):
        """
        Initialize noise error model.

        Args:
            noise_rate: Probability of bit flip (0 <= noise_rate <= 0.5)
            random_seed: Random seed for reproducible noise
        """
        if not (0 <= noise_rate <= 0.5):
            raise ValueError("Noise rate must be in [0, 0.5]")

        self.noise_rate = noise_rate
        self.rng = np.random.default_rng(random_seed)
        self.reliability = 1 - 2 * noise_rate  # Reliability decreases with noise

    def apply_error(self, result: Union[bool, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Apply bit-flip noise to Boolean results.

        Args:
            result: Boolean result(s)

        Returns:
            Result(s) with noise applied
        """
        if self.noise_rate == 0:
            return result

        if isinstance(result, bool):
            # Single boolean
            if self.rng.random() < self.noise_rate:
                return not result
            return result
        elif isinstance(result, np.ndarray):
            # Array of booleans
            noise_mask = self.rng.random(result.shape) < self.noise_rate
            noisy_result = result.copy()
            noisy_result[noise_mask] = ~noisy_result[noise_mask]
            return noisy_result
        else:
            # For other types, add noise metadata
            return {  # type: ignore[unreachable]
                "value": result,
                "noise_applied": True,
                "noise_rate": self.noise_rate,
                "reliability": self.reliability,
            }

    def get_confidence(self, result: Any) -> float:
        """Return confidence based on noise level."""
        return max(0.5, self.reliability)  # At least 50% confidence

    def is_reliable(self, result: Any) -> bool:
        """Check if noise level allows reliable results."""
        return self.noise_rate <= 0.1  # Reliable if noise <= 10%

    def __repr__(self) -> str:
        return f"NoiseErrorModel(noise_rate={self.noise_rate})"


class LinearErrorModel(ErrorModel):
    """
    Linear error propagation model using the ``uncertainties`` library.

    Provides automatic-differentiation-based error propagation.

    Args:
        std_dev: Default standard deviation used by :meth:`apply_error`
                 when no per-call value is given.
    """

    def __init__(self, std_dev: float = 0.01):
        """Initialize linear error model."""
        if not HAS_UNCERTAINTIES:
            raise ImportError(
                "uncertainties library required for LinearErrorModel. "
                "Install it with: pip install uncertainties"
            )
        self.std_dev = std_dev

    def apply_error(self, result: Any, **kwargs: Any) -> Any:
        """
        Apply linear error propagation.

        Args:
            result: Computation result.
            **kwargs: Optional ``std_dev`` override for this call.

        Returns:
            Result with uncertainty information.
        """
        sd = kwargs.get("std_dev", self.std_dev)
        if isinstance(result, (int, float)):
            return ufloat(result, sd)
        elif isinstance(result, np.ndarray):
            return unp.uarray(result, np.full_like(result, sd))
        else:
            return result

    def propagate_binary_op(self, left_error: Any, right_error: Any, operation: Callable) -> Any:
        """
        Automatic error propagation for binary operations.

        Args:
            left_error: Left operand with uncertainty
            right_error: Right operand with uncertainty
            operation: Operation function

        Returns:
            Result with propagated uncertainty
        """
        if HAS_UNCERTAINTIES:
            return operation(left_error, right_error)
        else:
            # Fallback without uncertainty propagation
            return operation(left_error, right_error)

    def get_confidence(self, result: Any) -> float:
        """Return confidence based on relative error."""
        if HAS_UNCERTAINTIES and hasattr(result, "std_dev"):
            if result.nominal_value != 0:
                relative_error = result.std_dev / abs(result.nominal_value)
                return max(0.5, 1 - relative_error)
        return 0.9  # Default confidence

    def is_reliable(self, result: Any) -> bool:
        """Check if relative error is acceptable."""
        if HAS_UNCERTAINTIES and hasattr(result, "std_dev"):
            if result.nominal_value != 0:
                relative_error = result.std_dev / abs(result.nominal_value)
                return relative_error <= 0.1  # 10% relative error threshold
        return True

    def __repr__(self) -> str:
        return "LinearErrorModel()"


# Factory function for creating error models
def create_error_model(model_type: str, **kwargs) -> ErrorModel:
    """
    Factory function to create error models.

    Args:
        model_type: Type of error model ('exact', 'pac', 'noise', 'linear')
        **kwargs: Model-specific parameters

    Returns:
        Configured error model
    """
    if model_type.lower() == "exact":
        return ExactErrorModel()
    elif model_type.lower() == "pac":
        return PACErrorModel(**kwargs)
    elif model_type.lower() == "noise":
        return NoiseErrorModel(**kwargs)
    elif model_type.lower() == "linear":
        return LinearErrorModel(**kwargs)
    else:
        raise ValueError(f"Unknown error model type: {model_type}")


# Export main classes
__all__ = [
    "ErrorModel",
    "ExactErrorModel",
    "PACErrorModel",
    "NoiseErrorModel",
    "LinearErrorModel",
    "create_error_model",
]
