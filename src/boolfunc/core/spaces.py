from enum import Enum, auto
from typing import Union
import numpy as np


class Space(Enum):
    BOOLEAN_CUBE = auto()  # {0,1}^n
    PLUS_MINUS_CUBE = auto()  # {-1,1}^n
    REAL = auto()  # ℝ^n
    LOG = auto()  # Log space
    GAUSSIAN = auto()  # Gaussian space

    @staticmethod
    def translate(
        input: Union[int, float, np.ndarray],
        source_space: "Space",
        target_space: "Space",
    ) -> Union[int, float, np.ndarray]:
        """
        Translate a scalar or array from one space to another.
        
        Args:
            input: Input value(s) to translate
            source_space: Source mathematical space
            target_space: Target mathematical space
            
        Returns:
            Translated value(s) in target space
            
        Examples:
            >>> Space.translate([0, 1], Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)
            array([-1,  1])
            >>> Space.translate([-1, 1], Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE) 
            array([0, 1])
        """
        if source_space == target_space:
            return input

        input = np.asarray(input)

        # Boolean (0/1) → ±1
        if source_space == Space.BOOLEAN_CUBE and target_space == Space.PLUS_MINUS_CUBE:
            return 2 * input - 1

        # ±1 → Boolean (0/1)
        if source_space == Space.PLUS_MINUS_CUBE and target_space == Space.BOOLEAN_CUBE:
            return ((input + 1) // 2).astype(int)

        # Real-valued input → Boolean (mod 2)
        if source_space == Space.REAL and target_space == Space.BOOLEAN_CUBE:
            return (np.round(input) % 2).astype(int)

        # Boolean → Real (just cast)
        if source_space == Space.BOOLEAN_CUBE and target_space == Space.REAL:
            return input.astype(float)

        # ±1 → Real
        if source_space == Space.PLUS_MINUS_CUBE and target_space == Space.REAL:
            return input.astype(float)

        # Real → ±1 (e.g., sign function)
        if source_space == Space.REAL and target_space == Space.PLUS_MINUS_CUBE:
            return np.where(input >= 0, 1, -1)

        # LOG space conversions
        if source_space == Space.LOG and target_space == Space.BOOLEAN_CUBE:
            # Interpret log probabilities: exp(log_prob) > 0.5
            return (np.exp(input) > 0.5).astype(int)
        
        if source_space == Space.BOOLEAN_CUBE and target_space == Space.LOG:
            # Convert to log probabilities (avoid log(0))
            prob = np.clip(input.astype(float), 1e-10, 1 - 1e-10)
            return np.log(prob)

        # GAUSSIAN space conversions  
        if source_space == Space.GAUSSIAN and target_space == Space.BOOLEAN_CUBE:
            # Standard normal CDF > 0.5
            from scipy.stats import norm
            return (norm.cdf(input) > 0.5).astype(int)
        
        if source_space == Space.BOOLEAN_CUBE and target_space == Space.GAUSSIAN:
            # Inverse normal CDF
            from scipy.stats import norm
            prob = np.clip(input.astype(float), 1e-10, 1 - 1e-10)
            return norm.ppf(prob)

        # Cross-conversions through intermediate spaces
        if source_space == Space.LOG and target_space == Space.PLUS_MINUS_CUBE:
            # LOG → BOOLEAN → ±1
            boolean = Space.translate(input, Space.LOG, Space.BOOLEAN_CUBE)
            return Space.translate(boolean, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)
        
        if source_space == Space.PLUS_MINUS_CUBE and target_space == Space.LOG:
            # ±1 → BOOLEAN → LOG
            boolean = Space.translate(input, Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE)
            return Space.translate(boolean, Space.BOOLEAN_CUBE, Space.LOG)

        raise NotImplementedError(
            f"Translation from {source_space.name} to {target_space.name} not implemented."
        )
    
    @staticmethod
    def get_canonical_space() -> "Space":
        """Get the canonical space for internal computations."""
        return Space.BOOLEAN_CUBE
    
    @staticmethod
    def is_discrete(space: "Space") -> bool:
        """Check if space uses discrete values."""
        return space in {Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE}
    
    @staticmethod
    def is_continuous(space: "Space") -> bool:
        """Check if space uses continuous values.""" 
        return space in {Space.REAL, Space.LOG, Space.GAUSSIAN}
    
    @staticmethod
    def get_default_threshold(space: "Space") -> float:
        """Get default threshold for converting continuous to discrete."""
        if space == Space.REAL:
            return 0.5
        elif space == Space.LOG:
            return 0.0  # log(0.5) ≈ -0.693, but 0.0 is simpler
        elif space == Space.GAUSSIAN:
            return 0.0  # Standard normal median
        else:
            return 0.5
