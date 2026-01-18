"""
Built-in function families with known asymptotic behavior.

Each family has:
- Generation for any valid n
- Known theoretical formulas for key properties
- Universal properties that always hold
"""

import numpy as np
from typing import Optional, List, Callable, TYPE_CHECKING
from .base import FunctionFamily, FamilyMetadata, WeightPatternFamily

if TYPE_CHECKING:
    from ..core.base import BooleanFunction


class MajorityFamily(FunctionFamily):
    """
    Majority function family: MAJ_n(x) = 1 if Σx_i > n/2.
    
    Well-known asymptotics:
    - Total influence: I[MAJ_n] ≈ √(2/π) · √n ≈ 0.798√n
    - Each influence: Inf_i[MAJ_n] ≈ √(2/(πn))
    - Noise stability: Stab_ρ[MAJ_n] → (1/2) + (1/π)arcsin(ρ)
    - Fourier degree: n (but most weight on lower degrees)
    """
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="Majority",
            description="MAJ_n(x) = 1 if Σx_i > n/2, else 0",
            parameters={},
            asymptotics={
                "total_influence": lambda n: np.sqrt(2/np.pi) * np.sqrt(n),
                "influence_i": lambda n, i=0: np.sqrt(2/(np.pi * n)),
                "noise_stability": lambda n, rho=0.5: 0.5 + (1/np.pi) * np.arcsin(rho),
                "fourier_degree": lambda n: n,
                "regularity": lambda n: 1/np.sqrt(n),  # τ → 0 as n → ∞
            },
            universal_properties=["monotone", "symmetric", "balanced", "is_ltf"],
            n_constraints=lambda n: n % 2 == 1,
            n_constraint_description="n must be odd for unambiguous majority",
        )
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        return bf.majority(n)
    
    def validate_n(self, n: int) -> bool:
        return n >= 1 and n % 2 == 1


class ParityFamily(FunctionFamily):
    """
    Parity/XOR function family: XOR_n(x) = Σx_i mod 2.
    
    Parity is the "opposite" of Majority in many ways:
    - NOT an LTF (not linearly separable)
    - All Fourier weight on a single coefficient
    - Maximum noise sensitivity
    
    Asymptotics:
    - Total influence: I[XOR_n] = n (each variable is pivotal always)
    - Noise stability: Stab_ρ[XOR_n] = ρ^n → 0 for ρ < 1
    - Fourier: f̂(S) = 1 only for S = [n], else 0
    """
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="Parity",
            description="XOR_n(x) = x_1 ⊕ x_2 ⊕ ... ⊕ x_n",
            parameters={},
            asymptotics={
                "total_influence": lambda n: float(n),
                "influence_i": lambda n, i=0: 1.0,
                "noise_stability": lambda n, rho=0.5: rho**n,
                "fourier_degree": lambda n: n,
                "fourier_sparsity": lambda n: 1,  # Only one non-zero coeff
            },
            universal_properties=["linear", "balanced", "symmetric"],
        )
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        return bf.parity(n)


class TribesFamily(FunctionFamily):
    """
    Tribes function family: Balanced DNF with k tribes of size s.
    
    Standard choice: s = log(n) - log(log(n)) for k = n/s tribes.
    This achieves Pr[TRIBES = 1] ≈ 1/2.
    
    Asymptotics:
    - Total influence: I[TRIBES] ≈ log(n)/n · n = log(n)
    - Each influence: Inf_i[TRIBES] ≈ log(n)/n  
    - Noise stability: Complex, depends on parameters
    """
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="Tribes",
            description="TRIBES_{w,k}(x) = OR of k ANDs of width w",
            parameters={"k": "number of tribes", "w": "width of each tribe"},
            asymptotics={
                "total_influence": lambda n, k=None, w=None: np.log(n) if k is None else k * w * (1/2)**(w-1),
                "influence_i": lambda n, k=None, w=None: np.log(n)/n if k is None else (1/2)**(w-1),
            },
            universal_properties=["monotone", "balanced"],
        )
    
    def generate(self, n: int, k: Optional[int] = None, w: Optional[int] = None, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        
        if k is not None and w is not None:
            return bf.tribes(k, w)
        
        # Default: standard tribes with n variables
        # Choose w ≈ log2(n) - log2(log2(n)) for balance
        if n < 4:
            return bf.AND(n)
        
        log_n = np.log2(n)
        w = max(2, int(log_n - np.log2(max(1, log_n))))
        k = n // w
        
        return bf.tribes(k, w)


class ThresholdFamily(FunctionFamily):
    """
    Threshold function family: THR_k(x) = 1 if Σx_i ≥ k.
    
    This is a symmetric LTF (uniform weights).
    
    Special cases:
    - k = 1: OR function
    - k = n: AND function
    - k = (n+1)/2: Majority (for odd n)
    """
    
    def __init__(self, k_function: Optional[Callable[[int], int]] = None):
        """
        Initialize threshold family.
        
        Args:
            k_function: Function n -> k specifying threshold for each n.
                       Default: k = n//2 + 1 (majority-like)
        """
        self._k_fn = k_function or (lambda n: n // 2 + 1)
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="Threshold",
            description="THR_k(x) = 1 if Σx_i ≥ k",
            parameters={"k": "threshold value"},
            universal_properties=["monotone", "symmetric", "is_ltf"],
        )
    
    def generate(self, n: int, k: Optional[int] = None, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        
        if k is None:
            k = self._k_fn(n)
        
        return bf.threshold(n, k)


class ANDFamily(FunctionFamily):
    """
    AND function family: AND_n(x) = 1 iff all x_i = 1.
    
    Extreme threshold function (k = n).
    
    Asymptotics:
    - Total influence: I[AND_n] = n · 2^{-(n-1)} → 0
    - Each influence: Inf_i[AND_n] = 2^{-(n-1)}
    - Pr[AND_n = 1] = 2^{-n}
    """
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="AND",
            description="AND_n(x) = x_1 ∧ x_2 ∧ ... ∧ x_n",
            parameters={},
            asymptotics={
                "total_influence": lambda n: n * 2**(-(n-1)),
                "influence_i": lambda n, i=0: 2**(-(n-1)),
                "expectation": lambda n: 2**(-n),  # Pr[AND = 1]
            },
            universal_properties=["monotone", "symmetric", "is_ltf"],
        )
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        return bf.AND(n)


class ORFamily(FunctionFamily):
    """
    OR function family: OR_n(x) = 1 iff at least one x_i = 1.
    
    Extreme threshold function (k = 1).
    Dual of AND: OR_n(x) = ¬AND_n(¬x).
    
    Asymptotics:
    - Total influence: I[OR_n] = n · 2^{-(n-1)} → 0
    - Each influence: Inf_i[OR_n] = 2^{-(n-1)}
    - Pr[OR_n = 1] = 1 - 2^{-n}
    """
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="OR",
            description="OR_n(x) = x_1 ∨ x_2 ∨ ... ∨ x_n",
            parameters={},
            asymptotics={
                "total_influence": lambda n: n * 2**(-(n-1)),
                "influence_i": lambda n, i=0: 2**(-(n-1)),
                "expectation": lambda n: 1 - 2**(-n),
            },
            universal_properties=["monotone", "symmetric", "is_ltf"],
        )
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        return bf.OR(n)


class DictatorFamily(FunctionFamily):
    """
    Dictator function family: DICT_i(x) = x_i.
    
    The "simplest" Boolean function - just returns one variable.
    
    Asymptotics:
    - Total influence: I[DICT] = 1 (only one influential variable)
    - Inf_i[DICT] = 1, Inf_j[DICT] = 0 for j ≠ i
    - Noise stability: Stab_ρ[DICT] = ρ
    """
    
    def __init__(self, variable: int = 0):
        """
        Initialize dictator family.
        
        Args:
            variable: Which variable to use (default: 0)
        """
        self._variable = variable
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="Dictator",
            description=f"DICT_i(x) = x_{self._variable}",
            parameters={"variable": str(self._variable)},
            asymptotics={
                "total_influence": lambda n: 1.0,
                "influence_i": lambda n, i=0: 1.0 if i == self._variable else 0.0,
                "noise_stability": lambda n, rho=0.5: rho,
                "fourier_degree": lambda n: 1,
            },
            universal_properties=["is_ltf", "is_junta"],
        )
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        var = kwargs.get("variable", self._variable)
        return bf.dictator(n, var)


class LTFFamily(WeightPatternFamily):
    """
    General LTF (Linear Threshold Function) family.
    
    LTF_w(x) = sign(w₁x₁ + ... + wₙxₙ - θ)
    
    This is a more flexible version of WeightPatternFamily
    with additional convenience methods.
    """
    
    def __init__(
        self,
        weight_pattern: Callable[[int, int], float] = lambda i, n: 1.0,
        threshold_pattern: Optional[Callable[[int], float]] = None,
        name: str = "LTF",
    ):
        """
        Initialize LTF family.
        
        Args:
            weight_pattern: Function (i, n) -> weight of variable i
            threshold_pattern: Function n -> threshold
            name: Family name
        """
        super().__init__(weight_pattern, threshold_pattern, name)
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name=self._name,
            description="LTF with custom weight pattern",
            parameters={"weight_pattern": "custom"},
            asymptotics={
                # General LTF asymptotics depend on weights
                "total_influence": self._estimate_total_influence,
                "regularity": self._compute_regularity,
            },
            universal_properties=["is_ltf"],
        )
    
    def _estimate_total_influence(self, n: int) -> float:
        """Estimate total influence using CLT formula."""
        weights = self.get_weights(n)
        tau = np.max(np.abs(weights)) / np.linalg.norm(weights)
        
        # For regular LTFs: I[f] ≈ √(2/π) · √n
        # Adjust for irregularity
        regular_estimate = np.sqrt(2/np.pi) * np.sqrt(n)
        return regular_estimate * (1 - tau) + tau  # Interpolate to dictator
    
    def _compute_regularity(self, n: int) -> float:
        """Compute regularity parameter τ."""
        weights = self.get_weights(n)
        norm = np.linalg.norm(weights)
        if norm < 1e-10:
            return 0.0
        return np.max(np.abs(weights)) / norm
    
    @classmethod
    def uniform(cls, name: str = "UniformLTF") -> "LTFFamily":
        """Create LTF with uniform weights (= Majority)."""
        return cls(lambda i, n: 1.0, name=name)
    
    @classmethod
    def geometric(cls, ratio: float = 0.5, name: str = "GeometricLTF") -> "LTFFamily":
        """Create LTF with geometrically decaying weights."""
        return cls(lambda i, n: ratio**i, name=name)
    
    @classmethod
    def harmonic(cls, name: str = "HarmonicLTF") -> "LTFFamily":
        """Create LTF with harmonic weights 1/(i+1)."""
        return cls(lambda i, n: 1.0/(i+1), name=name)
    
    @classmethod  
    def power_law(cls, power: float = 2.0, name: str = "PowerLTF") -> "LTFFamily":
        """Create LTF with power-law weights."""
        return cls(lambda i, n: (n-i)**power if n > i else 1.0, name=name)


class RecursiveMajority3Family(FunctionFamily):
    """
    Recursive Majority of 3 function family.
    
    REC_MAJ3 on n = 3^k variables is defined recursively:
    - Base case: n=3 is MAJ_3
    - Recursive: REC_MAJ3(x) = MAJ_3(REC_MAJ3(x[0:m]), REC_MAJ3(x[m:2m]), REC_MAJ3(x[2m:3m]))
    
    This is a key function in complexity theory, with interesting properties:
    - Total influence: I[REC_MAJ3] = Θ(n^(log_3(2))) ≈ n^0.631
    - More "noise sensitive" than flat majority
    - Used in lower bounds for branching programs
    
    Note: Only defined for n = 3^k (k ≥ 1).
    """
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name="RecursiveMajority3",
            description="REC_MAJ3_n = MAJ_3(REC_MAJ3, REC_MAJ3, REC_MAJ3)",
            parameters={},
            asymptotics={
                "total_influence": lambda n: n ** (np.log(2) / np.log(3)),  # n^0.631
                "influence_max": lambda n: (2/3) ** int(np.log(n) / np.log(3)),
                "noise_stability": lambda n, rho=0.5: self._noise_stability_approx(n, rho),
            },
            universal_properties=["monotone", "balanced"],
            n_constraints=lambda n: self._is_power_of_3(n),
            n_constraint_description="n must be a power of 3 (3, 9, 27, 81, ...)",
        )
    
    @staticmethod
    def _is_power_of_3(n: int) -> bool:
        """Check if n is a power of 3."""
        if n < 3:
            return False
        while n > 1:
            if n % 3 != 0:
                return False
            n //= 3
        return True
    
    @staticmethod
    def _noise_stability_approx(n: int, rho: float) -> float:
        """Approximate noise stability for recursive majority."""
        # Recursive formula: Stab_ρ[REC_MAJ3_n] ≈ 3ρ(Stab_ρ[REC_MAJ3_{n/3}])² - 2(Stab_ρ[...])³
        # For simplicity, use known asymptotic
        k = int(np.log(n) / np.log(3))
        # Converges to fixed point of 3ρx² - 2x³ = x
        # For ρ = 0.5, this gives ≈ 0.5
        return 0.5 + 0.3 * rho * np.exp(-0.5 * k)
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        """Generate recursive majority of 3 function."""
        import boolfunc as bf
        
        if not self._is_power_of_3(n):
            raise ValueError(f"n must be a power of 3, got {n}")
        
        # Build truth table recursively
        def rec_maj3(bits: tuple) -> int:
            """Recursive majority on a tuple of bits."""
            if len(bits) == 3:
                return int(sum(bits) >= 2)
            m = len(bits) // 3
            return int(sum([
                rec_maj3(bits[:m]),
                rec_maj3(bits[m:2*m]),
                rec_maj3(bits[2*m:])
            ]) >= 2)
        
        # Generate truth table
        truth_table = []
        for x in range(2**n):
            bits = tuple((x >> i) & 1 for i in range(n))
            truth_table.append(rec_maj3(bits))
        
        return bf.create(truth_table)
    
    def theoretical_properties(self, n: int) -> dict:
        """Return known theoretical properties."""
        k = int(np.log(n) / np.log(3))
        return {
            "depth": k,
            "total_influence_theory": n ** (np.log(2) / np.log(3)),
            "max_influence_theory": (2/3) ** k,
            "is_balanced": True,
            "is_monotone": True,
        }


__all__ = [
    "MajorityFamily",
    "ParityFamily",
    "TribesFamily",
    "ThresholdFamily",
    "ANDFamily",
    "ORFamily",
    "DictatorFamily",
    "LTFFamily",
    "RecursiveMajority3Family",
]
