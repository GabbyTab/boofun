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
        """
        Generate tribes function.
        
        Args:
            n: Total number of variables
            k: Number of tribes (optional, auto-computed if not provided)
            w: Width of each tribe (optional, auto-computed if not provided)
            
        Note: bf.tribes(tribe_size, total_vars) so we call bf.tribes(w, n)
        """
        import boolfunc as bf
        
        if k is not None and w is not None:
            # User specified k tribes of width w, total vars = k * w
            total_vars = k * w
            return bf.tribes(w, total_vars)
        
        # Default: standard tribes with n variables
        # Choose w ≈ log2(n) - log2(log2(n)) for balance
        if n < 4:
            return bf.AND(n)
        
        log_n = np.log2(n)
        w = max(2, int(log_n - np.log2(max(1, log_n))))
        # Make sure n is divisible by w
        actual_n = (n // w) * w
        if actual_n < w:
            actual_n = w
        
        return bf.tribes(w, actual_n)


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


class IteratedMajorityFamily(FunctionFamily):
    """
    Iterated Majority function family.
    
    ITER_MAJ builds majority functions in layers:
    - Layer 0: n input variables
    - Layer i: majority of groups from layer i-1
    - Continue until single output
    
    Different from RecursiveMajority3 which specifically uses groups of 3.
    This allows general group sizes.
    """
    
    def __init__(self, group_size: int = 3):
        """
        Initialize iterated majority.
        
        Args:
            group_size: Size of each majority group (must be odd)
        """
        if group_size % 2 == 0:
            raise ValueError("Group size must be odd")
        self._group_size = group_size
    
    @property
    def metadata(self) -> FamilyMetadata:
        k = self._group_size
        return FamilyMetadata(
            name=f"IteratedMajority{k}",
            description=f"Iterated majority with groups of {k}",
            parameters={"group_size": str(k)},
            asymptotics={
                "total_influence": lambda n: n ** (np.log((k+1)/2) / np.log(k)),
                "depth": lambda n: int(np.log(n) / np.log(k)) + 1,
            },
            universal_properties=["monotone", "balanced"],
            n_constraints=lambda n: self._is_valid_n(n),
            n_constraint_description=f"n must be {k}^depth for integer depth",
        )
    
    def _is_valid_n(self, n: int) -> bool:
        """Check if n is a power of group_size."""
        if n < self._group_size:
            return n == 1
        k = self._group_size
        while n > 1:
            if n % k != 0:
                return False
            n //= k
        return True
    
    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        
        k = self._group_size
        
        def iterated_maj(bits: tuple) -> int:
            if len(bits) <= k:
                return int(sum(bits) >= (len(bits) + 1) // 2)
            
            # Group and compute majority of each group
            num_groups = len(bits) // k
            new_bits = tuple(
                int(sum(bits[i*k:(i+1)*k]) >= (k + 1) // 2)
                for i in range(num_groups)
            )
            return iterated_maj(new_bits)
        
        # Adjust n to be valid if needed
        depth = max(1, int(np.ceil(np.log(n) / np.log(k))))
        actual_n = k ** depth
        
        truth_table = []
        for x in range(2**actual_n):
            bits = tuple((x >> i) & 1 for i in range(actual_n))
            truth_table.append(iterated_maj(bits))
        
        return bf.create(truth_table)
    
    def validate_n(self, n: int) -> bool:
        return self._is_valid_n(n)


class RandomDNFFamily(FunctionFamily):
    """
    Random DNF (Disjunctive Normal Form) function family.
    
    Generates random k-DNF functions with m terms.
    
    A k-DNF is an OR of terms, where each term is an AND of at most k literals.
    """
    
    def __init__(self, term_width: int = 3, num_terms: Optional[int] = None):
        """
        Initialize random DNF family.
        
        Args:
            term_width: Maximum number of literals per term (k)
            num_terms: Number of terms (default: 2^{n/2})
        """
        self._term_width = term_width
        self._num_terms = num_terms
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name=f"RandomDNF_{self._term_width}",
            description=f"Random {self._term_width}-DNF",
            parameters={
                "term_width": str(self._term_width),
                "num_terms": str(self._num_terms or "auto")
            },
            asymptotics={
                # Random k-DNF properties depend on parameters
            },
            universal_properties=[],  # Properties vary by instance
        )
    
    def generate(self, n: int, seed: Optional[int] = None, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        
        rng = np.random.RandomState(seed)
        
        k = kwargs.get('term_width', self._term_width)
        m = kwargs.get('num_terms', self._num_terms)
        if m is None:
            m = max(1, 2 ** (n // 2))
        
        # Generate random terms
        terms = []
        for _ in range(m):
            # Choose term width
            width = rng.randint(1, min(k, n) + 1)
            # Choose variables
            vars_in_term = rng.choice(n, size=width, replace=False)
            # Choose polarities (True = positive literal)
            polarities = rng.randint(0, 2, size=width).astype(bool)
            terms.append(list(zip(vars_in_term, polarities)))
        
        # Build truth table
        def evaluate_dnf(x: int) -> bool:
            bits = [(x >> i) & 1 for i in range(n)]
            for term in terms:
                # Check if term is satisfied
                satisfied = True
                for var, polarity in term:
                    if polarity:  # Positive literal
                        if not bits[var]:
                            satisfied = False
                            break
                    else:  # Negative literal
                        if bits[var]:
                            satisfied = False
                            break
                if satisfied:
                    return True
            return False
        
        truth_table = [int(evaluate_dnf(x)) for x in range(2**n)]
        return bf.create(truth_table)


class SboxFamily(FunctionFamily):
    """
    Cryptographic S-box component function family.
    
    S-boxes are nonlinear components in block ciphers.
    This family provides access to standard S-boxes and their
    component Boolean functions.
    """
    
    # AES S-box (first 32 values, full box has 256)
    AES_SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
        0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
        0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
        0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
        0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
        0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
        0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
        0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
        0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
        0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
        0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
        0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
        0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ]
    
    def __init__(self, sbox: Optional[List[int]] = None, bit: int = 0):
        """
        Initialize S-box family.
        
        Args:
            sbox: Custom S-box (default: AES S-box)
            bit: Output bit to extract (0-7 for 8-bit S-box)
        """
        self._sbox = sbox or self.AES_SBOX
        self._bit = bit
        self._n_bits = int(np.log2(len(self._sbox)))
    
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            name=f"Sbox_bit{self._bit}",
            description=f"S-box component function (bit {self._bit})",
            parameters={"bit": str(self._bit), "sbox_size": str(len(self._sbox))},
            asymptotics={
                "nonlinearity": lambda n: 2**(n-1) - 2**(n//2-1),  # Optimal for bent-ish
            },
            universal_properties=["balanced"],
            n_constraints=lambda n: n == self._n_bits,
            n_constraint_description=f"n must equal S-box input bits ({self._n_bits})",
        )
    
    def generate(self, n: int = None, **kwargs) -> "BooleanFunction":
        import boolfunc as bf
        
        bit = kwargs.get('bit', self._bit)
        
        # Extract component function
        truth_table = [(self._sbox[x] >> bit) & 1 for x in range(len(self._sbox))]
        
        return bf.create(truth_table)
    
    def get_component(self, bit: int) -> "BooleanFunction":
        """Get specific bit component."""
        import boolfunc as bf
        truth_table = [(self._sbox[x] >> bit) & 1 for x in range(len(self._sbox))]
        return bf.create(truth_table)
    
    def all_components(self) -> List["BooleanFunction"]:
        """Get all component functions."""
        return [self.get_component(b) for b in range(self._n_bits)]
    
    @classmethod
    def aes(cls, bit: int = 0) -> "SboxFamily":
        """Create AES S-box family."""
        return cls(cls.AES_SBOX, bit)
    
    def validate_n(self, n: int) -> bool:
        return n == self._n_bits


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
    "IteratedMajorityFamily",
    "RandomDNFFamily",
    "SboxFamily",
]
