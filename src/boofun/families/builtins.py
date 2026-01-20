"""
Built-in function families with known asymptotic behavior.

Each family has:
- Generation for any valid n
- Known theoretical formulas for key properties
- Universal properties that always hold
"""

from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from .base import FamilyMetadata, FunctionFamily, WeightPatternFamily

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
                "total_influence": lambda n: np.sqrt(2 / np.pi) * np.sqrt(n),
                "influence_i": lambda n, i=0: np.sqrt(2 / (np.pi * n)),
                "noise_stability": lambda n, rho=0.5: 0.5 + (1 / np.pi) * np.arcsin(rho),
                "fourier_degree": lambda n: n,
                "regularity": lambda n: 1 / np.sqrt(n),  # τ → 0 as n → ∞
            },
            universal_properties=["monotone", "symmetric", "balanced", "is_ltf"],
            n_constraints=lambda n: n % 2 == 1,
            n_constraint_description="n must be odd for unambiguous majority",
        )

    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boofun as bf

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
        import boofun as bf

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
                "total_influence": lambda n, k=None, w=None: (
                    np.log(n) if k is None else k * w * (1 / 2) ** (w - 1)
                ),
                "influence_i": lambda n, k=None, w=None: (
                    np.log(n) / n if k is None else (1 / 2) ** (w - 1)
                ),
            },
            universal_properties=["monotone", "balanced"],
        )

    def generate(
        self, n: int, k: Optional[int] = None, w: Optional[int] = None, **kwargs
    ) -> "BooleanFunction":
        """
        Generate tribes function.

        Args:
            n: Total number of variables
            k: Number of tribes (optional, auto-computed if not provided)
            w: Width of each tribe (optional, auto-computed if not provided)

        Note: bf.tribes(tribe_size, total_vars) so we call bf.tribes(w, n)
        """
        import boofun as bf

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
        import boofun as bf

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
                "total_influence": lambda n: n * 2 ** (-(n - 1)),
                "influence_i": lambda n, i=0: 2 ** (-(n - 1)),
                "expectation": lambda n: 2 ** (-n),  # Pr[AND = 1]
            },
            universal_properties=["monotone", "symmetric", "is_ltf"],
        )

    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boofun as bf

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
                "total_influence": lambda n: n * 2 ** (-(n - 1)),
                "influence_i": lambda n, i=0: 2 ** (-(n - 1)),
                "expectation": lambda n: 1 - 2 ** (-n),
            },
            universal_properties=["monotone", "symmetric", "is_ltf"],
        )

    def generate(self, n: int, **kwargs) -> "BooleanFunction":
        import boofun as bf

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
        import boofun as bf

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
        regular_estimate = np.sqrt(2 / np.pi) * np.sqrt(n)
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
        return cls(lambda i, n: 1.0 / (i + 1), name=name)

    @classmethod
    def power_law(cls, power: float = 2.0, name: str = "PowerLTF") -> "LTFFamily":
        """Create LTF with power-law weights."""
        return cls(lambda i, n: (n - i) ** power if n > i else 1.0, name=name)


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
                "influence_max": lambda n: (2 / 3) ** int(np.log(n) / np.log(3)),
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
        import boofun as bf

        if not self._is_power_of_3(n):
            raise ValueError(f"n must be a power of 3, got {n}")

        # Build truth table recursively
        def rec_maj3(bits: tuple) -> int:
            """Recursive majority on a tuple of bits."""
            if len(bits) == 3:
                return int(sum(bits) >= 2)
            m = len(bits) // 3
            return int(
                sum([rec_maj3(bits[:m]), rec_maj3(bits[m : 2 * m]), rec_maj3(bits[2 * m :])]) >= 2
            )

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
            "max_influence_theory": (2 / 3) ** k,
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
                "total_influence": lambda n: n ** (np.log((k + 1) / 2) / np.log(k)),
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
        import boofun as bf

        k = self._group_size

        def iterated_maj(bits: tuple) -> int:
            if len(bits) <= k:
                return int(sum(bits) >= (len(bits) + 1) // 2)

            # Group and compute majority of each group
            num_groups = len(bits) // k
            new_bits = tuple(
                int(sum(bits[i * k : (i + 1) * k]) >= (k + 1) // 2) for i in range(num_groups)
            )
            return iterated_maj(new_bits)

        # Adjust n to be valid if needed
        depth = max(1, int(np.ceil(np.log(n) / np.log(k))))
        actual_n = k**depth

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
                "num_terms": str(self._num_terms or "auto"),
            },
            asymptotics={
                # Random k-DNF properties depend on parameters
            },
            universal_properties=[],  # Properties vary by instance
        )

    def generate(self, n: int, seed: Optional[int] = None, **kwargs) -> "BooleanFunction":
        import boofun as bf

        rng = np.random.RandomState(seed)

        k = kwargs.get("term_width", self._term_width)
        m = kwargs.get("num_terms", self._num_terms)
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
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
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
                "nonlinearity": lambda n: 2 ** (n - 1) - 2 ** (n // 2 - 1),  # Optimal for bent-ish
            },
            universal_properties=["balanced"],
            n_constraints=lambda n: n == self._n_bits,
            n_constraint_description=f"n must equal S-box input bits ({self._n_bits})",
        )

    def generate(self, n: int = None, **kwargs) -> "BooleanFunction":
        import boofun as bf

        bit = kwargs.get("bit", self._bit)

        # Extract component function
        truth_table = [(self._sbox[x] >> bit) & 1 for x in range(len(self._sbox))]

        return bf.create(truth_table)

    def get_component(self, bit: int) -> "BooleanFunction":
        """Get specific bit component."""
        import boofun as bf

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
