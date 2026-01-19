"""
Theoretical bounds and asymptotic formulas for Boolean function properties.

This module provides known theoretical results that can be compared
against computed values to validate understanding and implementation.
"""

import numpy as np
from typing import Callable, Dict, Optional


class TheoreticalBounds:
    """
    Collection of theoretical bounds and asymptotic formulas.
    
    These are from O'Donnell's book and other sources.
    """
    
    # ==========================================================
    # Total Influence Bounds
    # ==========================================================
    
    @staticmethod
    def majority_total_influence(n: int) -> float:
        """
        Total influence of Majority_n.
        
        I[MAJ_n] = √(2/π) · √n + O(1/√n)
        """
        return np.sqrt(2 / np.pi) * np.sqrt(n)
    
    @staticmethod
    def parity_total_influence(n: int) -> float:
        """Total influence of Parity_n = n."""
        return float(n)
    
    @staticmethod
    def and_total_influence(n: int) -> float:
        """Total influence of AND_n = n · 2^{-(n-1)}."""
        return n * 2**(-(n-1))
    
    @staticmethod
    def tribes_total_influence(n: int, w: Optional[int] = None) -> float:
        """
        Total influence of Tribes.
        
        For standard balanced Tribes: I[TRIBES_n] ≈ log(n)
        """
        if w is None:
            log_n = np.log2(max(2, n))
            w = int(log_n - np.log2(max(1, log_n)))
            w = max(2, w)
        
        k = n // w
        # Each variable has influence ≈ (1/2)^{w-1}
        return k * w * (1/2)**(w-1)
    
    # ==========================================================
    # Noise Stability Bounds
    # ==========================================================
    
    @staticmethod
    def majority_noise_stability(rho: float) -> float:
        """
        Noise stability of Majority (Sheppard's formula).
        
        Stab_ρ[MAJ_n] → (1/2) + (1/π)·arcsin(ρ) as n → ∞
        """
        return 0.5 + (1/np.pi) * np.arcsin(rho)
    
    @staticmethod
    def parity_noise_stability(n: int, rho: float) -> float:
        """
        Noise stability of Parity.
        
        Stab_ρ[XOR_n] = ρ^n → 0 for ρ < 1
        """
        return rho ** n
    
    @staticmethod
    def dictator_noise_stability(rho: float) -> float:
        """
        Noise stability of Dictator.
        
        Stab_ρ[DICT] = ρ
        """
        return rho
    
    @staticmethod  
    def and_noise_stability(n: int, rho: float) -> float:
        """
        Noise stability of AND.
        
        Stab_ρ[AND_n] = Pr[AND(x) = AND(y)] where y = ρ-correlated with x
        """
        # Complex formula involving sums
        # Simplified: approximately ((1+rho)/2)^n for large n
        return ((1 + rho) / 2) ** n
    
    # ==========================================================
    # Influence Bounds
    # ==========================================================
    
    @staticmethod
    def majority_influence_i(n: int, i: int = 0) -> float:
        """
        Influence of variable i in Majority_n.
        
        By symmetry: Inf_i[MAJ_n] = √(2/(πn)) for all i
        """
        return np.sqrt(2 / (np.pi * n))
    
    @staticmethod
    def parity_influence_i(n: int, i: int = 0) -> float:
        """Each variable has influence 1 in Parity."""
        return 1.0
    
    @staticmethod
    def and_influence_i(n: int, i: int = 0) -> float:
        """Influence of each variable in AND_n = 2^{-(n-1)}."""
        return 2**(-(n-1))
    
    # ==========================================================
    # Poincaré and KKL Bounds
    # ==========================================================
    
    @staticmethod
    def poincare_lower_bound(variance: float) -> float:
        """
        Poincaré inequality: Var[f] ≤ I[f]
        
        So I[f] ≥ Var[f]
        """
        return variance
    
    @staticmethod
    def kkl_lower_bound(n: int, variance: float) -> float:
        """
        KKL Theorem: max_i Inf_i[f] ≥ Var[f] · c·log(n)/n
        
        For balanced f: max Inf_i ≥ Ω(log(n)/n)
        """
        c = 1.0  # Constant factor
        return variance * c * np.log(n) / n
    
    @staticmethod
    def friedgut_junta_bound(total_influence: float, epsilon: float) -> float:
        """
        Friedgut's Junta Theorem.
        
        If I[f] ≤ k, then f is ε-close to a 2^{O(k/ε)}-junta.
        
        Returns the junta size bound.
        """
        return 2 ** (4 * total_influence / epsilon)
    
    # ==========================================================
    # Fourier Concentration Bounds
    # ==========================================================
    
    @staticmethod
    def decision_tree_fourier_support(depth: int) -> int:
        """
        Decision tree of depth d has at most 4^d non-zero Fourier coefficients.
        """
        return 4 ** depth
    
    @staticmethod
    def decision_tree_spectral_norm(size: int) -> float:
        """
        Decision tree of size s has Σ|f̂(S)| ≤ s.
        """
        return float(size)
    
    @staticmethod
    def mansour_spectral_concentration(depth: int, epsilon: float) -> int:
        """
        Mansour's Theorem: DNF of depth d has (1-ε) of Fourier weight
        on O(1/ε^2 · 2^{O(d)}) coefficients.
        """
        return int((1/epsilon)**2 * 2**(2*depth))
    
    # ==========================================================
    # LTF-specific Bounds
    # ==========================================================
    
    @staticmethod
    def ltf_total_influence(n: int, regularity: float = 0.0) -> float:
        """
        Total influence of an LTF.
        
        For regular LTFs (τ small): I[f] ≈ √(2/π)·√n
        For irregular (τ → 1): I[f] → 1
        """
        regular = np.sqrt(2/np.pi) * np.sqrt(n)
        return regular * (1 - regularity) + regularity
    
    @staticmethod
    def ltf_noise_stability(rho: float, regularity: float = 0.0) -> float:
        """
        Noise stability of LTF.
        
        Regular LTFs: Stab_ρ → (1/2) + (1/π)arcsin(ρ)
        Dictator (τ=1): Stab_ρ = ρ
        """
        regular = 0.5 + (1/np.pi) * np.arcsin(rho)
        return regular * (1 - regularity) + rho * regularity
    
    # ==========================================================
    # Query Complexity Bounds
    # ==========================================================
    
    @staticmethod
    def sensitivity_vs_block_sensitivity(s: float) -> float:
        """
        Sensitivity Theorem (Huang 2019): bs(f) ≤ s(f)^4
        
        Returns upper bound on block sensitivity.
        """
        return s ** 4
    
    @staticmethod
    def certificate_vs_sensitivity(s: float) -> float:
        """
        C(f) ≤ s(f) · bs(f) ≤ s(f)^5
        """
        return s ** 5
    
    @staticmethod
    def degree_vs_sensitivity(s: float) -> float:
        """
        deg(f) ≤ s(f)^2 (from Sensitivity Theorem)
        """
        return s ** 2
    
    # ==========================================================
    # Helper: Get all bounds for a family
    # ==========================================================
    
    @classmethod
    def get_bounds_for_family(cls, family_name: str) -> Dict[str, Callable]:
        """
        Get all applicable theoretical bounds for a named family.
        
        Returns dictionary of {property_name: bound_function}
        """
        family_name = family_name.lower()
        
        if family_name == "majority":
            return {
                "total_influence": cls.majority_total_influence,
                "influence_i": cls.majority_influence_i,
                "noise_stability": lambda n, rho=0.5: cls.majority_noise_stability(rho),
            }
        
        elif family_name == "parity":
            return {
                "total_influence": cls.parity_total_influence,
                "influence_i": cls.parity_influence_i,
                "noise_stability": cls.parity_noise_stability,
            }
        
        elif family_name in ["and", "or"]:
            return {
                "total_influence": cls.and_total_influence,
                "influence_i": cls.and_influence_i,
                "noise_stability": cls.and_noise_stability,
            }
        
        elif family_name == "tribes":
            return {
                "total_influence": cls.tribes_total_influence,
            }
        
        elif family_name == "dictator":
            return {
                "total_influence": lambda n: 1.0,
                "noise_stability": lambda n, rho=0.5: cls.dictator_noise_stability(rho),
            }
        
        elif family_name == "ltf":
            return {
                "total_influence": cls.ltf_total_influence,
                "noise_stability": lambda n, rho=0.5, tau=0: cls.ltf_noise_stability(rho, tau),
            }
        
        return {}


__all__ = ["TheoreticalBounds"]
