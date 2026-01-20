#!/usr/bin/env python3
"""
Tutorial 3: Common Function Families
=====================================

This tutorial covers:
- Standard Boolean function families
- Their asymptotic properties
- When to use each family
- Comparing families visually

References:
- O'Donnell, "Analysis of Boolean Functions", Chapter 2
"""

import numpy as np

import boofun as bf
from boofun.families.builtins import (
    ANDFamily,
    DictatorFamily,
    MajorityFamily,
    ORFamily,
    ParityFamily,
    TribesFamily,
)


def expectation(f):
    """Compute E[f] = Pr[f(x) = 1]."""
    tt = f.get_representation("truth_table")
    return sum(1 for x in tt if x) / len(tt)


print("=" * 60)
print("Tutorial 3: Common Function Families")
print("=" * 60)


# =============================================================================
# 1. AND and OR Functions
# =============================================================================
print("\n--- 1. AND and OR Functions ---\n")

print(
    """
AND_n(x) = 1 iff ALL inputs are 1
OR_n(x) = 1 iff ANY input is 1

Properties:
- Highly unbalanced for large n
- Very low total influence (vanishes exponentially)
- Monotone functions
"""
)

for n in [3, 5, 8]:
    f = bf.AND(n)
    g = bf.OR(n)
    print(f"n={n}:")
    print(f"  AND: E[f]={expectation(f):.4f}, I[f]={f.total_influence():.4f}")
    print(f"  OR:  E[f]={expectation(g):.4f}, I[f]={g.total_influence():.4f}")


# =============================================================================
# 2. Majority Function
# =============================================================================
print("\n--- 2. Majority Function ---\n")

print(
    """
MAJ_n(x) = 1 iff more than half of inputs are 1

Properties:
- Balanced (for odd n)
- Total influence ≈ √(2/π) · √n ≈ 0.798√n
- Each variable has equal influence ≈ √(2/(πn))
- Most important symmetric threshold function
"""
)

print(f"{'n':<5} {'E[f]':<10} {'I[f]':<10} {'Theory':<10} {'Inf_i':<10}")
print("-" * 45)
for n in [3, 5, 7, 9, 11]:
    f = bf.majority(n)
    theory = np.sqrt(2 / np.pi) * np.sqrt(n)
    inf_i = f.influences()[0]
    print(
        f"{n:<5} {expectation(f):<10.3f} {f.total_influence():<10.3f} {theory:<10.3f} {inf_i:<10.4f}"
    )


# =============================================================================
# 3. Parity Function
# =============================================================================
print("\n--- 3. Parity Function ---\n")

print(
    """
PARITY_n(x) = XOR of all inputs = Σx_i mod 2

Properties:
- Balanced
- Maximum total influence (I[f] = n)
- Each variable has influence 1
- Most noise-sensitive function
- Single Fourier coefficient: f̂({1,...,n}) = ±1
"""
)

print(f"{'n':<5} {'I[f]':<10} {'Inf_i':<10} {'Degree':<10} {'Noise Stab(0.9)':<15}")
print("-" * 50)
for n in [3, 5, 7, 9]:
    f = bf.parity(n)
    print(
        f"{n:<5} {f.total_influence():<10.1f} {f.influences()[0]:<10.1f} "
        f"{f.degree():<10} {f.noise_stability(0.9):<15.4f}"
    )


# =============================================================================
# 4. Tribes Function
# =============================================================================
print("\n--- 4. Tribes Function ---\n")

print(
    """
TRIBES is a balanced DNF: OR of ANDs (tribes)

TRIBES(x) = OR(AND(x_1..x_w), AND(x_{w+1}..x_{2w}), ...)

Properties:
- Balanced when w ≈ log(n) - log(log(n))
- Total influence ≈ log(n)
- Used in randomness and derandomization
"""
)

# Using the tribes builder
for n in [4, 8, 16]:
    try:
        f = bf.tribes(2, n)  # width 2 tribes
        print(f"TRIBES(w=2, n={n}): E[f]={expectation(f):.3f}, I[f]={f.total_influence():.3f}")
    except:
        print(f"TRIBES(w=2, n={n}): n not divisible by w")


# =============================================================================
# 5. Dictator Functions
# =============================================================================
print("\n--- 5. Dictator Functions ---\n")

print(
    """
DICTATOR_i(x) = x_i (just returns the i-th variable)

Properties:
- Trivially computable
- Total influence = 1 (only x_i matters)
- Noise stability = ρ
- The FKN theorem says: low-influence functions are close to dictators
"""
)

f = bf.dictator(5, 0)  # Returns x_0
print(f"Dictator_0 on 5 variables:")
print(f"  Influences: {f.influences()}")
print(f"  Total influence: {f.total_influence()}")
print(f"  Noise stability (ρ=0.9): {f.noise_stability(0.9):.3f}")


# =============================================================================
# 6. Threshold Functions
# =============================================================================
print("\n--- 6. Threshold Functions ---\n")

print(
    """
THR_k(x) = 1 iff Σx_i ≥ k

Special cases:
- k = 1: OR
- k = n: AND
- k = (n+1)/2: Majority
"""
)

n = 5
print(f"Threshold functions on n={n} variables:")
print(f"{'k':<5} {'Function':<15} {'E[f]':<10} {'I[f]':<10}")
print("-" * 40)
for k in range(1, n + 1):
    f = bf.threshold(n, k)
    name = "OR" if k == 1 else ("AND" if k == n else ("MAJ" if k == 3 else f"THR_{k}"))
    print(f"{k:<5} {name:<15} {expectation(f):<10.3f} {f.total_influence():<10.4f}")


# =============================================================================
# 7. Using Family Classes
# =============================================================================
print("\n--- 7. Using Family Classes for Analysis ---\n")

print("Families provide theoretical asymptotics for comparison:")

majority_family = MajorityFamily()
print(f"\nMajority family metadata:")
print(f"  Name: {majority_family.metadata.name}")
print(f"  Properties: {majority_family.metadata.universal_properties}")
print(
    f"  Theoretical I[f] for n=9: {majority_family.metadata.asymptotics['total_influence'](9):.3f}"
)

# Generate and verify
f = majority_family.generate(9)
print(f"  Computed I[f] for n=9: {f.total_influence():.3f}")


# =============================================================================
# 8. Comparison Table
# =============================================================================
print("\n--- 8. Family Comparison (n=9) ---\n")

n = 9
families = [
    ("AND", bf.AND(n)),
    ("OR", bf.OR(n)),
    ("Majority", bf.majority(n)),
    ("Parity", bf.parity(n)),
    ("Dictator", bf.dictator(n, 0)),
]

print(f"{'Function':<12} {'E[f]':<8} {'Var[f]':<8} {'I[f]':<8} {'Degree':<8} {'Max Inf':<8}")
print("-" * 60)
for name, f in families:
    print(
        f"{name:<12} {expectation(f):<8.3f} {f.variance():<8.3f} "
        f"{f.total_influence():<8.3f} {f.degree():<8} {f.max_influence():<8.4f}"
    )


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- AND/OR: extreme threshold, vanishing influence")
print("- Majority: balanced threshold, I[f] ≈ √n")
print("- Parity: maximum influence (n), most noise-sensitive")
print("- Tribes: balanced DNF, I[f] ≈ log(n)")
print("- Dictator: minimal I[f] = 1, FKN theorem baseline")
print("- Threshold: generalize AND/OR/Majority")
print("=" * 60)
