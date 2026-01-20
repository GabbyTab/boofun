#!/usr/bin/env python3
"""
Tutorial 5: Query Complexity with BooFun
==========================================

Query complexity measures how many input bits must be queried
to compute a Boolean function. This is fundamental to understanding
computational complexity.

Topics covered:
- Deterministic query complexity D(f)
- Sensitivity s(f) and block sensitivity bs(f)
- Certificate complexity C(f)
- Quantum query complexity Q(f)
- Relationships between measures
"""

import numpy as np

import boofun as bf
from boofun.analysis.query_complexity import QueryComplexityProfile

print("=" * 60)
print("Tutorial 5: Query Complexity")
print("=" * 60)

# =============================================================================
# 1. Introduction to Query Complexity
# =============================================================================
print("\n--- 1. What is Query Complexity? ---\n")

print(
    """
Query complexity asks: How many input bits must we look at to
determine f(x)?

Key measures:
- D(f): Deterministic query complexity (worst-case queries)
- s(f): Sensitivity (max bits that flip output)
- bs(f): Block sensitivity (max disjoint blocks)
- C(f): Certificate complexity (min bits to prove output)
- Q(f): Quantum query complexity

For all Boolean functions:
  s(f) ≤ bs(f) ≤ C(f) ≤ D(f) ≤ n
"""
)

# =============================================================================
# 2. Complete Profile for AND
# =============================================================================
print("\n--- 2. Query Complexity Profile ---\n")

print("Complete profile for AND_4:\n")

f = bf.AND(4)
profile = QueryComplexityProfile(f)
print(profile.summary())

# =============================================================================
# 3. Sensitivity (from BooleanFunction)
# =============================================================================
print("\n--- 3. Sensitivity s(f) ---\n")

print(
    """
Sensitivity at input x: number of bits i where f(x) ≠ f(x ⊕ eᵢ)
(flipping bit i changes the output)

s(f) = max over all x of sensitivity at x

This is available directly on BooleanFunction.
"""
)

functions = [
    ("AND_4", bf.AND(4)),
    ("OR_4", bf.OR(4)),
    ("Parity_4", bf.parity(4)),
    ("Majority_5", bf.majority(5)),
    ("Dictator_4", bf.dictator(4, 0)),
]

print(f"{'Function':<15} {'s(f)':<8} {'Interpretation'}")
print("-" * 50)

for name, f in functions:
    s = f.sensitivity()

    if "Parity" in name:
        interp = "All bits matter equally"
    elif "AND" in name or "OR" in name:
        interp = "Only 1 critical input"
    elif "Majority" in name:
        interp = "Depends on balance point"
    else:
        interp = "Single bit determines output"

    print(f"{name:<15} {s:<8} {interp}")

# =============================================================================
# 4. Comparing via Profiles
# =============================================================================
print("\n--- 4. Comparing Functions via Profiles ---\n")

print(
    """
For more complexity measures, use QueryComplexityProfile.
It computes sensitivity, block sensitivity, certificate complexity,
decision tree depth, and more.
"""
)

# Extract key measures from profiles
print(f"{'Function':<12} {'s(f)':<6} {'D(f)':<6} {'deg':<6} {'Q₂(f)'}")
print("-" * 40)

for name, f in functions:
    profile = QueryComplexityProfile(f)
    summary = profile.summary()

    # Parse key values from summary (simplified approach)
    s = f.sensitivity()
    deg = f.degree()

    # Get quantum estimate (approximate)
    q2_approx = np.sqrt(s) * 1.5  # rough approximation

    print(f"{name:<12} {s:<6} {f.n_vars:<6} {deg:<6} ~{q2_approx:.1f}")

# =============================================================================
# 5. Degree and Sensitivity
# =============================================================================
print("\n--- 5. Degree vs Sensitivity ---\n")

print(
    """
The degree of f (in the Fourier representation) relates to sensitivity.

Key relationship (Nisan-Szegedy):
  deg(f) ≤ 2 · s(f)²

Higher degree → potentially higher sensitivity.
"""
)

print(f"{'Function':<15} {'s(f)':<8} {'deg(f)':<8} {'2·s(f)²'}")
print("-" * 40)

for name, f in functions:
    s = f.sensitivity()
    deg = f.degree()
    bound = 2 * s * s

    ok = "✓" if deg <= bound else "✗"
    print(f"{name:<15} {s:<8} {deg:<8} {bound:<8} {ok}")

# =============================================================================
# 6. Influences and Sensitivity
# =============================================================================
print("\n--- 6. Influences and Sensitivity Connection ---\n")

print(
    """
Sensitivity is closely related to influences:

- Sensitivity at x = number of influential variables at x
- max_influence ≤ s(f) / n
- Total influence relates to average sensitivity
"""
)

print(f"{'Function':<12} {'s(f)':<6} {'Total Inf':<10} {'Max Inf'}")
print("-" * 40)

for name, f in functions:
    s = f.sensitivity()
    total_inf = f.total_influence()
    max_inf = f.max_influence()

    print(f"{name:<12} {s:<6} {total_inf:<10.3f} {max_inf:.4f}")

# =============================================================================
# 7. Quantum Advantage
# =============================================================================
print("\n--- 7. Quantum Query Complexity ---\n")

print(
    """
Quantum computers can sometimes compute f with fewer queries.

Q₂(f) = bounded-error quantum (succeeds with prob ≥ 2/3)

Key bounds:
- Q₂(f) ≥ Ω(√s(f))  [polynomial relationship]
- Q₂(f) ≤ O(√(n·D(f)))  [Grover-type speedup]

Grover's algorithm shows:
- For OR_n: classical D(f) = n, quantum Q₂(f) = O(√n)
"""
)

# Show quantum advantage for OR (search)
print("Quantum speedup for OR (unstructured search):\n")

for n in [4, 9, 16, 25]:
    classical = n  # D(OR_n) = n
    quantum = np.sqrt(n)  # Grover's bound
    speedup = classical / quantum

    print(f"  OR_{n}: D(f)={classical}, Q₂(f)≈{quantum:.1f}, speedup≈{speedup:.1f}x")

# =============================================================================
# 8. Full Profiles for Key Functions
# =============================================================================
print("\n--- 8. Full Profile Examples ---\n")

print("=" * 50)
print("Parity_4 - Maximum sensitivity function")
print("=" * 50)
print(QueryComplexityProfile(bf.parity(4)).summary())

print("\n" + "=" * 50)
print("Majority_5 - Balanced threshold function")
print("=" * 50)
print(QueryComplexityProfile(bf.majority(5)).summary())

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- s(f): How sensitive is f to single-bit changes?")
print("- f.sensitivity() gives max sensitivity directly")
print("- QueryComplexityProfile(f).summary() shows all measures")
print("- Key chain: s(f) ≤ bs(f) ≤ C(f) ≤ D(f) ≤ n")
print("- deg(f) ≤ 2·s(f)² (Nisan-Szegedy)")
print("- Quantum can achieve quadratic speedup (Grover)")
print("=" * 60)
