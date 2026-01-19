#!/usr/bin/env python3
"""
Tutorial 2: Fourier Analysis Basics
====================================

This tutorial covers:
- Walsh-Hadamard Transform (WHT)
- Fourier coefficients and their meaning
- Parseval's identity
- Fourier degree and spectral weight

References:
- O'Donnell, "Analysis of Boolean Functions", Chapters 1-2
"""

import numpy as np
import boofun as bf


def get_expectation(f):
    """Compute E[f] = Pr[f(x) = 1] for uniform random x."""
    tt = f.get_representation("truth_table")
    return sum(1 for x in tt if x) / len(tt)

print("=" * 60)
print("Tutorial 2: Fourier Analysis Basics")
print("=" * 60)


# =============================================================================
# 1. What is Fourier Analysis of Boolean Functions?
# =============================================================================
print("\n--- 1. Introduction ---\n")

print("""
Every Boolean function f: {0,1}^n → {0,1} can be written as:

    f(x) = Σ_S f̂(S) · χ_S(x)

where:
- S ranges over all subsets of {1,...,n}
- f̂(S) is the Fourier coefficient for subset S
- χ_S(x) = (-1)^{Σ_{i∈S} x_i} is the parity function on S

The Fourier coefficients tell us about the function's structure!
""")


# =============================================================================
# 2. Computing Fourier Coefficients
# =============================================================================
print("\n--- 2. Computing Fourier Coefficients ---\n")

# Parity function (XOR)
f = bf.parity(3)
fourier = f.fourier()

print("Parity_3 Fourier coefficients:")
for i, coef in enumerate(fourier):
    if abs(coef) > 0.001:
        bits = format(i, '03b')
        print(f"  f̂({bits}) = {coef:.4f}")

print("\n→ Parity has ONE non-zero coefficient: f̂(111) = ±1")
print("  This means parity is a 'pure' character function.")

# AND function
f = bf.AND(3)
fourier = f.fourier()

print("\nAND_3 Fourier coefficients:")
for i, coef in enumerate(fourier):
    if abs(coef) > 0.001:
        bits = format(i, '03b')
        print(f"  f̂({bits}) = {coef:.4f}")


# =============================================================================
# 3. Parseval's Identity
# =============================================================================
print("\n--- 3. Parseval's Identity ---\n")

print("""
Parseval's Identity: Σ_S f̂(S)² = 1

In the ±1 domain, f: {-1,+1}^n → {-1,+1}, we have E[f²] = 1.
So Σ_S f̂(S)² = 1 always (the L2 norm is preserved).
""")

f = bf.majority(5)
fourier = f.fourier()

# Compute sum of squared coefficients
sum_squared = sum(c**2 for c in fourier)

print(f"Majority_5:")
print(f"  Σ f̂(S)² = {sum_squared:.4f}")
print(f"  Should equal 1.0: {abs(sum_squared - 1.0) < 0.01}")


# =============================================================================
# 4. Variance and the f̂(∅) Coefficient
# =============================================================================
print("\n--- 4. Variance and Expectation ---\n")

print("""
Key facts:
- f̂(∅) = E[f] (the 'DC component')
- Var[f] = Σ_{S≠∅} f̂(S)² = E[f²] - E[f]²
""")

f = bf.majority(5)
fourier = f.fourier()

f_hat_empty = fourier[0]  # Index 0 = empty set
variance = sum(c**2 for c in fourier[1:])  # Exclude empty set

print(f"Majority_5:")
print(f"  f̂(∅) = {f_hat_empty:.4f}")
print(f"  E[f]  = {get_expectation(f):.4f}")
print(f"  Var   = {variance:.4f} (from Fourier)")
print(f"  Var   = {f.variance():.4f} (direct)")


# =============================================================================
# 5. Fourier Degree
# =============================================================================
print("\n--- 5. Fourier Degree ---\n")

print("""
Fourier degree = max |S| such that f̂(S) ≠ 0

- Degree 1: dictators, linear functions
- Degree n: can depend on all variables (like parity)
""")

functions = {
    "Dictator": bf.dictator(5, 0),
    "Parity_5": bf.parity(5),
    "Majority_5": bf.majority(5),
    "AND_5": bf.AND(5),
}

for name, f in functions.items():
    degree = f.degree()
    print(f"  {name}: degree = {degree}")


# =============================================================================
# 6. Spectral Weight by Degree
# =============================================================================
print("\n--- 6. Spectral Weight by Degree ---\n")

print("""
Spectral weight at level k: W^{=k}[f] = Σ_{|S|=k} f̂(S)²

This tells us how much of the function's 'variance' is at each level.
""")

f = bf.majority(5)
weights = f.spectral_weight_by_degree()

print("Majority_5 spectral weight:")
for k, w in sorted(weights.items()):
    bar = "█" * int(w * 30)
    print(f"  Level {k}: {w:.4f} {bar}")

print("\n→ Majority has most weight on level 1 (individual influences)")


# =============================================================================
# 7. Total Influence from Fourier
# =============================================================================
print("\n--- 7. Total Influence ---\n")

print("""
Total Influence: I[f] = Σ_S |S| · f̂(S)²

This measures how 'complex' or 'noise-sensitive' a function is.
""")

functions = {
    "Dictator_5": bf.dictator(5, 0),
    "Majority_5": bf.majority(5),
    "Parity_5": bf.parity(5),
    "AND_5": bf.AND(5),
}

print(f"{'Function':<15} {'I[f]':<10} {'Interpretation'}")
print("-" * 50)
for name, f in functions.items():
    inf = f.total_influence()
    if "Dictator" in name:
        interp = "One variable matters"
    elif "Parity" in name:
        interp = "All variables matter equally"
    elif "AND" in name:
        interp = "Very low influence"
    else:
        interp = "Moderate influence"
    print(f"{name:<15} {inf:<10.3f} {interp}")


# =============================================================================
# 8. Example: Comparing Functions
# =============================================================================
print("\n--- 8. Comparing Fourier Structure ---\n")

f1 = bf.majority(5)
f2 = bf.parity(5)

print("Majority_5 vs Parity_5:")
print()
print(f"{'Property':<25} {'Majority':<15} {'Parity':<15}")
print("-" * 55)
print(f"{'E[f]':<25} {get_expectation(f1):<15.3f} {get_expectation(f2):<15.3f}")
print(f"{'Var[f]':<25} {f1.variance():<15.3f} {f2.variance():<15.3f}")
print(f"{'Degree':<25} {f1.degree():<15} {f2.degree():<15}")
print(f"{'Total influence I[f]':<25} {f1.total_influence():<15.3f} {f2.total_influence():<15.3f}")
print(f"{'Max influence':<25} {f1.max_influence():<15.3f} {f2.max_influence():<15.3f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- f.fourier() returns all Fourier coefficients")
print("- f̂(∅) = E[f], Var[f] = Σ_{S≠∅} f̂(S)²")
print("- f.degree() = highest level with non-zero coefficient")
print("- f.total_influence() = Σ_S |S| · f̂(S)²")
print("- f.spectral_weight_by_degree() shows weight distribution")
print("=" * 60)
