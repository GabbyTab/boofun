#!/usr/bin/env python3
"""
Tutorial 6: Noise and Stability with BooFun
==============================================

Understanding how Boolean functions behave under random noise
is crucial for applications in voting, coding, and learning.

Topics covered:
- Noise stability and noise sensitivity
- Influences and total influence
- The connection to Fourier analysis
- Applications to voting and learning
"""

import numpy as np

import boofun as bf

print("=" * 60)
print("Tutorial 6: Noise and Stability")
print("=" * 60)

# =============================================================================
# 1. Introduction to Noise
# =============================================================================
print("\n--- 1. What is Noise? ---\n")

print(
    """
Consider input x ∈ {0,1}ⁿ. With noise parameter ρ ∈ [0,1]:
- Each bit is kept with probability ρ
- Each bit is flipped with probability 1-ρ

This creates a "noisy" version y of x.

Key question: How often does f(x) ≠ f(y)?
"""
)

# =============================================================================
# 2. Noise Stability
# =============================================================================
print("\n--- 2. Noise Stability ---\n")

print(
    """
Noise Stability: NS_ρ(f) = Pr[f(x) = f(y)] where y is ρ-correlated with x

Equivalently: NS_ρ(f) = Σ_S ρ^|S| · f̂(S)²

- NS_ρ(f) close to 1 → f is stable (robust to noise)
- NS_ρ(f) close to 0.5 → f is unstable (sensitive to noise)
"""
)

# Compare noise stability of different functions
functions = [
    ("Dictator_5", bf.dictator(5, 0)),
    ("Majority_5", bf.majority(5)),
    ("Parity_5", bf.parity(5)),
    ("AND_5", bf.AND(5)),
]

print("Noise stability at different noise levels:\n")
print(f"{'Function':<15} {'ρ=0.99':<10} {'ρ=0.9':<10} {'ρ=0.5':<10} {'ρ=0.1'}")
print("-" * 55)

for name, f in functions:
    stabilities = []
    for rho in [0.99, 0.9, 0.5, 0.1]:
        ns = f.noise_stability(rho)
        stabilities.append(f"{ns:.3f}")

    print(
        f"{name:<15} {stabilities[0]:<10} {stabilities[1]:<10} "
        f"{stabilities[2]:<10} {stabilities[3]}"
    )

print(
    """
Observations:
- Dictator: Very stable (output depends on just one bit)
- Parity: Very unstable (all bits matter equally)
- Majority: Moderate stability
- AND: Stable for outputs near 0
"""
)

# =============================================================================
# 3. Noise Sensitivity
# =============================================================================
print("\n--- 3. Noise Sensitivity ---\n")

print(
    """
Noise Sensitivity: NS^ρ(f) = Pr[f(x) ≠ f(y)] = 1 - NS_ρ(f)

This measures the probability that noise changes the output.
"""
)

print(f"{'Function':<15} {'Noise Sens (ρ=0.9)':<20} {'Interpretation'}")
print("-" * 60)

for name, f in functions:
    ns = f.noise_stability(0.9)
    noise_sens = 1 - ns

    if noise_sens < 0.05:
        interp = "Very stable"
    elif noise_sens < 0.15:
        interp = "Moderately stable"
    elif noise_sens < 0.3:
        interp = "Somewhat sensitive"
    else:
        interp = "Very sensitive"

    print(f"{name:<15} {noise_sens:<20.3f} {interp}")

# =============================================================================
# 4. Influences
# =============================================================================
print("\n--- 4. Influences ---\n")

print(
    """
Influence of variable i: Inf_i(f) = Pr[f(x) ≠ f(x ⊕ eᵢ)]

This measures how much variable i "matters" to the function.

In Fourier terms: Inf_i(f) = Σ_{S∋i} f̂(S)²
"""
)

print("Influences for Majority_5:\n")

f = bf.majority(5)
influences = f.influences()

for i, inf in enumerate(influences):
    bar = "█" * int(inf * 20)
    print(f"  x_{i}: {inf:.4f} {bar}")

print(f"\nTotal influence I[f] = {f.total_influence():.4f}")
print(f"Max influence = {f.max_influence():.4f}")

# =============================================================================
# 5. Total Influence and Noise
# =============================================================================
print("\n--- 5. Total Influence and Noise Connection ---\n")

print(
    """
Total Influence: I[f] = Σᵢ Inf_i(f) = Σ_S |S| · f̂(S)²

Key relationship with noise sensitivity:
  NS^ρ(f) ≤ (1-ρ) · I[f]   (for small 1-ρ)

High total influence → more noise sensitive!
"""
)

print(f"{'Function':<15} {'I[f]':<10} {'NS^0.9':<10} {'Bound'}")
print("-" * 45)

for name, f in functions:
    total_inf = f.total_influence()
    noise_sens = 1 - f.noise_stability(0.9)
    bound = 0.1 * total_inf  # (1-0.9) * I[f]

    print(f"{name:<15} {total_inf:<10.3f} {noise_sens:<10.3f} {bound:.3f}")

# =============================================================================
# 6. Spectral Weight Distribution
# =============================================================================
print("\n--- 6. Spectral Weight Distribution ---\n")

print(
    """
The Fourier spectrum shows how "spread out" f's representation is.

Weight at level k: W^k[f] = Σ_{|S|=k} f̂(S)²

- Low degree concentration → stable
- High degree concentration → sensitive
"""
)

print("Spectral weight distribution:\n")

for name, f in [
    ("Dictator_5", bf.dictator(5, 0)),
    ("Majority_5", bf.majority(5)),
    ("Parity_5", bf.parity(5)),
]:
    weights = f.spectral_weight_by_degree()

    print(f"{name}:")
    for k, w in enumerate(weights):
        if w > 0.001:
            bar = "█" * int(w * 30)
            print(f"  Level {k}: {w:.4f} {bar}")
    print()

# =============================================================================
# 7. Application: Voting Systems
# =============================================================================
print("\n--- 7. Application: Voting Systems ---\n")

print(
    """
In voting theory:
- f(x) = 1 if candidate wins, 0 otherwise
- x_i = 1 if voter i votes for the candidate

Noise models voter error/manipulation.
Stability measures voting system robustness.
"""
)

print("Comparing voting rules (5 voters):\n")

voting_rules = [
    ("Majority", bf.majority(5)),
    ("Dictator (1 voter)", bf.dictator(5, 0)),
    ("Unanimity (AND)", bf.AND(5)),
    ("Any vote (OR)", bf.OR(5)),
]

print(f"{'Rule':<20} {'Stability':<12} {'Total Inf':<12} {'Max Inf'}")
print("-" * 55)

for name, f in voting_rules:
    ns = f.noise_stability(0.9)
    total_inf = f.total_influence()
    max_inf = f.max_influence()

    print(f"{name:<20} {ns:<12.3f} {total_inf:<12.3f} {max_inf:.3f}")

print(
    """
Analysis:
- Majority: Fair (equal influences), moderately stable
- Dictator: Unfair (one person decides), but very stable
- Unanimity: Requires consensus, very stable for "no" outcome
- Any vote: Low threshold, stable for "yes" outcome
"""
)

# =============================================================================
# 8. Asymptotic Behavior
# =============================================================================
print("\n--- 8. Asymptotic Behavior ---\n")

print(
    """
How does noise stability scale with n?

For balanced functions (E[f] = 0.5):
- Majority: NS_ρ → 2/π · arcsin(ρ) as n → ∞
- Parity: NS_ρ → 1/2 + 1/2 · ρⁿ → 1/2 (unstable!)
- Tribes: NS_ρ → constant (designed to be marginally stable)
"""
)

print("Majority stability as n grows:\n")
print(f"{'n':<6} {'NS_0.9':<10} {'NS_0.7':<10} {'NS_0.5'}")
print("-" * 35)

# Theoretical limit for ρ=0.9
import math

limit_0_9 = 2 / math.pi * math.asin(0.9)
limit_0_7 = 2 / math.pi * math.asin(0.7)
limit_0_5 = 2 / math.pi * math.asin(0.5)

for n in [3, 5, 7, 9, 11, 21]:
    f = bf.majority(n)
    ns_9 = f.noise_stability(0.9)
    ns_7 = f.noise_stability(0.7)
    ns_5 = f.noise_stability(0.5)

    print(f"{n:<6} {ns_9:<10.4f} {ns_7:<10.4f} {ns_5:.4f}")

print(f"{'Limit':<6} {limit_0_9:<10.4f} {limit_0_7:<10.4f} {limit_0_5:.4f}")

# =============================================================================
# 9. Influence Bounds
# =============================================================================
print("\n--- 9. Important Influence Bounds ---\n")

print(
    """
Key theorems about influences:

1. KKL Theorem: max_i Inf_i(f) ≥ Var[f] · Ω(log n / n)
   (Balanced functions have at least one influential variable)

2. Friedgut's Theorem: Functions with I[f] = O(1) are juntas
   (approximately depend on few variables)

3. For monotone functions: Inf_i(f) = Pr[x_i is pivotal]
"""
)

print("Verifying KKL-type bounds:\n")

for n in [5, 9, 15]:
    f = bf.majority(n)
    max_inf = f.max_influence()
    var = f.variance()
    kkl_bound = var * np.log(n) / n

    print(f"Majority_{n}: max_inf={max_inf:.4f}, KKL-type bound={kkl_bound:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- Noise stability NS_ρ(f) = Pr[f(x) = f(y)] with ρ-correlated y")
print("- Influence Inf_i(f) = probability variable i is pivotal")
print("- Total influence I[f] = Σ Inf_i(f) = noise sensitivity driver")
print("- Fourier: NS_ρ(f) = Σ ρ^|S| f̂(S)², I[f] = Σ |S| f̂(S)²")
print("- High degree Fourier weight → noise sensitive")
print("- Applications: voting robustness, learning, coding")
print("=" * 60)
