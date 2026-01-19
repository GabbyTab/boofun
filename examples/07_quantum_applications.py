#!/usr/bin/env python3
"""
Tutorial 7: Quantum Applications with BoolFunc
===============================================

Boolean function analysis has deep connections to quantum computing.
This tutorial explores quantum query complexity, Grover's algorithm,
and quantum property testing.

Topics covered:
- Quantum query complexity
- Grover's search algorithm analysis
- Quantum walk algorithms
- Quantum advantage estimation
"""

import numpy as np
import boolfunc as bf
from boolfunc.quantum import (
    grover_speedup, quantum_walk_analysis, 
    element_distinctness_analysis
)

print("=" * 60)
print("Tutorial 7: Quantum Applications")
print("=" * 60)

# =============================================================================
# 1. Introduction to Quantum Query Complexity
# =============================================================================
print("\n--- 1. Quantum Query Complexity ---\n")

print("""
In quantum computing, we can query multiple inputs in superposition.

Key measures:
- Q(f): Exact quantum query complexity
- Q₂(f): Bounded-error quantum (succeeds with prob ≥ 2/3)

Key relationships:
- Q₂(f) ≥ Ω(√(bs(f)))   [polynomial method]
- Q₂(f) ≥ Ω(√(D(f)))    [in many cases]
- Q₂(f) ≤ O(√(n·D(f)))  [Grover-type speedup]
""")

# =============================================================================
# 2. Grover's Algorithm Speedup
# =============================================================================
print("\n--- 2. Grover's Algorithm Analysis ---\n")

print("""
Grover's algorithm finds marked items in unstructured search.

For f: {0,1}ⁿ → {0,1}, Grover finds x where f(x) = 1.

Classical: O(N) queries where N = 2ⁿ
Quantum:   O(√N) queries - quadratic speedup!
""")

# Analyze different functions with Grover
functions = [
    ("AND_4", bf.AND(4)),
    ("AND_6", bf.AND(6)),
    ("OR_4", bf.OR(4)),
    ("Majority_5", bf.majority(5)),
]

print(f"{'Function':<12} {'n':<4} {'Grover Speedup':<15} {'Interpretation'}")
print("-" * 55)

for name, f in functions:
    result = grover_speedup(f)
    speedup = result['speedup']
    n = f.n_vars
    
    if speedup > 3:
        interp = "High speedup"
    elif speedup > 1.5:
        interp = "Moderate speedup"
    else:
        interp = "Minimal speedup"
    
    print(f"{name:<12} {n:<4} {speedup:<15.2f}x {interp}")

print("""
Observations:
- AND has 1 solution → maximum Grover speedup
- OR has many solutions → less speedup needed
- Majority has 50% solutions → modest speedup
""")

# =============================================================================
# 3. Quantum Advantage Estimation
# =============================================================================
print("\n--- 3. Quantum Advantage Estimation ---\n")

print("""
The estimate_quantum_advantage function provides a comprehensive
analysis of potential quantum speedups for a given function.
""")

f = bf.AND(4)
grover_result = grover_speedup(f)

print(f"AND_4 Quantum Advantage Analysis:")
print(f"  Function: AND with {f.n_vars} variables")
print(f"  Solutions: {grover_result['num_solutions']}")
print(f"  Classical queries: {grover_result['classical_queries']:.1f}")
print(f"  Grover queries: {grover_result['grover_queries']:.2f}")
print(f"  Speedup: {grover_result['speedup']:.2f}x")
print(f"  Optimal iterations: {grover_result['optimal_iterations']}")

# =============================================================================
# 4. Quantum Walk Algorithms
# =============================================================================
print("\n--- 4. Quantum Walk Algorithms ---\n")

print("""
Quantum walks generalize classical random walks.
They can achieve polynomial speedups for various problems.

Key applications:
- Element distinctness: O(n^(2/3)) vs classical O(n)
- Spatial search on graphs
- Triangle finding
""")

# Analyze quantum walks
print("Quantum walk analysis:\n")

for name, f in [("AND_4", bf.AND(4)), ("OR_4", bf.OR(4))]:
    walk = quantum_walk_analysis(f)
    
    print(f"{name}:")
    print(f"  Classical hitting time: {walk['classical_hitting_time']:.2f}")
    print(f"  Quantum hitting time: {walk['quantum_hitting_time']:.2f}")
    print(f"  Speedup: {walk['speedup_over_classical']:.2f}x")
    print()

# =============================================================================
# 5. Element Distinctness
# =============================================================================
print("\n--- 5. Element Distinctness Problem ---\n")

print("""
Element Distinctness: Given a list, determine if all elements are distinct.

Classical: O(n) or O(n log n) depending on model
Quantum: O(n^(2/3)) using Ambainis' quantum walk algorithm!

This is optimal - matches the quantum lower bound.
""")

# Analyze element distinctness for different sizes
print("Element distinctness complexity:\n")
print(f"{'n':<8} {'Classical':<12} {'Quantum':<12} {'Speedup'}")
print("-" * 40)

for n in [8, 16, 64, 256, 1024]:
    classical = n
    quantum = n ** (2/3)
    speedup = classical / quantum
    
    print(f"{n:<8} {classical:<12.0f} {quantum:<12.1f} {speedup:.2f}x")

# =============================================================================
# 6. Quantum Query Complexity Bounds
# =============================================================================
print("\n--- 6. Quantum Query Complexity Bounds ---\n")

print("""
Key lower bounds for quantum query complexity:

1. Polynomial Method: Q₂(f) ≥ deg(f)/2
   where deg(f) is the approximate degree

2. Adversary Method: Q₂(f) ≥ Adv(f)
   General adversary gives tight bounds for many functions

3. For symmetric functions:
   Q₂(OR_n) = Θ(√n)
   Q₂(Majority_n) = Θ(n)  (no quantum speedup!)
   Q₂(Parity_n) = Θ(n)    (no quantum speedup!)
""")

print("Quantum vs Classical complexity:\n")

complexity_data = [
    ("OR", "√n", "n", "Quadratic"),
    ("AND", "√n", "n", "Quadratic"),
    ("Majority", "n", "n", "None"),
    ("Parity", "n", "n", "None"),
    ("Element Dist.", "n^(2/3)", "n", "Polynomial"),
]

print(f"{'Function':<15} {'Q₂(f)':<10} {'D(f)':<10} {'Speedup Type'}")
print("-" * 50)

for name, q, d, speedup in complexity_data:
    print(f"{name:<15} {q:<10} {d:<10} {speedup}")

# =============================================================================
# 7. Grover Speedup Scaling
# =============================================================================
print("\n--- 7. Grover Speedup Scaling ---\n")

print("""
Grover's algorithm achieves quadratic speedup.
The speedup grows as √N where N = 2ⁿ.
""")

print("Grover speedup for OR_n (unstructured search):\n")
print(f"{'n':<6} {'N=2^n':<12} {'Classical':<12} {'Quantum':<12} {'Speedup'}")
print("-" * 55)

for n in [4, 6, 8, 10, 12]:
    N = 2**n
    classical = N  # Expected queries to find 1 element
    quantum = np.sqrt(N)
    speedup = classical / quantum
    
    print(f"{n:<6} {N:<12} {classical:<12} {quantum:<12.1f} {speedup:.1f}x")

# =============================================================================
# 8. Practical Quantum Advantage
# =============================================================================
print("\n--- 8. When Does Quantum Help? ---\n")

print("""
Summary of quantum advantages for Boolean functions:

DEFINITE SPEEDUP:
- Unstructured search (OR): quadratic
- Element distinctness: n^(2/3) vs n
- Collision finding: polynomial

NO SPEEDUP:
- Parity: requires reading all bits
- Majority: inherently hard
- Most "structured" problems

DEPENDS ON STRUCTURE:
- Property testing: often polynomial speedup in ε
- Learning: depends on function class
- Optimization: problem-dependent

Key insight: Quantum helps most when the problem has
"unstructured" or "symmetric" components.
""")

# =============================================================================
# 9. Comparing Functions
# =============================================================================
print("\n--- 9. Quantum Analysis Summary ---\n")

functions = [
    ("AND_4", bf.AND(4)),
    ("OR_4", bf.OR(4)),
    ("Parity_4", bf.parity(4)),
    ("Majority_5", bf.majority(5)),
]

print(f"{'Function':<12} {'Grover':<10} {'Walk':<10} {'Best Speedup'}")
print("-" * 45)

for name, f in functions:
    grover_result = grover_speedup(f)
    grover_sp = grover_result['speedup']
    walk = quantum_walk_analysis(f)
    walk_sp = walk['speedup_over_classical']
    best = max(grover_sp, walk_sp)
    
    print(f"{name:<12} {grover_sp:<10.2f}x {walk_sp:<10.2f}x {best:.2f}x")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- grover_speedup(f): Estimates Grover's algorithm advantage")
print("- quantum_walk_analysis(f): Analyzes quantum walk speedups")
print("- element_distinctness_analysis(): O(n^(2/3)) algorithm")
print("- Grover's algorithm: quadratic speedup for search")
print("- Not all functions benefit: Parity, Majority have no speedup")
print("- Best speedups for 'unstructured' components")
print("=" * 60)
