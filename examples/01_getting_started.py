#!/usr/bin/env python3
"""
Tutorial 1: Getting Started with BoolFunc
==========================================

This tutorial covers:
- Creating Boolean functions from truth tables
- Using built-in function generators
- Basic evaluation and properties
- Converting between representations

Prerequisites: pip install boolfunc
"""

import numpy as np

# Import the library
import boolfunc as bf

print("=" * 60)
print("Tutorial 1: Getting Started with BoolFunc")
print("=" * 60)


# =============================================================================
# 1. Creating Boolean Functions
# =============================================================================
print("\n--- 1. Creating Boolean Functions ---\n")

# From a truth table (most direct way)
# XOR function: f(x) = x_0 ⊕ x_1
xor_table = [0, 1, 1, 0]  # f(00)=0, f(01)=1, f(10)=1, f(11)=0
xor = bf.create(xor_table)
print(f"XOR function created with {xor.n_vars} variables")

# Using built-in generators (recommended for standard functions)
and_3 = bf.AND(3)        # AND of 3 variables
or_3 = bf.OR(3)          # OR of 3 variables
maj_5 = bf.majority(5)   # Majority of 5 variables
par_4 = bf.parity(4)     # Parity (XOR) of 4 variables
print(f"AND_3: {and_3.n_vars} vars, MAJ_5: {maj_5.n_vars} vars")


# =============================================================================
# 2. Evaluating Functions
# =============================================================================
print("\n--- 2. Evaluating Functions ---\n")

# Evaluate at a specific input
# Input as integer index (0 to 2^n - 1)
result = and_3.evaluate(7)  # 7 = 111 in binary
print(f"AND_3(1,1,1) = {result}")

result = and_3.evaluate(3)  # 3 = 011 in binary
print(f"AND_3(0,1,1) = {result}")

# Input as binary array
result = maj_5.evaluate([1, 1, 1, 0, 0])
print(f"MAJ_5(1,1,1,0,0) = {result}")

result = maj_5.evaluate([1, 0, 0, 0, 0])
print(f"MAJ_5(1,0,0,0,0) = {result}")


# =============================================================================
# 3. Basic Properties
# =============================================================================
print("\n--- 3. Basic Properties ---\n")

f = bf.majority(5)

# Number of variables
print(f"Number of variables: {f.n_vars}")

# Get truth table to compute expectation
tt = f.get_representation("truth_table")
exp = sum(1 for x in tt if x) / len(tt)
print(f"Expectation E[f]: {exp:.3f}")

# Variance
var = f.variance()
print(f"Variance Var[f]: {var:.3f}")

# Is the function balanced? (outputs 0 and 1 equally often)
is_balanced = f.is_balanced()
print(f"Is balanced: {is_balanced}")


# =============================================================================
# 4. Truth Table Access
# =============================================================================
print("\n--- 4. Truth Table Access ---\n")

f = bf.AND(3)
truth_table = f.get_representation("truth_table")

print("AND_3 truth table:")
for i in range(2**3):
    bits = format(i, '03b')
    print(f"  {bits} -> {truth_table[i]}")


# =============================================================================
# 5. Comparing Functions
# =============================================================================
print("\n--- 5. Comparing Functions ---\n")

# Create two versions of the same function
f1 = bf.parity(3)
f2 = bf.create([0, 1, 1, 0, 1, 0, 0, 1])  # Parity truth table

# Compare by truth tables
tt1 = f1.get_representation("truth_table")
tt2 = f2.get_representation("truth_table")

are_equal = np.array_equal(tt1, tt2)
print(f"f1 == f2: {are_equal}")


# =============================================================================
# 6. Common Function Types
# =============================================================================
print("\n--- 6. Common Function Types ---\n")

functions = {
    "AND_4": bf.AND(4),
    "OR_4": bf.OR(4),
    "Parity_4": bf.parity(4),
    "Majority_5": bf.majority(5),
    "Threshold_3/5": bf.threshold(5, 3),  # At least 3 of 5
    "Dictator_0": bf.dictator(4, 0),      # Just returns x_0
}

print(f"{'Function':<15} {'n_vars':<8} {'Balanced':<10} {'Var[f]':<10}")
print("-" * 45)
for name, f in functions.items():
    print(f"{name:<15} {f.n_vars:<8} {str(f.is_balanced()):<10} {f.variance():<10.3f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- Create functions with bf.create(truth_table) or bf.AND(n), etc.")
print("- Evaluate with f.evaluate(index) or f.evaluate([bits])")
print("- Get properties: f.n_vars, f.is_balanced(), f.variance()")
print("- Access truth table: f.get_representation('truth_table')")
print("=" * 60)
