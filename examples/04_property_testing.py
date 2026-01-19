#!/usr/bin/env python3
"""
Tutorial 4: Property Testing with BoolFunc
==========================================

Property testing allows us to quickly determine if a Boolean function
has certain properties (like linearity, monotonicity) using only a 
small number of queries.

Topics covered:
- BLR linearity test
- Junta testing
- Monotonicity testing  
- Dictator/anti-dictator testing
- Symmetry and balance testing
"""

import numpy as np
import boolfunc as bf
from boolfunc.testing import PropertyTester

print("=" * 60)
print("Tutorial 4: Property Testing")
print("=" * 60)

# =============================================================================
# 1. Introduction to Property Testing
# =============================================================================
print("\n--- 1. What is Property Testing? ---\n")

print("""
Property testing asks: "Does this function have property P?"

Instead of checking ALL inputs (exponential), we sample a few
random inputs and make a probabilistic determination.

Key guarantee: If f has property P, we accept with high probability.
              If f is ε-far from P, we reject with high probability.
""")

# =============================================================================
# 2. BLR Linearity Test
# =============================================================================
print("\n--- 2. BLR Linearity Test ---\n")

print("""
The BLR test checks if f(x) ⊕ f(y) = f(x ⊕ y) for random x, y.

A function is linear if f(x) = a·x (inner product with some vector a).
Parity functions are linear; AND/OR/Majority are not.
""")

# Test various functions
functions = [
    ("Parity_5", bf.parity(5)),
    ("AND_5", bf.AND(5)),
    ("OR_5", bf.OR(5)),
    ("Majority_5", bf.majority(5)),
]

print(f"{'Function':<15} {'Expected Linear':<15} {'BLR Result'}")
print("-" * 45)

for name, f in functions:
    tester = PropertyTester(f)
    
    # Run BLR test (returns bool)
    blr_result = tester.blr_linearity_test(num_queries=100)
    
    # Check if actually linear (parity is linear)
    expected = "parity" in name.lower()
    
    status = "✓" if blr_result == expected else "✗"
    print(f"{name:<15} {str(expected):<15} {blr_result} {status}")

# =============================================================================
# 3. Junta Testing
# =============================================================================
print("\n--- 3. Junta Testing ---\n")

print("""
A k-junta depends on at most k variables.
- Dictator (depends on 1 variable) is a 1-junta
- AND_n depends on all n variables, not a junta for k < n

The junta test identifies functions with few relevant variables.
""")

# Create functions with different junta properties
dictator = bf.dictator(5, 0)  # 1-junta
and_5 = bf.AND(5)             # 5-junta (needs all vars)

# Simple relevance check via influences
print("Checking variable relevance via influences:\n")

for name, f in [("Dictator_0", dictator), ("AND_5", and_5)]:
    influences = f.influences()
    relevant = sum(1 for i in influences if i > 0.01)
    print(f"{name}:")
    print(f"  Influences: {[f'{x:.3f}' for x in influences]}")
    print(f"  Relevant variables: {relevant}")
    print()

# Junta test
print("Junta test results:")
for name, f in [("Dictator_0", dictator), ("AND_5", and_5)]:
    tester = PropertyTester(f)
    result = tester.junta_test(k=2)
    print(f"  {name}: is 2-junta? {result}")

# =============================================================================
# 4. Monotonicity Testing
# =============================================================================
print("\n--- 4. Monotonicity Testing ---\n")

print("""
A function is monotone if x ≤ y implies f(x) ≤ f(y).
(Here ≤ means coordinate-wise comparison)

Examples:
- AND, OR, Majority are monotone
- XOR, Parity are NOT monotone
""")

test_functions = [
    ("AND_4", bf.AND(4), True),
    ("OR_4", bf.OR(4), True),
    ("Majority_5", bf.majority(5), True),
    ("Parity_4", bf.parity(4), False),
]

print(f"{'Function':<15} {'Expected':<12} {'Test Result'}")
print("-" * 40)

for name, f, expected in test_functions:
    tester = PropertyTester(f)
    result = tester.monotonicity_test()
    
    status = "✓" if result == expected else "✗"
    print(f"{name:<15} {str(expected):<12} {result} {status}")

# =============================================================================
# 5. Symmetry Testing
# =============================================================================
print("\n--- 5. Symmetry Testing ---\n")

print("""
A symmetric function depends only on the Hamming weight of its input.
f(x) = g(|x|) for some function g.

Examples:
- AND, OR, Majority, Parity, Threshold are symmetric
- Dictator is NOT symmetric (depends on which bit)
""")

test_functions = [
    ("Majority_5", bf.majority(5), True),
    ("Parity_4", bf.parity(4), True),
    ("Dictator_0", bf.dictator(4, 0), False),
    ("AND_4", bf.AND(4), True),
]

print(f"{'Function':<15} {'Expected':<12} {'Test Result'}")
print("-" * 40)

for name, f, expected in test_functions:
    tester = PropertyTester(f)
    result = tester.symmetry_test()
    
    status = "✓" if result == expected else "✗"
    print(f"{name:<15} {str(expected):<12} {result} {status}")

# =============================================================================
# 6. Balance Testing
# =============================================================================
print("\n--- 6. Balance Testing ---\n")

print("""
A balanced function has equal number of 0s and 1s in its truth table.
Equivalently, E[f] = 0.5 (or f̂(∅) = 0 in ±1 representation).

Examples:
- Parity, Majority (odd n), XOR are balanced
- AND, OR are NOT balanced
""")

test_functions = [
    ("Parity_4", bf.parity(4)),
    ("Majority_5", bf.majority(5)),
    ("AND_4", bf.AND(4)),
    ("OR_4", bf.OR(4)),
]

print(f"{'Function':<15} {'Balanced?':<12} {'Bias'}")
print("-" * 40)

for name, f in test_functions:
    is_balanced = f.is_balanced()
    bias = f.bias()
    
    tester = PropertyTester(f)
    test_result = tester.balanced_test()
    
    print(f"{name:<15} {str(is_balanced):<12} {bias:.3f}")

# =============================================================================
# 7. Dictator and Anti-Dictator Testing
# =============================================================================
print("\n--- 7. Dictator Testing ---\n")

print("""
A dictator function is f(x) = x_i for some variable i.
An anti-dictator is f(x) = NOT x_i.

By the FKN theorem, functions close to dictators have:
- Low total influence (close to 1)
- One variable with influence close to 1
""")

test_functions = [
    ("Dictator_0", bf.dictator(5, 0)),
    ("Majority_5", bf.majority(5)),
    ("Parity_5", bf.parity(5)),
]

print(f"{'Function':<15} {'Max Inf':<10} {'Total Inf':<12} {'Dictator Test'}")
print("-" * 55)

for name, f in test_functions:
    max_inf = f.max_influence()
    total_inf = f.total_influence()
    
    tester = PropertyTester(f)
    is_dict = tester.dictator_test()
    
    print(f"{name:<15} {max_inf:<10.3f} {total_inf:<12.3f} {is_dict}")

# =============================================================================
# 8. Running All Tests
# =============================================================================
print("\n--- 8. Complete Test Suite ---\n")

print("Running all property tests on Majority_5:\n")

f = bf.majority(5)
tester = PropertyTester(f)

# Run all tests
results = tester.run_all_tests()

print(f"{'Property':<20} {'Result'}")
print("-" * 30)

for prop, result in results.items():
    print(f"{prop:<20} {result}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Key Takeaways:")
print("- BLR test: O(1/ε) queries to test linearity")
print("- Monotonicity test: O(n/ε) queries")
print("- Junta test: Identifies functions with few relevant variables")
print("- Symmetry test: Checks if f depends only on Hamming weight")
print("- Use PropertyTester(f).run_all_tests() for complete analysis")
print("=" * 60)
