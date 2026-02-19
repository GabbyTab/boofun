# Query Complexity Guide

Deterministic, randomized, and quantum query complexity measures for Boolean functions.

## Overview

Query complexity measures how many input bits must be queried to compute a Boolean function. BooFun provides comprehensive tools including:

- Decision tree complexity (D, D_avg)
- Randomized complexity (R₀, R₁, R₂)
- Quantum complexity (Q₂, QE) and lower bounds
- Sensitivity measures (s, bs, es)
- Certificate complexity (C, C₀, C₁)
- Degree measures (exact, approximate, threshold)
- Decision tree algorithms (DP, enumeration)

## Decision Tree Complexity

The fundamental deterministic query measure.

| Measure | Function | Description |
|---------|----------|-------------|
| D(f) | `complexity.decision_tree_depth(f)` | Deterministic depth |
| D_avg(f) | `complexity.average_decision_tree_depth(f)` | Average depth |
| Optimal tree (DP) | `decision_trees.decision_tree_depth_dp(f)` | DP algorithm |
| Randomized depth | `decision_trees.compute_randomized_complexity(f)` | R(f) |
| Count optimal trees | `decision_trees.count_decision_trees(f)` | Enumeration |

### Example: Decision Tree Analysis

```python
import boofun as bf
from boofun.analysis import complexity
from boofun.analysis.decision_trees import (
    decision_tree_depth_dp,
    count_decision_trees
)

f = bf.majority(5)

# Deterministic decision tree depth
D_f = complexity.decision_tree_depth(f)
print(f"D(MAJ_5) = {D_f}")

# Using DP algorithm
D_dp = decision_tree_depth_dp(f)
print(f"D(f) via DP = {D_dp}")

# Count number of optimal decision trees
count = count_decision_trees(f)
print(f"Number of optimal trees: {count}")
```

## Sensitivity Measures

How sensitive is f to single-bit changes?

| Measure | Function | Description |
|---------|----------|-------------|
| s(f) | `complexity.max_sensitivity(f)` | Sensitivity |
| bs(f) | `complexity.block_sensitivity(f)` | Block sensitivity |
| es(f) | `complexity.everywhere_sensitivity(f)` | Everywhere sensitivity |

### Sensitivity vs Block Sensitivity

- **Sensitivity s(f)**: Maximum over all x of the number of single-bit flips that change f(x)
- **Block sensitivity bs(f)**: Maximum over all x of the number of *disjoint* blocks whose flip changes f(x)

```python
from boofun.analysis import complexity

f = bf.AND(5)

s = complexity.max_sensitivity(f)
bs = complexity.block_sensitivity(f)

print(f"s(AND_5) = {s}")    # = 1
print(f"bs(AND_5) = {bs}")  # = 5

# Note: bs(f) ≥ s(f) always, with possible polynomial gap
```

## Certificate Complexity

Minimum number of bits that "prove" a function value.

| Measure | Function | Description |
|---------|----------|-------------|
| C(f) | `complexity.certificate_complexity(f)` | Certificate complexity |
| C₀(f) | `complexity.max_certificate_complexity(f, target=0)` | 0-certificate |
| C₁(f) | `complexity.max_certificate_complexity(f, target=1)` | 1-certificate |

### Example: Certificates

```python
from boofun.analysis.complexity import (
    certificate_complexity,
    max_certificate_complexity
)

f = bf.OR(5)

C_f = certificate_complexity(f)
C_0 = max_certificate_complexity(f, target=0)
C_1 = max_certificate_complexity(f, target=1)

print(f"C(OR_5) = {C_f}")
print(f"C_0(OR_5) = {C_0}")  # Need to see all 0s
print(f"C_1(OR_5) = {C_1}")  # Just need one 1
```

## Quantum Complexity

Quantum query complexity and lower bounds.

| Measure | Function | Description |
|---------|----------|-------------|
| Ambainis bound | `query_complexity.ambainis_complexity(f)` | Quantum lower bound |
| Spectral adversary | `query_complexity.spectral_adversary(f)` | Spectral method |
| Polynomial method | `query_complexity.polynomial_method_bound(f)` | deg(f) bound |

### Example: Quantum Bounds

```python
from boofun.analysis import query_complexity as qc

f = bf.OR(4)

# Ambainis adversary method lower bound
amb = qc.ambainis_complexity(f)
print(f"Ambainis bound: Q(OR_4) ≥ {amb:.2f}")

# For comparison, classical deterministic
D_f = complexity.decision_tree_depth(f)
print(f"D(OR_4) = {D_f}")

# Quantum can achieve sqrt speedup for OR
```

## Degree Measures

Polynomial degree measures related to query complexity.

| Measure | Function | Description |
|---------|----------|-------------|
| deg(f) | `complexity.exact_degree(f)` | Exact degree |
| deg̃(f) | `complexity.approximate_degree(f)` | Approximate degree |
| deg_th(f) | `complexity.threshold_degree(f)` | Threshold degree |

### Example: Degree Analysis

```python
from boofun.analysis import complexity

f = bf.parity(4)

deg = complexity.exact_degree(f)
print(f"deg(PAR_4) = {deg}")  # = 4 (full degree)

f = bf.OR(4)
deg = complexity.exact_degree(f)
print(f"deg(OR_4) = {deg}")   # = 4
```

## Huang's Theorem

The celebrated result connecting sensitivity to degree.

| Function | Description |
|----------|-------------|
| `huang.sensitivity_lower_bound(f)` | s(f) ≥ √deg(f) |
| `huang.verify_huang_theorem(f)` | Verify the relationship |

### Example: Verifying Huang's Theorem

```python
from boofun.analysis import huang

f = bf.AND(6)

# Verify Huang's theorem: s(f) >= sqrt(deg(f))
result = huang.verify_huang_theorem(f)
print(f"s(f) = {result['sensitivity']}")
print(f"deg(f) = {result['degree']}")
print(f"sqrt(deg(f)) = {result['sqrt_degree']:.2f}")
print(f"Huang satisfied: {result['satisfied']}")
```

## Complexity Relationships

Known relationships between measures (all polynomial):

```
       s(f) ≤ bs(f) ≤ C(f) ≤ D(f)
         ↓
    deg(f) ≤ D(f)
         ↓
      Q(f) ≤ D(f)

Key results:
- D(f) ≤ bs(f)² (classical)
- s(f) ≥ √deg(f) (Huang 2019)
- Q(f) = Θ(√D(f)) for some functions (Grover)
```

## Full Complexity Profile

Get all measures at once:

```python
from boofun.analysis.query_complexity import QueryComplexityProfile

f = bf.majority(5)

profile = QueryComplexityProfile(f)
print(profile.summary())

# Access individual measures
print(f"D(f) = {profile.deterministic_depth}")
print(f"s(f) = {profile.sensitivity}")
print(f"bs(f) = {profile.block_sensitivity}")
print(f"C(f) = {profile.certificate_complexity}")
```

## Decision Tree Algorithms

Advanced algorithms for decision tree analysis.

### DP Algorithm

Compute optimal decision tree depth via dynamic programming:

```python
from boofun.analysis.decision_trees import decision_tree_depth_dp

f = bf.tribes(2, 4)  # 2 tribes of 4

depth = decision_tree_depth_dp(f)
print(f"D(TRIBES) = {depth}")
```

### Tree Enumeration

Count the number of optimal decision trees:

```python
from boofun.analysis.decision_trees import count_decision_trees

f = bf.majority(3)

count = count_decision_trees(f)
print(f"Number of optimal trees for MAJ_3: {count}")
```

### Randomized Complexity

Compute randomized decision tree complexity:

```python
from boofun.analysis.decision_trees import compute_randomized_complexity

f = bf.OR(4)

R_f = compute_randomized_complexity(f)
print(f"R(OR_4) = {R_f:.2f}")
```

## See Also

- [Spectral Analysis Guide](spectral_analysis.md) - Fourier analysis and influences
- [Hypercontractivity Guide](hypercontractivity.md) - Advanced influence bounds
- Aaronson, "Algorithms for Boolean Function Query Measures" (2000)
- Buhrman & de Wolf, "Complexity Measures and Decision Tree Complexity" (2002)
- Huang, "Induced subgraphs of hypercubes and a proof of the Sensitivity Conjecture" (2019)
