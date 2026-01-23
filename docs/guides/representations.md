# Representations Guide

BooFun supports 10+ representations of Boolean functions with automatic conversion between them.

## Overview

Boolean functions can be represented in many ways, each suited for different tasks:

| Representation | Best For |
|---------------|----------|
| Truth table | Fast evaluation, small n |
| Fourier expansion | Spectral analysis |
| ANF | Algebraic analysis, cryptography |
| DNF/CNF | Logic, SAT solving |
| BDD | Compact storage, verification |
| Circuit | Complexity analysis |
| LTF | Weighted voting, learning |

## Available Representations

### Truth Tables

Dense array of function outputs.

| Name | Description | Memory | Access |
|------|-------------|--------|--------|
| `truth_table` | Dense NumPy array | O(2^n) | O(1) |
| `sparse_truth_table` | Only true inputs | O(k) | O(1) |
| `packed_truth_table` | Bit-packed array | O(2^n / 8) | O(1) |

```python
import boofun as bf

# Dense truth table (default)
f = bf.create([0, 1, 1, 0])

# Access truth table
tt = f.get_representation("truth_table")
print(f"Truth table: {tt}")

# For large sparse functions
sparse_f = bf.create({(0, 1), (1, 0)}, n=2)  # Only true inputs
sparse_tt = sparse_f.get_representation("sparse_truth_table")
```

### Fourier Expansion

Fourier coefficients over the Boolean hypercube.

```python
f = bf.majority(5)

# Get Fourier expansion
fourier = f.get_representation("fourier_expansion")

# Access coefficients
print(f"f̂(∅) = {f.fourier_coefficient(frozenset()):.4f}")
print(f"f̂({{0}}) = {f.fourier_coefficient(frozenset([0])):.4f}")
```

### Algebraic Normal Form (ANF)

Polynomial over GF(2) - XOR of AND terms.

```python
f = bf.create([0, 0, 0, 1])  # AND function

anf = f.get_representation("anf")
print(f"ANF: {anf}")
# Output: x0 ∧ x1  (or similar notation)

# Algebraic degree
from boofun.analysis import cryptographic
deg = cryptographic.algebraic_degree(f)
print(f"Algebraic degree: {deg}")
```

### Real Polynomial

Polynomial over the reals (multilinear).

```python
f = bf.majority(3)

poly = f.get_representation("polynomial")
# Represents f as sum of monomials with real coefficients
```

### DNF/CNF

Disjunctive/Conjunctive Normal Forms.

```python
f = bf.create([0, 0, 0, 1, 0, 1, 1, 1])  # Threshold function

# Get DNF (OR of ANDs)
dnf = f.get_representation("dnf")

# Get CNF (AND of ORs)
cnf = f.get_representation("cnf")
```

### Binary Decision Diagram (BDD)

Compact graph representation.

```python
f = bf.tribes(2, 4)

bdd = f.get_representation("bdd")
print(f"BDD nodes: {bdd.node_count()}")
```

### Circuit

Boolean circuit representation.

```python
f = bf.majority(5)

circuit = f.get_representation("circuit")
print(f"Circuit depth: {circuit.depth()}")
print(f"Circuit size: {circuit.size()}")
```

### Linear Threshold Function (LTF)

Weighted voting representation: f(x) = sign(w·x - θ).

```python
# Create weighted majority
weights = [3, 2, 1, 1, 1]
f = bf.weighted_majority(weights)

ltf = f.get_representation("ltf")
print(f"Weights: {ltf.weights}")
print(f"Threshold: {ltf.threshold}")
```

### Symbolic

Human-readable string expression.

```python
f = bf.create("x0 and (x1 or not x2)", n=3)

symbolic = f.get_representation("symbolic")
print(f"Expression: {symbolic}")
```

## Automatic Conversion

BooFun automatically converts between representations as needed.

### Getting Any Representation

```python
f = bf.majority(5)

# Get any representation - auto-converts if needed
fourier = f.get_representation("fourier_expansion")
anf = f.get_representation("anf")
dnf = f.get_representation("dnf")

# Converted representations are cached
```

### Convert In Place

```python
f = bf.create([0, 1, 1, 0])

# Convert to different representation
f.convert_to("fourier_expansion")

# Now Fourier is the primary representation
```

### Check Conversion Paths

```python
from boofun.core.conversion_graph import get_conversion_path, can_convert

# Check if conversion is possible
if can_convert("truth_table", "bdd"):
    path = get_conversion_path("truth_table", "bdd")
    print(f"Conversion path: {' -> '.join(path)}")
```

## Storage Hints

For large functions, specify storage format hints.

### Packed Truth Tables

8x memory savings for large n.

```python
# Create with packed storage
f = bf.create(large_truth_table, storage="packed")

# Or hint during creation
from boofun import create
f = create(data, storage="packed")
```

### Sparse Storage

Efficient when few inputs are true.

```python
# For functions with <30% true inputs
f = bf.create(sparse_data, storage="sparse")

# Auto-detection
f = bf.create(data, storage="auto")  # Chooses best format
```

### Memory Comparison

```python
from boofun.core.representations.packed_truth_table import memory_comparison

# Compare memory usage for n=20
comparison = memory_comparison(20)
print(comparison)
# packed_bitarray: 131,072 bytes (128.0 KB)
# numpy_bool: 1,048,576 bytes (1024.0 KB)
# savings: 8x
```

## Conversion Graph

The conversion graph shows all possible paths between representations.

```
              truth_table
             /     |     \
      fourier    bdd     sparse
         |        |
        anf   circuit
         |
       dnf/cnf
```

### Direct Conversions

| From | To | Method |
|------|-----|--------|
| truth_table | fourier | Walsh-Hadamard Transform |
| fourier | truth_table | Inverse WHT |
| truth_table | anf | Möbius transform |
| anf | truth_table | Inverse Möbius |
| truth_table | bdd | Shannon expansion |
| any | symbolic | String formatting |

## Choosing Representations

### For Evaluation

- Small n (≤ 14): `truth_table`
- Large n, sparse: `sparse_truth_table`
- Large n, dense: `packed_truth_table`

### For Analysis

- Spectral analysis: `fourier_expansion`
- Algebraic analysis: `anf`
- Complexity: `circuit`, `bdd`
- Cryptography: `anf`, `truth_table`

### For Storage

- Compact: `bdd`, `sparse_truth_table`
- Fast load: `truth_table`
- Human-readable: `symbolic`

## Creating from Different Sources

### From Truth Table

```python
# List
f = bf.create([0, 1, 1, 0])

# NumPy array
import numpy as np
f = bf.create(np.array([0, 1, 1, 0]))
```

### From Callable

```python
f = bf.create(lambda x: x[0] ^ x[1], n=2)
```

### From True Inputs

```python
# Set of inputs where f=1
f = bf.create({(0, 1), (1, 0)}, n=2)
```

### From Polynomial

```python
# Dict mapping monomials to coefficients
f = bf.create({
    frozenset(): 0,           # constant term
    frozenset([0]): 1,        # x0
    frozenset([1]): 1,        # x1
    frozenset([0, 1]): -2     # -2·x0·x1
}, n=2)
```

### From String

```python
f = bf.create("x0 and x1", n=2)
f = bf.create("x0 XOR x1", n=2)
f = bf.create("(x0 | x1) & ~x2", n=3)
```

### From File

```python
f = bf.load("function.json")   # JSON format
f = bf.load("function.bf")     # Aaronson format
f = bf.load("function.cnf")    # DIMACS CNF
```

## Saving Functions

```python
f = bf.majority(5)

# Save to various formats
bf.save(f, "maj5.json")    # JSON with metadata
bf.save(f, "maj5.bf")      # Aaronson format
```

## See Also

- [Spectral Analysis Guide](spectral_analysis.md) - Fourier representation
- [Cryptographic Guide](cryptographic.md) - ANF and algebraic analysis
- [Performance Guide](../performance.md) - Memory optimization
