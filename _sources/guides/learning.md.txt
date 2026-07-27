# Learning Theory Guide

Algorithms for learning Boolean functions from queries or samples.

## Overview

BooFun provides implementations of key learning algorithms from computational learning theory:

- **Goldreich-Levin**: Find heavy Fourier coefficients with query access
- **PAC Learning**: Probably Approximately Correct learning framework
- **Junta Learning**: Learn functions that depend on few variables
- **LMN Algorithm**: Learn decision trees from uniform samples
- **Sparse Fourier Learning**: Learn functions with few Fourier coefficients

## Goldreich-Levin Algorithm

The Goldreich-Levin algorithm finds all "heavy" Fourier coefficients (|f̂(S)| ≥ τ) using only O(n/τ²) queries.

```python
from boofun.analysis.learning import goldreich_levin, find_heavy_coefficients

f = bf.majority(7)

# Find all coefficients with |f̂(S)| ≥ 0.1
heavy = goldreich_levin(f, threshold=0.1, num_samples=5000)
print(f"Found {len(heavy)} heavy coefficients")

for S, coeff in heavy:
    print(f"  f̂({S}) ≈ {coeff:.4f}")
```

### Estimating Single Coefficients

```python
from boofun.analysis.learning import estimate_fourier_coefficient

f = bf.parity(5)

# Estimate f̂({0,1,2,3,4}) - should be ±1 for parity
S = 0b11111  # All variables
estimate, std_err = estimate_fourier_coefficient(f, S, num_samples=10000)
print(f"Estimated f̂(S) = {estimate:.4f} ± {std_err:.4f}")
```

### GoldreichLevinLearner Class

For more control over the learning process:

```python
from boofun.analysis.learning import GoldreichLevinLearner

learner = GoldreichLevinLearner(threshold=0.1, confidence=0.95)

# Learn from query access
heavy_coeffs = learner.learn(f, num_samples=10000)

# Get learning statistics
print(f"Queries used: {learner.query_count}")
print(f"Coefficients found: {len(heavy_coeffs)}")
```

## PAC Learning

The PAC (Probably Approximately Correct) framework:
- **Given:** Sample access to f (can draw (x, f(x)) for random x)
- **Goal:** Output hypothesis h such that Pr[h(x) != f(x)] <= epsilon
- **With probability:** at least 1 - delta

### Learning Low-Degree Functions

Functions with spectral concentration at low degrees:

```python
from boofun.analysis.pac_learning import pac_learn_low_degree

f = bf.majority(5)

# Learn degree-2 approximation
hypothesis, error = pac_learn_low_degree(
    f,
    degree=2,
    epsilon=0.1,
    delta=0.05,
    num_samples=10000
)

print(f"Approximation error: {error:.4f}")
```

### Learning Juntas

A k-junta depends on at most k variables:

```python
from boofun.analysis.pac_learning import pac_learn_junta

# Create a 3-junta (depends on variables 0, 2, 4 only)
def junta_func(x):
    return x[0] ^ x[2] ^ x[4]

f = bf.create(junta_func, n=7)

# Learn the junta
hypothesis, relevant_vars = pac_learn_junta(
    f,
    k=3,
    epsilon=0.1,
    delta=0.05
)

print(f"Relevant variables: {relevant_vars}")
```

### LMN Algorithm

Learn decision trees from uniform samples (Linial-Mansour-Nisan):

```python
from boofun.analysis.pac_learning import lmn_algorithm

f = bf.tribes(2, 3)  # A read-once DNF

# Learn using LMN
hypothesis = lmn_algorithm(
    f,
    epsilon=0.1,
    delta=0.05
)

# Test accuracy
from boofun.analysis.pac_learning import sample_function
test_samples = sample_function(f, 1000)
correct = sum(1 for x, y in test_samples if hypothesis(x) == y)
print(f"Accuracy: {correct/1000:.1%}")
```

### Learning Sparse Fourier Functions

Functions with few non-zero Fourier coefficients:

```python
from boofun.analysis.pac_learning import pac_learn_sparse_fourier

f = bf.parity(4)  # Has only one non-zero coefficient

hypothesis, support = pac_learn_sparse_fourier(
    f,
    sparsity=5,
    epsilon=0.1,
    delta=0.05
)

print(f"Learned Fourier support: {support}")
```

### Learning Monotone Functions

```python
from boofun.analysis.pac_learning import pac_learn_monotone

f = bf.AND(4)  # Monotone function

hypothesis = pac_learn_monotone(
    f,
    epsilon=0.1,
    delta=0.05
)
```

## PACLearner Class

General-purpose PAC learner:

```python
from boofun.analysis.pac_learning import PACLearner

learner = PACLearner(epsilon=0.1, delta=0.05)

# Learn from samples
hypothesis = learner.learn(f, num_samples=10000)

# Evaluate
error = learner.evaluate(hypothesis, f, num_test=1000)
print(f"Test error: {error:.4f}")

# Get statistics
print(f"Samples used: {learner.sample_count}")
```

## Sample Complexity

| Function Class | Sample Complexity | Algorithm |
|---------------|-------------------|-----------|
| k-juntas | O(2^k log n / ε) | Influence-based |
| Degree-d | O(n^d / ε²) | Low-degree learning |
| s-sparse Fourier | O(s log n / ε²) | Sparse recovery |
| Decision trees (size s) | O(s log n / ε²) | LMN |
| Monotone | O(n / ε²) | Monotone learning |

## Connections to Other Topics

### Influences and Learning

High-influence variables are likely relevant:

```python
# Find influential variables for junta learning
infs = f.influences()
top_vars = sorted(range(len(infs)), key=lambda i: infs[i], reverse=True)[:k]
```

### Noise Stability and Learning

Noise-stable functions are easier to learn:

```python
stability = f.noise_stability(0.99)
# High stability → few high-degree coefficients → easier to learn
```

### Hypercontractivity and Learning

KKL theorem bounds help with junta identification:

```python
from boofun import max_influence_bound
max_inf, kkl_bound, total = max_influence_bound(f)
# If max_inf >> kkl_bound, function might be close to a junta
```

## Mathematical Background

### PAC Model

A concept class C is PAC-learnable if there exists an algorithm A such that:
- For any f ∈ C and distribution D
- Given m = poly(n, 1/ε, 1/δ) samples from D
- A outputs h with Pr_D[h(x) ≠ f(x)] ≤ ε
- With probability at least 1 - δ

### Goldreich-Levin Theorem

**Theorem**: Given query access to f, we can find all S with |f̂(S)| ≥ τ using O(n/τ²) queries.

### LMN Theorem

**Theorem** (Linial-Mansour-Nisan): Decision trees of size s can be ε-approximated by degree O(log(s/ε)) polynomials.

## See Also

- [Spectral Analysis Guide](spectral_analysis.md): Fourier coefficients
- [Query Complexity Guide](query_complexity.md): Query models
- `notebooks/lecture7_goldreich_levin.ipynb`: Goldreich-Levin tutorial
- `notebooks/lecture8_learning_juntas.ipynb`: Junta learning tutorial
- O'Donnell, *Analysis of Boolean Functions*, Chapter 3
