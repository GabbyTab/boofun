# Spectral Analysis Guide

Fourier analysis of Boolean functions following O'Donnell's conventions.

## Overview

BooFun provides comprehensive tools for spectral analysis of Boolean functions, including:
- Fourier coefficients and Walsh-Hadamard transform
- Influence measures
- Noise stability
- p-biased analysis
- Sensitivity analysis
- Monte Carlo estimation

## Core Fourier Analysis

The foundation of Boolean function analysis.

| Task | Function | Reference |
|------|----------|-----------|
| All Fourier coefficients | `f.fourier()` | O'Donnell Ch 1 |
| Single coefficient f̂(S) | `f.fourier_coefficient(S)` | |
| Per-variable influences | `f.influences()` | O'Donnell 2.2 |
| Total influence I[f] | `f.total_influence()` | O'Donnell 2.3 |
| Noise stability Stab_ρ[f] | `f.noise_stability(rho)` | O'Donnell 2.4 |
| Fourier degree | `f.degree()` | |
| Spectral weight at degree d | `f.spectral_weight(d)` | |

### Example: Basic Fourier Analysis

```python
import boofun as bf

# Create a function
maj = bf.majority(5)

# Get all Fourier coefficients
coeffs = maj.fourier()
print(f"Number of non-zero coefficients: {len(coeffs)}")

# Per-variable influences
infs = maj.influences()
print(f"Influences: {infs}")

# Total influence
I_f = maj.total_influence()
print(f"Total influence I[f] = {I_f:.4f}")

# Noise stability
stab = maj.noise_stability(0.9)
print(f"Noise stability Stab_0.9[f] = {stab:.4f}")
```

## Inner Products and Correlations

The inner product ⟨f, g⟩ = E[f·g] measures how "similar" two functions are.

| Task | Function | Description |
|------|----------|-------------|
| Inner product ⟨f, g⟩ | `fourier.plancherel_inner_product(f, g)` | E[f·g] = Σ f̂(S)·ĝ(S) |
| Correlation | `fourier.correlation(f, g)` | Same as inner product |
| Convolution | `fourier.convolution(f, g)` | (f * g)(x) = E_y[f(y)·g(x⊕y)] |

### Three Ways to Compute Inner Products

```python
import boofun as bf
from boofun.analysis.fourier import plancherel_inner_product

f = bf.AND(3)
g = bf.OR(3)

# Method 1: Library function
ip1 = plancherel_inner_product(f, g)

# Method 2: NumPy dot product (Plancherel's identity!)
ip2 = f.fourier() @ g.fourier()

# Method 3: Probabilistic view using XOR
# Because XOR = multiplication in ±1, and f̂(∅) = E[f]
ip3 = (f ^ g).fourier()[0]

# All three give the same answer: -0.5
```

### Interpretation

- **⟨f, f⟩ = 1** for any Boolean function (Parseval's identity)
- **⟨f, g⟩ > 0** means f and g are positively correlated (tend to agree)
- **⟨f, g⟩ < 0** means f and g are negatively correlated (tend to disagree)
- **⟨f, g⟩ = 0** means f and g are uncorrelated (orthogonal)

## Advanced Spectral Features

Additional tools for deeper spectral analysis.

| Task | Function | Reference |
|------|----------|-----------|
| Annealed/noisy influence | `fourier.annealed_influence(f, i, rho)` | |
| Truncate to degree d | `fourier.truncate_to_degree(f, d)` | |
| Weight distribution | `fourier.fourier_weight_distribution(f)` | |
| Min coefficient size | `fourier.min_fourier_coefficient_size(f)` | |

### Example: Advanced Spectral Analysis

```python
from boofun.analysis import fourier

f = bf.majority(5)
g = bf.parity(5)

# Correlation between functions
corr = fourier.correlation(f, g)
print(f"Correlation(MAJ, PAR) = {corr:.4f}")

# Fourier weight distribution by degree
weights = fourier.fourier_weight_distribution(f)
for d, w in enumerate(weights):
    print(f"  Weight at degree {d}: {w:.4f}")
```

## p-Biased Analysis

For non-uniform product measures μ_p where each bit is 1 with probability p.

| Task | Function |
|------|----------|
| p-biased expectation | `p_biased.p_biased_expectation(f, p)` |
| p-biased influence | `p_biased.p_biased_influence(f, i, p)` |
| p-biased total influence | `p_biased.p_biased_total_influence(f, p)` |
| p-biased Fourier coefficient | `p_biased.p_biased_fourier_coefficient(f, S, p)` |
| p-biased average sensitivity | `p_biased.p_biased_average_sensitivity(f, p)` |
| Full analyzer | `PBiasedAnalyzer(f, p)` |

### Example: p-Biased Analysis

```python
from boofun.analysis.p_biased import PBiasedAnalyzer

f = bf.majority(5)

# Analyze under different biases
for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    analyzer = PBiasedAnalyzer(f, p)
    print(f"p={p}: E_p[f]={analyzer.expectation():.3f}, "
          f"I_p[f]={analyzer.total_influence():.3f}")
```

## Sensitivity Analysis

Measures of how sensitive a function is to input changes.

| Task | Function |
|------|----------|
| Average sensitivity | `sensitivity.average_sensitivity(f)` |
| Max sensitivity s(f) | `sensitivity.max_sensitivity(f)` |
| Min sensitivity | `sensitivity.min_sensitivity(f)` |
| Sensitivity at input x | `sensitivity.pointwise_sensitivity(f, x)` |
| Sensitive coordinates | `sensitivity.sensitive_coordinates(f, x)` |
| Sensitivity histogram | `sensitivity.sensitivity_histogram(f)` |
| t-th moment of sensitivity | `sensitivity.average_sensitivity_moment(f, t)` |
| Find max sensitivity input | `sensitivity.arg_max_sensitivity(f)` |
| Find min sensitivity input | `sensitivity.arg_min_sensitivity(f)` |

### Example: Sensitivity Analysis

```python
from boofun.analysis import sensitivity

f = bf.majority(5)

# Basic sensitivity measures
print(f"Average sensitivity: {sensitivity.average_sensitivity(f):.3f}")
print(f"Max sensitivity s(f): {sensitivity.max_sensitivity(f)}")

# Find the input with maximum sensitivity
x_max, s_max = sensitivity.arg_max_sensitivity(f)
print(f"Max sensitivity {s_max} achieved at input {x_max}")

# Sensitivity histogram
hist = sensitivity.sensitivity_histogram(f)
for i, count in enumerate(hist):
    if count > 0:
        print(f"  {int(count)} inputs have sensitivity {i}")

# Higher moments
for t in [1, 2, 3]:
    moment = sensitivity.average_sensitivity_moment(f, t)
    print(f"  {t}-th moment: {moment:.4f}")
```

## Probabilistic View

Boolean functions can be viewed as random variables over uniformly random inputs.

### Key Identities

| Quantity | Formula | Computation |
|----------|---------|-------------|
| Expectation E[f] | f̂(∅) | `f.fourier()[0]` |
| Variance Var[f] | Σ_{S≠∅} f̂(S)² | `1 - f.fourier()[0]**2` |
| Inner product E[f·g] | Σ f̂(S)·ĝ(S) | `f.fourier() @ g.fourier()` |
| Product expectation E[f·g] | (f ^ g)^ (∅) | `(f ^ g).fourier()[0]` |

### The XOR Trick

In ±1 notation, XOR corresponds to multiplication:
- f(x) ⊕ g(x) in {0,1} becomes f(x) · g(x) in {±1}

This gives a way to compute products:

```python
f = bf.majority(3)
g = bf.parity(3)

# E[f·g] using the XOR trick
inner_product = (f ^ g).fourier()[0]

# Verify: this equals the Fourier inner product
assert inner_product == f.fourier() @ g.fourier()
```

### Fourier Coefficients as Expectations

The inversion formula says f̂(S) = E[f·χ_S]:

```python
maj = bf.majority(3)

# Build character χ_{0} = x₀ (just a dictator!)
chi_0 = bf.dictator(3, 0)

# f̂({0}) = E[f·χ_{0}] = (f ^ χ_{0}).fourier()[0]
coeff = (maj ^ chi_0).fourier()[0]
assert coeff == maj.fourier()[1]  # Index 1 = 2^0 = {0}
```

## Sampling & Monte Carlo

Estimate spectral properties via sampling when exact computation is expensive.

| Task | Function |
|------|----------|
| Estimate Fourier coefficient | `sampling.estimate_fourier_coefficient(f, S, n)` |
| Estimate influence | `sampling.estimate_influence(f, i, n)` |
| Estimate total influence | `sampling.estimate_total_influence(f, n)` |
| Random variable view | `RandomVariableView(f, p)` |
| Spectral distribution | `SpectralDistribution.from_function(f)` |

### Example: Monte Carlo Estimation

```python
from boofun.analysis.sampling import (
    estimate_fourier_coefficient,
    estimate_influence,
    RandomVariableView
)

f = bf.majority(11)  # Larger function

# Estimate Fourier coefficient for S = {0, 1}
S = frozenset([0, 1])
estimate = estimate_fourier_coefficient(f, S, n_samples=10000)
print(f"Estimated f̂({set(S)}) ≈ {estimate:.4f}")

# Estimate influence of variable 0
inf_est = estimate_influence(f, 0, n_samples=10000)
print(f"Estimated Inf_0[f] ≈ {inf_est:.4f}")

# Random variable view for statistical analysis
rv = RandomVariableView(f, p=0.5)
print(f"E[f] = {rv.expectation():.4f}")
print(f"Var[f] = {rv.variance():.4f}")
```

## Mathematical Background

### Fourier Expansion

Every Boolean function f: {0,1}^n → {±1} has a unique Fourier expansion:

$$f(x) = \sum_{S \subseteq [n]} \hat{f}(S) \chi_S(x)$$

where χ_S(x) = ∏_{i∈S} x_i are the parity functions.

### Parseval's Identity

$$\sum_{S \subseteq [n]} \hat{f}(S)^2 = \mathbb{E}[f^2] = 1$$

### Influence

The influence of variable i is:

$$\text{Inf}_i[f] = \Pr_{x}[f(x) \neq f(x^{\oplus i})] = \sum_{S \ni i} \hat{f}(S)^2$$

### Total Influence

$$I[f] = \sum_{i=1}^{n} \text{Inf}_i[f] = \sum_{S} |S| \cdot \hat{f}(S)^2$$

## See Also

- [Hypercontractivity Guide](hypercontractivity.md) - KKL theorem, Bonami's Lemma
- [Query Complexity Guide](query_complexity.md) - Sensitivity and complexity measures
- O'Donnell, *Analysis of Boolean Functions*, Chapters 1-3
