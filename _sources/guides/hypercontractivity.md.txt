# Hypercontractivity Guide

Tools for hypercontractivity analysis from O'Donnell Chapter 9 and extensions including global hypercontractivity.

## Overview

Hypercontractivity is a fundamental tool in Boolean function analysis, providing:

- **Noise operator** T_ρ and its properties
- **Bonami's Lemma** for bounding noisy function norms
- **Hypercontractive inequality** relating L_p and L_q norms
- **KKL theorem** bounding maximum influence
- **Friedgut's junta theorem** showing low-influence functions are close to juntas
- **Global hypercontractivity** (Keevash et al.) for p-biased analysis

## When to Use This

- Proving concentration inequalities
- Bounding influences via KKL theorem
- Analyzing noise sensitivity
- Junta approximation via Friedgut
- Threshold phenomena analysis

## Classical Hypercontractivity

### Noise Operator and Norms

| Task | Function | Reference |
|------|----------|-----------|
| Noise operator T_ρ | `bf.noise_operator(f, rho)` | O'Donnell 9.1 |
| L_q norm | `bf.lq_norm(f, q)` | O'Donnell 9.2 |

The noise operator T_ρ "smooths" a function by applying ρ-correlated noise:

$$(T_\rho f)(x) = \mathbb{E}_{y \sim N_\rho(x)}[f(y)]$$

### Example: Noise Operator

```python
import boofun as bf
import numpy as np

f = bf.majority(5)

# Apply noise operator
for rho in [0.9, 0.5, 0.1]:
    noisy_f = bf.noise_operator(f, rho)
    print(f"T_{rho} applied, L2 norm: {bf.lq_norm(noisy_f, 2):.4f}")

# Observe: as rho → 0, T_ρ f → E[f] (constant function)
```

### Bonami's Lemma

Bonami's Lemma bounds the L_q norm of the noisy function.

| Task | Function | Reference |
|------|----------|-----------|
| Check Bonami bound | `bf.bonami_lemma_bound(f, q, rho)` | O'Donnell 9.3 |

**Theorem (Bonami's Lemma):** For q ≥ 2 and ρ ≤ 1/√(q-1):

$$\|T_\rho f\|_q \leq \|f\|_2$$

```python
import boofun as bf

f = bf.majority(5)

# Check Bonami's Lemma for q=4, rho=0.5
# Requires rho <= 1/sqrt(3) ≈ 0.577
lq_noisy, l2, satisfied = bf.bonami_lemma_bound(f, q=4, rho=0.5)

print(f"||T_0.5 f||_4 = {lq_noisy:.4f}")
print(f"||f||_2 = {l2:.4f}")
print(f"Bonami satisfied: {satisfied}")
```

### Hypercontractive Inequality

The general hypercontractive inequality.

| Task | Function | Reference |
|------|----------|-----------|
| Check hypercontractivity | `bf.hypercontractive_inequality(f, rho, p, q)` | O'Donnell 9.5 |

**Theorem:** For 1 ≤ p ≤ q and ρ ≤ √((p-1)/(q-1)):

$$\|T_\rho f\|_q \leq \|f\|_p$$

```python
lq, lp, satisfied = bf.hypercontractive_inequality(f, rho=0.5, p=2, q=4)
print(f"||T_ρ f||_q = {lq:.4f}, ||f||_p = {lp:.4f}")
print(f"Hypercontractive inequality satisfied: {satisfied}")
```

### Level-d Inequality

Bounds the L_q norm of degree-d part of f.

| Task | Function | Reference |
|------|----------|-----------|
| Level-d bound | `bf.level_d_inequality(f, d, q)` | O'Donnell 9.7 |

```python
# Check level-d inequality for degree 2 part
lq_d, bound, satisfied = bf.level_d_inequality(f, d=2, q=4)
print(f"||f^(=d)||_q = {lq_d:.4f}")
print(f"Bound: {bound:.4f}")
```

## KKL Theorem

The Kahn-Kalai-Linial theorem bounds the maximum influence.

| Task | Function | Reference |
|------|----------|-----------|
| KKL bound | `bf.max_influence_bound(f)` | O'Donnell 9.6 |

**Theorem (KKL):** For any Boolean function f:

$$\max_i \text{Inf}_i[f] \geq c \cdot \frac{I[f] \cdot \log n}{n}$$

```python
f = bf.majority(15)

max_inf, kkl_bound, total_inf = bf.max_influence_bound(f)

print(f"Max influence: {max_inf:.4f}")
print(f"KKL lower bound: {kkl_bound:.4f}")
print(f"Total influence I[f]: {total_inf:.4f}")
print(f"KKL satisfied: {max_inf >= kkl_bound}")
```

## Friedgut's Junta Theorem

Functions with low total influence are close to juntas.

| Task | Function | Reference |
|------|----------|-----------|
| Junta size bound | `bf.friedgut_junta_bound(I, eps)` | O'Donnell 9.4 |
| Junta approximation error | `bf.junta_approximation_error(f, junta_vars)` | |

**Theorem (Friedgut):** If I[f] ≤ k, then f is ε-close to a 2^O(k/ε)-junta.

```python
# Estimate junta size needed for given total influence and error
I_f = 2.0  # total influence
eps = 0.1  # approximation error

junta_size = bf.friedgut_junta_bound(I_f, eps)
print(f"For I[f]={I_f}, eps={eps}: f is {eps}-close to a {junta_size}-junta")

# Check approximation error for a specific junta
f = bf.majority(7)
junta_vars = [0, 1, 2, 3]  # Approximate using first 4 variables
error = bf.junta_approximation_error(f, junta_vars)
print(f"Approximation error using variables {junta_vars}: {error:.4f}")
```

## Global Hypercontractivity (Keevash et al.)

For analyzing Boolean functions under p-biased measures μ_p.

### GlobalHypercontractivityAnalyzer

Comprehensive analysis under p-biased measures.

```python
f = bf.majority(7)

# Create analyzer
analyzer = bf.GlobalHypercontractivityAnalyzer(f, p=0.3)

# Get summary of all measures
print(analyzer.summary())
```

### α-Global Functions

A function is α-global if no small set has large generalized influence.

| Task | Function |
|------|----------|
| Check α-global | `bf.is_alpha_global(f, alpha, max_set_size)` |

```python
f = bf.majority(7)

# Check if f is 0.01-global (no set of size ≤3 has influence > 0.01)
is_global, details = bf.is_alpha_global(f, alpha=0.01, max_set_size=3)
print(f"Is 0.01-global: {is_global}")
if not is_global:
    print(f"Violating set: {details['violating_set']}")
    print(f"Influence: {details['influence']}")
```

### Generalized Influence

The influence of a set S under μ_p.

| Task | Function |
|------|----------|
| Generalized influence | `bf.generalized_influence(f, S, p)` |

```python
f = bf.tribes(2, 4)

# Generalized influence of set {0, 1} under p=0.3
S = {0, 1}
gen_inf = bf.generalized_influence(f, S, p=0.3)
print(f"Generalized influence of {S} under μ_0.3: {gen_inf:.4f}")
```

### Threshold Phenomena

Analyze how function behavior changes with bias p.

| Task | Function |
|------|----------|
| Threshold curve | `bf.threshold_curve(f, p_range)` |
| Find critical p | `bf.find_critical_p(f)` |
| Hypercontractivity bound | `bf.hypercontractivity_bound(f, p)` |

```python
import numpy as np

f = bf.tribes(3, 3)  # Tribes function

# Get threshold curve
p_range = np.linspace(0.01, 0.99, 50)
curve = bf.threshold_curve(f, p_range)

# Find critical p (where E_p[f] = 0.5)
p_crit = bf.find_critical_p(f)
print(f"Critical p: {p_crit:.4f}")

# Plot threshold curve
import matplotlib.pyplot as plt
plt.plot(p_range, curve)
plt.axvline(p_crit, color='r', linestyle='--', label=f'p_crit={p_crit:.2f}')
plt.xlabel('p')
plt.ylabel('E_p[f]')
plt.title('Threshold Curve')
plt.legend()
plt.show()
```

## Applications

### 1. Proving Concentration Inequalities

Use hypercontractivity to show functions don't deviate far from their mean:

```python
f = bf.majority(11)

# The hypercontractive inequality implies concentration
# ||f - E[f]||_4 is bounded, giving tail bounds
```

### 2. Bounding Influences

Use KKL to show some variable must have significant influence:

```python
f = bf.tribes(3, 3)

max_inf, kkl_bound, total = bf.max_influence_bound(f)
print(f"Some variable has influence ≥ {kkl_bound:.4f}")
```

### 3. Junta Approximation

Find a small set of variables that approximately determines f:

```python
f = bf.majority(9)

# If total influence is low, f is close to a junta
total = f.total_influence()
junta_size = bf.friedgut_junta_bound(total, eps=0.1)
print(f"MAJ_9 is 0.1-close to a {junta_size}-junta")
```

### 4. Threshold Phenomena

Study sharp transitions in random graph properties:

```python
# Tribes exhibits sharp threshold
f = bf.tribes(4, 4)

p_crit = bf.find_critical_p(f)
print(f"Tribes threshold at p ≈ {p_crit:.3f}")

# Compare to balanced function (no sharp threshold)
g = bf.parity(4)
# Parity has no threshold - linear in p
```

## Mathematical Background

### Noise Operator

The noise operator T_ρ acts on Fourier coefficients:

$$(T_\rho f)^\wedge(S) = \rho^{|S|} \hat{f}(S)$$

This attenuates high-degree Fourier coefficients.

### Hypercontractivity

The (2,q)-hypercontractivity constant is:

$$\rho^*(2,q) = \frac{1}{\sqrt{q-1}}$$

For ρ ≤ ρ*(2,q), we have ||T_ρ f||_q ≤ ||f||_2.

### KKL Theorem

The KKL theorem states:

$$\max_i \text{Inf}_i[f] \geq \text{Var}[f] \cdot \frac{c \log n}{n}$$

where c > 0 is a universal constant.

## See Also

- [Spectral Analysis Guide](spectral_analysis.md) - Fourier basics and influences
- [Query Complexity Guide](query_complexity.md) - Sensitivity measures
- O'Donnell, *Analysis of Boolean Functions*, Chapter 9
- Keevash, Lifshitz, Long & Minzer, "Global Hypercontractivity"
