# Advanced Topics Guide

This guide covers advanced analysis modules not covered in the main guides.

## Social Choice Theory

### Arrow's Theorem

Analysis of voting functions and impossibility results.

```python
from boofun.analysis.arrow import (
    is_unanimous,
    is_dictator,
    is_iia,
    arrow_analysis,
    distance_to_dictator,
    SocialChoiceAnalyzer
)

maj = bf.majority(5)

# Check Arrow's conditions
print(f"Unanimous: {is_unanimous(maj)}")
print(f"Dictator: {is_dictator(maj)}")

# Full social choice analysis
analysis = arrow_analysis(maj)
print(f"Distance to nearest dictator: {analysis['min_distance_to_dictator']:.4f}")

# Use the analyzer class
analyzer = SocialChoiceAnalyzer(maj)
print(analyzer.report())
```

### FKN Theorem

Functions with low influence are close to dictators (Friedgut-Kalai-Naor).

```python
from boofun.analysis.fkn import (
    distance_to_dictator,
    nearest_dictator,
    fkn_distance,
    fkn_analysis
)

f = bf.majority(5)

# Distance to specific dictator
d = distance_to_dictator(f, i=0)
print(f"Distance to x_0: {d:.4f}")

# Find nearest dictator
best_i, min_dist = nearest_dictator(f)
print(f"Nearest dictator: x_{best_i} (distance {min_dist:.4f})")

# FKN theorem bound
analysis = fkn_analysis(f)
```

## Gaussian Analysis

Connection between Boolean functions and Gaussian space (O'Donnell Chapter 10).

```python
from boofun.analysis.gaussian import (
    hermite_polynomial,
    hermite_coefficients,
    gaussian_noise_stability,
    ornstein_uhlenbeck_operator,
    multilinear_extension,
    gaussian_expectation,
    GaussianAnalyzer
)

maj = bf.majority(5)

# Hermite polynomials (Gaussian analog of Fourier characters)
H3 = hermite_polynomial(3)  # H_3(x)
print(f"H_3(0) = {H3(0)}")

# Gaussian noise stability
rho = 0.9
stability = gaussian_noise_stability(maj, rho)
print(f"Gaussian stability at rho={rho}: {stability:.4f}")

# Multilinear extension
mle = multilinear_extension(maj)
print(f"MLE at (0.5, 0.5, 0.5, 0.5, 0.5) = {mle([0.5]*5):.4f}")

# Full Gaussian analysis
analyzer = GaussianAnalyzer(maj)
print(f"Hermite degree: {analyzer.hermite_degree()}")
```

## Invariance Principle

The invariance principle connects Boolean and Gaussian behavior (O'Donnell Chapter 11).

```python
from boofun.analysis.invariance import (
    invariance_distance,
    multilinear_extension_gaussian_expectation,
    majority_is_stablest_bound,
    max_influence_for_invariance,
    InvarianceAnalyzer
)

maj = bf.majority(7)

# How well does invariance hold?
dist = invariance_distance(maj, num_samples=10000)
print(f"Invariance distance: {dist:.4f}")

# "Majority is Stablest" theorem bound
rho = 0.9
bound = majority_is_stablest_bound(maj, rho)
print(f"Majority is Stablest bound: {bound:.4f}")

# Check if influences are low enough for invariance
max_inf = max_influence_for_invariance(maj)
print(f"Max influence: {max_inf:.4f}")
```

## Communication Complexity

Analysis of communication complexity for Boolean functions.

```python
from boofun.analysis.communication_complexity import (
    deterministic_cc,
    log_rank_bound,
    fooling_set_bound,
    rectangle_partition_bound,
    discrepancy,
    CommunicationMatrix,
    CommunicationComplexityProfile
)

# Simple function (inner product)
f = bf.parity(4)

# Communication matrix
matrix = CommunicationMatrix(f)
print(f"Matrix rank: {matrix.rank()}")

# Various bounds
print(f"Log-rank bound: {log_rank_bound(f)}")
print(f"Fooling set bound: {fooling_set_bound(f)}")

# Full complexity profile
profile = CommunicationComplexityProfile(f)
print(f"Deterministic CC: {profile.deterministic}")
print(f"Discrepancy: {profile.discrepancy:.4f}")
```

## LTF Analysis

Linear Threshold Function analysis (O'Donnell Chapter 5).

```python
from boofun.analysis.ltf_analysis import (
    is_ltf,
    chow_parameters,
    critical_index,
    regularity,
    ltf_noise_sensitivity,
    LTFAnalysis,
    fit_ltf
)

maj = bf.majority(5)

# Check if LTF
print(f"Is LTF: {is_ltf(maj)}")

# Chow parameters (uniquely identify LTFs)
chow = chow_parameters(maj)
print(f"Chow parameters: {chow}")

# Critical index (important for LTF analysis)
ci = critical_index(maj)
print(f"Critical index: {ci}")

# Full LTF analysis
analysis = LTFAnalysis(maj)
print(f"Weights: {analysis.weights}")
print(f"Threshold: {analysis.threshold}")

# Fit an LTF to approximate a function
f = bf.tribes(2, 3)
approx_ltf = fit_ltf(f)
print(f"Approximation error: {approx_ltf.error:.4f}")
```

## Random Restrictions

Random restrictions and the switching lemma (O'Donnell Chapter 4).

```python
from boofun.analysis.restrictions import (
    Restriction,
    random_restriction,
    apply_restriction,
    restriction_shrinkage,
    average_restricted_decision_tree_depth,
    switching_lemma_probability,
    batch_random_restrictions
)

f = bf.tribes(3, 4)

# Create a p-random restriction
rho = random_restriction(n=12, p=0.5)
print(f"Free variables: {rho.free_variables}")

# Apply restriction
f_restricted = apply_restriction(f, rho)
print(f"Original vars: {f.n_vars}, Restricted vars: {f_restricted.n_vars}")

# Average decision tree depth after restriction
avg_depth = average_restricted_decision_tree_depth(f, p=0.5, num_trials=100)
print(f"Avg DT depth after p=0.5 restriction: {avg_depth:.2f}")

# Switching lemma probability
prob = switching_lemma_probability(width=3, p=0.5, s=2)
print(f"Pr[DT depth > 2]: {prob:.4f}")
```

## Symmetry Analysis

Symmetry properties and symmetrization.

```python
from boofun.analysis.symmetry import (
    is_symmetric,
    symmetrize,
    symmetrize_profile,
    degree_sym,
    sens_sym,
    sens_sym_by_weight,
    shift_function,
    find_monotone_shift,
    symmetric_representation
)

maj = bf.majority(5)

# Check if symmetric
print(f"Is symmetric: {is_symmetric(maj)}")

# Symmetrize any function
f = bf.tribes(2, 3)
f_sym = symmetrize(f)

# Symmetric profile (value at each Hamming weight)
profile = symmetrize_profile(maj)
print(f"Profile: {profile}")

# Symmetric degree
sd = degree_sym(maj)
print(f"Symmetric degree: {sd}")

# Symmetric sensitivity
sens = sens_sym(maj)
print(f"Symmetric sensitivity: {sens}")

# Find shift to make function monotone
shift = find_monotone_shift(f)
if shift is not None:
    f_monotone = shift_function(f, shift)
    print(f"Shifted to monotone by mask: {bin(shift)}")
```

## Fourier Sparsity

Analysis of Fourier sparsity.

```python
from boofun.analysis.sparsity import (
    fourier_sparsity,
    fourier_sparsity_up_to_constants,
    granularity,
    sparse_representation,
    SparsityAnalyzer
)

parity = bf.parity(5)

# Sparsity (number of non-zero coefficients)
s = fourier_sparsity(parity)
print(f"Fourier sparsity: {s}")  # 1 for parity!

# Sparsity up to constant multiples
s_const = fourier_sparsity_up_to_constants(parity)

# Granularity (GCD structure)
g = granularity(parity)
print(f"Granularity: {g}")

# Full sparsity analysis
analyzer = SparsityAnalyzer(parity)
print(f"Sparsity: {analyzer.sparsity}")
print(f"Non-zero coefficients: {analyzer.non_zero_sets}")
```

## Canalization

Canalization analysis from systems biology.

```python
from boofun.analysis.canalization import (
    is_canalizing,
    canalizing_variables,
    nested_canalizing_depth,
    canalizing_values,
    CanalizationAnalysis
)

and_func = bf.AND(4)

# Check if canalizing
print(f"Is canalizing: {is_canalizing(and_func)}")  # True

# Which variables are canalizing?
vars = canalizing_variables(and_func)
print(f"Canalizing variables: {vars}")

# Nested canalizing depth
depth = nested_canalizing_depth(and_func)
print(f"Nested depth: {depth}")

# Full analysis
analysis = CanalizationAnalysis(and_func)
for var, (canalizing_input, canalized_output) in analysis.canalizing_info.items():
    print(f"Variable {var}: input={canalizing_input} -> output={canalized_output}")
```

## Module Reference

| Module | Topic | O'Donnell Chapter |
|--------|-------|-------------------|
| `arrow` | Social choice, voting | 2 |
| `fkn` | FKN theorem | 2 |
| `gaussian` | Gaussian analysis, Hermite | 10 |
| `invariance` | Invariance principle | 11 |
| `communication_complexity` | Communication complexity | 6 |
| `ltf_analysis` | Linear threshold functions | 5 |
| `restrictions` | Random restrictions | 4 |
| `symmetry` | Symmetric functions | 4 |
| `sparsity` | Fourier sparsity | 3 |
| `canalization` | Canalization | - |

## See Also

- [Spectral Analysis Guide](spectral_analysis.md): Core Fourier analysis
- [Hypercontractivity Guide](hypercontractivity.md): Noise and KKL
- [Query Complexity Guide](query_complexity.md): Decision tree complexity
- `notebooks/lecture10_fourier_concentration.ipynb`: Gaussian analysis
- `notebooks/lecture11_invariance_principle.ipynb`: Invariance
