# Migration from Tal's BooleanFunc.py

If you've used Avishay Tal's `BooleanFunc.py` toolkit (from CS 294-92 at UC Berkeley), this guide maps his functions to boofun equivalents. The coverage is ~90%; a few generic utilities (tqdm, binary_search) are available from standard Python.

---

## Quick Start

```python
# Tal's BooleanFunc
f = BooleanFunc("x0&x1 ^ x2")
f.real_fourier()
f.sensitivity(5)
f.decision_tree()

# boofun equivalent
import boofun as bf
f = bf.create([0, 0, 0, 1, 1, 0, 1, 0])  # truth table
f.fourier()                                 # WHT coefficients
f.sensitivity(5)                            # sensitivity at input 5 (via analysis module)
bf.analysis.complexity.D(f)                 # decision tree depth
```

---

## Fourier Analysis

| Tal | boofun | Notes |
|-----|--------|-------|
| `f.real_fourier()` | `f.fourier()` | Returns array of WHT coefficients |
| `f.xor_fourier()` | `from boofun.analysis.gf2 import gf2_fourier_transform` | GF(2) / ANF coefficients |
| `FourierCoef(f, S)` | `SpectralAnalyzer(f).get_fourier_coefficient(S)` | Single coefficient |
| `f.degree()` | `f.degree()` | Fourier degree |
| `f.deg_F2()` | `f.degree(gf2=True)` or `gf2_degree(f)` | Algebraic degree over GF(2) |
| `f.bias()` | `f.bias()` | E[f] in ±1 convention |
| `f.l1()` | `spectral_norm(f, p=1)` | L1 spectral norm |
| `f.fourier_weights()` | `f.spectral_weight_by_degree()` or `fourier_weight_distribution(f)` | W^k = sum f_hat(S)^2 by degree |
| `f.truncated_degree_d(d)` | `truncate_to_degree(f, d)` | Zero out degree > d |
| `f.ann_influence(i, rho)` | `annealed_influence(f, i, rho)` | Noisy influence |
| `f.norm_influence(i)` | `normalized_influence(f, i)` | sum f_hat(S)^2/|S| for S containing i |

```python
from boofun.analysis.fourier import (
    spectral_norm, fourier_weight_distribution, fourier_level_lp_norm,
    fourier_tail_profile, truncate_to_degree, annealed_influence,
    normalized_influence,
)
from boofun.analysis.gf2 import gf2_fourier_transform, gf2_degree
```

---

## Complexity Measures

| Tal | boofun | Notes |
|-----|--------|-------|
| `f.sensitivity(x)` | `sensitivity.sensitivity_at(f, x)` | Sensitivity at input x |
| `f.max_sensitivity(val)` | `sensitivity.max_sensitivity(f, output_value=val)` | Max sensitivity |
| `f.min_sensitivity(val)` | `sensitivity.min_sensitivity(f, output_value=val)` | Min sensitivity |
| `f.average_sensitivity()` | `sensitivity.average_sensitivity(f)` | = total influence |
| `f.average_sensitivity_moment(t)` | `sensitivity.average_sensitivity_moment(f, t)` | t-th moment |
| `f.max_block_sensitivity(val)` | `complexity.bs(f)` | Block sensitivity |
| `f.certificate(x)` | `certificates.certificate(f, x)` | Returns (size, vars) |
| `f.max_certificate(val)` | `complexity.C(f)` | Certificate complexity |
| `f.min_certificate(val)` | `certificates.min_certificate_size(f, value=val)` | Min certificate |
| `f.decision_tree()` | `complexity.D(f)` | Decision tree depth |
| `f.influence(i)` | `f.influence(i)` or `f.influences()[i]` | Variable influence |
| `f.total_influence()` | `f.total_influence()` | Total influence |

```python
from boofun.analysis import complexity, sensitivity, certificates
complexity.D(f)      # Decision tree depth
complexity.s(f)      # Max sensitivity
complexity.bs(f)     # Block sensitivity
complexity.C(f)      # Certificate complexity
```

---

## Restrictions and Composition

| Tal | boofun | Notes |
|-----|--------|-------|
| `f.fix(var, val)` | `f.fix(var, val)` | Fix single variable |
| `f.fix(vars, vals)` | `f.fix(vars, vals)` | Fix multiple variables |
| `f.shift(s)` | `f.shift(s)` | XOR shift: f(x XOR s) |
| `f.compose(g)` | `f.compose(g)` | Compose on disjoint blocks |

---

## P-Biased Analysis

| Tal | boofun | Notes |
|-----|--------|-------|
| `f.probability_sat(p)` | `p_biased_expectation(f, p)` | E_{mu_p}[f] (exact, ±1 convention) |
| `FourierCoefMuP(f, p, S)` | `p_biased_fourier_coefficient(f, p, S)` | Single p-biased Fourier coeff |
| `asMuP(f, p)` | `p_biased_average_sensitivity(f, p)` | P-biased average sensitivity |
| `asFourierMuP(f, p)` | `p_biased_total_influence_fourier(f, p)` | Via Fourier with normalization |
| `parity_biased(n, k, i)` | `parity_biased_coefficient(n, k, i)` | Parity coefficient under bias |

```python
from boofun.analysis.p_biased import (
    p_biased_expectation, p_biased_fourier_coefficient,
    p_biased_average_sensitivity, p_biased_total_influence_fourier,
    PBiasedAnalyzer,
)
```

---

## Special Functions

| Tal | boofun | Notes |
|-----|--------|-------|
| `Krawchouk(n, k, x)` | `krawchouk(n, k, x)` | From `utils.math` |
| `over(n, k)` | `over(n, k)` | Binomial coefficient |
| `prime_sieve(n)` | `prime_sieve(n)` | From `utils.number_theory` |
| `factor(n)` | `factor(n)` | From `utils.number_theory` |
| `is_prime(n)` | `is_prime(n)` | From `utils.number_theory` |
| `tensor_product(A, B)` | `tensor_product(f, g)` | From `analysis.fourier` |
| `popcnt(x)` | `popcnt(x)` | From `utils.math` |
| `poppar(x)` | `poppar(x)` | From `utils.math` |

```python
from boofun.utils.math import popcnt, poppar, over, krawchouk
from boofun.utils.number_theory import prime_sieve, factor, is_prime
```

---

## Construction

| Tal | boofun | Notes |
|-----|--------|-------|
| `BooleanFunc(truth_table)` | `bf.create(truth_table)` | From list |
| `BooleanFunc("x0&x1 ^ x2")` | `bf.create("x0 and x1", n=3)` | Symbolic (Python syntax) |
| `BooleanFunc.sym(vals)` | `bf.threshold(n, k)` | Symmetric functions |
| `BooleanFunc.random_func(n, d)` | `bf.random(n)` | Random function |
| `BooleanFunc.from_str(expr)` | `bf.create(expr, n=...)` | Expression parsing |

---

## What boofun Adds

Features in boofun not in Tal's BooleanFunc.py:

- **Multiple representations**: truth table, Fourier, ANF, BDD, DNF/CNF, circuits, LTF
- **Automatic conversion** between representations via conversion graph
- **Query complexity**: R(f), Q(f), Ambainis bound, spectral adversary
- **Property testing**: BLR, junta, monotonicity with probability bounds
- **Hypercontractivity**: KKL, Bonami, Friedgut, global hypercontractivity
- **Cryptographic analysis**: nonlinearity, bent functions, Walsh spectrum, LAT/DDT
- **Invariance principle**: Gaussian analysis, Berry-Esseen, Majority is Stablest
- **Visualization**: influence plots, Fourier spectrum, hypercube, dashboards
- **Family tracking**: asymptotic growth analysis across function families
- **Monte Carlo estimation**: sampling-based analysis for large n

---

## Not in boofun (use standard Python)

| Tal | Alternative |
|-----|-------------|
| `tqdm` | `pip install tqdm` |
| `binary_search` | `bisect.bisect_left` from stdlib |
| `lagrange_interpolation` | `numpy.polynomial.polynomial.polyfit` or `scipy.interpolate` |
| `FourierTransform_2d` | `numpy.fft.fft2` |
