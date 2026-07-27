# Library Comparison

How BooFun compares to other Boolean function libraries.

## Summary

BooFun focuses on theoretical computer science: Fourier analysis (O'Donnell style), property testing, query complexity. Other libraries have different strengths.

| Library | Focus | Fourier | Property Testing | Query Complexity |
|---------|-------|---------|------------------|------------------|
| BooFun | TCS theory | ✓ | ✓ | ✓ |
| SageMath | Cryptography | Walsh only | ✗ | ✗ |
| VBF (C++) | Cryptography (vectorial) | Walsh only | ✗ | ✗ |
| sboxU | S-box cryptanalysis | Walsh only | ✗ | ✗ |
| fbool | TCS (Rust + Python, young) | ✓ | ✗ | partial |
| pyeda | Logic/SAT/BDD | ✗ | ✗ | ✗ |
| BoolForge | Biology | influences only¹ | ✗ | ✗ |
| CANA | Network control | ✗ | ✗ | ✗ |

¹ BoolForge computes exact activities (= influences) and average sensitivity —
BooFun cross-validates against both in CI — but no Fourier spectrum beyond that.

## What BooFun Has

**Query Complexity** (based on Aaronson's Boolean Function Wizard):
- Deterministic: D(f), D_avg(f)
- Randomized: R₀(f), R₁(f), R₂(f), nondeterministic variants
- Quantum: Q₂(f), QE(f), nondeterministic variants
- Sensitivity: s(f), bs(f), es(f) (everywhere sensitivity)
- Certificates: C(f), C₀(f), C₁(f)
- Lower bounds: Ambainis, spectral adversary, polynomial method, general adversary
- Degree measures: exact, approximate, threshold, nondeterministic
- Decision tree algorithms: DP optimal depth, tree enumeration, randomized complexity

**Property Testing:**
- BLR linearity
- Junta testing
- Monotonicity, unateness, symmetry

**Fourier Analysis:**
- Influences, total influence
- Noise stability
- Spectral weight by degree
- KKL theorem bounds
- p-biased Fourier analysis
- Annealed influence, truncation, correlation

**Sensitivity Analysis:**
- Sensitivity moments and histograms
- p-biased sensitivity
- Pointwise sensitivity, sensitive coordinates
- arg_max/arg_min sensitivity

**Hypercontractivity (v1.1):**
- Noise operator T_ρ, L_q norms
- Bonami's Lemma, hypercontractive inequality
- KKL theorem, Friedgut's junta theorem
- Level-d inequality

**Global Hypercontractivity (v1.1, unique):**
- GlobalHypercontractivityAnalyzer
- α-global function detection
- Generalized influence under μ_p
- Threshold curves, critical p

**Cryptographic Analysis (v1.1):**
- Nonlinearity, bent function detection
- Walsh transform and spectrum
- Algebraic Normal Form, algebraic degree
- Correlation immunity, resiliency
- Strict Avalanche Criterion (SAC)
- Linear Approximation Table (LAT)
- Difference Distribution Table (DDT)
- S-box analyzer

**Quantum Complexity Bounds** (experimental playground — classical computation of quantum query estimates):
- Grover complexity bounds (closed-form formulas)
- Quantum walk complexity bounds (analytical)
- Element distinctness analysis
- *Actual quantum simulation planned for v2.0.0*

## What BooFun Lacks

Features better served by other libraries:
- SAT solving, advanced BDD operations → pyeda
- Boolean networks, attractors → BoolForge, biobalm
- Network control theory → CANA
- Canalizing layer structure → BoolForge

Note: As of v1.1, BooFun includes canalization analysis (depth, nested canalizing detection, essential variables) and cryptographic analysis (bent functions, nonlinearity, correlation immunity, LAT/DDT).

## BoolForge Comparison (Systems Biology)

BoolForge (Kadelka & Coberly, 2025) focuses on Boolean **networks** for systems biology, while BooFun focuses on Boolean **functions** for theoretical CS.

### What BoolForge Does Well

**Random Generation with Constraints:**
```python
# BoolForge can generate functions with specific properties
random_k_canalizing_function(n, k)  # Specific canalizing depth
random_NCF(n, layer_structure)       # Nested canalizing with structure
random_non_degenerated_function(n, bias)  # Specific bias
```

**Boolean Networks:**
- Networks of interconnected Boolean functions
- Attractor analysis (steady states, limit cycles)
- Network robustness metrics
- Modular structure detection

**Null Model Generation:**
- Generate ensembles for statistical comparison
- Control for degree distribution, canalization, bias

### Feature Comparison

| Feature | BooFun | BoolForge |
|---------|--------|-----------|
| **Canalization** | | |
| is_canalizing | ✓ | ✓ |
| canalizing_depth | ✓ | ✓ |
| is_nested_canalizing | ✓ | ✓ |
| get_layer_structure | ✗ | ✓ |
| canalizing_strength | ✗ | ✓ |
| **Random Generation** | | |
| Random k-canalizing | ✗ | ✓ |
| Random with bias | ✗ | ✓ |
| Random layer structure | ✗ | ✓ |
| **Analysis** | | |
| Monotonicity | ✓ | ✓ |
| Symmetry groups | ✓ | ✓ |
| Sensitivity | ✓ | ✓ |
| Influences (exact) | ✓ | ✓ (activities)² |
| Average sensitivity / total influence | ✓ | ✓² |
| Essential variables | ✓ | ✓ |
| **Networks** | | |
| Network representation | ✗ | ✓ |
| Attractor analysis | ✗ | ✓ |
| Network motifs | ✗ | ✓ |
| **Unique to BooFun** | | |
| Full Fourier spectrum | ✓ | ✗ |
| Noise stability | ✓ | ✗ |
| Query complexity | ✓ | ✗ |
| Property testing | ✓ | ✗ |
| Hypercontractivity | ✓ | ✗ |
| Cryptographic analysis | ✓ | ✗ |

² Cross-validated live in CI: BooFun's influences and total influence are
compared against BoolForge's exact activities and average sensitivity at
atol 1e-10 (`tests/cross_validation/test_boolforge.py`).

### When to Use Which

**Use BoolForge when:**
- Modeling gene regulatory networks
- Need to generate ensembles with specific canalization properties
- Studying network dynamics and attractors
- Comparing biological networks to null models

**Use BooFun when:**
- Studying theoretical properties (Fourier, query complexity)
- Following O'Donnell's textbook
- Property testing algorithms
- Cryptographic analysis of Boolean functions

## Comparison Tables

### Fourier Analysis

| Feature | BooFun | SageMath |
|---------|--------|----------|
| Walsh-Hadamard | ✓ | ✓ |
| Influences | ✓ | ✗ |
| Total influence | ✓ | ✗ |
| Noise stability | ✓ | ✗ |
| Bent functions | ✓ | ✓ |
| Correlation immunity | ✓ | ✓ |
| Hypercontractivity | ✓ | ✗ |
| p-biased analysis | ✓ | ✗ |

BooFun now covers both O'Donnell-style analysis and cryptographic properties.

### Property Testing

| Test | BooFun | BoolForge |
|------|--------|-----------|
| Linearity (BLR) | ✓ | ✗ |
| Junta | ✓ | ✗ |
| Monotonicity | ✓ (exact check + probabilistic tester) | ✓ (exact) |
| Dictator proximity | ✓ | ✗ |

### Representations

| Format | BooFun | pyeda |
|--------|--------|-------|
| Truth table | ✓ | ✓ |
| BDD | ✓ (basic) | ✓ (full ROBDD) |
| CNF/DNF | ✓ | ✓ |
| Fourier | ✓ | ✗ |

pyeda's BDD implementation is more mature.

## When to Use What

**BooFun:**
- Studying Boolean function theory (O'Donnell book)
- Query complexity research
- Property testing algorithms
- Influence/noise stability analysis
- Hypercontractivity and threshold phenomena
- Cryptographic analysis (nonlinearity, bent, LAT/DDT, S-box)

**SageMath:**
- Deeper algebraic cryptanalysis
- Finite field computations

**sboxU / VBF:**
- Serious S-box cryptanalysis at scale (C++/optimized cores)
- APN / almost-bent classification, CCZ/EA equivalence (sboxU)

**pyeda:**
- SAT solving
- BDD manipulation
- Logic minimization

**BoolForge:**
- Gene regulatory networks
- Canalization

**CANA:**
- Network control theory

## Cross-Validation

BooFun's agreement with the libraries on this page is not asserted, it is
tested — every claim has an executable test against a pinned reference
version:

- **SageMath 10.9**: a committed fixture corpus of 303 functions (standard
  families plus all AES S-box components) generated inside a pinned Docker
  image; Walsh spectra, nonlinearity, algebraic degree, balancedness, and
  correlation immunity are compared exactly.
- **BoolForge 1.0.1**: a live CI job (on every push to main and weekly)
  installs a commit-pinned BoolForge and compares canalization, monotonicity,
  symmetry, influences, and average sensitivity.
- **Published values**: AES and PRESENT S-box properties, bent function
  spectra, and counting results are asserted against their literature
  citations.
- **Closed forms and internal consistency**: theoretical bounds (Huang,
  Nisan–Szegedy) and independent computation paths (FWHT vs direct
  correlation sums) are checked against each other.

See the [claim matrix](cross_validation.md) for the full list of validated
claims, reference versions, function families, and tolerances.

## Installation

```bash
pip install boofun      # BooFun (PyPI)
pip install boolforge   # BoolForge (on PyPI since v1.0)
pip install sboxU       # sboxU
pip install cana        # CANA
pip install pyeda       # pyeda
```

VBF is a C++/NTL library built from [source](https://github.com/jacubero/VBF).

## Prior Art

BooFun's query complexity module builds on:
- **Scott Aaronson's Boolean Function Wizard** (2000): C implementation of D(f), R(f), Q(f), sensitivity, block sensitivity, certificate complexity, approximate degrees. See Aaronson, "Algorithms for Boolean Function Query Measures."
- **Avishay Tal's library**: Python implementation of Fourier transforms, sensitivity, decision trees, polynomial representations over F₂ and reals.

These tools inspired BooFun's design but were either no longer maintained or not publicly distributed. BooFun aims to provide a modern, documented, tested implementation of these ideas.

Other prior art in adjacent niches:

- **VBF** (Álvarez-Cubero & Zufiria, ACM TOMS 2016): peer-reviewed C++/NTL
  library for *vectorial* Boolean functions in cryptography — Walsh spectrum,
  nonlinearity, algebraic degree, linear structures, autocorrelation. The
  strongest academic prior art on the cryptographic side; no Fourier-analytic
  TCS tooling (influences, noise stability, property testing).
- **fbool** (2026): young Rust library with Python bindings; the closest
  *Fourier-analytic* neighbor (influences, Walsh/Fourier coefficients, degree,
  nonlinearity, certificate complexity) plus its own entropy/fragmentation
  measures. No property testing or query-complexity suite yet.
- **JOSS landscape**: no published JOSS package does Fourier-analytic Boolean
  function analysis. Adjacent JOSS tools operate on different objects —
  CircuitGraph (Boolean *circuits*), Biddy (BDDs), sboxgates (S-box gate
  synthesis) — or on Boolean *networks* for biology (BoolForge, PyDrugLogics,
  emba, NORDic).

## References

- Aaronson, S. (2000). "Algorithms for Boolean Function Query Measures."
- O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge.
- Buhrman, H. & de Wolf, R. (2002). "Complexity Measures and Decision Tree Complexity."
- Correia et al. (2018). CANA. Frontiers in Physiology.
- Álvarez-Cubero, J. A. & Zufiria, P. J. (2016). "Algorithm 959: VBF: A Library
  of C++ Classes for Vector Boolean Functions in Cryptography." ACM TOMS 42(2).
- Kadelka, C. & Coberly, B. (2025). BoolForge.
