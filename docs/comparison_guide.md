# Library Comparison

How BooFun compares to other Boolean function libraries.

## Summary

BooFun focuses on theoretical computer science: Fourier analysis (O'Donnell style), property testing, query complexity. Other libraries have different strengths.

| Library | Focus | Fourier | Property Testing | Query Complexity |
|---------|-------|---------|------------------|------------------|
| BooFun | TCS theory | ✓ | ✓ | ✓ |
| SageMath | Cryptography | Walsh only | ✗ | ✗ |
| pyeda | Logic/SAT/BDD | ✗ | ✗ | ✗ |
| BoolForge | Biology | ✗ | ✗ | ✗ |
| CANA | Network control | ✗ | ✗ | ✗ |

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

**Quantum** (theoretical estimation only):
- Grover speedup
- Quantum walk analysis

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
| Essential variables | ✓ | ✓ |
| **Networks** | | |
| Network representation | ✗ | ✓ |
| Attractor analysis | ✗ | ✓ |
| Network motifs | ✗ | ✓ |
| **Unique to BooFun** | | |
| Fourier analysis | ✓ | ✗ |
| Query complexity | ✓ | ✗ |
| Property testing | ✓ | ✗ |
| Hypercontractivity | ✓ | ✗ |
| Cryptographic analysis | ✓ | ✗ |

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
| Monotonicity | ✓ (probabilistic) | ✓ (exact) |
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

We've validated against known results where possible:
- Parseval's identity
- Majority function influences (compare to theoretical √(2/πn))
- Parity function properties

See `tests/test_cross_validation.py` for details. Not everything has been cross-validated.

## Installation

```bash
pip install boofun      # BooFun (PyPI)
pip install git+https://github.com/ckadelka/BoolForge  # BoolForge
pip install cana        # CANA
pip install pyeda       # pyeda
```

## Prior Art

BooFun's query complexity module builds on:
- **Scott Aaronson's Boolean Function Wizard** (2000): C implementation of D(f), R(f), Q(f), sensitivity, block sensitivity, certificate complexity, approximate degrees. See Aaronson, "Algorithms for Boolean Function Query Measures."
- **Avishay Tal's library**: Python implementation of Fourier transforms, sensitivity, decision trees, polynomial representations over F₂ and reals.

These tools inspired BooFun's design but were either no longer maintained or not publicly distributed. BooFun aims to provide a modern, documented, tested implementation of these ideas.

## References

- Aaronson, S. (2000). "Algorithms for Boolean Function Query Measures."
- O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge.
- Buhrman, H. & de Wolf, R. (2002). "Complexity Measures and Decision Tree Complexity."
- Correia et al. (2018). CANA. Frontiers in Physiology.
