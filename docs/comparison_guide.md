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

**Property Testing:**
- BLR linearity
- Junta testing
- Monotonicity, unateness, symmetry

**Fourier Analysis:**
- Influences, total influence
- Noise stability
- Spectral weight by degree
- KKL theorem bounds

**Quantum** (theoretical estimation only):
- Grover speedup
- Quantum walk analysis

## What BooFun Lacks

Features better served by other libraries:
- Cryptographic analysis (bent, correlation immunity) → SageMath
- SAT solving, BDD operations → pyeda
- Boolean networks, attractors → BoolForge
- Network control theory → CANA
- Canalization analysis → BoolForge/CANA

## Comparison Tables

### Fourier Analysis

| Feature | BooFun | SageMath |
|---------|--------|----------|
| Walsh-Hadamard | ✓ | ✓ |
| Influences | ✓ | ✗ |
| Total influence | ✓ | ✗ |
| Noise stability | ✓ | ✗ |
| Bent functions | ✗ | ✓ |
| Correlation immunity | ✗ | ✓ |

Different focus: BooFun follows O'Donnell; SageMath focuses on cryptographic properties.

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

**SageMath:**
- Cryptographic analysis
- Bent functions, nonlinearity

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
pip install git+https://github.com/GabbyTab/boofun  # BooFun
pip install git+https://github.com/ckadelka/BoolForge  # BoolForge
pip install cana  # CANA
pip install pyeda  # pyeda
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
