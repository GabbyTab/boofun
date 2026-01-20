# Roadmap

**Version:** 0.2.0  
**Updated:** January 2025

## Current State

Core functionality works. Test coverage is low (38%). Edge cases remain untested. API may change before v1.0.

## What Exists

### Representations
Truth table (dense, sparse, packed), Fourier expansion, ANF, DNF/CNF, polynomial, LTF, circuit, BDD, symbolic. Automatic conversion between formats.

### Analysis
- Property testing: BLR, junta, monotonicity, unateness, symmetry, balance, dictator, affine, constant
- Query complexity: D(f), R₀, R₂, Q₂, QE, s(f), bs(f), C(f), Ambainis bound, spectral adversary
- Fourier: WHT, influences, noise stability, spectral concentration
- Specialized: FKN analysis, communication complexity, Goldreich-Levin

### Families
Majority, Parity, AND, OR, Tribes, Threshold, Dictator, weighted LTF, RecursiveMajority, IteratedMajority, RandomDNF, Sbox

### Performance
NumPy vectorization, Numba JIT (optional), CuPy GPU (optional), sparse representations, memoization

### Quantum
Grover speedup estimation, quantum walk analysis. Theoretical only—Qiskit required for actual oracles.

### Infrastructure
GitHub Actions CI, pytest, Hypothesis property tests, cross-validation against known results.

## Before v1.0.0

| Task | Status |
|------|--------|
| Increase test coverage to 60%+ | Incomplete (38%) |
| Document API stability policy | Not done |
| Publish to PyPI | Blocked on token |

## Test Coverage

| Module | Coverage |
|--------|----------|
| core/base.py | ~75% |
| analysis/*.py | 60-80% |
| families/*.py | 18-46% |
| visualization/*.py | 0-13% |
| quantum/*.py | ~11% |

Coverage numbers are approximate. Low-coverage modules are less reliable.

## Nice to Have (No Timeline)

- Mutation testing
- Manim animations
- Dask distributed computation
- conda-forge package

## Fourier Convention

O'Donnell standard:
- Boolean 0 → +1
- Boolean 1 → −1
- f̂(∅) = E[f]

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

High-value contributions:
- Tests for untested paths
- Bug reports with reproducible examples
- Corrections to mathematical errors
