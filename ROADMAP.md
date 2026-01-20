# Roadmap

**Version:** 1.0.0
**Updated:** January 2026

## Current State

Core functionality is production-ready. Test coverage has reached 60%. API is stabilizing with structured exception hierarchy and comprehensive error handling. The library supports the full O'Donnell textbook curriculum.

## What Exists

### Representations
Truth table (dense, sparse, packed), Fourier expansion, ANF, DNF/CNF, polynomial, LTF, circuit, BDD, symbolic. Automatic conversion between formats via conversion graph.

### Analysis
- **Property testing**: BLR linearity, junta, monotonicity, unateness, symmetry, quasisymmetry, balance, dictator, affine, constant, primality
- **Query complexity**: D(f), D_avg(f), R₀(f), R₁(f), R₂(f), Q₂(f), QE(f), s(f), bs(f), es(f), C(f), Ambainis bound, spectral adversary, polynomial method bound, general adversary bound
- **Fourier analysis**: WHT, influences, noise stability, spectral concentration, degree, p-biased Fourier coefficients, approximate degree, threshold degree
- **Specialized**: FKN theorem, hypercontractivity, invariance principle, Gaussian analysis, communication complexity
- **Learning**: PAC learning, Goldreich-Levin algorithm
- **LTF analysis**: Weight finding, critical index, Chow parameters

### Visualization
- **Static plots**: Influences, Fourier spectrum, truth table heatmaps, noise stability curves
- **Hypercube visualization**: 3D Boolean hypercube with function coloring
- **Growth plots**: Asymptotic behavior, family comparisons, convergence rates
- **Decision trees**: Tree visualization and export
- **Dashboards**: Comprehensive analysis dashboards (matplotlib/plotly)

### Families
Majority, Parity, AND, OR, Tribes, Threshold, Dictator, weighted LTF, RecursiveMajority, IteratedMajority, RandomDNF, Sbox. Growth tracking for asymptotic analysis.

### Performance
NumPy vectorization, Numba JIT (optional), CuPy GPU (optional), sparse representations, memoization, batch processing.

### Error Handling
Structured exception hierarchy with error codes, debug logging, and production-ready error messages.

### Quantum
Grover speedup estimation, quantum walk analysis. Theoretical only; Qiskit required for actual oracles.

### Infrastructure
GitHub Actions CI, pytest, Hypothesis property tests, mutation testing config, pre-commit hooks, cross-validation against known results.

## v1.0.0 Milestone (Complete)

| Task | Status |
|------|--------|
| Increase test coverage to 60%+ | ✓ Complete |
| Structured exception handling | ✓ Complete |
| Fix bit-ordering bugs (MSB/LSB) | ✓ Complete |
| Document API stability policy | ✓ Complete |
| Publish to PyPI | ✓ Complete |
| File I/O (JSON, .bf, DIMACS CNF) | ✓ Complete |
| Partial representations | ✓ Complete |
| Flexible inputs & oracle pattern | ✓ Complete |

## Test Coverage

| Module | Coverage |
|--------|----------|
| core/base.py | ~80% |
| analysis/*.py | 60-85% |
| families/*.py | 40-60% |
| visualization/*.py | 40-60% |
| quantum/*.py | ~20% |

Coverage achieved 60% milestone. Visualization and quantum modules have improved significantly.

## Nice to Have (No Timeline)

- Manim animations for educational content
- Dask distributed computation for large n
- conda-forge package
- Interactive Jupyter widgets documentation

## Recently Completed

### File I/O (v0.2.1)
- `bf.load(path)` and `bf.save(func, path)` at top level
- JSON format with full metadata
- `.bf` format (Scott Aaronson's Boolean Function Wizard)
- DIMACS CNF format (SAT solver standard)
- Auto-detection from extension or content
- `bf.create("file.json")` works directly

### Partial Representations (v0.2.1)
- `PartialRepresentation` class for incomplete data
- Confidence tracking for known/unknown values
- Estimation for unknown values using neighbor voting
- Incremental value addition
- Conversion to complete with estimation

### Flexible Inputs & Oracle Pattern (v0.2.1)
- `bf.create()` auto-detects: lists, numpy, callables, symbolic strings, dicts, sets, files
- `f.evaluate()` accepts: int index, list, tuple, numpy array, batch (2D)
- New tutorial: `notebooks/flexible_inputs_and_oracles.ipynb`
- **Oracle pattern**: analyze huge functions (n=100+) without materializing 2^n truth table
  - BLR linearity testing via random queries
  - Fourier coefficient estimation via sampling
  - PAC learning with sample access only
  - Real-world examples: ML classifiers, external APIs as Boolean functions

## Fourier Convention

O'Donnell standard (Analysis of Boolean Functions, Chapter 1):
- Boolean 0 → +1
- Boolean 1 → -1
- f̂(∅) = E[f]

## Prior Art

The query complexity module builds on ideas from:
- **Scott Aaronson's Boolean Function Wizard** (2000): Implemented D(f), R(f), Q(f), sensitivity, block sensitivity, certificate complexity, and degree measures.
- **Avishay Tal's library**: Fourier transforms, sensitivity, decision trees, polynomial representations.

See `src/boofun/analysis/query_complexity.py` for specific citations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

High-value contributions:
- Tests for untested paths
- Bug reports with reproducible examples
- Corrections to mathematical errors
- Notebook improvements and examples
