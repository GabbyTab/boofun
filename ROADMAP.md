# Roadmap

**Version:** 1.0.0
**Updated:** January 2026

## Current State

Core functionality is production-ready. Test coverage at 65%. API is stable with structured exception hierarchy and comprehensive error handling. The library supports the full O'Donnell textbook curriculum with unique features like global hypercontractivity analysis.

## What Exists

### Representations
Truth table (dense, sparse, packed), Fourier expansion, ANF, DNF/CNF, polynomial, LTF, circuit, BDD, symbolic. Automatic conversion between formats via conversion graph.

### Analysis
- **Property testing**: BLR linearity, junta, monotonicity, unateness, symmetry, quasisymmetry, balance, dictator, affine, constant, primality
- **Query complexity**: D(f), D_avg(f), R₀(f), R₁(f), R₂(f), Q₂(f), QE(f), s(f), bs(f), es(f), C(f), Ambainis bound, spectral adversary, polynomial method bound, general adversary bound
- **Fourier analysis**: WHT, influences, noise stability, spectral concentration, degree, p-biased Fourier coefficients, approximate degree, threshold degree
- **Specialized**: FKN theorem, hypercontractivity (global), invariance principle, Gaussian analysis, communication complexity
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
NumPy vectorization, Numba JIT (optional), CuPy GPU (optional), sparse representations, memoization, batch processing. **Note:** NumPy array inputs can be interpreted as batches for vectorized evaluation.

### Error Handling
Structured exception hierarchy with error codes, debug logging, and production-ready error messages.

### Quantum
Grover speedup estimation, quantum walk analysis. Theoretical only; Qiskit required for actual oracles.

### Infrastructure
GitHub Actions CI, pytest, Hypothesis property tests, mutation testing config, pre-commit hooks (black, isort, flake8, codespell, mypy), cross-validation against known results.

---

## v1.1.0 Goals (In Progress)

### 1. Avishay Tal's PhD Library Integration

**Status:** ✅ Mostly Complete (Jan 2026)

The `BooleanFunc.py` and `library.py` files contain valuable functionality from **Avishay Tal's PhD-era library** (shared via email, August 2025). The integration strategy was to **enhance existing modules** where possible.

#### ✅ Completed (102 new tests)

**New Modules Created:**
- `analysis/decision_trees.py` - DP algorithms, tree enumeration, randomized complexity
- `analysis/sparsity.py` - Fourier sparsity measures and support analysis

**Enhanced `analysis/sensitivity.py`:**
- ✅ `average_sensitivity_moment(t)` - t-th moment of sensitivity distribution
- ✅ `sensitive_coordinates(x)` - Return sensitive coordinates at input x
- ✅ `sensitivity_histogram()` - Distribution of sensitivity values
- ✅ `arg_max_sensitivity()`, `arg_min_sensitivity()` - Find extremal inputs

**Enhanced `analysis/p_biased.py`:**
- ✅ `p_biased_fourier_coefficient()` - Tal's efficient formula
- ✅ `p_biased_average_sensitivity()` - Average sensitivity under μ_p
- ✅ `p_biased_total_influence_fourier()` - Via Fourier (with correct 1/4p(1-p) normalization)
- ✅ `parity_biased_coefficient()` - Bias of parity function
- ✅ `PBiasedAnalyzer.validate()` - Cross-validation method

**Enhanced `analysis/fourier.py`:**
- ✅ `annealed_influence(i, rho)` - Annealed/noisy influence
- ✅ `truncate_to_degree(d)` - Truncate Fourier to degree d
- ✅ `correlation(f, g)` - Correlation between functions
- ✅ `fourier_weight_distribution()` - Weight distribution by degree
- ✅ `min_fourier_coefficient_size()` - Minimum |S| with non-zero f̂(S)

#### Recently Added (Jan 2026)

**Enhanced `analysis/symmetry.py`:**
- ✅ `is_symmetric(f)` - Check if function is symmetric
- ✅ `symmetrize_profile(f)` - Detailed profile by Hamming weight
- ✅ `sens_sym_by_weight(f)` - Sensitivity at each weight class
- ✅ `shift_function(f, shift)` - XOR shift transformation
- ✅ `find_monotone_shift(f)` - Find shift to make monotone
- ✅ `symmetric_representation(f)` - Symmetric specification

**Enhanced `analysis/restrictions.py`:**
- ✅ `min_fixing_to_constant(f, target)` - Greedy fixing to constant
- ✅ `shift_by_mask(f, mask)` - XOR transformation

#### Remaining (Low Priority)

**`utils/math.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `lagrange_interpolation` | Polynomial interpolation | Medium |

**`core/polynomials.py`** (Low Priority):
- `Polynomial` class - Real coefficients with arithmetic
- `FiniteFieldPolynomial` class - GF(q) coefficients

### 2. Boolean Functions as Random Variables

**Status:** ✅ Complete (Jan 2026)

New module `analysis/sampling.py` provides probabilistic treatment of Boolean functions. Reference: O'Donnell, Chapters 1-3.

#### ✅ Implemented Features

| Feature | Function | Status |
|---------|----------|--------|
| Uniform sampling | `sample_uniform(n, n_samples)` | ✅ |
| P-biased sampling | `sample_biased(n, p, n_samples)` | ✅ |
| Spectral sampling | `sample_spectral(f, n_samples)` | ✅ |
| Input-output pairs | `sample_input_output_pairs(f, n_samples)` | ✅ |
| Fourier estimation | `estimate_fourier_coefficient(f, S, n_samples)` | ✅ |
| Influence estimation | `estimate_influence(f, i, n_samples)` | ✅ |
| Total influence est. | `estimate_total_influence(f, n_samples)` | ✅ |
| Random variable view | `RandomVariableView(f, p)` | ✅ |
| Spectral distribution | `SpectralDistribution.from_function(f)` | ✅ |

#### RandomVariableView Features
- `rv.expectation()`, `rv.variance()` - exact values
- `rv.estimate_expectation(n)`, `rv.estimate_variance(n)` - Monte Carlo estimates  
- `rv.sample(n)` - sample (input, output) pairs
- `rv.sample_spectral(n)` - sample from Fourier weight distribution
- `rv.validate_estimates(n)` - cross-validate estimates vs exact
- `rv.summary()` - human-readable summary

#### Tests
- 43 comprehensive tests covering all functions
- Statistical convergence tests (law of large numbers)
- Cross-validation against exact computations

#### Remaining
- [ ] Create tutorial notebook: `notebooks/boolean_functions_as_random_variables.ipynb`

### 3. Increase Test Coverage to 70-75%

**Status:** In Progress (currently ~68%, target 70-75%)

| Module | Current | Target | Notes |
|--------|---------|--------|-------|
| core/base.py | ~80% | 85% | |
| analysis/*.py | 70-85% | 75-85% | +145 new tests (Tal + sampling) |
| families/*.py | 55-70% | 65-75% | +28 new tests |
| visualization/*.py | 40-60% | 60-70% | |
| quantum/*.py | ~20% | 40-50% | |
| **Overall** | **~69%** | **70-75%** | **2878 tests** |

#### Recent Progress (Jan 2026)
- ✅ Added 145+ tests for Tal library integration
- ✅ Added 43 tests for sampling module
- ✅ Added 28 tests for families module  
- ✅ Added 35 tests for symmetry/restrictions enhancements
- ✅ Added 58 tests for cryptographic module (LAT/DDT, algebraic immunity)
- ✅ Created tutorial notebook for random variables

#### Priority Areas
- [ ] Edge cases in representation conversions
- [ ] Error handling paths
- [ ] Visualization module (mock matplotlib)
- [ ] Quantum module basic coverage

### 4. Documentation Improvements

**Status:** Planning

#### README Updates
- [ ] Document batch processing: "NumPy array matrices can be interpreted as batches"
- [ ] Highlight unique features: "Global hypercontractivity analysis"
- [ ] Add library comparison section

#### API Documentation
- [ ] Complete docstrings for all public functions
- [ ] Add mathematical notation (LaTeX)
- [ ] Include "See Also" cross-references

### 5. Library Comparison & Cross-Validation

**Status:** ✅ In Progress (Jan 2026)

#### New Cryptographic Module

Created `analysis/cryptographic.py` with cryptographic measures that match thomasarmel/boolean_function:

| Feature | boofun | thomasarmel | Status |
|---------|--------|-------------|--------|
| Nonlinearity | `nonlinearity(f)` | `f.nonlinearity()` | ✅ Cross-validated |
| Bent detection | `is_bent(f)` | `f.is_bent()` | ✅ Cross-validated |
| Balanced check | `is_balanced(f)` | `f.is_balanced()` | ✅ Cross-validated |
| Walsh transform | `walsh_transform(f)` | Internal | ✅ |
| Walsh spectrum | `walsh_spectrum(f)` | - | ✅ |
| Algebraic degree | `algebraic_degree(f)` | `f.algebraic_normal_form().degree()` | ✅ |
| ANF | `algebraic_normal_form(f)` | `f.algebraic_normal_form()` | ✅ |
| Correlation immunity | `correlation_immunity(f)` | - | ✅ |
| Resiliency | `resiliency(f)` | - | ✅ |
| SAC | `strict_avalanche_criterion(f)` | - | ✅ |

#### Cross-Validation Tests (37 tests)
- ✅ 0xac90 (4-var bent): both libraries agree
- ✅ 0x0113077C165E76A8 (6-var bent): both libraries agree  
- ✅ Balanced count: C(16,8) = 12870

#### Libraries to Analyze

| Library | Language | Focus | Notable Features |
|---------|----------|-------|------------------|
| **thomasarmel/boolean_function** | Rust | Cryptographic | ANF, bent, nonlinearity, **parallel** |
| **BooLSPLG** | CUDA C++ | GPU | S-box analysis, LAT/DDT, **massively parallel** |
| **SageMath** | Python | General | Comprehensive but slow |

#### Potential Rust FFI Integration
thomasarmel's library provides:
- `SmallBooleanFunction` (≤6 vars, u64 storage)
- `BigBooleanFunction` (>6 vars, BigUint)
- Native CPU optimizations (POPCNT instruction)

We could use it via PyO3/Rust FFI for:
- Large function exhaustive search
- Performance-critical cryptographic computations
- Cross-validation of our implementations

#### What We Have That Others Don't
- ✅ **Global hypercontractivity** analysis
- ✅ **Query complexity** (D, R, Q, certificates, Ambainis)
- ✅ **Property testing** with probability bounds
- ✅ **O'Donnell curriculum** alignment
- ✅ **Family tracking** for asymptotic analysis
- ✅ **Monte Carlo estimation** (sampling module)

#### ✅ Recently Completed (Jan 2026)
- ✅ LAT/DDT tables (Linear Approximation Table, Difference Distribution Table)
- ✅ Differential uniformity and linearity measures
- ✅ Algebraic immunity computation
- ✅ SBoxAnalyzer class for comprehensive S-box analysis
- ✅ README library comparison section

#### Remaining Tasks
- [ ] Consider Rust FFI for large computations

---

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

---

## Nice to Have (No Timeline)

- Manim animations for educational content
- Dask distributed computation for large n
- ~~conda-forge package~~ (PR submitted, awaiting review)
- Interactive Jupyter widgets documentation
- GPU acceleration via CuPy/CUDA (BooLSPLG-inspired)
- Rust FFI for performance-critical paths

---

## Recently Completed

### OpenSSF Best Practices Badge (v1.0.0)
- Achieved **Passing** level (100%)
- Security scanning: CodeQL, Dependabot, OSSF Scorecard
- Branch protection, signed commits ready

### CI/CD Improvements (v1.0.0)
- Path-based filtering (skip tests for docs-only changes)
- Codespell integration for spell checking
- Docker image SHA pinning
- Benchmark runs only on releases

---

## Fourier Convention

O'Donnell standard (Analysis of Boolean Functions, Chapter 1):
- Boolean 0 → +1
- Boolean 1 → -1
- f̂(∅) = E[f]

---

## Prior Art

The query complexity module builds on ideas from:
- **Scott Aaronson's Boolean Function Wizard** (2000): Implemented D(f), R(f), Q(f), sensitivity, block sensitivity, certificate complexity, and degree measures.
- **Avishay Tal's library**: Fourier transforms, sensitivity, decision trees, polynomial representations, Krawchouk polynomials, Galois field operations.

See `src/boofun/analysis/query_complexity.py` for specific citations.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

High-value contributions:
- Tests for untested paths (priority: visualization, quantum)
- Avishay Tal library integration (Krawchouk, p-biased, decision tree DP)
- Cross-validation tests against other libraries
- Bug reports with reproducible examples
- Corrections to mathematical errors
- Notebook improvements and examples
