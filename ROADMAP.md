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

**Status:** Planning

The `BooleanFunc.py` and `library.py` files contain valuable functionality from **Avishay Tal's PhD-era library** (shared via email, August 2025). The integration strategy is to **enhance existing modules** where possible rather than creating parallel "legacy" code. Query complexity measures (D, R, Q, certificates, sensitivity, block sensitivity) are already implemented in the modern `analysis/` modules.

#### Integration Map: Tal's Code → Existing Modules

**Enhance `analysis/sensitivity.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `average_sensitivity_moment(t)` | t-th moment of sensitivity distribution | Medium |
| `sens_coor(x)` | Return sensitive coordinates at input x | Low |
| `ts_2`, `ts_2_max` | 2-step transitive sensitivity | Low |

**Enhance `analysis/p_biased.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `FourierCoefMuP` | Alternative p-biased formula (cross-validate) | Medium |
| `asMuP` | Average sensitivity under μ_p | High |
| `asFourierMuP` | Total influence via p-biased Fourier | High |
| `parity_biased(n,k,i)` | Bias of parity function | Medium |

**Enhance `analysis/fourier.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `ann_influence(i, rho)` | Annealed/noisy influence | Medium |
| `truncated_degree_d(d)` | Truncate Fourier to degree d | Medium |
| `correlation(other)` | Correlation between functions | High |
| `fourier_weights()` | Weight distribution by degree | High |

**Enhance `analysis/restrictions.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `min_fixing(go_up)` | Minimum fixing to constant | Medium |
| `fix_func(func, val)` | Advanced restriction by function | Low |
| `shift(sh)` | Shift by integer mask | Medium |

**Enhance `analysis/symmetry.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `degree_sym(f)` | Degree for symmetric functions (fast) | Medium |
| `sens_sym(f)` | Sensitivity for symmetric functions | Medium |
| `shift_to_mono()` | Shift to monotone form | Low |

**Enhance `utils/math.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `lagrange_interpolation` | Polynomial interpolation | Medium |
| `span_nums` | Linear span over GF(2) | Low |

**Enhance `utils/finite_fields.py`:**
| Function | Description | Priority |
|----------|-------------|----------|
| `GF` class | Standalone GF(2^k) (no galois dep) | Medium |

#### New Modules to Create

**`analysis/decision_trees.py`** (High Priority):
- `calc_decision_tree_DP` - Basic DP for optimal tree
- `calc_decision_tree_DP_uniform` - Uniform distribution  
- `calc_decision_tree_DP_with_prob` - Custom input distribution
- `all_decision_trees` - Enumerate all trees
- `randomized_decision_tree` - Game-theoretic R(f) analysis
- `minimax` - LP solver for randomized complexity

**`analysis/sparsity.py`** (Medium Priority):
- `fourier_sparsity(f)` - Number of non-zero coefficients
- `fourier_sparsity_up_to_constants(f)` - Ignoring constant multiples
- `min_size_fourier_coef(f)` - Minimum |S| with non-zero f̂(S)

**`core/polynomials.py`** (Low Priority):
- `Polynomial` class - Real coefficients with arithmetic
- `FiniteFieldPolynomial` class - GF(q) coefficients

**`families/patterns.py`** (Low Priority):
- `pattern_function(pattern)` - Function from bit pattern
- `pattern_cyclic_function(pattern)` - Cyclic pattern families

#### Already Implemented (Cross-Validate Only)

The following exist in both codebases—use Tal's for test cases:
- Krawchouk polynomials (`utils/math.py` ✓)
- Sensitivity, block sensitivity (`analysis/sensitivity.py`, `analysis/block_sensitivity.py` ✓)
- Certificates (`analysis/certificates.py` ✓)
- Fourier/ANF transforms (`analysis/fourier.py`, `analysis/gf2.py` ✓)
- Influences (`analysis/fourier.py` ✓)
- Automorphisms (`analysis/symmetry.py` ✓)
- Basic utilities: `popcnt`, `poppar`, `over`, `subsets` (`utils/math.py` ✓)

#### Integration Workflow
1. **Cross-validate**: Run Tal's code against modern implementations, collect test cases
2. **Enhance existing**: Add missing functions to appropriate existing modules
3. **Create new sparingly**: Only create new modules when no natural home exists
4. **Type + document**: Add type hints, docstrings, O'Donnell references
5. **Test**: Property tests + Tal's known values as regression tests

### 2. Boolean Functions as Random Variables

**Status:** Planning

Flesh out the probabilistic treatment of Boolean functions and sampling from spectral distributions. Reference: O'Donnell, Chapters 1-3.

#### Why This Matters (from O'Donnell)
- Boolean functions can be viewed as random variables on the hypercube {-1, +1}^n
- The Fourier expansion f = Σ f̂(S)χ_S is an orthonormal decomposition
- Parseval: E[f²] = Σ f̂(S)² gives "energy" distribution across frequencies
- Sampling from the spectrum enables:
  - Estimating Fourier coefficients without full enumeration
  - Testing structural properties (juntas, low-degree)
  - Learning algorithms (Goldreich-Levin, KM algorithm)

#### New Features to Implement

| Feature | Description |
|---------|-------------|
| `f.sample(n_samples)` | Sample uniformly from {-1,+1}^n, return (x, f(x)) pairs |
| `f.spectral_sample(n_samples)` | Sample S with probability ∝ f̂(S)² |
| `f.estimate_fourier_coef(S, n_samples, confidence)` | Monte Carlo estimation of f̂(S) |
| `f.estimate_influence(i, n_samples)` | Estimate Inf_i[f] via sampling |
| `f.as_random_variable()` | Return object supporting E[], Var[], covariance |
| `spectral_distribution(f)` | Return scipy-like distribution object |
| `biased_sample(f, p, n_samples)` | Sample from μ_p distribution |

#### Tasks
- [ ] Implement `RandomVariableView` class
- [ ] Add spectral sampling with exact and approximate modes
- [ ] Implement coefficient estimation with confidence intervals
- [ ] Add cross-validation tests against exact computation
- [ ] Create tutorial notebook: `notebooks/boolean_functions_as_random_variables.ipynb`
- [ ] Document connection to O'Donnell Chapters 1-3

### 3. Increase Test Coverage to 70-75%

**Status:** In Progress (currently 65%)

| Module | Current | Target |
|--------|---------|--------|
| core/base.py | ~80% | 85% |
| analysis/*.py | 60-85% | 75-85% |
| families/*.py | 40-60% | 65-75% |
| visualization/*.py | 40-60% | 60-70% |
| quantum/*.py | ~20% | 40-50% |
| **Overall** | **65%** | **70-75%** |

#### Priority Areas
- [ ] Edge cases in representation conversions
- [ ] Error handling paths
- [ ] Visualization module (mock matplotlib)
- [ ] Quantum module basic coverage
- [ ] Tal library code after integration (Krawchouk, p-biased, etc.)

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

**Status:** Research

#### Libraries to Analyze

| Library | Language | Focus | Notable Features |
|---------|----------|-------|------------------|
| **thomasarmel/boolean_function** | Rust | Cryptographic analysis | ANF, bent detection, nonlinearity, fast/parallelizable |
| **BooLSPLG** | CUDA C++ | GPU acceleration | S-box analysis, LAT/DDT tables, massively parallel |
| **SageMath** | Python | General math | Comprehensive but slow |
| **SET (S-box Evaluation Tool)** | C++ | S-box analysis | Linearity, differential uniformity |

#### What We Have That Others Don't
- ✅ **Global hypercontractivity** analysis
- ✅ **Query complexity** (D, R, Q, certificates, Ambainis)
- ✅ **Property testing** with probability bounds
- ✅ **O'Donnell curriculum** alignment (educational notebooks)
- ✅ **Flexible inputs** (oracle pattern, streaming)
- ✅ **Family tracking** for asymptotic analysis

#### What Others Have That We Could Add
- ❓ **Bent function detection** (thomasarmel)
- ❓ **Nonlinearity computation** (cryptographic measure)
- ❓ **Linear Approximation Table (LAT)** (BooLSPLG)
- ❓ **Difference Distribution Table (DDT)** (BooLSPLG)
- ❓ **S-box differential uniformity** (BooLSPLG)
- ❓ **Algebraic immunity** (cryptographic)
- ❓ **GPU acceleration** (BooLSPLG style)

#### Tasks
- [ ] Install and test thomasarmel/boolean_function
- [ ] Review BooLSPLG paper for missing features
- [ ] Add cross-validation tests against other libraries
- [ ] Implement missing cryptographic measures
- [ ] Document feature comparison in README

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
