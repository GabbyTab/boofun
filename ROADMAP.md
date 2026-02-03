# Roadmap

**Version:** 1.1.0
**Updated:** January 2026

## Current State

Core functionality is production-ready. Test coverage at **72%** with 3056+ tests. API is stable with structured exception hierarchy and comprehensive error handling. The library supports the full O'Donnell textbook curriculum with unique features like global hypercontractivity analysis.

## What Exists

### Representations
Truth table (dense, sparse, packed), Fourier expansion, ANF, DNF/CNF, polynomial, LTF, circuit, BDD, symbolic. Automatic conversion between formats via conversion graph.

### Analysis
- **Property testing**: BLR linearity, junta, monotonicity, unateness, symmetry, quasisymmetry, balance, dictator, affine, constant, primality, **local correction** (BLR self-correction)
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

## v1.1.0 Goals (Complete)

### 1. Advanced Analysis Features

**Status:** ✅ Complete (Jan 2026)

New analysis capabilities for sensitivity, p-biased measures, decision trees, and Fourier sparsity. These features enhance the library's coverage of O'Donnell's textbook and provide research-grade tools for Boolean function analysis.

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
- ✅ `p_biased_fourier_coefficient()` - Efficient formula via Fourier transform
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
- [x] Create tutorial notebook: `notebooks/boolean_functions_as_random_variables.ipynb`

### 3. Increase Test Coverage to 70-75%

**Status:** ✅ Complete (currently 71%, target 70-75%)

| Module | Current | Target | Notes |
|--------|---------|--------|-------|
| core/base.py | ~80% | 85% | |
| analysis/*.py | 70-85% | 75-85% | +145 new tests (advanced analysis + sampling) |
| analysis/hypercontractivity.py | ~70% | 70% | ✅ 76 tests |
| analysis/global_hypercontractivity.py | ~70% | 70% | ✅ 28 tests |
| families/*.py | 55-70% | 65-75% | +28 new tests |
| visualization/*.py | 60-70% | 60-70% | ✅ 176 tests |
| quantum/*.py | ~40% | 40-50% | ✅ 21 tests |
| **Overall** | **72%** | **70-75%** | **3056+ tests** ✅ |

#### Recent Progress (Jan 2026)
- ✅ Added `PropertyTester.local_correct()` - BLR self-correction (7 tests)
- ✅ Added 145+ tests for advanced analysis features
- ✅ Added 43 tests for sampling module
- ✅ Added 28 tests for families module
- ✅ Added 35 tests for symmetry/restrictions enhancements
- ✅ Added 58 tests for cryptographic module (LAT/DDT, algebraic immunity)
- ✅ Added 46 tests for partial functions (streaming, hex I/O)
- ✅ Created tutorial notebook for random variables
- ✅ Registered PackedTruthTableRepresentation in factory
- ✅ Exposed hypercontractivity API at top level

#### Priority Areas
- [x] Hypercontractivity module tests (Chapter 9 O'Donnell) - 76 tests covering all functions
- [x] Global hypercontractivity tests (Keevash et al.) - 28 tests including analyzer class
- [x] Edge cases in representation conversions - 97+ tests covering roundtrips, BDD, bit ordering
- [x] Error handling paths - 30 tests covering exception hierarchy, validation, error codes
- [x] Visualization module (mock matplotlib) - 176 tests covering all visualization modules
- [x] Quantum module basic coverage - 21 tests covering Grover, quantum walks, element distinctness

### 4. Documentation Improvements

**Status:** Planning

#### README Updates
- [x] Document batch processing: "NumPy array matrices can be interpreted as batches" (Jan 2026)
- [x] Highlight unique features: "Global hypercontractivity analysis" (already documented)
- [x] Add library comparison section (already present)

#### API Documentation
- [x] Complete docstrings for all public functions (104 public API functions documented)
- [x] Add mathematical notation (LaTeX) to key functions (Jan 2026)
- [x] Include "See Also" cross-references in hypercontractivity module (Jan 2026)

#### Examples & Tutorials (Jan 2026)
- [x] `08_cryptographic_analysis.py` - S-box analysis, LAT/DDT, bent functions
- [x] `09_partial_functions.py` - Streaming, hex I/O, storage hints
- [x] `10_sensitivity_decision_trees.py` - Sensitivity moments, decision tree DP
- [x] Updated `examples/README.md` with new examples and notebook index
- [x] Updated `docs/index.rst` with v1.1 features

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
- [x] Documented Rust FFI consideration for future performance optimization

### 6. Partial Boolean Functions API

**Status:** ✅ Complete (Jan 2026)

User-friendly API for working with partial/streaming Boolean function specification, inspired by thomasarmel/boolean_function's partial representation capabilities.

#### ✅ Implemented Features

| Feature | API | Description |
|---------|-----|-------------|
| Partial function | `bf.partial(n=20)` | Create with only some known outputs |
| Incremental add | `p.add(idx, value)` | Add values one at a time (streaming) |
| Batch add | `p.add_batch({...})` | Add multiple values at once |
| Completeness | `p.completeness` | Fraction of known values |
| Confidence estimation | `p.evaluate_with_confidence(idx)` | Estimate unknown values |
| Convert to full | `p.to_function()` | Convert to BooleanFunction |
| Hex input | `bf.from_hex("ac90", n=4)` | thomasarmel-compatible |
| Hex output | `bf.to_hex(f)` | Export to hex string |
| Storage hints | `bf.create(tt, storage='packed')` | Choose storage strategy |

#### PartialBooleanFunction Class

```python
# Create partial - only some outputs known
p = bf.partial(n=30, known_values={0: True, 1: False})

# Stream in data
p.add(5, True)
p.add_batch({10: True, 11: False, 12: True})

# Query status
p.completeness      # 0.0000001 (fraction known)
p.num_known         # 6
p.is_known(5)       # True

# Evaluate
p.evaluate(0)       # True (known)
p.evaluate(100)     # None (unknown)
p[0]                # Indexing syntax

# Estimation
val, conf = p.evaluate_with_confidence(100)  # (False, 0.3)

# Convert when ready
f = p.to_function(fill_unknown=False)
```

#### Hex String I/O (thomasarmel-compatible)

```python
# From thomasarmel examples
f = bf.from_hex("0xac90", n=4)        # 4-bit bent function
f = bf.from_hex("0113077C165E76A8", n=6)  # 6-bit bent

# Export
bf.to_hex(f)  # "ac90"
```

#### Storage Hints

```python
bf.create(tt, storage='auto')    # Default - selects best
bf.create(tt, storage='dense')   # Standard truth table
bf.create(tt, storage='packed')  # 1 bit per entry (n > 14)
bf.create(tt, storage='sparse')  # Only store exceptions
bf.create(oracle, n=20, storage='lazy')  # Compute on demand
```

### 7. API Improvements & Feature Exposure

**Status:** ✅ In Progress (Jan 2026)

Exposing existing but hidden functionality through the public API.

#### ✅ Completed

**Representation Registration:**
- ✅ `packed_truth_table` - Now registered via `representations/__init__.py`
- ✅ `sparse_truth_table` - Already registered
- ✅ `adaptive_truth_table` - Already registered
- ✅ Factory updated to accept new representation types

**Hypercontractivity API (Chapter 9 O'Donnell):**

Now accessible at top-level via `bf.*`:

```python
import boofun as bf

# KKL Theorem bounds
bf.kkl_lower_bound(total_influence, n)
bf.max_influence_bound(f)

# Bonami's Lemma and hypercontractivity
bf.noise_operator(f, rho)
bf.bonami_lemma_bound(f, q, rho)
bf.hypercontractive_inequality(f, rho, p, q)
bf.level_d_inequality(f, d, q)
bf.lq_norm(f, q)

# Friedgut's Junta Theorem
bf.friedgut_junta_bound(total_influence, epsilon)
bf.junta_approximation_error(f, junta_vars)
```

**Global Hypercontractivity (Keevash, Lifshitz, Long & Minzer):**

```python
# Analyze p-biased measures and global functions
bf.GlobalHypercontractivityAnalyzer(f, p=0.5)
bf.is_alpha_global(f, alpha, max_set_size)
bf.generalized_influence(f, S, p)

# p-biased analysis
bf.p_biased_expectation(f, p, samples)
bf.p_biased_influence(f, i, p, samples)
bf.p_biased_total_influence(f, p, samples)
bf.noise_stability_p_biased(f, rho, p, samples)

# Threshold phenomena
bf.threshold_curve(f, p_range, samples)
bf.find_critical_p(f, samples, tolerance)
bf.hypercontractivity_bound(f, p)
```

#### Remaining Tasks

- [x] Add tests for hypercontractivity functions - 76 tests covering all functions
- [x] Add tests for global hypercontractivity - 28 tests including analyzer class
- [x] Fix sparse_truth_table evaluation bug (fixed Jan 2026 - `_binary_to_index` now uses LSB=x₀ convention)
- [x] Document hypercontractivity API in README (Jan 2026)

#### Tests
- 46 comprehensive tests for partial functions
- Cross-validation with thomasarmel/boolean_function examples

---

## v1.2.0 Goals (Proposed)

### 1. Function Family Enhancements

**Status:** Planning

Improve the families module for easier growth analysis and visualization.

#### N-Value Generators

Currently, users manually specify `n_values=[3, 5, 7, ...]`. Add convenience generators:

```python
from boofun.families import n_values

# Built-in generators
n_values.odd(3, 15)           # [3, 5, 7, 9, 11, 13, 15]
n_values.powers(2, max_n=64)  # [2, 4, 8, 16, 32, 64]
n_values.adaptive(3, 100)     # [3, 6, 12, 24, 48, 96] - doubling
n_values.stepped(3, 100, [10, 50])  # every 1 up to 10, every 5 up to 50, every 10 after
```

#### Parameter Sweep Visualization

Plot properties vs parameters other than n:

```python
# Vary tribe size k while fixing n
tracker.sweep_parameter('k', values=[2, 3, 4, 5], fixed_n=20)
viz.plot_parameter_sweep(tracker, 'influence_max')

# Compare LTF weight patterns
trackers = {
    'Uniform': LTFFamily.uniform(),
    'Geometric(0.5)': LTFFamily.geometric(0.5),
    'Harmonic': LTFFamily.harmonic(),
}
viz.plot_family_comparison(trackers, 'total_influence')
```

#### Perturbation and Constraint Families

```python
# Add noise to any family
noisy_maj = NoisyFamily(MajorityFamily(), noise_rate=0.05)

# Apply constraints
restricted = ConstrainedFamily(TribesFamily(),
    constraint=lambda f: f.is_balanced())

# Custom growth rules
class MyFamily(InductiveFamily):
    def step(self, f_prev, n, n_prev):
        return f_prev.extend(n, fill='random')
```

### 2. Lazy Property System (Architecture)

**Status:** Planning

A general architecture for lazy computation and caching of function properties.

#### Core Concept

Each BooleanFunction has a **PropertyStore** that:
1. Caches computed properties (avoid recomputation)
2. Knows which algorithms can compute each property
3. Selects the best algorithm based on context

```python
# User just asks for a property
inf = f.total_influence()  # Internally:
                           # 1. Check cache - return if computed
                           # 2. Select algorithm based on:
                           #    - Function size (n_vars)
                           #    - Available representations
                           #    - User hints
                           # 3. Compute, cache, return
```

#### Algorithm Registry

For each property, register multiple computation methods:

```python
# Pseudocode for registry
PropertyRegistry.register(
    name='total_influence',
    algorithms=[
        Algorithm(
            name='exact_fourier',
            compute=lambda f: sum(f.influences()),
            requires=['fourier'],
            complexity=O(2**n),
            exact=True,
        ),
        Algorithm(
            name='sample_pivotal',
            compute=lambda f, samples: estimate_total_influence(f, samples),
            requires=[],  # Works with any representation
            complexity=O(samples * n),
            exact=False,
        ),
    ]
)
```

#### Smart Selection

The system picks algorithms based on:

| Factor | Small n (≤12) | Medium n (13-20) | Large n (>20) |
|--------|---------------|------------------|---------------|
| Default | Exact | Exact if cached, else approximate | Approximate |
| Has Fourier | Use spectral formula | Use spectral formula | Use spectral formula |
| Query-only | Sample-based | Sample-based | Sample-based |

#### User Control

```python
# Let system decide
f.total_influence()

# Force exact (may be slow)
f.total_influence(exact=True)

# Force approximate with sample count
f.total_influence(approximate=True, samples=10000)

# Check what's cached
f.properties.cached()  # ['fourier', 'total_influence', ...]

# Clear cache (e.g., after mutation)
f.properties.clear()
```

#### Where to Store

Options being considered:

| Location | Pros | Cons |
|----------|------|------|
| `BooleanFunction.properties` | Direct access, intuitive | Clutters base class |
| `core/properties.py` (new) | Clean separation | Extra import |
| Extend `core/representations/` | Properties derived from reps | Conceptually different |

**Recommendation**: New `core/properties.py` module with `PropertyStore` class attached to each function instance.

#### Integration with Families

```python
# In GrowthTracker, properties are computed lazily
tracker.mark('total_influence')  # Registers intent
tracker.observe(n_values=[...])  # Computes, using best algorithm for each n
```

### 3. Families Documentation

**Status:** Complete

| Task | Status |
|------|--------|
| Create `docs/guides/families.md` guide | ✅ Done |
| Add families examples to notebooks | Medium priority |
| Document theoretical bounds | Medium priority |
| Add growth visualization tutorial | Medium priority |

### 4. Dashboard and Visualization Improvements

**Status:** In Progress

- [x] Summary statistics: expectation, variance, degree, sparsity
- [x] Truth table legend and axis labels
- [x] Vertical bars with short labels
- [ ] Interactive parameter sliders (Jupyter widgets)
- [ ] Export to LaTeX/TikZ

### 5. Globality Properties in PropertyStore

**Status:** Planning

Add α-globality as a first-class property in the `PropertyStore`, enabling lazy computation and caching of global hypercontractivity measures.

#### Motivation

Currently, `is_alpha_global()` is a standalone function. Integrating it into `PropertyStore` would:
1. Allow caching of computed generalized influences
2. Enable property-based testing (`f.is_global(alpha)`)
3. Support automatic algorithm selection (exact vs Monte Carlo)

#### Proposed API

```python
# Simple check
f.is_global(alpha=4.0)  # Returns bool

# Full details with caching
f.globality_alpha(max_set_size=3)  # Returns minimum α for which f is α-global

# Property access
f.properties.get('alpha_global')  # Cached minimum α
f.properties.get('max_generalized_influence')  # Cached max I_S
f.properties.get('worst_influence_set')  # The set S achieving max I_S

# Computed generalized influences
f.generalized_influence({0, 1})  # I_{0,1}(f) with caching
```

#### Implementation Tasks

| Task | Priority | Complexity |
|------|----------|------------|
| Add `alpha_global` property to PropertyStore | High | Medium |
| Add `f.is_global(alpha)` method to BooleanFunction | High | Low |
| Add `f.generalized_influence(S)` method | Medium | Low |
| Cache generalized influences in PropertyStore | Medium | Medium |
| Add tests for globality properties | High | Medium |

### 6. P-Biased Measures in Spaces Module

**Status:** Planning

Move p-biased measure functionality into the `core/spaces.py` module, treating p-biased measures as a fundamental space alongside the existing boolean/±1 cubes.

#### Motivation

Currently, p-biased analysis is scattered across modules. A unified approach would:
1. Treat `μ_p` as a first-class measure on the hypercube
2. Enable consistent p-biased expectation, variance, norms across the library
3. Support space-aware property computation

#### Proposed Architecture

```python
from boofun.core.spaces import Space, Measure

# Measures as first-class objects
uniform = Measure.uniform()        # p = 0.5
biased = Measure.p_biased(p=0.1)   # μ_p with p = 0.1

# Function properties under different measures
f.expectation(measure=biased)       # E_μp[f]
f.variance(measure=biased)          # Var_μp[f]  
f.total_influence(measure=biased)   # I^p[f]
f.noise_stability(rho, measure=biased)  # S_ρ^p[f]

# Fourier analysis under measures
analyzer = SpectralAnalyzer(f, measure=biased)
analyzer.fourier_coefficients()  # p-biased Fourier coefficients

# Space translation with measures
Space.translate(x, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE, measure=biased)
```

#### Implementation Tasks

| Task | Priority | Complexity |
|------|----------|------------|
| Add `Measure` class to `core/spaces.py` | High | Medium |
| Add `Measure.uniform()` and `Measure.p_biased(p)` | High | Low |
| Extend `BooleanFunction` methods to accept `measure` param | High | High |
| Migrate `analysis/p_biased.py` to use Measure | Medium | Medium |
| Update `SpectralAnalyzer` to support measures | Medium | High |
| Add tests for measure-aware computations | High | Medium |

### 7. PAC Learning and Estimation Enhancements

**Status:** Planning

Enhance the PAC learning module with richer estimation support, confidence intervals, and adaptive sampling. Reference: O'Donnell Chapter 3, Lecture 6.

#### EstimatedBooleanFunction Class

A function backed by estimated Fourier coefficients with confidence intervals:

```python
from boofun.analysis.pac_learning import EstimatedBooleanFunction

# Create from samples (query access to unknown function)
est_f = EstimatedBooleanFunction.from_oracle(oracle, n=10, initial_samples=1000)

# Coefficients are estimates with confidence intervals
coeff, stderr = est_f.fourier_coefficient(S)  # Returns (estimate, standard_error)

# Automatically increase samples when needed
est_f.refine(target_stderr=0.01)  # Add more samples until stderr < 0.01

# Access confidence information
est_f.confidence_level        # Current confidence (e.g., 0.95)
est_f.total_samples           # Total samples used
est_f.coefficient_errors      # Dict of {S: stderr} for all estimated coefficients

# Convert to regular function when sufficiently confident
f = est_f.to_function(threshold=0.05)  # Zero out coefficients with |est| < threshold
```

#### Adaptive Sampling

Automatically determine sample complexity based on target accuracy:

```python
from boofun.analysis.learning import estimate_fourier_coefficient

# Current API
est, stderr = estimate_fourier_coefficient(f, S, num_samples=1000)

# Proposed: adaptive sampling
est, stderr = estimate_fourier_coefficient(
    f, S,
    target_error=0.01,      # Stop when stderr < target_error
    confidence=0.95,         # Confidence level
    max_samples=100000,      # Budget limit
)

# Returns additional info
result = estimate_fourier_coefficient(f, S, target_error=0.01, return_info=True)
# result.estimate, result.stderr, result.samples_used, result.converged
```

#### Integration with `uncertainties` Library

The `errormodels.py` module already has optional `uncertainties` support. Extend to learning:

```python
from boofun.analysis.pac_learning import pac_learn_low_degree

# With uncertainties integration
coeffs = pac_learn_low_degree(f, max_degree=3, epsilon=0.1, with_uncertainties=True)
# coeffs[S] is a ufloat with value and uncertainty

# Propagates through computations
hypothesis_value = sum(c * chi_S(x) for S, c in coeffs.items())
# hypothesis_value.nominal_value, hypothesis_value.std_dev
```

#### Estimation Convergence Visualization

```python
from boofun.visualization import plot_estimation_convergence

# Show how estimate improves with samples
fig = plot_estimation_convergence(
    f, S,
    sample_sizes=[50, 100, 200, 500, 1000, 2000],
    true_value=f.fourier()[S],  # Optional: show true value
)
```

#### Implementation Priority

| Feature | Priority | Complexity | Dependencies |
|---------|----------|------------|--------------|
| Adaptive sampling | High | Low | None |
| `EstimatedBooleanFunction` | Medium | Medium | None |
| Convergence visualization | Medium | Low | matplotlib |
| `uncertainties` integration | Low | Low | uncertainties (optional) |

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

**Infrastructure:**
- Manim animations for educational content
- Dask distributed computation for large n
- ~~conda-forge package~~ (PR submitted, awaiting review)
- Interactive Jupyter widgets documentation
- GPU acceleration via CuPy/CUDA (BooLSPLG-inspired)
- Rust FFI for performance-critical paths

**Inspired by BoolForge (Kadelka & Coberly, 2025):**
- Random function generators with constraints:
  - `random_k_canalizing(n, k)` - specific canalizing depth
  - `random_with_bias(n, bias)` - specific bias/Hamming weight
  - `random_layer_structure(n, layers)` - nested canalizing with structure
- Canalizing layer structure analysis (`get_layer_structure()`)
- Canalizing strength metric
- Null model generation for ensemble experiments
- Consider: Basic Boolean network support (out of scope for TCS focus?)

---

## Recently Completed

### OpenSSF Best Practices Badge (v1.0.0)
- Achieved **Passing** level (100%)
- Security scanning: CodeQL, Dependabot, OSSF Scorecard
- Branch protection, signed commits ready

### CI/CD Improvements (v1.0.0)
- Path-based filtering (skip tests for docs-only changes)
- Notebook validation runs only on notebook changes
- Codespell integration for spell checking
- Docker image SHA pinning
- Benchmark runs only on releases

### Representation Bit Ordering Fixes (Jan 2026)
- Fixed `_binary_to_index` in truth_table, sparse_truth_table, packed_truth_table, polynomial (now uses LSB=x₀)
- Fixed `_index_to_binary` in dnf_form, cnf_form, circuit, bdd, ltf (now uses LSB=x₀)
- Fixed BDD Shannon expansion to work correctly with LSB-first truth tables
- Fixed API storage hints to not incorrectly apply to callable data
- Added 5 new tests for direct binary vector evaluation (`TestDirectBinaryVectorEvaluation`)

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
- **Avishay Tal**: Sensitivity moments, p-biased analysis, decision tree algorithms, polynomial representations.

See `src/boofun/analysis/query_complexity.py` for specific citations.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

High-value contributions:
- Tests for untested paths (priority: visualization, quantum)
- Advanced analysis features (Krawchouk polynomials, p-biased analysis, decision tree DP)
- Cross-validation tests against other libraries
- Bug reports with reproducible examples
- Corrections to mathematical errors
- Notebook improvements and examples
