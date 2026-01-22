# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [1.1.0] - 2026-01-22

### Added

**Hypercontractivity Module (Chapter 9 O'Donnell)**
- `noise_operator(f, rho)` - Apply noise operator T_ρ to Boolean functions
- `lq_norm(f, q)` - Compute L_q norms
- `bonami_lemma_bound(f, q, rho)` - Bonami's Lemma bounds
- `kkl_lower_bound(total_influence, n)` - KKL theorem lower bound on max influence
- `max_influence_bound(f)` - Compute and compare max influence with KKL bound
- `friedgut_junta_bound(total_influence, epsilon)` - Friedgut's junta theorem
- `junta_approximation_error(f, junta_vars)` - Junta approximation error
- `hypercontractive_inequality(f, rho, p, q)` - Hypercontractive inequality check
- `level_d_inequality(f, d, q)` - Level-d Fourier inequality

**Global Hypercontractivity (Keevash et al.)**
- `GlobalHypercontractivityAnalyzer` class for p-biased analysis
- `is_alpha_global(f, alpha, max_set_size)` - Check if function is α-global
- `generalized_influence(f, S, p)` - Generalized influence of set S
- `p_biased_expectation(f, p)` - P-biased expectation
- `p_biased_influence(f, i, p)` - P-biased influence
- `p_biased_total_influence(f, p)` - P-biased total influence
- `threshold_curve(f, p_range)` - Threshold phenomena analysis
- `find_critical_p(f)` - Find critical probability
- `hypercontractivity_bound(f, p)` - Hypercontractivity bounds

**Cryptographic Analysis Module**
- `nonlinearity(f)` - Distance to nearest affine function
- `is_bent(f)` - Bent function detection
- `walsh_transform(f)` - Walsh transform coefficients
- `walsh_spectrum(f)` - Walsh spectrum analysis
- `algebraic_degree(f)` - Algebraic degree via ANF
- `algebraic_normal_form(f)` - ANF computation
- `correlation_immunity(f)` - Correlation immunity order
- `resiliency(f)` - Resiliency order
- `strict_avalanche_criterion(f)` - SAC check
- `linear_approximation_table(sbox)` - LAT for S-boxes
- `difference_distribution_table(sbox)` - DDT for S-boxes
- `SBoxAnalyzer` class for comprehensive S-box analysis

**Partial Boolean Functions**
- `bf.partial(n, known_values)` - Create partial Boolean functions
- `PartialBooleanFunction.add(idx, value)` - Streaming specification
- `PartialBooleanFunction.add_batch(values)` - Batch addition
- `PartialBooleanFunction.evaluate_with_confidence(idx)` - Confidence-based evaluation
- `PartialBooleanFunction.to_function()` - Convert to full BooleanFunction
- `bf.from_hex(hex_str, n)` - Create from hex string (thomasarmel-compatible)
- `bf.to_hex(f)` - Export to hex string
- Storage hints: `bf.create(data, storage='packed'|'sparse'|'auto'|'lazy')`

**Sensitivity Analysis Enhancements**
- `average_sensitivity_moment(f, t)` - t-th moment of sensitivity distribution
- `sensitive_coordinates(f, x)` - Return sensitive coordinates at input x
- `sensitivity_histogram(f)` - Distribution of sensitivity values
- `arg_max_sensitivity(f)` / `arg_min_sensitivity(f)` - Find extremal inputs

**Decision Tree Algorithms**
- `decision_tree_depth_dp(f)` - DP algorithm for optimal depth
- `compute_randomized_complexity(f)` - Randomized decision tree complexity
- `count_decision_trees(f)` - Count optimal decision trees
- `enumerate_decision_trees(f)` - Enumerate all optimal trees

**Sampling Module**
- `sample_uniform(n, n_samples)` - Uniform sampling
- `sample_biased(n, p, n_samples)` - P-biased sampling
- `sample_spectral(f, n_samples)` - Spectral sampling
- `estimate_fourier_coefficient(f, S, n_samples)` - Monte Carlo Fourier estimation
- `estimate_influence(f, i, n_samples)` - Monte Carlo influence estimation
- `RandomVariableView` class for probabilistic analysis
- `SpectralDistribution` class

**New Examples**
- `08_cryptographic_analysis.py` - S-box analysis, LAT/DDT, bent functions
- `09_partial_functions.py` - Streaming, hex I/O, storage hints
- `10_sensitivity_decision_trees.py` - Sensitivity moments, decision tree DP

### Fixed
- **Bit ordering consistency**: Fixed `_binary_to_index` in 4 representation files to use LSB=x₀ convention
- **Bit ordering consistency**: Fixed `_index_to_binary` in 5 representation files to use LSB=x₀ convention
- **BDD Shannon expansion**: Fixed to work correctly with LSB-first truth tables
- **API storage hints**: Fixed to not incorrectly apply packed_truth_table to callable data
- **Empty data handling**: Fixed `np.log2(0)` warning in API for empty truth tables

### Changed
- Updated documentation to describe features rather than implementation sources
- Added LaTeX mathematical notation to hypercontractivity docstrings
- Added "See Also" cross-references to key public API functions
- Updated `docs/index.rst` with v1.1 feature highlights
- Test count increased to 2931 (from ~2500)
- Test coverage at ~70%

---

## [1.0.0] - 2026-01-15

First stable release with production-ready API.

### Features
- Structured exception hierarchy with error codes
- File I/O (JSON, .bf, DIMACS CNF)
- Flexible input handling and oracle pattern
- OpenSSF Best Practices badge (Passing level)
- CI/CD with path-based filtering

---

## [0.2.1] - 2026-01-20

### Added
- Backwards-compatible property aliases: `num_variables` and `num_vars` as aliases for `n_vars`
- Missing `_compare_fourier_matplotlib()` function for function comparison plotting
- Notebook validation in CI pipeline (validates key notebooks on every push)
- Separate lint job in CI that enforces black, isort, and flake8

### Fixed
- **CRT no-solution handling**: SymPy's `crt()` returns None for unsolvable systems; now raises proper ValueError
- **Numba prange compatibility**: Removed `parallel=True` from WHT function to fix Numba's variable step size error
- **Function name collisions**: Renamed `test_function_expectation` → `compute_test_function_expectation` and `test_representation` → `validate_representation` to avoid pytest collection conflicts
- **Notebook API consistency**: Fixed all notebooks to use `n_vars` instead of `num_variables`
- **Notebook imports**: Added missing `SpectralAnalyzer` and `fourier` imports to notebooks that needed them
- **Notebook argument order**: Fixed `dictator(i, n)` → `dictator(n, i)` in hw1_fourier_expansion.ipynb
- **Canalization bug**: Fixed undefined `fixed_vars_values` in `_compute_canalizing_depth_recursive()`
- **Type hint forward references**: Added proper TYPE_CHECKING imports for `QueryModel`, `AccessType`, `DNFFormula`, `CNFFormula`, `BooleanFunction`

### Changed
- **CI workflow restructured**: Separate lint job that must pass before tests run
- **Linting enforced**: Black and flake8 checks now fail CI instead of continue-on-error
- **Code formatting**: Ran black and isort on entire codebase (172 files reformatted)
- **Unused imports removed**: Ran autoflake to clean up unused imports across codebase

### Deprecated
- `test_representation()` function - use `validate_representation()` instead
- `test_function_expectation()` function - use `compute_test_function_expectation()` instead

---

## [0.2.0] - Previous Release

Initial public release with core functionality.

### Features
- Multiple Boolean function representations (12+ types)
- Spectral analysis (Fourier transform, influences, noise stability)
- Property testing (BLR linearity, monotonicity, junta testing)
- Query complexity analysis (BFW-style measures)
- Function families with growth tracking
- Visualization tools
- Educational notebooks (16+ aligned with O'Donnell textbook)

---

## [1.0.0] - TBD

First stable release. See [ROADMAP.md](ROADMAP.md) for v1.0.0 milestone checklist.
