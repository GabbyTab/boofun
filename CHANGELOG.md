# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-18

### Added

#### Simplified API
- Direct methods on BooleanFunction: `f.fourier()`, `f.influences()`, `f.degree()`, `f.total_influence()`, `f.max_influence()`, `f.variance()`, `f.noise_stability()`, `f.analyze()`
- Property testing methods: `f.is_linear()`, `f.is_monotone()`, `f.is_balanced()`, `f.is_junta(k)`
- Top-level shortcuts: `bf.majority()`, `bf.parity()`, `bf.tribes()`, `bf.dictator()`, `bf.constant()`, `bf.AND()`, `bf.OR()`, `bf.threshold()`, `bf.weighted_majority()`, `bf.random()`, `bf.from_weights()`

#### New Analysis Modules
- `global_hypercontractivity.py` - Keevash et al. p-biased hypercontractivity analysis
- `arrow.py` - Arrow's Theorem verification and Shapley values
- `fkn.py` - FKN Theorem (functions close to dictators)
- `huang.py` - Huang's Sensitivity Theorem implementation
- `invariance.py` - Invariance principle and Gaussian analysis
- `pac_learning.py` - PAC learning framework with membership/equivalence queries
- `ltf_analysis.py` - Linear Threshold Function analysis

#### Function Families
- `boolfunc.families` package with `FunctionFamily`, `InductiveFamily` base classes
- Built-in families: `MajorityFamily`, `ParityFamily`, `TribesFamily`, `ThresholdFamily`, `ANDFamily`, `ORFamily`, `DictatorFamily`, `LTFFamily`
- `GrowthTracker` for asymptotic behavior visualization with theoretical comparisons

#### Representations
- DNF Form properly registered
- CNF Form properly registered
- Improved conversion graph with Dijkstra-based optimal path finding

#### Educational Content
- 16 Jupyter notebooks covering O'Donnell book (Lectures 1-11, HW1-4)
- Research paper notebooks (global hypercontractivity, asymptotic visualization)
- Real-world applications notebook

#### Testing
- Property-based tests with Hypothesis (21 tests for Fourier identities, representation conversions)
- External benchmarks against theoretical results
- Gold standard tests for known function properties

#### Infrastructure
- GitHub Actions CI with multi-OS (Linux, macOS, Windows) and multi-Python (3.9-3.12) testing
- GitHub Pages documentation deployment
- Pre-commit hooks configuration
- Codecov integration

### Changed
- Improved dictator function signature: `bf.dictator(n, i=0)` instead of `bf.dictator(i, n)`
- Performance example updated to use direct API
- Documentation updated for GitHub Pages (was ReadTheDocs)

### Fixed
- Operator overloading to use string literals instead of operator objects
- Truth table representation evaluation bugs
- Import system consistency issues

## [0.1.0] - 2024-12-01

### Added
- Initial library structure and architecture
- Core `BooleanFunction` class with truth table representation
- Built-in Boolean functions: `majority`, `dictator`, `parity`, `constant`
- `SpectralAnalyzer` for Fourier analysis and influence computation
- Basic `PropertyTester` framework
- Factory pattern for function creation with `bf.create()`
- Space translation utilities ({0,1} ↔ {-1,+1})
- Documentation framework with Sphinx

### Notes
- First development release
- API subject to change in future versions
