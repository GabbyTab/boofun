# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Pre-v1.0.0

This project is in active development. The API may change before v1.0.0.

### Development Notes
- Library renamed from `boolfunc` to `boofun`
- Fourier analysis uses O'Donnell convention (Boolean 0 → +1, Boolean 1 → -1)
- See [ROADMAP.md](ROADMAP.md) for current status and planned features

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
