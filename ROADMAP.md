# Roadmap

**Version:** 1.2.0 (planned)
**Updated:** February 2026

## Vision

BooFun is the computational companion to O'Donnell's *Analysis of Boolean Functions*. The library makes every theorem runnable: create a function, compute its Fourier coefficients, verify KKL, visualize thresholds. 23 interactive notebooks cover Chapters 1-11.

**Primary audience**: TCS graduate students and researchers.
**Secondary**: Cryptographers analyzing S-boxes. Complexity theorists exploring conjectures.

---

## Current State (v1.1.1)

72% test coverage, 3200+ tests, 13 examples, 23 notebooks. Published on PyPI. Cross-validated against SageMath, thomasarmel/boolean_function, and BoolForge. See [CHANGELOG.md](CHANGELOG.md) for what shipped.

---

## v1.2.0 Goals

### Phase 1: Reduce Friction (1-2 weeks)

| Item | Status | Why |
|------|--------|-----|
| Make Numba optional dependency | ADR-004 | Biggest install friction. Move to `[performance]` extra. |
| Fix `__new__`/`_init` pattern | ADR-009 | Prevents normal subclassing. Replace with standard `__init__`. |
| Consolidate GPU modules | ADR-005 | Merge `gpu_acceleration.py` and `gpu.py` into one. Remove unused stubs, keep CuPy WHT. |
| Remove quantum from public API | ADR-005 | Stub-only. Keep as internal placeholder. |
| Delete `setup.cfg` | Cleanup | Consolidate into `pyproject.toml`. |
| conda-forge merge | PR #31964 | Fix python_min pin and pypi.org URL, request review. |

### Phase 2: Strengthen Core (1-2 months)

| Item | Why |
|------|-----|
| `f.is_global(alpha)` method on BooleanFunction | Natural API for global hypercontractivity. Simple delegation. |
| `Measure` class in `core/spaces.py` | Unify p-biased code: `f.expectation(measure=biased)` instead of scattered modules. |
| `bf.f2_polynomial(n, monomials)` | Create F2-polynomials directly. Needed for pseudorandomness notebooks. |
| Adaptive sampling in `estimate_fourier_coefficient` | `target_error=0.01` stops early. Low complexity. |
| Representation round-trip tests | Convert A -> B -> A for every conversion graph edge. |
| Lazy imports in `__init__.py` | Only if import time exceeds 5 seconds. |

### Phase 3: Quality (3+ months)

| Item | Why |
|------|-----|
| Enable mypy module-by-module | Start with `core/base.py`, `core/factory.py`. Remove `py.typed` until done. |
| Simplify conversion graph | Replace Dijkstra with two-level dispatch (source -> truth_table -> target). |
| Replace `ComputeCache` with `OrderedDict` | O(n) eviction -> O(1). |

---

## Nice to Have (No Timeline)

**Pseudorandomness** (CHLT ITCS 2019):
- Fractional PRG utilities: truncated Gaussian sampler, polarizing walk
- Epsilon-biased distribution construction
- Fooling verification

**Performance** (for research at larger n):
- GPU-accelerated WHT via CuPy for n > 20
- Parallel batch processing for exhaustive search
- Lazy evaluation for oracle-access functions
- Rust FFI for performance-critical paths (thomasarmel integration)

**Infrastructure**:
- ~~conda-forge~~ (PR #31964 submitted)
- Interactive Jupyter widgets
- Manim animations for educational content

**Function families**:
- N-value generators (`n_values.odd()`, `n_values.powers()`)
- Parameter sweep visualization
- BoolForge-inspired constrained random generation

---

## Architecture Decisions

See [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) for documented design choices (god object rationale, eager imports, conversion graph costs, etc.).

## Fourier Convention

O'Donnell standard (Chapter 1): Boolean 0 -> +1, Boolean 1 -> -1, f-hat(empty) = E[f].

## Prior Art

- **Scott Aaronson's Boolean Function Wizard** (2000): D(f), R(f), Q(f), sensitivity measures.
- **Avishay Tal's BooleanFunc.py**: Sensitivity, p-biased analysis, decision trees. See [migration guide](docs/guides/migration_from_tal.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). High-value contributions: bug reports with reproducible examples, mathematical error corrections, cross-validation tests, notebook improvements.
