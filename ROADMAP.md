# BoolFunc Roadmap

This document tracks planned features, improvements, and known gaps in the boolfunc library.

**Last Updated**: January 2025  
**Current Version**: 0.2.0

---

## Current Status Summary

### What's Working
- Core BooleanFunction class with 12+ representations
- Simplified API with direct methods (`f.fourier()`, `f.influences()`, etc.)
- Spectral analysis, property testing, noise stability
- Function families with growth tracking
- 16 educational notebooks (all O'Donnell lectures + homework)
- GitHub Actions CI/CD with multi-OS, multi-Python testing
- GitHub Pages documentation

### Next Priorities
1. Publish to PyPI (package ready, needs token)
2. Increase test coverage
3. Update notebooks to use simplified API
4. Performance optimizations

---

## Priority Matrix

### P0 - Immediate
| Task | Status | Notes |
|------|--------|-------|
| Publish to PyPI | Ready | Need to configure PYPI_TOKEN secret |
| Add Codecov badge | ✅ DONE | Badge added to README |
| Update notebooks to direct API | ✅ DONE | Migrated to f.fourier(), f.influences(), etc. |

### P1 - High Priority
| Task | Status | Notes |
|------|--------|-------|
| Increase test coverage to 60%+ | ✅ DONE | Achieved 60% (was ~24%) |
| Enable Numba by default | ✅ DONE | Added to core dependencies |
| Add dependabot | ✅ DONE | Configuration added |
| Create tutorial series | TODO | Common workflows |

### P2 - Medium Priority
| Task | Status | Notes |
|------|--------|-------|
| GPU acceleration (CuPy) | TODO | For large transforms |
| Fluent/chainable API | TODO | `f.restrict().fourier()` |
| Animation of growth | TODO | As n increases |
| Comparison guide | TODO | vs BoolForge, pyeda, Sage |

### P3 - Future
| Task | Notes |
|------|-------|
| LaTeX/TikZ export | For Fourier diagrams |
| Manim animations | Video content |
| Distributed computation | Dask support |
| conda-forge recipe | Alternative installation |

---

## Educational Notebooks

All core educational content is complete:

| Category | Status |
|----------|--------|
| Homework notebooks (HW1-4) | Complete |
| Lecture notebooks (1-11) | Complete |
| Global Hypercontractivity | Complete |
| Asymptotic Visualization | Complete |
| Real-World Applications | Complete |

### Potential Additions
- O'Donnell book exercises (selected problems)
- Additional research paper implementations

---

## Testing

### Current Structure
```
tests/
├── analysis/          # Spectral analysis, complexity
├── benchmarks/        # Performance, external benchmarks
├── integration/       # End-to-end tests
├── property/          # Property-based tests (Hypothesis)
└── unit/              # Core functionality
```

### TODO
- [x] Increase coverage to 60%+ (achieved!)
- [x] Add mutation testing (mutmut) - configured in setup.cfg, scripts/run_mutation_tests.py
- [ ] Cross-validate with Sage/Mathematica
- [x] Test large n more thoroughly - tests/test_large_n.py added

---

## Missing Features

### Representations
| Representation | Status |
|----------------|--------|
| Decision Tree | TODO |
| ROBDD (Reduced Ordered BDD) | TODO |
| Circuit (AND/OR/NOT gates) | TODO |

### Analysis Methods
| Method | Status |
|--------|--------|
| Polynomial threshold functions | TODO |
| AC⁰ circuit complexity bounds | TODO |
| Communication complexity | TODO |
| Majority is Stablest (verification) | TODO |

### Function Families
| Family | Status |
|--------|--------|
| Iterated majority | TODO |
| Random DNF families | TODO |
| Cryptographic S-boxes | TODO |

---

## Performance Optimizations

### High Priority
- [x] Profile and optimize Walsh-Hadamard for n > 20 - scripts/profile_performance.py, core/optimizations.py
- [x] Optimize influence computation - vectorized + Numba implementations in core/optimizations.py
- [x] Add lazy evaluation for chained operations - LazyFourierCoefficients class
- [ ] Memory optimization (bitarray for truth tables?)

### Medium Priority
- [ ] GPU acceleration via CuPy
- [ ] Sparse representation auto-selection for n > 14
- [ ] Parallel computation for influences
- [ ] Aggressive memoization

---

## Infrastructure

### CI/CD Status
| Feature | Status |
|---------|--------|
| GitHub Actions workflow | Done |
| Multi-OS testing | Done |
| Multi-Python (3.9-3.12) | Done |
| Pre-commit hooks | Done |
| GitHub Pages docs | Done |
| PyPI publishing | Ready (needs token) |
| Codecov integration | Configured |

### TODO
- [ ] Configure PYPI_TOKEN for automated releases
- [x] Add dependabot for dependency updates
- [ ] Create Docker image

---

## v1.0.0 Milestone

Requirements for stable release:
- [x] 60%+ test coverage (achieved!)
- [ ] Published on PyPI
- [ ] API stability guarantee
- [ ] Performance benchmarks documented
- [ ] At least 3 real-world usage examples verified

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.

### Good First Issues
- Add tests for a specific module
- Improve docstrings
- Add type hints where missing
- Documentation improvements

### Intermediate
- Implement a missing algorithm
- Add a new visualization type
- Optimize a performance bottleneck

### Advanced
- Implement a research result
- Add GPU support for an operation
- Design new representation strategy

---

*This roadmap is a living document. Please open issues or PRs to suggest additions!*
