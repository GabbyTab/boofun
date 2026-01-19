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
| GPU acceleration (CuPy) | ✅ DONE | core/gpu.py with CPU fallback |
| Fluent/chainable API | ✅ DONE | xor(), and_(), permute(), extend(), apply_noise(), etc. |
| Animation of growth | ✅ DONE | visualization/animation.py, visualization/growth_plots.py |
| Comparison guide | ✅ DONE | docs/comparison_guide.md + canalization module |

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
- [x] Cross-validate with Sage/Mathematica - tests/test_theoretical_validation.py (50 tests)
- [x] Test large n more thoroughly - tests/test_large_n.py added

---

## Feature Inventory

### Property Testing (UNIQUE - No competitor has these)
| Test | Status | Module |
|------|--------|--------|
| BLR Linearity | ✅ Done | `PropertyTester.blr_linearity_test()` |
| Junta (k-junta) | ✅ Done | `PropertyTester.junta_test(k)` |
| Monotonicity | ✅ Done | `PropertyTester.monotonicity_test()` |
| **Unateness** | ✅ Done | `PropertyTester.unateness_test()` |
| Symmetry | ✅ Done | `PropertyTester.symmetry_test()` |
| Balanced | ✅ Done | `PropertyTester.balanced_test()` |
| Dictator/Anti-dictator | ✅ Done | `PropertyTester.dictator_test()` |
| Affine | ✅ Done | `PropertyTester.affine_test()` |
| Constant | ✅ Done | `PropertyTester.constant_test()` |

### Query Complexity (UNIQUE - No competitor has these)
| Measure | Status | Function |
|---------|--------|----------|
| D(f) - Deterministic | ✅ Done | `deterministic_query_complexity()` |
| R0(f) - Zero-error | ✅ Done | `zero_error_randomized_complexity()` |
| R2(f) - Bounded-error | ✅ Done | `bounded_error_randomized_complexity()` |
| Q2(f) - Quantum bounded | ✅ Done | `quantum_query_complexity()` |
| QE(f) - Quantum exact | ✅ Done | `exact_quantum_complexity()` |
| s(f) - Sensitivity | ✅ Done | `sensitivity_lower_bound()` |
| bs(f) - Block sensitivity | ✅ Done | `block_sensitivity_lower_bound()` |
| C(f) - Certificate | ✅ Done | `certificate_lower_bound()` |
| Ambainis bound | ✅ Done | `ambainis_complexity()` |
| Spectral adversary | ✅ Done | `spectral_adversary_bound()` |
| **Polynomial method** | ✅ Done | `polynomial_method_bound()` |
| **General adversary** | ✅ Done | `general_adversary_bound()` |
| Approximate degree | ✅ Done | `approximate_degree()` |
| Threshold degree | ✅ Done | `threshold_degree()` |

### FKN/Dictatorship Analysis
| Feature | Status | Module |
|---------|--------|--------|
| Distance to dictator | ✅ Done | `fkn.distance_to_dictator()` |
| Closest dictator | ✅ Done | `fkn.closest_dictator()` |
| FKN theorem bounds | ✅ Done | `fkn.fkn_theorem_bound()` |
| Is close to dictator | ✅ Done | `fkn.is_close_to_dictator()` |
| Spectral gap | ✅ Done | `fkn.spectral_gap()` |
| Dictator proximity analysis | ✅ Done | `fkn.analyze_dictator_proximity()` |

### Quantum Module (UNIQUE bridge to Qiskit)
| Feature | Status | Class/Function |
|---------|--------|----------------|
| Quantum oracle creation | ✅ Done | `QuantumBooleanFunction.create_quantum_oracle()` |
| Quantum Fourier analysis | ✅ Done | `QuantumBooleanFunction.quantum_fourier_analysis()` |
| Quantum property testing | ✅ Done | `QuantumBooleanFunction.quantum_property_testing()` |
| Quantum resource estimation | ✅ Done | `QuantumBooleanFunction.get_quantum_resources()` |
| Quantum advantage estimation | ✅ Done | `estimate_quantum_advantage()` |
| **Grover analysis** | ✅ Done | `QuantumBooleanFunction.grover_analysis()` |
| **Grover amplitude evolution** | ✅ Done | `QuantumBooleanFunction.grover_amplitude_analysis()` |

### Canalization (from BoolForge/CANA concepts)
| Feature | Status | Function |
|---------|--------|----------|
| Is canalizing | ✅ Done | `is_canalizing()` |
| K-canalizing | ✅ Done | `is_k_canalizing()` |
| Canalizing depth | ✅ Done | `get_canalizing_depth()` |
| Symmetry groups | ✅ Done | `get_symmetry_groups()` |
| Input redundancy | ✅ Done | `input_redundancy()` |
| Edge effectiveness | ✅ Done | `edge_effectiveness()` |

---

## Still Missing

### Representations
| Representation | Status | Priority |
|----------------|--------|----------|
| Decision Tree (export) | TODO | Low |
| ROBDD | TODO | Low (use pyeda) |
| Circuit (gates) | TODO | Low |

### Analysis Methods
| Method | Status | Priority |
|--------|--------|----------|
| Polynomial method lower bounds | ✅ Done | `polynomial_method_bound()` |
| General adversary method | ✅ Done | `general_adversary_bound()` |
| **Goldreich-Levin algorithm** | ✅ Done | `analysis/learning.py` |
| Communication complexity | TODO | Low |

### Visualization
| Feature | Status | Module |
|---------|--------|--------|
| Animation of growth | ✅ Done | `visualization/animation.py` |
| Growth plots | ✅ Done | `visualization/growth_plots.py` |
| Decision tree viz | ✅ Done | `visualization/decision_tree.py` |
| Interactive widgets | ✅ Done | `visualization/widgets.py` |
| **Interactive Fourier spectrum** | ✅ Done | `visualization/interactive.py` |
| **Influence heatmaps** | ✅ Done | `visualization/interactive.py` |
| **FourierExplorer dashboard** | ✅ Done | `visualization/interactive.py` |

### Quantum Algorithms
| Feature | Status | Module |
|---------|--------|--------|
| Grover analysis | ✅ Done | `quantum/__init__.py` |
| **Quantum walk analysis** | ✅ Done | `quantum/__init__.py` |
| **Element distinctness** | ✅ Done | `quantum/__init__.py` |
| Quantum walk search | ✅ Done | `quantum/__init__.py` |

### Function Families
| Family | Status | Priority |
|--------|--------|----------|
| Iterated majority | TODO | Low |
| Random DNF families | TODO | Low |
| Cryptographic S-boxes | TODO | Low (use SageMath) |

---

## Performance Optimizations

### High Priority
- [x] Profile and optimize Walsh-Hadamard for n > 20 - scripts/profile_performance.py, core/optimizations.py
- [x] Optimize influence computation - vectorized + Numba implementations in core/optimizations.py
- [x] Add lazy evaluation for chained operations - LazyFourierCoefficients class
- [x] **Memory optimization (bitarray)** - core/representations/packed_truth_table.py

### Medium Priority
- [x] GPU acceleration via CuPy - core/gpu.py module
- [x] **Sparse representation auto-selection for n > 14** - core/auto_representation.py
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
