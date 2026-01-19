# BooFun Roadmap

**Last Updated**: January 2025  
**Current Version**: 0.2.0

---

## 🎯 Status: Primary Functionality Complete!

All core features are implemented. Remaining work focuses on polish, testing, and documentation.

---

## ✅ What's Done (Collapsed)

<details>
<summary><strong>Click to expand completed features</strong></summary>

### Representations (12+)
- Truth table, sparse, packed (bitarray)
- Fourier expansion, ANF, DNF/CNF, polynomial
- LTF, circuit, BDD, symbolic
- Decision tree export (ASCII, DOT, JSON, TikZ)

### Analysis
- Property testing: BLR, junta, monotonicity, unateness, symmetry, balanced, dictator, affine, constant
- Query complexity: D(f), R0, R2, Q2, QE, s(f), bs(f), C(f), Ambainis, spectral adversary, polynomial method, general adversary
- FKN/dictatorship analysis
- Communication complexity
- Goldreich-Levin sparse learning

### Quantum Speedup Analysis
- Grover speedup estimation, quantum walks, element distinctness
- Theoretical analysis tools (no Qiskit required for speedup estimation)
- Oracle creation available with Qiskit installed

### Function Families
- Majority, Parity, AND, OR, Tribes, Threshold, Dictator
- LTF (weighted), RecursiveMajority3
- IteratedMajority, RandomDNF, SboxFamily

### Visualization
- Animation, growth plots, decision tree viz
- Interactive Plotly: spectrum, heatmaps, dashboard
- LaTeX/TikZ export for all diagrams

### Performance
- WHT optimization, Numba JIT, GPU (CuPy)
- Memory optimization (bitarray, sparse)
- Parallel batch operations, memoization

### Infrastructure
- GitHub Actions CI/CD (multi-OS, multi-Python)
- GitHub Pages docs, Codecov, dependabot
- Docker (Dockerfile + docker-compose)

### Recent Fixes (v0.2.x)
- **Fourier Sign Convention**: Fixed to use O'Donnell standard (Boolean 0 → +1, Boolean 1 → -1)
- **Test Determinism**: All PropertyTester instances now use seeded randomness
- **GPU Module**: Aligned with O'Donnell convention
- **Theoretical Bounds Tests**: Huang theorem, Nisan-Szegedy, complexity chain verification

</details>

---

## 📋 Remaining Tasks

### P0 - Before v1.0.0

| Task | Status | Notes |
|------|--------|-------|
| PyPI publication | ⏳ Blocked | Need PYPI_TOKEN configured |
| API stability guarantee | TODO | Document public API, add deprecation policy |
| Tutorial series | ✅ Done | 7 tutorials covering beginner to advanced |
| Test coverage 60%+ | TODO | Currently at 38%, need to increase |

### P1 - Quality & Polish

| Task | Status | Notes |
|------|--------|-------|
| Integration tests for new features | ✅ Done | 60+ tests in integration/ |
| Revamp examples/ folder | ✅ Done | 7 tutorials, cleaned old files |
| Clean up Jupyter notebooks | ✅ Done | Already use simplified API |
| Fuzz testing | ✅ Done | 18 Hypothesis-based fuzz tests |
| Theoretical bounds verification | ✅ Done | Huang, Nisan-Szegedy, certificate bounds |
| Cross-validation tests | ✅ Done | 31 tests against known values |

### P2 - Nice to Have

| Task | Notes |
|------|-------|
| Mutation testing | Run mutmut on critical paths |
| Manim animations | Video content |
| Distributed computation | Dask support |
| conda-forge recipe | Alternative installation |

---

## 🎯 v1.0.0 Milestone Checklist

- [ ] 60%+ test coverage (currently 38%)
- [x] Performance benchmarks documented
- [x] 3+ real-world usage examples
- [x] LaTeX/TikZ export
- [ ] Published on PyPI
- [ ] API stability guarantee documented
- [x] Tutorial series complete (7 tutorials)
- [x] Fuzz testing in place
- [x] Theoretical bounds verified

---

## 📝 API Stability Guarantee

For v1.0.0, we commit to:

1. **Stable Public API**: Methods in `BooleanFunction`, `PropertyTester`, `QueryComplexityProfile`, and family classes won't change signatures without deprecation warnings.

2. **Deprecation Policy**: 
   - Deprecated features get warnings for at least one minor version
   - Breaking changes only in major versions (2.0, 3.0, etc.)

3. **Documented API**: All public methods have docstrings and type hints.

**Affected modules:**
- `boofun.core.base.BooleanFunction`
- `boofun.analysis` (all public functions)
- `boofun.families` (all public classes)
- `boofun.quantum` (QuantumBooleanFunction)

---

## 📊 Test Coverage Status

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| core/base.py | 75% | 80% | High |
| analysis/*.py | 60-80% | 75% | High |
| families/*.py | 18-46% | 70% | Medium |
| visualization/*.py | 0-13% | 60% | Low |
| quantum/*.py | 11% | 70% | Medium |

**Total: ~1620 tests, 38% line coverage**

### Test Categories
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Property-based (Hypothesis): `tests/property/`
- Fuzz tests: `tests/fuzz/`
- Benchmarks: `tests/benchmarks/`
- Cross-validation: `tests/test_cross_validation.py`

---

## 📂 Mathematical Foundation

### Fourier Convention (O'Donnell Standard)
This library uses the O'Donnell convention for Fourier analysis:
- Boolean 0 → +1 (in ±1 domain)
- Boolean 1 → -1 (in ±1 domain)

This ensures `f̂(∅) = E[f]` and aligns with *Analysis of Boolean Functions* (O'Donnell, 2014).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.

### Quick Wins
- Add tests for low-coverage modules (visualization, quantum)
- Improve docstrings
- Documentation improvements

### Medium Effort  
- Implement a missing algorithm
- Add visualization type
- Optimize performance

---

*This roadmap is a living document. Please open issues or PRs to suggest additions!*
