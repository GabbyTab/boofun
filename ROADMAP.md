# BoolFunc Roadmap

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

### Quantum Module
- Oracle creation, Fourier analysis, property testing
- Grover analysis, quantum walks, element distinctness

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

</details>

---

## 📋 Remaining Tasks

### P0 - Before v1.0.0

| Task | Status | Notes |
|------|--------|-------|
| PyPI publication | ⏳ Blocked | Need PYPI_TOKEN configured |
| API stability guarantee | TODO | Document public API, add deprecation policy |
| Tutorial series | ✅ Done | 7 tutorials covering beginner to advanced |

### P1 - Quality & Polish

| Task | Status | Notes |
|------|--------|-------|
| Integration tests for new features | ✅ Done | 30 tests in test_new_features.py |
| Revamp examples/ folder | ✅ Done | 7 tutorials, cleaned old files |
| Clean up Jupyter notebooks | ✅ Done | Already use simplified API |
| Increase test coverage | TODO | Target 70%+ |

### P2 - Nice to Have

| Task | Notes |
|------|-------|
| Manim animations | Video content |
| Distributed computation | Dask support |
| conda-forge recipe | Alternative installation |

---

## 🎯 v1.0.0 Milestone Checklist

- [x] 60%+ test coverage
- [x] Performance benchmarks documented
- [x] 3+ real-world usage examples
- [x] LaTeX/TikZ export
- [ ] Published on PyPI
- [ ] API stability guarantee documented
- [x] Tutorial series complete (7 tutorials)

---

## 📝 API Stability Guarantee

For v1.0.0, we commit to:

1. **Stable Public API**: Methods in `BooleanFunction`, `PropertyTester`, `QueryComplexityProfile`, and family classes won't change signatures without deprecation warnings.

2. **Deprecation Policy**: 
   - Deprecated features get warnings for at least one minor version
   - Breaking changes only in major versions (2.0, 3.0, etc.)

3. **Documented API**: All public methods have docstrings and type hints.

**Affected modules:**
- `boolfunc.core.base.BooleanFunction`
- `boolfunc.analysis` (all public functions)
- `boolfunc.families` (all public classes)
- `boolfunc.quantum` (QuantumBooleanFunction)

---

## 📂 Tutorial Series Plan

### Beginner Tutorials
1. **Getting Started** - Installation, basic usage, first function
2. **Fourier Analysis Basics** - WHT, coefficients, Parseval's identity
3. **Common Function Families** - AND, OR, majority, parity, tribes

### Intermediate Tutorials
4. **Property Testing** - BLR, junta, monotonicity testing
5. **Query Complexity** - Sensitivity, block sensitivity, certificate
6. **Noise and Stability** - Noise stability, influences

### Advanced Tutorials
7. **Quantum Applications** - Grover, quantum walks, advantage
8. **Cryptographic Analysis** - S-box analysis, nonlinearity
9. **Research Applications** - FKN theorem, communication complexity

---

## 📊 Test Coverage Goals

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| core/base.py | ~65% | 80% | High |
| analysis/*.py | ~60% | 75% | High |
| families/*.py | ~50% | 70% | Medium |
| visualization/*.py | ~40% | 60% | Low |
| quantum/*.py | ~55% | 70% | Medium |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.

### Quick Wins
- Add tests for specific modules
- Improve docstrings
- Documentation improvements

### Medium Effort  
- Implement a missing algorithm
- Add visualization type
- Optimize performance

---

*This roadmap is a living document. Please open issues or PRs to suggest additions!*
