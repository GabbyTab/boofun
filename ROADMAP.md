# BoolFunc Roadmap & TODO

This document tracks planned features, improvements, and known gaps in the boolfunc library.

**Last Updated**: January 2026

---

## 📚 Educational Notebooks

### Homework Notebooks
| Notebook | Status | File |
|----------|--------|------|
| HW1: Fourier Expansion | ✅ Done | `hw1_fourier_expansion.ipynb` |
| HW2: LTFs & Decision Trees | ✅ Done | `hw2_ltf_decision_trees.ipynb` |
| HW3: DNFs & Restrictions | ✅ Done | `hw3_dnf_restrictions.ipynb` |
| HW4: Hypercontractivity | ✅ Done | `hw4_hypercontractivity.ipynb` |

### Lecture Notebooks
| Lecture | Topic | Status | File |
|---------|-------|--------|------|
| 1 | Fourier Expansion, Orthogonality | ✅ Done | `lecture1_fourier_expansion.ipynb` |
| 2 | BLR Linearity Testing | ✅ Done | `lecture2_linearity_testing.ipynb` |
| 3 | Social Choice, Influences | ✅ Done | `lecture3_social_choice_influences.ipynb` |
| 4 | Influences, Effects | ✅ Done | `lecture4_influences_effects.ipynb` |
| 5 | Isoperimetric, Noise Stability, Arrow's | ✅ Done | `lecture5_noise_stability.ipynb` |
| 6 | Spectral Concentration & Learning | ✅ Done | `lecture6_spectral_concentration.ipynb` |
| 7 | Goldreich-Levin Algorithm | ✅ Done | `lecture7_goldreich_levin.ipynb` |
| 8 | Learning Juntas | ✅ Done | `lecture8_learning_juntas.ipynb` |
| 9 | DNFs, Random Restrictions | ✅ Done | `lecture9_dnf_restrictions.ipynb` |
| 10 | Fourier Concentration of DNFs | ✅ Done | `lecture10_fourier_concentration.ipynb` |
| 11 | Gaussian Analysis & Invariance | ✅ Done | `lecture11_invariance_principle.ipynb` |

### Paper Notebooks
| Paper | Status | Notes |
|-------|--------|-------|
| Global Hypercontractivity (Keevash et al.) | ✅ Done | `global_hypercontractivity.ipynb` |
| 1907.00847v2.pdf | ❌ TODO | Need to identify paper topic |
| 1908.08483v3.pdf | ❌ TODO | Need to identify paper topic |
| 1910.13433v2.pdf | ❌ TODO | Need to identify paper topic |
| v015a010.pdf | ❌ TODO | Need to identify paper topic |

### Additional Notebooks
| Topic | Status | File |
|-------|--------|------|
| Asymptotic Visualization | ✅ Done | `asymptotic_visualization.ipynb` |
| O'Donnell Book Exercises | ❌ TODO | Selected problems from the book |
| Real-World Applications | ✅ Done | `real_world_applications.ipynb` |

---

## 📖 Documentation

### Critical - Library Docs Update
The library has evolved significantly. Need comprehensive update:
- [ ] **Update README** with new features (simplified API, LTFs, growth tracking, global hypercontractivity)
- [ ] **Update docstrings** to show new direct methods (`f.influences()` not `SpectralAnalyzer(f).influences()`)
- [ ] **Update quickstart guide** with modern API examples
- [ ] **Simplify existing notebooks** to use direct API (43+ uses of SpectralAnalyzer to update)

### High Priority
- [x] **Host documentation online** ✅ GitHub Pages configured in CI
- [ ] Add API reference for new modules:
  - [ ] `global_hypercontractivity.py`
  - [ ] `families/` package
  - [ ] `visualization/growth_plots.py`
- [ ] Create tutorial series for common workflows

### Medium Priority
- [ ] Add mathematical background section (Fourier analysis primer)
- [ ] Document p-biased analysis comprehensively
- [ ] Add examples gallery with rendered outputs
- [ ] Create comparison guide vs other libraries (BoolForge, pyeda, Sage)

### Nice to Have
- [ ] Add LaTeX/TikZ export for Fourier diagrams
- [ ] Create cheat sheet / quick reference PDF
- [ ] Video tutorials

---

## 🧪 Testing

### Current Coverage
```
tests/
├── analysis/          # Spectral analysis, complexity
├── benchmarks/        # Performance, external benchmarks, gold standard
├── integration/       # End-to-end tests
├── property/          # (empty - needs property-based tests)
└── unit/              # Core functionality
```

### High Priority
- [x] **Add property-based tests with Hypothesis** ✅ DONE
  - [x] Fourier identities hold for random functions
  - [x] Representation conversions are invertible
  - [x] Influences are non-negative and sum correctly
- [ ] Add tests for new modules:
  - [ ] `test_global_hypercontractivity.py`
  - [ ] `test_families.py`
  - [ ] `test_growth_plots.py`
- [ ] Increase coverage to 80%+ (current estimate: ~60%)

### Medium Priority
- [ ] Add mutation testing (mutmut)
- [ ] Add fuzz testing for edge cases
- [ ] Test large n (query-access mode) more thoroughly
- [ ] Cross-validate with Sage/Mathematica on known functions

### Nice to Have
- [ ] Continuous benchmarking (track performance regression)
- [ ] Memory profiling tests

---

## ⚡ Performance & Efficiency

### High Priority
- [ ] **Enable Numba by default** (add to core dependencies)
- [ ] Profile and optimize critical paths:
  - [ ] Walsh-Hadamard transform for n > 20
  - [ ] Influence computation
  - [ ] Generalized influence (currently O(2^n) for all sets)
- [ ] Add lazy evaluation for chained operations
- [ ] Optimize memory for truth table storage (use bitarray?)

### Medium Priority
- [ ] GPU acceleration via CuPy for large transforms
- [ ] Sparse representation auto-selection for n > 14
- [ ] Parallel computation for influence calculations
- [ ] Cache more aggressively (memoization decorators)

### Nice to Have
- [ ] JIT compilation for user-defined functions
- [ ] Distributed computation support (Dask?)
- [ ] SIMD optimizations for bitwise operations

---

## 🔧 API Improvements

### From Notebook Feedback
Based on creating the educational notebooks, these API improvements would help:

- [x] **Add `bf.from_weights()` for LTFs** ✅ DONE
- [x] **Add `bf.random(n, balanced=True)`** ✅ DONE
- [x] **Add `f.restriction(fixed_vars)`** ✅ DONE
- [x] **Add `f.cofactor(var, val)`** ✅ DONE
- [x] **Improve dictator signature**: `bf.dictator(n, i=0)` ✅ DONE
- [x] **Add `f.hamming_weight()`** ✅ DONE
- [x] **Add `f.support()`** ✅ DONE
- [x] **Add `f.sensitivity()`** ✅ DONE (bonus!)

### General API
- [ ] Add fluent/chainable API: `f.restrict({0: 1}).fourier().influences()`
- [ ] Add context managers for p-biased analysis: `with bf.p_biased(0.3):`
- [ ] Add `BooleanFunction.from_circuit()` constructor
- [ ] Add `BooleanFunction.from_bdd()` constructor
- [ ] Add operator overloads for composition: `f @ g` for f(g(x))

---

## 🆕 Missing Features

### Representations
- [x] Truth Table
- [x] Fourier Coefficients
- [x] ANF (Algebraic Normal Form)
- [x] BDD (Binary Decision Diagram)
- [x] LTF (Linear Threshold Function)
- [x] **DNF (Disjunctive Normal Form)** ✅ Registered
- [x] **CNF (Conjunctive Normal Form)** ✅ Registered
- [ ] Decision Tree representation
- [ ] ROBDD (Reduced Ordered BDD)
- [ ] Circuit representation (AND/OR/NOT gates)

### Analysis Methods
- [x] Fourier analysis
- [x] Influences
- [x] Noise stability
- [x] Spectral concentration
- [x] Query complexity (BFW-style)
- [x] Gaussian analysis (O'Donnell Ch 10)
- [x] Invariance principle (O'Donnell Ch 11)
- [x] Global hypercontractivity
- [x] p-biased analysis
- [ ] **Arrow's Theorem verification**
- [ ] **Shapley values** (game theory connection)
- [ ] **Polynomial threshold functions**
- [ ] **AC⁰ circuit complexity bounds**
- [ ] **Communication complexity**

### Learning Algorithms
- [x] Goldreich-Levin (heavy Fourier coefficients)
- [x] LMN algorithm (low-degree learning)
- [x] **Junta learning** (Mossel-O'Donnell-Servedio) ✅ DONE
- [x] **Monotone function learning** ✅ DONE
- [x] **PAC learning framework** ✅ DONE

### Families & Growth
- [x] BooleanFamily base class
- [x] InductiveFamily for user-defined growth
- [x] GrowthTracker
- [x] Built-in families (Majority, Parity, AND, Tribes, LTF)
- [x] **Recursive Majority of 3** ✅ DONE
- [ ] **Iterated majority**
- [ ] **Random DNF families**
- [ ] **Cryptographic function families** (S-boxes)

---

## 📊 Visualization

### Current
- [x] Truth table plots
- [x] Fourier spectrum bar charts
- [x] Influence bar charts
- [x] Noise stability curves
- [x] Growth tracking plots
- [x] Interactive HTML (Plotly)

### High Priority
- [ ] **Interactive Jupyter widgets** (ipywidgets)
  - [ ] Slider for n to see how function changes
  - [ ] Toggle between representations
  - [ ] Real-time influence visualization
- [x] **3D hypercube visualization** for n ≤ 5 ✅ DONE
- [x] **Sensitivity heatmap visualization** ✅ DONE (bonus!)
- [x] **Decision tree structure visualization** ✅ DONE

### Medium Priority
- [ ] Animation of growth as n increases
- [ ] Heatmap for influence matrix
- [ ] Network graph for function composition

### Nice to Have
- [ ] Export to LaTeX/TikZ
- [ ] Export to Manim for animations
- [ ] D3.js interactive web visualizations

---

## 🔬 Research Features

### Landmark Results to Implement
- [x] KKL Theorem
- [x] Friedgut's Junta Theorem
- [x] Bonami's Lemma (hypercontractivity)
- [x] Global Hypercontractivity (Keevash et al.)
- [x] **Huang's Sensitivity Theorem** (complete implementation) ✅ DONE
- [x] **FKN Theorem** (functions close to dictators) ✅ DONE
- [ ] **Majority is Stablest** (complete proof verification)
- [ ] **Kahn-Kalai Conjecture** (expectation thresholds)
- [ ] **Park-Pham threshold result**

### Inverse/Cauchy-Schwarz Results
- [ ] Inverse Cauchy-Schwarz with noise (from Global Hypercontractivity)
- [ ] Level-k inequalities
- [ ] Hypercontractive inequalities for different norms

---

## 🏗️ Infrastructure

### CI/CD
- [x] GitHub Actions workflow
- [x] Multi-OS testing (Linux, macOS, Windows)
- [x] Multi-Python testing (3.9-3.12)
- [ ] **Fix PyPI publishing** (configure PYPI_TOKEN)
- [ ] Add Codecov badge to README
- [x] Add pre-commit hooks (black, isort, mypy) ✅ DONE
- [ ] Add dependabot for dependency updates

### Documentation Hosting
- [ ] Set up ReadTheDocs
- [ ] Or GitHub Pages with sphinx-github-pages
- [ ] Auto-deploy on release

### Package
- [ ] Publish to PyPI (v0.2.0 ready)
- [ ] Add conda-forge recipe
- [ ] Create Docker image for reproducible environment

---

## 🎯 Priority Matrix

### P0 - Critical (This Week)
1. ~~Register DNF/CNF strategies properly~~ ✅ DONE
2. ~~Add property-based tests~~ ✅ DONE (21 tests with Hypothesis)
3. ~~Fix dictator argument order API~~ ✅ DONE (now `dictator(n, i=0)`)
4. ~~Host documentation online~~ ✅ DONE (GitHub Pages in CI)
5. ~~Create lecture notebooks 3-11~~ ✅ DONE (all 11 lectures complete)
6. **Update docs/README for new simplified API** ← IN PROGRESS

### P1 - High (This Month)
1. ~~Create remaining lecture notebooks~~ ✅ ALL DONE
2. **Publish to PyPI** ← READY (package builds, needs token)
3. Add Jupyter widgets for interactivity
4. Enable Numba in default install
5. Update notebooks to use direct API

### P2 - Medium (This Quarter)
1. 3D hypercube visualization
2. Junta learning algorithm
3. Decision tree representation
4. GPU acceleration

### P3 - Nice to Have
1. LaTeX/TikZ export
2. Manim animations
3. Distributed computation
4. Formal verification connection

---

## 📝 Notes for Contributors

### Good First Issues
- Add tests for a specific function (e.g., `test_global_hypercontractivity.py`)
- Create a notebook for one lecture
- Improve docstrings in a module
- Add type hints where missing

### Intermediate
- Implement a missing learning algorithm
- Add a new visualization type
- Optimize a performance bottleneck

### Advanced
- Implement a research result
- Add GPU support for a specific operation
- Design new representation strategy

---

## 🏆 Milestone: v1.0.0

To reach v1.0.0, we need:
- [ ] 90%+ test coverage
- [ ] All O'Donnell chapters covered
- [ ] Hosted documentation
- [ ] Published on PyPI and conda-forge
- [ ] At least 3 real-world usage examples
- [ ] Performance benchmarks published
- [ ] API stability guarantee

---

*This roadmap is a living document. Please open issues or PRs to suggest additions!*
