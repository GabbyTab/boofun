# Boolean Function Libraries Comparison

This guide identifies BooFun's unique niche and compares it with other Boolean function libraries.

## TL;DR: Where BooFun Excels

**BooFun is the only Python library that combines:**
1. **Query Complexity** - D(f), R(f), Q(f), bs(f), certificate complexity
2. **Property Testing** - BLR linearity, junta, monotonicity, dictatorship tests
3. **Theoretical Fourier Analysis** - Influences, noise stability, KKL, hypercontractivity
4. **Quantum Speedup Analysis** - Grover speedup, quantum walk analysis, Q₂(f) bounds

**Use BooFun when:**
- Researching computational complexity of Boolean functions
- Teaching/learning Boolean function theory (O'Donnell's book)
- Property testing algorithms development
- Quantum speedup estimation for Boolean functions
- Reproducing theoretical computer science results

---

## Libraries Compared

| Library | Primary Focus | Query Complexity | Property Testing | Fourier Analysis | Quantum Speedup |
|---------|--------------|------------------|------------------|------------------|-----------------|
| **BooFun** | Theory & Complexity | ✅ Full suite | ✅ BLR, junta, etc. | ✅ O'Donnell style | ✅ (theoretical) |
| **SageMath** | Cryptography | ❌ | ❌ | ⚠️ Walsh only | ❌ |
| **pyeda** | Logic/SAT/BDD | ❌ | ❌ | ❌ | ❌ |
| **BoolForge** | Biology/Networks | ❌ | ❌ | ❌ | ❌ |
| **CANA** | Network Control | ❌ | ❌ | ❌ | ❌ |

---

## BooFun's Unique Features

### 1. Query Complexity (UNIQUE - No Other Library Has This)

```python
from boofun.analysis.query_complexity import QueryComplexityProfile

profile = QueryComplexityProfile(f)
measures = profile.compute()

# Available measures:
# - D(f): Deterministic query complexity
# - R0(f): Zero-error randomized complexity
# - R2(f): Bounded-error randomized complexity
# - Q(f): Quantum query complexity
# - bs(f): Block sensitivity
# - s(f): Sensitivity
# - C(f): Certificate complexity
# - Approximate degree, threshold degree
# - Ambainis bound (quantum lower bound)
```

**Why this matters:** Query complexity is fundamental to computational complexity theory. No other Python library provides these measures.

### 2. Property Testing (UNIQUE - No Other Library Has This)

```python
from boofun.analysis import PropertyTester

tester = PropertyTester(f)

# BLR Linearity Test (Blum-Luby-Rubinfeld)
is_linear = tester.blr_linearity_test(num_queries=100)

# Junta Testing
is_junta = tester.junta_test(k=3)

# Monotonicity Testing
is_monotone = tester.monotonicity_test()

# Dictatorship Testing
is_dictator = tester.dictatorship_test()
```

**Why this matters:** Property testing is a major area of TCS research. SageMath and pyeda don't have these algorithms.

### 3. Theoretical Fourier Analysis (Different from SageMath)

| Concept | BooFun | SageMath |
|---------|----------|----------|
| Walsh-Hadamard Transform | ✅ | ✅ |
| Influences (Inf_i[f]) | ✅ | ❌ |
| Total Influence (I[f]) | ✅ | ❌ |
| Noise Stability (Stab_ρ[f]) | ✅ | ❌ |
| Fourier Degree | ✅ | ✅ (algebraic_degree) |
| KKL Theorem verification | ✅ | ❌ |
| Hypercontractivity | ✅ | ❌ |
| Nonlinearity | ✅ | ✅ |
| Bent functions | ❌ | ✅ |
| Correlation immunity | ❌ | ✅ |

**SageMath focuses on cryptographic properties** (bent, plateaued, correlation immunity).
**BooFun focuses on O'Donnell-style theoretical analysis** (influences, noise stability).

### 4. Quantum Speedup Analysis (Educational)

```python
from boofun.quantum import grover_speedup, quantum_walk_analysis

# Analyze Grover speedup for a function
f = bf.AND(6)
result = grover_speedup(f)
print(f"Grover speedup: {result['speedup']:.2f}x")

# Quantum walk analysis
walk = quantum_walk_analysis(f)
print(f"Quantum walk speedup: {walk['speedup_over_classical']:.2f}x")

# These work without Qiskit - purely theoretical analysis
```

**What this provides:**
- **Theoretical speedup estimation**: Grover, quantum walks, element distinctness
- **Educational tools**: Understanding when quantum helps Boolean function problems
- **Query complexity**: Q₂(f) bounds via adversary method

**Note:** Actual quantum oracle creation requires Qiskit (`pip install qiskit`). Without Qiskit, the module provides theoretical/educational analysis only - all "quantum" methods fall back to classical algorithms for computation while still providing accurate speedup estimates.

---

## Feature Comparison

### Core Representations

| Feature | BooFun | BoolForge | CANA | pyeda |
|---------|----------|-----------|------|-------|
| Truth Table | ✅ | ✅ | ✅ | ✅ |
| ANF (Algebraic Normal Form) | ✅ | ✅ (polynomial) | ❌ | ❌ |
| Fourier Expansion | ✅ | ❌ | ❌ | ❌ |
| BDD | ✅ (basic) | ❌ | ❌ | ✅ (full) |
| CNF/DNF | ✅ | ✅ (to_logical) | ❌ | ✅ |
| Circuit | ✅ | ❌ | ❌ | ✅ |

### Analysis Methods

| Analysis | BooFun | BoolForge | CANA | Notes |
|----------|----------|-----------|------|-------|
| **Fourier Transform** | ✅ | ❌ | ❌ | BooFun specialty |
| **Influences** | ✅ | ✅ (activities) | ❌ | Same concept, different name |
| **Total Influence** | ✅ | ✅ (avg_sensitivity) | ❌ | Equivalent |
| **Noise Stability** | ✅ | ❌ | ❌ | BooFun specialty |
| **Spectral Analysis** | ✅ | ❌ | ❌ | BooFun specialty |
| **Canalization** | ❌ | ✅ | ✅ | Gap in BooFun |
| **k-Canalizing** | ❌ | ✅ | ❌ | Gap in BooFun |
| **Input Redundancy** | ❌ | ✅ (via CANA) | ✅ | Gap in BooFun |
| **Edge Effectiveness** | ❌ | ✅ (via CANA) | ✅ | Gap in BooFun |
| **Symmetry Groups** | ❌ | ✅ | ✅ | Gap in BooFun |
| **Essential Variables** | ✅ | ✅ | ✅ | All have this |
| **Sensitivity** | ✅ | ✅ | ❌ | Both have this |
| **Monotonicity** | ✅ | ✅ | ❌ | Both have this |

### Property Testing

| Property Test | BooFun | BoolForge | Notes |
|--------------|----------|-----------|-------|
| Linearity (BLR) | ✅ | ❌ | BooFun specialty |
| Junta Test | ✅ | ❌ | BooFun specialty |
| Monotonicity | ✅ | ✅ (exact) | Different approaches |
| Dictatorship | ✅ | ❌ | BooFun specialty |

### Function Families

| Family | BooFun | BoolForge | Notes |
|--------|----------|-----------|-------|
| Majority | ✅ | ❌ | |
| Parity | ✅ | ❌ | |
| Tribes | ✅ | ❌ | |
| AND/OR/XOR | ✅ | ❌ | |
| Random Functions | ✅ (basic) | ✅ (extensive) | BoolForge has many random generators |
| NCF (Nested Canalizing) | ❌ | ✅ | Gap in BooFun |

### Boolean Networks

| Feature | BooFun | BoolForge | CANA |
|---------|----------|-----------|------|
| Single Functions | ✅ | ✅ | ✅ |
| Networks | ❌ | ✅ | ✅ |
| Attractors | ❌ | ✅ | ✅ |
| State Transition Graph | ❌ | ✅ | ✅ |
| Control Analysis | ❌ | ❌ | ✅ |

---

## Unique Strengths

### BooFun
- **Fourier Analysis**: Walsh-Hadamard transform, spectral analysis
- **Theoretical Results**: KKL theorem, Poincaré inequality, hypercontractivity
- **Property Testing**: BLR linearity, junta testing
- **Educational Focus**: O'Donnell lecture notebooks
- **Performance**: GPU acceleration, Numba optimization

### BoolForge
- **Canalization**: Deep analysis of canalizing structure
- **Random Generation**: Many specialized random function generators
- **Boolean Networks**: Full network analysis with attractors
- **Biological Focus**: Loading models from Cell Collective

### CANA
- **Redundancy Analysis**: Input redundancy, effective connectivity
- **Symmetry Detection**: Schematodes algorithm for exact symmetry
- **Control Theory**: Driver variables, controllability
- **Network Simplification**: Effective graph extraction

### pyeda
- **BDD Operations**: Full reduced ordered BDD support
- **SAT Solving**: Integration with SAT solvers
- **Logic Minimization**: Espresso algorithm
- **Production Ready**: Mature, well-tested

---

## Cross-Validation Tests

### BooFun ↔ SageMath (Walsh-Hadamard)

```python
# Test: Walsh-Hadamard transforms should match
# SageMath uses different normalization, so we compare structure

# BooFun
import boofun as bf
f = bf.majority(5)
wht_boofun = f.fourier()

# SageMath equivalent:
# from sage.crypto.boolean_function import BooleanFunction
# sage_f = BooleanFunction([int(f.evaluate(x)) for x in range(32)])
# wht_sage = sage_f.walsh_hadamard_transform()

# Verify: same non-zero positions, proportional values
```

### BooFun ↔ BoolForge (Influences/Activities)

```python
# BooFun "influences" = BoolForge "activities"
import boofun as bf

# BooFun
f_bf = bf.majority(5)
influences = f_bf.influences()
total_influence = f_bf.total_influence()

# BoolForge equivalent:
# from boolforge import BooleanFunction
# f_forge = BooleanFunction([int(f_bf.evaluate(x)) for x in range(32)])
# activities = f_forge.get_activities(EXACT=True)
# avg_sensitivity = f_forge.get_average_sensitivity(EXACT=True, NORMALIZED=False)

# Expected: influences ≈ activities, total_influence ≈ avg_sensitivity
```

### Theoretical Validation (No External Library Needed)

Our `tests/test_theoretical_validation.py` validates against known mathematical results:

```python
# Parseval's Identity: Σ f̂(S)² = 1
# Poincaré Inequality: Var[f] ≤ I[f]
# KKL Theorem: max Inf_i ≥ Ω(Var·log(n)/I[f])
# Sheppard's Formula: Majority noise stability → (1/2) + arcsin(ρ)/π
```

---

## Who Should Use What

### Use BooFun For:

| Use Case | Why BooFun |
|----------|-------------|
| **Query complexity research** | Only library with D(f), R(f), Q(f), bs(f) |
| **Property testing algorithms** | BLR, junta, monotonicity tests |
| **Learning Boolean function theory** | O'Donnell lecture notebooks |
| **TCS paper reproduction** | Matches theoretical CS conventions |
| **Quantum speedup analysis** | Theoretical speedup estimation (Grover, walks) |
| **Influence/noise stability analysis** | Only library with full Fourier toolset |

### Use Other Libraries For:

| Use Case | Best Library | Why |
|----------|-------------|-----|
| Cryptographic analysis | **SageMath** | Bent functions, correlation immunity |
| SAT solving | **pyeda** | Mature SAT/BDD implementation |
| BDD manipulation | **pyeda** | Production-ready ROBDD |
| Gene regulatory networks | **BoolForge** | Biology-focused network tools |
| Network canalization | **BoolForge + CANA** | Specialized canalization tools |
| Control theory | **CANA** | Driver variables, controllability |
| Large-scale networks | **BoolForge** | Network attractors, state transitions |

### Overlap Zones:

| Feature | Libraries with Support |
|---------|----------------------|
| Truth tables | All |
| Walsh-Hadamard | BooFun, SageMath |
| Monotonicity | BooFun (test), BoolForge (exact) |
| Sensitivity | BooFun, BoolForge |
| Symmetry groups | BooFun (new), BoolForge, CANA |

---

## Strategic Roadmap for BooFun

### Core Niche (Strengthen What Makes Us Unique)

These are areas where **no other library competes**:

1. **Query Complexity** - ✅ COMPLETE (14 measures)
   - [x] D(f), R(f), Q(f), bs(f), C(f)
   - [x] Ambainis bound, spectral adversary
   - [x] **Polynomial method bound** (NEW)
   - [x] **General adversary bound** (NEW)
   - [x] Approximate degree, threshold degree
   - [ ] Add visualization of complexity measures

2. **Property Testing** - ✅ COMPLETE (9 tests)
   - [x] BLR linearity test
   - [x] Junta testing
   - [x] Monotonicity testing
   - [x] **Unateness testing** (NEW)
   - [x] Dictatorship testing (via `PropertyTester.dictator_test()`)
   - [x] Affine, balanced, symmetry, constant tests
   - [ ] Add tolerant testing variants

3. **FKN/Dictatorship Analysis** - ✅ COMPLETE
   - [x] `fkn.distance_to_dictator()`
   - [x] `fkn.closest_dictator()`
   - [x] `fkn.fkn_theorem_bound()`
   - [x] `fkn.analyze_dictator_proximity()`

4. **Quantum Speedup Analysis** - ✅ Educational/Theoretical
   - [x] Grover speedup estimation
   - [x] Quantum walk analysis
   - [x] Element distinctness analysis
   - [x] Quantum query complexity bounds (Q₂(f))
   - [x] Quantum resource estimation
   - [ ] Full Qiskit integration (oracle execution) - future, requires Qiskit

### Completed Additions (from BoolForge/CANA concepts)

- [x] `is_canalizing()` - detect canalization
- [x] `is_k_canalizing(k)` - nested canalization depth
- [x] `get_canalizing_depth()` - layer depth
- [x] `get_symmetry_groups()` - variable symmetry
- [x] `get_input_types()` - positive/negative/conditional
- [x] `input_redundancy()` - redundancy measure

### Future: Interoperability (Low Priority)

Consider adding if users request:
- [ ] `to_sage()` / `from_sage()` - SageMath interop
- [ ] `to_boolforge()` / `from_boolforge()` - BoolForge interop
- [ ] `to_cana()` / `from_cana()` - CANA interop

### NOT Adding (Outside Our Niche)

These are better served by specialized libraries:
- ❌ Full Boolean network support → Use BoolForge
- ❌ Bent/plateaued function analysis → Use SageMath
- ❌ SAT solving/BDD operations → Use pyeda
- ❌ Control theory → Use CANA

---

## Installation Commands

```bash
# BooFun (pending PyPI)
pip install git+https://github.com/GabbyTab/boofun

# BoolForge
pip install git+https://github.com/ckadelka/BoolForge

# CANA
pip install cana

# pyeda
pip install pyeda
```

---

## References

- O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge University Press.
- Correia et al. (2018). *CANA: A python package for quantifying control and canalization in Boolean networks*. Frontiers in Physiology.
- Kadelka et al. (2023). *Collectively canalizing Boolean functions*. Advances in Applied Mathematics.
