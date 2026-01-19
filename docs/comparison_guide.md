# Boolean Function Libraries Comparison

This guide compares BoolFunc with other Boolean function libraries to help users choose the right tool and understand interoperability options.

## Libraries Compared

| Library | Focus | PyPI | Active |
|---------|-------|------|--------|
| **BoolFunc** | Fourier analysis, property testing, educational | Pending | Yes |
| **[BoolForge](https://github.com/ckadelka/BoolForge)** | Canalization, Boolean networks, random generation | No (GitHub) | Yes |
| **[CANA](https://pypi.org/project/cana/)** | Control & redundancy in Boolean networks | Yes | Yes |
| **[pyeda](https://pyeda.readthedocs.io/)** | BDD, SAT, logic minimization | Yes | Maintenance |
| **[SageMath](https://www.sagemath.org/)** | Comprehensive math (includes Boolean functions) | Via conda | Yes |

---

## Feature Comparison

### Core Representations

| Feature | BoolFunc | BoolForge | CANA | pyeda |
|---------|----------|-----------|------|-------|
| Truth Table | ✅ | ✅ | ✅ | ✅ |
| ANF (Algebraic Normal Form) | ✅ | ✅ (polynomial) | ❌ | ❌ |
| Fourier Expansion | ✅ | ❌ | ❌ | ❌ |
| BDD | ✅ (basic) | ❌ | ❌ | ✅ (full) |
| CNF/DNF | ✅ | ✅ (to_logical) | ❌ | ✅ |
| Circuit | ✅ | ❌ | ❌ | ✅ |

### Analysis Methods

| Analysis | BoolFunc | BoolForge | CANA | Notes |
|----------|----------|-----------|------|-------|
| **Fourier Transform** | ✅ | ❌ | ❌ | BoolFunc specialty |
| **Influences** | ✅ | ✅ (activities) | ❌ | Same concept, different name |
| **Total Influence** | ✅ | ✅ (avg_sensitivity) | ❌ | Equivalent |
| **Noise Stability** | ✅ | ❌ | ❌ | BoolFunc specialty |
| **Spectral Analysis** | ✅ | ❌ | ❌ | BoolFunc specialty |
| **Canalization** | ❌ | ✅ | ✅ | Gap in BoolFunc |
| **k-Canalizing** | ❌ | ✅ | ❌ | Gap in BoolFunc |
| **Input Redundancy** | ❌ | ✅ (via CANA) | ✅ | Gap in BoolFunc |
| **Edge Effectiveness** | ❌ | ✅ (via CANA) | ✅ | Gap in BoolFunc |
| **Symmetry Groups** | ❌ | ✅ | ✅ | Gap in BoolFunc |
| **Essential Variables** | ✅ | ✅ | ✅ | All have this |
| **Sensitivity** | ✅ | ✅ | ❌ | Both have this |
| **Monotonicity** | ✅ | ✅ | ❌ | Both have this |

### Property Testing

| Property Test | BoolFunc | BoolForge | Notes |
|--------------|----------|-----------|-------|
| Linearity (BLR) | ✅ | ❌ | BoolFunc specialty |
| Junta Test | ✅ | ❌ | BoolFunc specialty |
| Monotonicity | ✅ | ✅ (exact) | Different approaches |
| Dictatorship | ✅ | ❌ | BoolFunc specialty |

### Function Families

| Family | BoolFunc | BoolForge | Notes |
|--------|----------|-----------|-------|
| Majority | ✅ | ❌ | |
| Parity | ✅ | ❌ | |
| Tribes | ✅ | ❌ | |
| AND/OR/XOR | ✅ | ❌ | |
| Random Functions | ✅ (basic) | ✅ (extensive) | BoolForge has many random generators |
| NCF (Nested Canalizing) | ❌ | ✅ | Gap in BoolFunc |

### Boolean Networks

| Feature | BoolFunc | BoolForge | CANA |
|---------|----------|-----------|------|
| Single Functions | ✅ | ✅ | ✅ |
| Networks | ❌ | ✅ | ✅ |
| Attractors | ❌ | ✅ | ✅ |
| State Transition Graph | ❌ | ✅ | ✅ |
| Control Analysis | ❌ | ❌ | ✅ |

---

## Unique Strengths

### BoolFunc
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

## Cross-Validation Opportunities

### BoolFunc ↔ BoolForge

```python
# Sensitivity/Activities comparison
import boolfunc as bf
from boolforge import BooleanFunction as BF_Forge

# BoolFunc
f_bf = bf.majority(5)
influences_bf = f_bf.influences()

# BoolForge  
f_forge = BF_Forge([int(f_bf.evaluate(x)) for x in range(32)])
activities_forge = f_forge.get_activities(EXACT=True)

# Should match (same concept, different names)
assert np.allclose(influences_bf, activities_forge)
```

### BoolFunc ↔ CANA

```python
# Effective degree comparison
import boolfunc as bf
import cana

# Our total influence ≈ CANA's effective degree
# (for balanced functions with unit-influence normalization)
```

---

## Recommended Use Cases

| Use Case | Best Library | Reason |
|----------|-------------|--------|
| Learning Boolean function theory | **BoolFunc** | Educational notebooks, direct API |
| Fourier/spectral analysis | **BoolFunc** | Only library with this focus |
| Canalization research | **BoolForge** + CANA | Specialized tools |
| Gene regulatory networks | **BoolForge** + CANA | Biology focus |
| SAT solving / BDD operations | **pyeda** | Mature BDD implementation |
| Formal verification | **pyeda** | SAT integration |
| Research paper reproduction | Depends on paper | Match the original |

---

## Integration Roadmap for BoolFunc

### Phase 1: Add Canalization (from BoolForge concepts)
- [ ] `is_canalizing()` - detect if any variable canalizes
- [ ] `is_k_canalizing(k)` - k-level nested canalization
- [ ] `get_canalizing_depth()` - canalizing layer depth
- [ ] `get_canalizing_variables()` - which variables canalize

### Phase 2: Add CANA-style Metrics
- [ ] `input_redundancy()` - fraction of redundant inputs
- [ ] `edge_effectiveness()` - per-variable effectiveness
- [ ] `effective_degree()` - sum of edge effectiveness
- [ ] `symmetry_groups()` - groups of interchangeable variables

### Phase 3: Interoperability
- [ ] `to_boolforge()` - export to BoolForge format
- [ ] `from_boolforge()` - import from BoolForge
- [ ] `to_cana()` - export to CANA BooleanNode
- [ ] `from_cana()` - import from CANA

### Phase 4: Cross-Validation Tests
- [ ] Compare influences vs activities (BoolForge)
- [ ] Compare total_influence vs average_sensitivity
- [ ] Verify monotonicity detection matches
- [ ] Compare sensitivity computations

---

## Installation Commands

```bash
# BoolFunc (pending PyPI)
pip install git+https://github.com/GabbyTab/boolfunc

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
