<p align="center">
  <img src="logos/boo_horizontal.png" alt="BooFun Logo" width="800"/>
</p>

<p align="center">
  <strong>Boolean Function Analysis in Python</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/boofun/"><img src="https://img.shields.io/pypi/v/boofun.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/boofun/"><img src="https://img.shields.io/pypi/dm/boofun" alt="PyPI Downloads"></a>
  <a href="https://github.com/GabbyTab/boofun/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/GabbyTab/boofun/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://gabbytab.github.io/boofun/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/GabbyTab/boofun"><img src="https://codecov.io/gh/GabbyTab/boofun/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/GabbyTab/boofun"><img src="https://img.shields.io/badge/typed-mypy-blue.svg" alt="Typed"></a>
  <a href="https://securityscorecards.dev/viewer/?uri=github.com/GabbyTab/boofun"><img src="https://api.securityscorecards.dev/projects/github.com/GabbyTab/boofun/badge" alt="OpenSSF Scorecard"></a>
</p>

## What This Is

A collection of tools for working with Boolean functions: representations, Fourier analysis, property testing, and complexity measures. I built this while studying O'Donnell's *Analysis of Boolean Functions* and wanted a unified toolkit that didn't exist.

**Intent:** Make the subject more approachable. If these tools save you time or help you understand something, that's the goal.

**Limitations:** This is a large codebase, partially AI-assisted. I've tested the core paths and verified mathematical properties where I could, but edge cases exist that I haven't found. If something breaks or gives wrong results, please report it.

**[Documentation](https://gabbytab.github.io/boofun/)**

## Installation

```bash
pip install boofun
```

Development:
```bash
git clone https://github.com/GabbyTab/boofun.git
cd boofun && pip install -e ".[dev]"
```

## Usage

```python
import boofun as bf

# Create functions
xor = bf.create([0, 1, 1, 0])    # From truth table
maj = bf.majority(5)             # Built-in
f = bf.random(4, seed=42)        # Random

# Evaluate
maj.evaluate([1, 1, 0, 0, 1])    # → 1

# Representations (automatic conversion)
maj.get_representation("fourier_expansion")
maj.get_representation("anf")    # Polynomial over GF(2)
maj.convert_to("dnf")

# Fourier analysis
maj.fourier()                    # Fourier coefficients
maj.influences()                 # Variable influences
maj.total_influence()            # I[f]
maj.noise_stability(0.9)         # Stab_ρ[f]
maj.degree()                     # Fourier degree

# Properties
maj.is_balanced()
maj.is_monotone()
maj.is_linear()
maj.is_junta(k=2)

# Query complexity
from boofun.analysis import complexity, query_complexity as qc
complexity.decision_tree_depth(maj)     # D(f)
complexity.max_sensitivity(maj)         # s(f)
qc.ambainis_complexity(maj)             # Quantum lower bound

# Quick summary
maj.analyze()  # dict with all metrics
```

## Flexible Input

Pass data in whatever form you have—BooFun infers the representation:

```python
bf.create([0, 1, 1, 0])                    # List → truth table
bf.create(np.array([0, 1, 1, 0]))          # NumPy array → truth table
bf.create(lambda x: x[0] ^ x[1], n=2)      # Callable → function
bf.create("x0 and not x1", n=2)            # String → symbolic
bf.create({frozenset([0]): 1}, n=2)        # Dict → polynomial
bf.create({(0,1), (1,0)}, n=2)             # Set → true inputs
bf.create(iter([0,1,1,0]))                 # Iterator → streaming
```

Also accepts scipy distributions, DNF/CNF formula objects, file paths, and adapts legacy code via `LegacyAdapter`.

```python
# Load from file (format auto-detected)
f = bf.load("function.json")
f = bf.load("function.bf")      # Aaronson format
f = bf.load("function.cnf")     # DIMACS CNF

# Or directly via create
f = bf.create("function.json")

# Save to file
bf.save(f, "output.json")
bf.save(f, "output.bf")
```

Evaluation is equally flexible:

```python
f.evaluate(3)                   # Integer index (binary: 011)
f.evaluate([0, 1, 1])           # List of bits
f.evaluate((0, 1, 1))           # Tuple
f.evaluate(np.array([0, 1, 1])) # NumPy array
f.evaluate([[0,0], [0,1], [1,0], [1,1]])  # Batch
```

## What's Included

### Representations
- Truth tables (dense, sparse, packed)
- Fourier expansion
- ANF (polynomial over GF(2))
- Real polynomial
- DNF/CNF
- Circuits, BDDs, LTFs
- Symbolic

Automatic conversion between any pair via conversion graph.

### Analysis
- **Fourier:** Walsh-Hadamard transform, spectral weight by degree, p-biased Fourier, spectral concentration
- **Influences:** Per-variable, total, average sensitivity
- **Noise stability:** Stab_ρ[f] for ρ ∈ [-1,1]
- **Property testing:** BLR linearity, junta, monotonicity, unateness, symmetry, quasisymmetry, balance, dictator, affine, constant
- **Query complexity:** D(f), D_avg(f), R₀(f), R₁(f), R₂(f), Q₂(f), QE(f), s(f), bs(f), es(f), C(f), Ambainis bound, spectral adversary, polynomial method
- **Structural:** Primality, dependence, decomposition
- **Advanced:** FKN theorem, hypercontractivity, Gaussian analysis, invariance principle, PAC learning, Goldreich-Levin, LTF analysis (Chow parameters, critical index), communication complexity, Huang's theorem

### Built-in Functions
`majority(n)`, `parity(n)`, `tribes(k, n)`, `threshold(n, k)`, `dictator(n, i)`, `AND(n)`, `OR(n)`, `weighted_majority(weights)`, `random(n)`

### Families
Track asymptotic behavior as n grows. Built-in: `MajorityFamily`, `ParityFamily`, `TribesFamily`, `ThresholdFamily`, `ANDFamily`, `ORFamily`, `DictatorFamily`, `LTFFamily`. Compare empirical results against theoretical bounds (e.g., I[MAJ_n] ≈ √(2/π)·√n).

### Visualization
- Influence bar plots, Fourier spectrum (box plots, spectral concentration)
- Truth table heatmaps, hypercube graphs (n ≤ 5)
- Noise stability curves, sensitivity heatmaps
- Decision tree visualization, growth plots
- Interactive dashboards (matplotlib/plotly)
- Animations for asymptotic growth

### Quantum
Grover speedup estimation, quantum walk analysis (theoretical bounds; oracle implementation requires Qiskit).

## Mathematical Convention

We follow O'Donnell's convention:
- Boolean 0 -> +1, Boolean 1 -> -1
- f̂(∅) = E[f] in the ±1 domain
- All formulas match the textbook

## Examples

`examples/` contains tutorials:
| File | Topic |
|------|-------|
| `01_getting_started.py` | Basics |
| `02_fourier_basics.py` | WHT, Parseval |
| `03_common_families.py` | Majority, Parity, Tribes |
| `04_property_testing.py` | BLR, junta tests |
| `05_query_complexity.py` | Sensitivity, certificates |
| `06_noise_stability.py` | Influences, voting |
| `07_quantum_applications.py` | Grover, quantum walks |

`notebooks/` has 18 Jupyter notebooks aligned with O'Donnell's course.

## Performance

- NumPy vectorization throughout
- Optional Numba JIT for WHT, influences
- Optional GPU via CuPy
- Sparse representations for large n

For n ≤ 14, most operations complete in milliseconds. For n > 18, consider sparse representations or GPU.

## Testing

```bash
pytest tests/
pytest --cov=boofun tests/
```

Test coverage is incomplete. Cross-validation against known results is in `tests/test_cross_validation.py`.

## API Stability

As of v1.0, the public API is stable:
- `bf.create()`, `bf.load()`, `bf.save()`, built-in functions (`bf.majority`, `bf.parity`, etc.)
- `f.evaluate()`, `f.fourier()`, `f.influences()`, `f.analyze()`
- `f.get_representation()`, `f.is_*()` property methods
- Analysis classes: `SpectralAnalyzer`, `PropertyTester`, `QueryComplexityProfile`

Breaking changes will increment the major version. Internal modules (`_*`, implementation details) may change between minor versions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and test cases are especially valuable as they help verify correctness where I couldn't.

## Acknowledgments

**Prior Art:**
- **Scott Aaronson's Boolean Function Wizard** (2000): The query complexity module draws on Aaronson's "Algorithms for Boolean Function Query Measures," which implemented D(f), R(f), Q(f), sensitivity, block sensitivity, and certificate complexity. His work established the algorithmic foundations we build on.
- **Avishay Tal's Boolean function library** (PhD-era): Professor Tal generously shared his Python library, which included Fourier transforms, sensitivity calculations, decision tree algorithms, and polynomial representations. His code informed several design decisions.

**Theoretical Foundation:**
- O'Donnell's *Analysis of Boolean Functions* (Cambridge, 2014)
- CS 294-92 at UC Berkeley (Spring 2025)

Partially developed with AI assistance; design and verification are human-led.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{boofun2026,
  title={BooFun: A Python Library for Boolean Function Analysis},
  author={Gabriel Taboada},
  year={2026},
  url={https://github.com/GabbyTab/boofun}
}
```

<p align="center">
  <img src="logos/boo_alt.png" alt="BooFun Logo" width="200"/>
</p>
