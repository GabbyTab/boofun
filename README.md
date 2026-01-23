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

A toolkit for Boolean function analysis: Fourier analysis, property testing, query complexity, and more. Built while studying O'Donnell's *Analysis of Boolean Functions*.

**[Full Documentation](https://gabbytab.github.io/boofun/)** · **[Quick Start](https://gabbytab.github.io/boofun/quickstart.html)**

## Installation

```bash
pip install boofun
```

## Quick Start

```python
import boofun as bf

# Create functions
maj = bf.majority(5)
xor = bf.create([0, 1, 1, 0])

# Evaluate
maj.evaluate([1, 1, 0, 0, 1])  # → 1

# Fourier analysis
maj.fourier()           # Fourier coefficients
maj.influences()        # Per-variable influences
maj.total_influence()   # I[f]
maj.noise_stability(0.9)

# Properties and complexity
maj.is_monotone()
maj.is_balanced()

from boofun.analysis import complexity
complexity.decision_tree_depth(maj)  # D(f)
complexity.max_sensitivity(maj)      # s(f)

# Full analysis
maj.analyze()  # dict with all metrics
```

## Features

| Category | What's Included |
|----------|-----------------|
| **Representations** | Truth tables (dense/sparse/packed), Fourier, ANF, DNF/CNF, BDD, circuits, LTF |
| **Fourier Analysis** | WHT, influences, noise stability, spectral concentration, p-biased analysis |
| **Query Complexity** | D(f), R(f), Q(f), sensitivity, block sensitivity, certificates, Ambainis bound |
| **Property Testing** | BLR linearity, junta, monotonicity, symmetry, balance |
| **Hypercontractivity** | Noise operator, Bonami's Lemma, KKL theorem, Friedgut's junta theorem |
| **Cryptographic** | Nonlinearity, bent functions, Walsh spectrum, LAT/DDT, S-box analysis |
| **Visualization** | Influence plots, Fourier spectrum, truth table heatmaps, decision trees |

## Guides

Detailed documentation for each topic:

- **[Spectral Analysis](https://gabbytab.github.io/boofun/guides/spectral_analysis.html)** — Fourier, influences, p-biased, sensitivity, sampling
- **[Query Complexity](https://gabbytab.github.io/boofun/guides/query_complexity.html)** — D/R/Q, certificates, decision trees, Huang's theorem
- **[Hypercontractivity](https://gabbytab.github.io/boofun/guides/hypercontractivity.html)** — KKL, Bonami, Friedgut, global hypercontractivity
- **[Cryptographic Analysis](https://gabbytab.github.io/boofun/guides/cryptographic.html)** — Nonlinearity, bent, LAT/DDT, S-box
- **[Representations](https://gabbytab.github.io/boofun/guides/representations.html)** — All formats, conversion graph, storage hints

## Flexible Input

```python
bf.create([0, 1, 1, 0])              # List → truth table
bf.create(lambda x: x[0] ^ x[1], n=2) # Callable
bf.create("x0 and not x1", n=2)      # String → symbolic
bf.load("function.cnf")              # DIMACS CNF
```

## Built-in Functions

`majority(n)`, `parity(n)`, `tribes(k, n)`, `threshold(n, k)`, `AND(n)`, `OR(n)`, `dictator(n, i)`, `weighted_majority(weights)`, `random(n)`

## Examples

| File | Topic |
|------|-------|
| `01_getting_started.py` | Basics |
| `02_fourier_basics.py` | WHT, Parseval |
| `03_common_families.py` | Majority, Parity, Tribes |
| `04_property_testing.py` | BLR, junta tests |
| `05_query_complexity.py` | Sensitivity, certificates |

Plus 18 Jupyter notebooks aligned with O'Donnell's course in `notebooks/`.

## Performance

- NumPy vectorization throughout
- Optional Numba JIT, CuPy GPU acceleration
- Sparse/packed representations for large n
- Most operations complete in milliseconds for n ≤ 14

## Testing

```bash
pytest tests/
pytest --cov=boofun tests/
```

3000+ tests with 71% coverage. Cross-validation against known results in `tests/test_cross_validation.py`.

## Convention

O'Donnell standard: Boolean 0 → +1, Boolean 1 → −1. This ensures f̂(∅) = E[f].

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and test cases are especially valuable.

## Acknowledgments

- **Scott Aaronson's Boolean Function Wizard** (2000) — Query complexity foundations
- **Avishay Tal** — Sensitivity, p-biased analysis, decision tree algorithms
- **O'Donnell's *Analysis of Boolean Functions*** (Cambridge, 2014)

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
