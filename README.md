<p align="center">
  <img src="logos/boo_horizontal.png" alt="BooFun Logo" width="800"/>
</p>

<p align="center">
  <strong>Boolean Function Analysis in Python</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/boofun/"><img src="https://img.shields.io/pypi/v/boofun.svg" alt="PyPI version"></a>
  <a href="https://github.com/GabbyTab/boofun/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/GabbyTab/boofun/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://gabbytab.github.io/boofun/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/GabbyTab/boofun"><img src="https://codecov.io/gh/GabbyTab/boofun/branch/main/graph/badge.svg" alt="codecov"></a>
</p>

## What This Is

A collection of tools for working with Boolean functions: representations, Fourier analysis, property testing, and complexity measures. I built this while studying O'Donnell's *Analysis of Boolean Functions* and wanted a unified toolkit that didn't exist.

**Intent:** Make the subject more approachable. If these tools save you time or help you understand something, that's the goal.

**Limitations:** This is a large codebase, partially AI-assisted. I've tested the core paths and verified mathematical properties where I could, but edge cases exist that I haven't found. If something breaks or gives wrong results, please report it.

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

# Analyze
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

# Quick summary
maj.analyze()  # dict with all metrics
```

## What's Included

### Representations
- Truth tables (dense, sparse, packed)
- Fourier expansion
- ANF (polynomial over GF(2))
- DNF/CNF
- Circuits, BDDs, LTFs

Automatic conversion between representations.

### Analysis
- **Fourier:** Walsh-Hadamard transform, spectral weight by degree
- **Influences:** Per-variable and total influence
- **Noise stability:** Stab_ρ[f] for ρ ∈ (0,1)
- **Property testing:** BLR linearity, junta, monotonicity, symmetry
- **Query complexity:** D(f), R(f), Q(f), sensitivity, block sensitivity, certificates

### Built-in Functions
`majority(n)`, `parity(n)`, `tribes(k, n)`, `threshold(n, k)`, `dictator(n, i)`, `AND(n)`, `OR(n)`, `weighted_majority(weights)`, `random(n)`

### Extras
- **Families:** Track asymptotic growth of function properties
- **Visualization:** Influence plots, Fourier spectra (requires matplotlib)
- **Quantum:** Grover speedup estimation, quantum walk analysis (theoretical; oracles require Qiskit)

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and test cases are especially valuable as they help verify correctness where I couldn't.

## Acknowledgments

Based on material from O'Donnell's *Analysis of Boolean Functions* and CS 294-92 (Spring 2025). Partially developed with AI assistance; design and verification are human-led.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{boofun2025,
  title={BooFun: A Python Library for Boolean Function Analysis},
  author={Gabriel Taboada},
  year={2025},
  url={https://github.com/GabbyTab/boofun}
}
```

<p align="center">
  <img src="logos/boo_alt.png" alt="BooFun Logo" width="200"/>
</p>
