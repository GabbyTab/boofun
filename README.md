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

Or run notebooks in Docker:
```bash
docker-compose up notebook  # localhost:8888, token: boofun
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
| **Built-in Functions** | Majority, Parity, AND, OR, Tribes, Threshold, Dictator, weighted LTF, random |
| **Representations** | Truth tables (dense/sparse/packed), Fourier, ANF, DNF/CNF, BDD, circuits, LTF |
| **Fourier Analysis** | WHT, influences, noise stability, spectral concentration, p-biased analysis |
| **Query Complexity** | D(f), R(f), Q(f), sensitivity, block sensitivity, certificates, Ambainis bound |
| **Property Testing** | BLR linearity, junta, monotonicity, symmetry, balance |
| **Hypercontractivity** | Noise operator, Bonami's Lemma, KKL theorem, Friedgut's junta theorem |
| **Learning Theory** | Goldreich-Levin, PAC learning, junta learning, LMN algorithm |
| **Cryptographic** | Nonlinearity, bent functions, Walsh spectrum, LAT/DDT, S-box analysis |
| **Advanced** | Gaussian analysis, invariance principle, communication complexity, LTF analysis |
| **Visualization** | Influence plots, Fourier spectrum, truth table heatmaps, decision trees |

## Guides

Detailed documentation for each topic:

- **[Spectral Analysis](https://gabbytab.github.io/boofun/guides/spectral_analysis.html)**: Fourier, influences, p-biased, sensitivity, sampling
- **[Query Complexity](https://gabbytab.github.io/boofun/guides/query_complexity.html)**: D/R/Q, certificates, decision trees, Huang's theorem
- **[Hypercontractivity](https://gabbytab.github.io/boofun/guides/hypercontractivity.html)**: KKL, Bonami, Friedgut, global hypercontractivity
- **[Learning Theory](https://gabbytab.github.io/boofun/guides/learning.html)**: Goldreich-Levin, PAC learning, junta learning, LMN
- **[Cryptographic Analysis](https://gabbytab.github.io/boofun/guides/cryptographic.html)**: Nonlinearity, bent, LAT/DDT, S-box
- **[Representations](https://gabbytab.github.io/boofun/guides/representations.html)**: All formats, conversion graph, storage hints
- **[Operations](https://gabbytab.github.io/boofun/guides/operations.html)**: Boolean operators, composition, restriction, permutation
- **[Advanced Topics](https://gabbytab.github.io/boofun/guides/advanced.html)**: Gaussian, invariance, communication complexity, LTF, restrictions

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

## Course Notebooks

Interactive notebooks following CS294-92 (Analysis of Boolean Functions). Click **Topic** to view, **Notes** for lecture notes, or **Play** to run in Colab.

<details>
<summary><strong>Lecture Notebooks (11)</strong></summary>

| Lecture | Topic | Notes | Play |
|---------|-------|-------|------|
| 1 | [Fourier Expansion](https://gabbytab.github.io/boofun/notebooks/lecture1_fourier_expansion.html) | [Joyce Lu](https://drive.google.com/file/d/1hL9brxvBAfJf8TlDf6v1OthrUamwmMAF/view?usp=share_link) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture1_fourier_expansion.ipynb) |
| 2 | [Linearity Testing](https://gabbytab.github.io/boofun/notebooks/lecture2_linearity_testing.html) | [Austin Pechan](https://drive.google.com/file/d/1ik5EzybeJiOAaS7R7tjV9R6AFgeKcCf-/view?usp=share_link) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture2_linearity_testing.ipynb) |
| 3 | [Social Choice & Influences](https://gabbytab.github.io/boofun/notebooks/lecture3_social_choice_influences.html) | [Vivien Goyal](https://drive.google.com/file/d/1-NYv-8OeHUuU7v8NPMi6kpw5Baszw0Ss/view?usp=share_link) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture3_social_choice_influences.ipynb) |
| 4 | [Influences & Effects](https://gabbytab.github.io/boofun/notebooks/lecture4_influences_effects.html) | [Patrick Bales](https://drive.google.com/file/d/1YYUHu6tz7eJghUzQIqanbGSZRQCEi015/view) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture4_influences_effects.ipynb) |
| 5 | [Noise Stability](https://gabbytab.github.io/boofun/notebooks/lecture5_noise_stability.html) | [Prastik Mohanraj](https://drive.google.com/file/d/1PIRoYDpXRd1F48fsyutyBFR_URrMMWnh/view?usp=sharing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture5_noise_stability.ipynb) |
| 6 | [Spectral Concentration](https://gabbytab.github.io/boofun/notebooks/lecture6_spectral_concentration.html) | *(upcoming)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture6_spectral_concentration.ipynb) |
| 7 | [Goldreich-Levin](https://gabbytab.github.io/boofun/notebooks/lecture7_goldreich_levin.html) | [Thomas Culhane](https://drive.google.com/file/d/1aGxilyDayIcPMB6TjfqmswhHqzFUj1ac/view?usp=sharing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture7_goldreich_levin.ipynb) |
| 8 | [Learning Juntas](https://gabbytab.github.io/boofun/notebooks/lecture8_learning_juntas.html) | [Timothe Kasriel](https://drive.google.com/file/d/1yaEXw3GDMCUrRhR8aODDjQaSl-djNae6/view?usp=sharing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture8_learning_juntas.ipynb) |
| 9 | [DNFs & Restrictions](https://gabbytab.github.io/boofun/notebooks/lecture9_dnf_restrictions.html) | [Qinggao Hong](https://drive.google.com/file/d/1pf6IS695K3EvytBzw58baHxgA0rfWhew/view?usp=sharing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture9_dnf_restrictions.ipynb) |
| 10 | [Fourier Concentration](https://gabbytab.github.io/boofun/notebooks/lecture10_fourier_concentration.html) | [Lucas Ericsson](https://drive.google.com/file/d/1pEuaXd0JoiD_S4b6OLJgdsKyYlrANie5/view?usp=share_link) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture10_fourier_concentration.ipynb) |
| 11 | [Invariance Principle](https://gabbytab.github.io/boofun/notebooks/lecture11_invariance_principle.html) | [Thapen](http://users.math.cas.cz/~thapen/switching.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/lecture11_invariance_principle.ipynb) |

</details>

<details>
<summary><strong>Homework Notebooks (4)</strong></summary>

| HW | Topic | Play |
|----|-------|------|
| 1 | [Fourier Expansion](https://gabbytab.github.io/boofun/notebooks/hw1_fourier_expansion.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/hw1_fourier_expansion.ipynb) |
| 2 | [LTFs & Decision Trees](https://gabbytab.github.io/boofun/notebooks/hw2_ltf_decision_trees.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/hw2_ltf_decision_trees.ipynb) |
| 3 | [DNFs & Restrictions](https://gabbytab.github.io/boofun/notebooks/hw3_dnf_restrictions.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/hw3_dnf_restrictions.ipynb) |
| 4 | [Hypercontractivity](https://gabbytab.github.io/boofun/notebooks/hw4_hypercontractivity.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabbyTab/boofun/blob/main/notebooks/hw4_hypercontractivity.ipynb) |

</details>

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

3000+ tests with 72% coverage. Cross-validation against known results in `tests/test_cross_validation.py`.

## Convention

O'Donnell standard: Boolean 0 → +1, Boolean 1 → −1. This ensures f̂(∅) = E[f].

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and test cases are especially valuable.

## Acknowledgments

- **[Avishay Tal](https://www2.eecs.berkeley.edu/Faculty/Homepages/atal.html)**: Course instructor, sensitivity analysis, p-biased measures, decision tree algorithms
- **[Patrick Bales](https://www.linkedin.com/in/patrickbbales/)**: Course materials and notebook review
- **O'Donnell's *Analysis of Boolean Functions*** (Cambridge, 2014): Theoretical foundation
- **[Scott Aaronson](https://scottaaronson.blog/)'s Boolean Function Wizard** (2000): Query complexity foundations

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
