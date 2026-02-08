# BooFun: Computational Companion for Boolean Function Analysis Courses

*For instructors teaching from O'Donnell's "Analysis of Boolean Functions" or related material.*

---

## What It Is

BooFun is a Python library with 23 interactive Jupyter notebooks that let students **run every major theorem computationally**. It was built during UC Berkeley's CS 294-92 (Analysis of Boolean Functions, Spring 2025) and covers Chapters 1-11 of O'Donnell's textbook.

## What Students Get

- **11 lecture notebooks** aligned to O'Donnell's chapters, with Colab links (zero install)
- **4 homework notebooks** with computational exercises
- **8 special topic notebooks** covering global hypercontractivity, pseudorandomness, cryptographic analysis, and more
- A Python library where `bf.majority(5).fourier()` replaces pages of NumPy boilerplate

## Coverage

| O'Donnell Chapter | Notebook | Key Computations |
|---|---|---|
| 1-2: Fourier expansion | lecture1, lecture2, hw1 | WHT, Parseval verification, BLR linearity test |
| 3: Social choice | lecture3 | Influences, Arrow's theorem, voting power |
| 4: Influences & effects | lecture4, hw2 | KKL theorem, Friedgut's junta theorem |
| 5: Noise stability | lecture5 | Noise operator, Sheppard's formula |
| 6: Pseudorandomness | fractional_prg | Fourier tails, spectral concentration |
| 7-8: Learning | lecture7, lecture8 | Goldreich-Levin, junta learning |
| 9: Hypercontractivity | hw4, global_hypercontractivity | Bonami's lemma, KKL, global hypercontractivity (Keevash et al.) |
| 10-11: Invariance | lecture11 | Berry-Esseen, Majority is Stablest, Gaussian analysis |

## How to Use It

**Option 1: Colab (no install)**
Share notebook links with students. Each notebook has a "Open in Colab" badge. Students click and run.

**Option 2: Local install**
```bash
pip install boofun
```

**Option 3: Docker**
```bash
docker-compose up notebook  # JupyterLab at localhost:8888
```

## What Makes It Different

- **Curriculum-aligned**: Notebooks follow O'Donnell's chapter structure, not arbitrary topics
- **Correct**: 3200+ tests cross-validated against known results, Tal's BooleanFunc.py, SageMath, and thomasarmel/boolean_function
- **Maintained**: Active development, PyPI published, CI/CD, Sphinx documentation
- **Extensible**: Students can create custom functions with `bf.create(lambda x: ..., n=10)` and analyze them with the same tools

## Links

- **GitHub**: [github.com/GabbyTab/boofun](https://github.com/GabbyTab/boofun)
- **Documentation**: [gabbytab.github.io/boofun](https://gabbytab.github.io/boofun)
- **PyPI**: [pypi.org/project/boofun](https://pypi.org/project/boofun)

## Contact

Gabriel Taboada -- gabtab@berkeley.edu
Built with guidance from [Avishay Tal](https://www2.eecs.berkeley.edu/Faculty/Homepages/atal.html) (UC Berkeley EECS).
