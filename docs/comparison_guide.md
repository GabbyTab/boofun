# Library Comparison

How BooFun compares to other Boolean function libraries.

## Summary

BooFun focuses on theoretical computer science: Fourier analysis (O'Donnell style), property testing, query complexity. Other libraries have different strengths.

| Library | Focus | Fourier | Property Testing | Query Complexity |
|---------|-------|---------|------------------|------------------|
| BooFun | TCS theory | TCS + cryptographic | ✓ | ✓ |
| SageMath | Computer algebra / cryptography | Walsh / cryptographic | ✗ | ✗ |
| VBF (C++) | Cryptography (vectorial) | Walsh + autocorrelation | ✗ | ✗ |
| sboxU | S-box cryptanalysis | Walsh / cryptographic | ✗ | ✗ |
| boolfun (R) | Cryptography (scalar) | Walsh / cryptographic | ✗ | ✗ |
| fbool | TCS-adjacent (Rust + Python, young) | Core Fourier metrics | ✗ | sensitivity + certificates |
| CircuitGraph | Boolean circuits | influences only¹ | ✗ | ✗ |
| pyeda | Logic/SAT/BDD | ✗ | ✗ | ✗ |
| BoolForge | Boolean networks / biology | activities only¹ | ✗ | ✗ |
| CANA | Canalization, dynamics, control | activities only¹ | ✗ | ✗ |

¹ Influence/activity is a Fourier-analytic quantity, but these packages do not
expose BooFun's broader Fourier suite. BooFun cross-validates BoolForge's exact
activities and average sensitivity in CI.

## What BooFun Has

**Query Complexity** (based on Aaronson's Boolean Function Wizard):
- Deterministic: D(f), D_avg(f)
- Randomized: R₀(f), R₁(f), R₂(f), nondeterministic variants
- Quantum: Q₂(f), QE(f), nondeterministic variants
- Sensitivity: s(f), bs(f), es(f) (everywhere sensitivity)
- Certificates: C(f), C₀(f), C₁(f)
- Exploratory quantum-complexity estimates: Ambainis, spectral adversary,
  polynomial method, and a combined general-adversary heuristic. These are
  not certified lower bounds or SDP solutions; correction is tracked in
  [#119](https://github.com/GabbyTab/boofun/issues/119).
- Degree measures: exact, approximate, threshold, nondeterministic
- Decision tree algorithms: DP optimal depth, tree enumeration, randomized complexity

**Property Testing:**
- BLR linearity
- Exact k-junta recognition (not a query-limited tester)
- Monotonicity, unateness, symmetry

**Fourier Analysis:**
- Influences, total influence
- Noise stability
- Spectral weight by degree
- KKL theorem bounds
- p-biased Fourier analysis
- Annealed influence, truncation, correlation

**Sensitivity Analysis:**
- Sensitivity moments and histograms
- p-biased sensitivity
- Pointwise sensitivity, sensitive coordinates
- arg_max/arg_min sensitivity

**Hypercontractivity (v1.1):**
- Noise operator T_ρ, L_q norms
- Bonami's Lemma, hypercontractive inequality
- KKL theorem, Friedgut's junta theorem
- Level-d inequality

**Global Hypercontractivity (v1.1):**
- GlobalHypercontractivityAnalyzer
- α-global function detection
- Generalized influence under μ_p
- Threshold curves, critical p

**Cryptographic Analysis (v1.1):**
- Nonlinearity, bent function detection
- Walsh transform and spectrum
- Algebraic Normal Form, algebraic degree
- Correlation immunity, resiliency
- Strict Avalanche Criterion (SAC)
- Linear Approximation Table (LAT)
- Difference Distribution Table (DDT)
- S-box analyzer

**Quantum Complexity Bounds** (experimental playground — classical computation of quantum query estimates):
- Grover complexity bounds (closed-form formulas)
- Quantum walk complexity bounds (analytical)
- Element distinctness analysis
- *Actual quantum simulation planned for v2.0.0*

## What BooFun Lacks

Features better served by other libraries:
- SAT solving, advanced BDD operations → pyeda
- Boolean networks, attractors → BoolForge, biobalm
- Logical redundancy, automata-network dynamics and control → CANA
- Canalizing layer structure → BoolForge

Note: As of v1.1, BooFun includes canalization analysis (depth, nested canalizing detection, essential variables) and cryptographic analysis (bent functions, nonlinearity, correlation immunity, LAT/DDT).

## BoolForge Comparison (Systems Biology)

BoolForge (Kadelka & Coberly, 2025) focuses on Boolean **networks** for systems biology, while BooFun focuses on Boolean **functions** for theoretical CS.

### What BoolForge Does Well

**Random Generation with Constraints:**
```python
# BoolForge can generate functions with specific properties
random_k_canalizing_function(n, k)  # Specific canalizing depth
random_NCF(n, layer_structure)       # Nested canalizing with structure
random_non_degenerated_function(n, bias)  # Specific bias
```

**Boolean Networks:**
- Networks of interconnected Boolean functions
- Attractor analysis (steady states, limit cycles)
- Network robustness metrics
- Modular structure detection

**Null Model Generation:**
- Generate ensembles for statistical comparison
- Control for degree distribution, canalization, bias

### Feature Comparison

| Feature | BooFun | BoolForge |
|---------|--------|-----------|
| **Canalization** | | |
| is_canalizing | ✓ | ✓ |
| canalizing_depth | ✓ | ✓ |
| is_nested_canalizing | ✓ | ✓ |
| get_layer_structure | ✗ | ✓ |
| canalizing_strength | ✗ | ✓ |
| **Random Generation** | | |
| Random k-canalizing | ✗ | ✓ |
| Random with bias | ✗ | ✓ |
| Random layer structure | ✗ | ✓ |
| **Analysis** | | |
| Monotonicity | ✓ | ✓ |
| Symmetry groups | ✓ | ✓ |
| Sensitivity | ✓ | ✓ |
| Influences (exact) | ✓ | ✓ (activities)² |
| Average sensitivity / total influence | ✓ | ✓² |
| Essential variables | ✓ | ✓ |
| **Networks** | | |
| Network representation | ✗ | ✓ |
| Attractor analysis | ✗ | ✓ |
| Network motifs | ✗ | ✓ |
| **Unique to BooFun** | | |
| Full Fourier spectrum | ✓ | ✗ |
| Noise stability | ✓ | ✗ |
| Query complexity | ✓ | ✗ |
| Property testing | ✓ | ✗ |
| Hypercontractivity | ✓ | ✗ |
| Cryptographic analysis | ✓ | ✗ |

² Cross-validated live in CI: BooFun's influences and total influence are
compared against BoolForge's exact activities and average sensitivity at
atol 1e-10 (`tests/cross_validation/test_boolforge.py`).

### When to Use Which

**Use BoolForge when:**
- Modeling gene regulatory networks
- Need to generate ensembles with specific canalization properties
- Studying network dynamics and attractors
- Comparing biological networks to null models

**Use BooFun when:**
- Studying theoretical properties (Fourier, query complexity)
- Following O'Donnell's textbook
- Property testing algorithms
- Cryptographic analysis of Boolean functions

## Comparison Tables

### Fourier Analysis

| Feature | BooFun | SageMath |
|---------|--------|----------|
| Walsh-Hadamard | ✓ | ✓ |
| Influences | ✓ | ✗ |
| Total influence | ✓ | ✗ |
| Noise stability | ✓ | ✗ |
| Bent functions | ✓ | ✓ |
| Correlation immunity | ✓ | ✓ |
| Hypercontractivity | ✓ | ✗ |
| p-biased analysis | ✓ | ✗ |

BooFun now covers both O'Donnell-style analysis and cryptographic properties.

### Property Testing

| Test | BooFun | BoolForge |
|------|--------|-----------|
| Linearity (BLR) | ✓ | ✗ |
| k-junta | ✓ (exact recognition) | ✗ |
| Monotonicity | ✓ (exact check + probabilistic tester)¹ | ✓ (exact) |
| Dictator proximity | ✓ | ✗ |

¹ The exact check (`analysis.basic_properties.is_monotone`, with `is_unate`
for unateness) is validated by an exhaustive census against published OEIS
counts (168 monotone and 2,170 unate four-variable functions); see the
[claim matrix](cross_validation.md).

### Representations

| Format | BooFun | pyeda |
|--------|--------|-------|
| Truth table | ✓ | ✓ |
| BDD | ✓ (basic) | ✓ (full ROBDD) |
| CNF/DNF | ✓ | ✓ |
| Fourier | ✓ | ✗ |

pyeda's BDD implementation is more mature.

## When to Use What

**BooFun:**
- Studying Boolean function theory (O'Donnell book)
- Query complexity research
- Property testing algorithms
- Influence/noise stability analysis
- Hypercontractivity and threshold phenomena
- Cryptographic analysis (nonlinearity, bent, LAT/DDT, S-box)

**SageMath:**
- Deeper algebraic cryptanalysis
- Finite field computations

**sboxU / VBF:**
- Specialized vectorial Boolean-function and S-box cryptanalysis
- APN / almost-bent classification, CCZ/EA equivalence (sboxU)

**fbool:**
- Rust-native analysis and Python bindings
- Entropy, fragmentation, and exact small-circuit data

**pyeda:**
- SAT solving
- BDD manipulation
- Logic minimization

**BoolForge:**
- Gene regulatory networks
- Canalization

**CANA:**
- Logical redundancy and input symmetry
- Automata-network dynamics and control

## Cross-Validation

BooFun's agreement with the libraries on this page is not asserted, it is
tested — every claim has an executable test against a pinned reference
version:

- **SageMath 10.9**: a committed fixture corpus of 303 functions (standard
  families plus all AES S-box components) generated inside a pinned Docker
  image; Walsh spectra, nonlinearity, algebraic degree, balancedness, and
  correlation immunity are compared exactly.
- **BoolForge 1.0.1**: a live CI job (on every push to main and weekly)
  installs a commit-pinned BoolForge and compares canalization, monotonicity,
  symmetry, influences, and average sensitivity.
- **Published values**: AES and PRESENT S-box properties, bent function
  spectra, and counting results are asserted against their literature
  citations.
- **Closed forms and internal consistency**: theoretical bounds (Huang,
  Nisan–Szegedy) and independent computation paths (FWHT vs direct
  correlation sums) are checked against each other.

See the [claim matrix](cross_validation.md) for the full list of validated
claims, reference versions, function families, and tolerances.

## Installation

```bash
pip install boofun      # BooFun (PyPI)
pip install boolforge   # BoolForge (on PyPI since v1.0)
pip install fbool       # fbool (Python 3.11+)
pip install sboxU       # sboxU
pip install cana        # CANA
pip install pyeda       # pyeda
```

VBF is a C++/NTL library built from [source](https://github.com/jacubero/VBF).
The older R package boolfun is archived on
[CRAN](https://cran.r-project.org/src/contrib/Archive/boolfun/).

## Prior Art

BooFun's query complexity module builds on:
- **Scott Aaronson's Boolean Function Wizard** (2000): C implementation of D(f), R(f), Q(f), sensitivity, block sensitivity, certificate complexity, approximate degrees. See Aaronson, "Algorithms for Boolean Function Query Measures."
- **Avishay Tal's library**: Python implementation of Fourier transforms, sensitivity, decision trees, polynomial representations over F₂ and reals.

These tools inspired BooFun's design but were either no longer maintained or not publicly distributed. BooFun aims to provide a modern, documented, tested implementation of these ideas.

Direct and materially overlapping software:

- **boolfun** (Lafitte, Van Heule & Van Hamme, R Journal 2011): peer-reviewed
  R package for scalar cryptographic Boolean functions, including truth
  tables, Walsh spectra, ANF, algebraic degree and immunity, nonlinearity,
  correlation immunity, and resiliency. Its last CRAN release was archived
  in 2012.
- **VBF** (Álvarez-Cubero & Zufiria, ACM TOMS 2016): peer-reviewed C++/NTL
  library for *vectorial* Boolean functions in cryptography, with Walsh and
  autocorrelation spectra, nonlinearity, algebraic degree, linear structures,
  and vectorial operations. It is a major cryptographic predecessor; its
  documented API does not provide BooFun's TCS-oriented influence, noise,
  property-testing, or query-complexity suite.
- **fbool** v0.2.0 (2026): young Rust library with Python bindings and one of
  the closest modern Fourier-analytic neighbors. It overlaps on influence,
  sensitivity, Walsh/Fourier metrics, Fourier degree, nonlinearity, and
  certificate complexity, and adds entropy, fragmentation, and exact
  five-variable circuit data. It is not peer reviewed.
- **[py-aiger-spectral](https://github.com/mvcisback/py-aiger-spectral)**: a
  small AIGER-based Python package for Fourier coefficients, degree weights,
  mean, variance, and covariance. It has no tagged release or paper, but is
  relevant direct software prior art.
- **[Boolan](https://github.com/JellePiepenbrock/Boolan)**: a small,
  unmaintained Python package computing influences, degree-weight profiles,
  variance, and noise sensitivity from the Fourier expansion. Minimal, but
  direct prior art for BooFun's core Fourier quantities.
- **[CircuitGraph](https://doi.org/10.21105/joss.02646)** (JOSS 2020):
  Boolean-circuit manipulation with exact or approximate model-counting
  routines for per-input influence and average sensitivity. This is material
  partial overlap, not merely a different object.
- **[QuantumQueryOptimizer](https://doi.org/10.4230/LIPIcs.ESA.2023.36)**
  (ESA 2023): solves general-adversary semidefinite programs and constructs
  query-optimal quantum algorithms. BooFun's `general_adversary_bound` is a
  combined heuristic estimate, not an SDP implementation or certified lower
  bound ([#119](https://github.com/GabbyTab/boofun/issues/119)).

Adjacent ecosystems:

- **[BooLSPLG](https://doi.org/10.3390/math11081864)** (Bikov, Bouyukliev &
  Dzhumalieva-Stoeva, *Mathematics* 2023) is a peer-reviewed CUDA C/C++
  library computing Walsh and autocorrelation spectra, nonlinearity,
  algebraic degree/ANF, and LAT/DDT for Boolean functions and S-boxes up to
  n = 20 on GPUs. **[BoolCrypt](https://github.com/ranea/BoolCrypt)** (2022)
  is a Sage-based library for vectorial Boolean functions focused on
  affine/CCZ equivalence via SAT solvers, and **PEIGEN** (IACR ToSC 2019)
  evaluates and generates S-boxes. All are cryptographic specialists
  complementary to BooFun's TCS suite.
- Transform kernels such as **[pyfwht](https://pypi.org/project/pyfwht/)**
  (CPU/OpenMP/CUDA fast Walsh–Hadamard transforms — an optional BooFun
  dependency) and Julia's Hadamard.jl provide the low-level transform
  without function-analysis semantics.
- **BoolForge** and **CANA** focus on canalization and Boolean/automata
  networks. CANA's prime-implicant redundancy and schema symmetry are not
  interchangeable with BooFun's classic canalizing depth and variable
  symmetry; function-level activities and sensitivity do overlap.
- **[Biddy](https://doi.org/10.21105/joss.01189)** (JOSS 2019) represents and
  manipulates Boolean functions through several BDD families, while
  **[sboxgates](https://doi.org/10.21105/joss.02946)** (JOSS 2021) synthesizes
  low-gate circuits for S-boxes. Biological-network applications such as
  PyDrugLogics, emba, and NORDic are farther from BooFun's function-level
  scope. SPbLA operates on sparse matrices over the Boolean semiring and is
  terminologically, rather than functionally, adjacent.

A survey of JOSS papers, indexed metadata, and package registries through
**26 July 2026** found no package with BooFun's *combined* focus on
Fourier-analytic Boolean-function measures, property testing, and query
complexity. CircuitGraph is the clearest published partial overlap, and
fbool the closest unpublished one. This is a dated, scoped search
result—not an exhaustive claim that no other overlapping software exists.
The exact sources, query strings, verification protocol, inclusion
criteria, and the full candidate ledger (including rejected candidates) are
documented in the reproducible survey log:
[Prior-art survey: method and ledger](prior_art_survey.md).

## References

- Aaronson, S. (2000). "Algorithms for Boolean Function Query Measures."
- O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge.
- Buhrman, H. & de Wolf, R. (2002). "Complexity Measures and Decision Tree Complexity."
- Lafitte, F., Van Heule, D. & Van Hamme, J. (2011). "Cryptographic Boolean
  Functions with R." *The R Journal*. <https://doi.org/10.32614/RJ-2011-007>
- Álvarez-Cubero, J. A. & Zufiria, P. J. (2016). "Algorithm 959: VBF: A Library
  of C++ Classes for Vector Boolean Functions in Cryptography." ACM TOMS 42(2).
  <https://doi.org/10.1145/2794077>
- Sweeney, J. et al. (2020). "CircuitGraph: A Python package for Boolean
  circuits." *JOSS* 5(56). <https://doi.org/10.21105/joss.02646>
- Czekanski, M., Kimmel, S. & Witter, R. T. (2023). "Robust and
  Space-Efficient Dual Adversary Quantum Query Algorithms." ESA 2023.
  <https://doi.org/10.4230/LIPIcs.ESA.2023.36>
- Bikov, D., Bouyukliev, I. & Dzhumalieva-Stoeva, M. (2023). "BooLSPLG: A
  Library with Parallel Algorithms for Boolean Functions and S-Boxes for
  GPU." *Mathematics* 11(8), 1864. <https://doi.org/10.3390/math11081864>
- Correia, R. B. et al. (2018). "CANA: A Python Package for Quantifying
  Control and Canalization in Boolean Networks." *Frontiers in Physiology*.
  <https://doi.org/10.3389/fphys.2018.01046>
- Marcus, A. M. et al. (2025). "CANA v1.0.0: efficient quantification of
  canalization in automata networks." *Bioinformatics* 41(10).
  <https://doi.org/10.1093/bioinformatics/btaf461>
- Kadelka, C. & Coberly, B. (2025). "BoolForge: Controlled Generation and
  Analysis of Boolean Functions and Networks." arXiv:2509.02496.
- González-Vaquero, E. & Maurizio Paul, R. (2026). "fbool: A Rust library for
  Boolean function entanglement analysis," v0.2.0.
