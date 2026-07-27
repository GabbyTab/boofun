# Prior-Art Survey: Method and Ledger

This page documents *how* the prior-art survey behind the
[comparison guide](comparison_guide.md), the
[statement of need](statement_of_need.md), and the JOSS paper's "State of
the field" section was carried out, so that the survey's central claim is
reproducible and datable rather than an assertion.

**Survey date: 26 July 2026.**

**Claim supported.** As of the survey date, no located package combines
BooFun's three-part focus — Fourier-analytic Boolean-function measures,
property testing, and query complexity — in one library. CircuitGraph is
the clearest *published* partial overlap (influence and average sensitivity
via model counting); fbool is the closest *unpublished* one (Fourier
metrics, influence, sensitivity, certificate complexity). This is a scoped,
dated search result, not proof of absence.

## Method

The survey ran keyword searches against the sources below, then verified
every candidate against primary sources before classifying it.

### Sources and query strings

| Source | Queries run |
|--------|-------------|
| JOSS paper search (`joss.theoj.org`) | `Boolean function`, `Boolean network`, `S-box`, `binary decision diagram`, `property testing`, `query complexity`, `Walsh`, `Fourier analysis` |
| Crossref REST API (`api.crossref.org/works`) | bibliographic queries for each candidate's title/authors, and direct DOI record fetches to verify venue, year, authors, and title of every citation used |
| PyPI (`pypi.org/search`) | `boolean function`, `walsh hadamard`, `boolean fourier`, `s-box`, `property testing`, `canalization` |
| CRAN (incl. archive) | `boolean`, `boolfun`; archive status checked at `cran.r-project.org/src/contrib/Archive/` |
| crates.io | `boolean function`, `walsh` |
| GitHub search | `boolean function analysis`, `boolean function library`, `walsh spectrum`, topic `boolean-functions`; plus direct lookups of every named candidate |
| Julia registry (`juliapackages.com`) | `walsh`, `hadamard`, `boolean` |
| General web search (dated engine results) | `Python package "analysis of Boolean functions" Fourier influence noise stability library`, `Julia package Boolean functions Walsh Hadamard influence sensitivity`, `cryptographic Boolean functions MATLAB toolbox nonlinearity Walsh spectrum algebraic immunity`, `property testing library junta monotonicity software implementation`, `software hypercontractivity noise operator Boolean cube implementation`, `Banzhaf Shapley power index Python library`, `py-aiger-spectral Boolan Fourier boolean functions`, `JOSS Boolean function property testing package 2022..2026` |

### Verification protocol

For each candidate surfaced by any query:

1. Locate the primary artifact (repository and/or package registry entry).
2. If a paper is claimed, verify its metadata (venue, year, DOI, authors)
   through Crossref or the publisher — not through secondary mentions.
3. Establish capabilities from the project's *documented public API*
   (API reference, README examples), not from abstracts or taglines.
4. Record maintenance status (last release/commit, archived flags).

### Inclusion criteria

A system enters the comparison documents if it computes function-level
analytic quantities of Boolean functions (spectra, influence, sensitivity,
complexity measures, testing, cryptographic criteria) through a public API.
Systems are classified as: **direct overlap** (shares part of BooFun's
Fourier/testing/complexity core), **specialized cryptographic**,
**networks/biology**, **BDD/circuits/synthesis**, **transform kernels**
(fast transforms without function-analysis semantics), or **out of scope**.

## Candidate ledger

Every system evaluated, including rejections — so a future re-run can diff
against this list.

### Direct or materially overlapping

| System | Evidence checked | Notes |
|--------|------------------|-------|
| boolfun (R) | R Journal 2011, [DOI](https://doi.org/10.32614/RJ-2011-007); CRAN archive | Peer-reviewed scalar crypto predecessor; archived 2012 |
| VBF (C++) | ACM TOMS 2016, [DOI](https://doi.org/10.1145/2794077); manual | Peer-reviewed vectorial crypto predecessor |
| fbool (Rust + Python) | GitHub README/API, PyPI, v0.2.0 tag | Closest modern Fourier-analytic neighbor; not peer reviewed |
| py-aiger-spectral | GitHub README | Fourier coefficients/degree weights over AIGs; no release or paper |
| Boolan | GitHub README | Small unmaintained package: influences, degree weights, variance, noise sensitivity |
| CircuitGraph | JOSS 2020, [DOI](https://doi.org/10.21105/joss.02646); API docs | Influence + average sensitivity via model counting |
| QuantumQueryOptimizer | ESA 2023, [DOI](https://doi.org/10.4230/LIPIcs.ESA.2023.36) | Exact general-adversary SDP; used as a pinned cross-validation reference for BooFun's certified adversary lower bounds |
| Aaronson's Boolean Function Wizard | paper (2000) | Historical query-complexity lineage; unmaintained |
| Avishay Tal's scripts | course materials | Reference implementations, not a versioned package |

### Specialized cryptographic

| System | Evidence checked | Notes |
|--------|------------------|-------|
| SageMath `sage.crypto.boolean_function` | Sage reference docs | Walsh spectrum, nonlinearity, algebraic immunity; used as a pinned fixture reference |
| sboxU | GitHub | Sage/C++ S-box cryptanalysis; APN, equivalence |
| BooLSPLG | Mathematics 2023, [DOI](https://doi.org/10.3390/math11081864) | Peer-reviewed CUDA C/C++ GPU library; Walsh/autocorrelation spectra, nonlinearity, degree, LAT/DDT, n ≤ 20 |
| BoolCrypt | GitHub; [ePrint 2022/428](https://eprint.iacr.org/2022/428) | Sage-based vectorial functions; affine/CCZ equivalence via SAT |
| PEIGEN | IACR ToSC 2019 | S-box evaluation/implementation/generation platform |
| BSbox-tools | project page | Menu-driven predecessor from the BooLSPLG group; subsumed by BooLSPLG here |
| `boolean_function` (Rust crate) | crates.io, docs.rs | Crypto properties in Rust; cross-validation candidate ([#116](https://github.com/GabbyTab/boofun/issues/116)) |
| boolbox | GitHub | ANF/Möbius + quantum-circuit synthesis; pre-release, not on PyPI — below maturity threshold |

### Boolean networks / systems biology

BoolForge (arXiv:2509.02496; live cross-validation reference), CANA
(Frontiers in Physiology 2018; Bioinformatics 2025), pyboolnet, biobalm,
PyDrugLogics, emba, NORDic. Network-level dynamics; function-level
activities/sensitivity overlap only.

### BDD / circuits / synthesis

pyeda, Biddy (JOSS 2019), sboxgates (JOSS 2021), CUDD, ABC, espresso.
Representation and synthesis, not spectral analysis.

### Transform kernels (no function-analysis semantics)

| System | Notes |
|--------|-------|
| pyfwht / libfwht | CPU/OpenMP/CUDA fast Walsh–Hadamard transforms; an optional BooFun dependency, not a competing analysis library |
| Hadamard.jl (absorbed Walsh.jl) | Julia FWHT kernels |
| SymPy `fwht` | Transform function only |
| SPbLA | Sparse linear algebra over the Boolean semiring; terminological, not functional, overlap |

### Out of scope / adjacent domains

- **Power-index packages** (powerindex, powerindices, tucoopy,
  CooperativeGames, power-bdd): Banzhaf/Shapley–Shubik for weighted voting
  games via dynamic programming or BDDs. No Fourier machinery; adjacent
  only through the influence ↔ Banzhaf correspondence.
- **Boolin.jl**: small helpers mirroring Mathematica's `BooleanFunction`.
- **Mathematica** (`BooleanFunction`, `BooleanConvert`,
  `BooleanMinimize`): symbolic manipulation and minimization; no
  documented spectral/influence analysis.
- **MATLAB**: no located dedicated Boolean-function analysis toolbox
  (`fwht` transform only).

## Limitations and how to re-run

The survey is limited to English-language, indexed sources on the survey
date; capability judgments rest on documented public APIs, and undocumented
features would be missed. To update it: re-run the queries in the table
above, diff new candidates against the ledger, verify any additions via the
protocol, and bump the survey date everywhere it appears (this page, the
comparison guide, the statement of need, and the paper).
