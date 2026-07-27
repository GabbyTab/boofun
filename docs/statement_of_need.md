# Statement of Need

## The gap

Analysis of Boolean functions is foundational in theoretical computer
science, with applications to property testing, learning theory,
pseudorandomness, circuit complexity, social choice, and cryptography. The
resources available to someone who wants to *compute* in this field are
split across four disconnected worlds:

- **Textbook exposition** (O'Donnell's *Analysis of Boolean Functions* and
  course notes) presents the mathematics but leaves computation to the
  reader.
- **One-off research scripts** implement exactly what one paper needed, are
  rarely packaged or versioned, and encode undocumented conventions.
- **Computer-algebra systems** such as SageMath cover parts of the area, but
  require adopting an entire environment rather than a `pip install`.
- **Domain-specific high-performance packages** go deep on one topic (e.g.
  S-box analysis) without covering the broader analysis toolkit.

The practical consequence: researchers and students reimplement truth-table
operations, Fourier transforms, influences, and complexity measures from
scratch before they can test an idea — and every reimplementation is a fresh
opportunity for convention bugs (`{0,1}` vs. `{-1,+1}` encodings, bit
ordering, normalization).

## What BooFun provides

BooFun fills the reproducible-experiment and teaching gap:

- `pip install boofun` gives a conventional Python package (Python 3.10+,
  NumPy-based core, everything else optional).
- A fluent API makes textbook operations directly executable in notebooks:
  `bf.majority(5).fourier()`, `.influences()`, `.noise_stability(0.9)`.
- A representation layer supports truth tables (dense, sparse, packed),
  Fourier expansions, algebraic normal form, DNF/CNF, decision diagrams,
  circuits, and threshold functions behind one interface, with automatic
  conversion along a cost-aware conversion graph.
- The test suite (3,800+ tests) encodes mathematical identities and
  cross-validates independent computation paths against closed-form results
  for standard families; see {doc}`cross_validation`.
- Twenty-five course-aligned notebooks connect the computations to
  O'Donnell's textbook and graduate course topics.

## Who it is for

- **Primary**: theoretical computer science students and researchers who
  need small- to medium-scale computational experiments (typical
  truth-table workloads: up to roughly n = 20 variables, depending on the
  operation and installed extras).
- **Secondary**: cryptographers evaluating Boolean functions and S-boxes,
  and instructors teaching Fourier analysis or property testing.

## Relation to existing software

BooFun is designed to complement, not replace, the existing ecosystem.

**SageMath** provides Boolean-function and S-box functionality inside a
broad computer-algebra environment. It is strong for algebraic and
cryptographic workflows but requires the Sage environment. BooFun's
differentiators are ordinary Python packaging, a fluent notebook-oriented
API, integrated coverage of query complexity, hypercontractivity, and
learning algorithms, and explicit cross-validation.

The R package
**[boolfun](https://doi.org/10.32614/RJ-2011-007)** and the C++ library
**[VBF](https://doi.org/10.1145/2794077)** are peer-reviewed predecessors
for scalar and vectorial cryptographic analysis, respectively. **sboxU**
(Perrin et al.) is a Sage/Python interface to a performance-oriented C++
library for vectorial Boolean functions, S-box analysis, APN functions, and
equivalence problems.
**[BooLSPLG](https://doi.org/10.3390/math11081864)** (2023) is a
peer-reviewed CUDA C/C++ library computing Walsh spectra, nonlinearity,
autocorrelation, algebraic degree, and LAT/DDT on GPUs. These tools are
deeper than BooFun in their cryptographic domains. BooFun's S-box functions are complementary and easier
to access from a standard scientific Python environment; first-class
vectorial Boolean functions are future work.

**[fbool](https://github.com/edugzlez/fbool)** v0.2.0 is a young Rust
library with Python bindings and one of the closest modern overlaps:
Fourier/Walsh coefficients, influences, sensitivity, Fourier degree,
nonlinearity, and certificate complexity. It additionally emphasizes
entropy, fragmentation, and exact small-circuit data. BooFun covers a
broader TCS curriculum and representation system; fbool supplies an
independently implemented comparison target for several exact quantities.

**Avishay Tal's course and research scripts** (`BooleanFunc.py` and
relatives) motivated part of BooFun's API. They are useful reference
implementations but not a maintained, versioned package with BooFun's
representation architecture, documentation, and validation infrastructure.
The {doc}`guides/migration_from_tal` documents correspondences and
convention differences.

**[CircuitGraph](https://doi.org/10.21105/joss.02646)** represents Boolean
circuits and computes per-input influence and average sensitivity using
exact or approximate model counting. This is material overlap, but its
primary object and purpose are circuit manipulation rather than a general
analysis-of-Boolean-functions toolkit.

**BoolForge** (Kadelka and Coberly) generates and analyzes Boolean functions
and networks with prescribed canalization structure.
**[CANA](https://doi.org/10.1093/bioinformatics/btaf461)** studies logical
redundancy, input symmetry, dynamics, and control in automata networks.
These projects target systems biology and network science. Their notions of
canalization are not all interchangeable with BooFun's classic canalizing
depth; BoolForge's overlapping function-level quantities are used for
cross-validation in BooFun's test suite.

**Scott Aaronson's Boolean Function Wizard** (2000) is part of the
query-complexity lineage and a historical inspiration for the complexity
module.
**[QuantumQueryOptimizer](https://doi.org/10.4230/LIPIcs.ESA.2023.36)**
solves general-adversary semidefinite programs and constructs query-optimal
quantum algorithms. BooFun instead exposes a broad complexity profile in
which every quantity is labeled exact, certified lower bound, or clamped
estimate, and its certified adversary lower bounds are cross-validated
against pinned QuantumQueryOptimizer SDP optima.

A survey of JOSS papers, indexed metadata, and package registries through
26 July 2026 found no package with BooFun's *combined* focus on
Fourier-analytic Boolean function measures, property testing, and query
complexity. CircuitGraph is a significant partial overlap; other JOSS
packages found in the survey focus on BDDs, S-box circuit synthesis,
Boolean-network applications, or Boolean-semiring linear algebra. This is a
dated, scoped survey result, not an assertion that no other partially
overlapping software exists. The sources, query strings, verification
protocol, and full candidate ledger are documented in the
{doc}`prior-art survey log <prior_art_survey>`.

## Design commitments

- **One convention, stated everywhere**: the O'Donnell standard (Boolean
  0 → +1, 1 → −1), so that f̂(∅) = E[f].
- **Verifiable claims**: numerical outputs are checked against closed-form
  results, redundant internal computation paths, and independent software
  where definitions overlap — against *pinned* reference versions
  (SageMath 10.9 via digest-pinned Docker fixtures; BoolForge v1.0.1
  commit-pinned in a scheduled CI job), with every claim, family range, and
  tolerance recorded in the {doc}`cross-validation claim matrix
  <cross_validation>`.
- **Quality gates in CI**: strict mypy (zero errors), a zero-warning Ruff
  profile, and an enforced line-and-branch coverage floor on every pull
  request.
