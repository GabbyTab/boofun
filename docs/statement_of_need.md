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

**sboxU** (Perrin et al.) is a Sage/Python interface to a
performance-oriented C++ library for vectorial Boolean functions, S-box
analysis, APN functions, and equivalence problems. It is substantially
deeper and faster than BooFun in its cryptographic domain. BooFun's S-box
functions are complementary and easier to access from a standard scientific
Python environment; first-class vectorial Boolean functions are future work.

**Avishay Tal's course and research scripts** (`BooleanFunc.py` and
relatives) motivated part of BooFun's API. They are useful reference
implementations but not a maintained, versioned package with BooFun's
representation architecture, documentation, and validation infrastructure.
The {doc}`guides/migration_from_tal` documents correspondences and
convention differences.

**BoolForge** (Kadelka et al.) generates and analyzes Boolean functions and
networks with prescribed canalization structure, aimed at systems biology
and network science. The projects are complementary: BooFun emphasizes
Fourier-analytic and TCS measures; BoolForge emphasizes canalization,
ensembles, and network dynamics. Their overlapping canalization and
sensitivity APIs are used for cross-validation in BooFun's test suite.

**Scott Aaronson's Boolean Function Wizard** (2000) is part of the
query-complexity lineage and a historical inspiration for the complexity
module.

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
