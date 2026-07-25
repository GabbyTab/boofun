---
title: 'BooFun: Reproducible Boolean Function Analysis in Python'
tags:
  - Python
  - Boolean functions
  - Fourier analysis
  - property testing
  - query complexity
  - theoretical computer science
authors:
  - name: Gabriel Taboada
    # TODO: register an ORCID iD before JOSS submission
    # orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: University of California, Berkeley, United States
    index: 1
date: 25 July 2026
bibliography: paper.bib
---

<!--
DRAFT SKELETON — not ready for submission.
Sections marked TODO need evidence or final numbers before this can go to
JOSS. See PUBLICATION_PLAN.md §2 for the full guidance behind each section.
-->

# Summary

BooFun is an open-source Python library for creating, transforming, and
analyzing Boolean functions. It provides a notebook-oriented interface for
Fourier analysis, influence and sensitivity measures, query complexity,
property testing, hypercontractivity, learning-theoretic algorithms, and
cryptographic metrics, following the conventions of @odonnell2014. Multiple
representations — truth tables (dense, sparse, and packed), Fourier
expansions, algebraic normal form, DNF/CNF, decision diagrams, circuits, and
threshold functions — share a common API with automatic, cost-aware
conversion. Numerical implementations are validated against closed-form
results, redundant internal computation paths, and independent software.

# Statement of need

Analysis of Boolean functions is foundational in theoretical computer
science, with applications to property testing, learning theory,
pseudorandomness, circuit complexity, social choice, and cryptography.
Existing resources are split between textbook exposition, one-off research
scripts, computer-algebra systems, and domain-specific high-performance
packages. Researchers and students often reimplement truth-table operations,
Fourier transforms, influences, and complexity measures before they can test
an idea — and every reimplementation is a fresh opportunity for convention
bugs ($\{0,1\}$ versus $\{-1,+1\}$ encodings, bit ordering, normalization).

BooFun fills the reproducible-experiment and teaching gap:

- `pip install boofun` provides a conventional Python package (Python 3.10+)
  with a NumPy-based core; performance, GPU, and visualization dependencies
  are optional extras.
- A fluent API makes textbook operations directly executable in notebooks.
- A representation layer supports multiple Boolean-function representations
  behind one interface.
- The test suite encodes mathematical identities and compares independent
  implementations.
- Twenty-five notebooks connect the computations to O'Donnell's textbook
  [@odonnell2014] and graduate course topics.

The primary audience is theoretical computer science students and
researchers who need small- to medium-scale computational experiments.
Secondary audiences include cryptographers evaluating Boolean functions and
S-boxes, and instructors teaching Fourier analysis or property testing.

# Cross-validation

<!-- TODO before submission: pin exact versions, families, parameter ranges,
and tolerances for each comparison, and link the executable tests that back
each claim. Do not claim cross-validation for areas without executable
tests. -->

BooFun treats verifiability as a design commitment. Outputs are checked
against:

- Known closed-form results for standard families (majority, parity, tribes,
  dictators, thresholds), including Fourier coefficients, influences, total
  influence, and noise stability.
- Independent internal computation paths (e.g., Fourier coefficients via the
  fast Walsh–Hadamard transform versus direct correlation sums).
- Independent software where definitions overlap: SageMath
  [@sagemath], Avishay Tal's research and teaching scripts, and BoolForge
  [@boolforge] for canalization and sensitivity measures.

All comparisons document convention conversions between the $\{0,1\}$ and
$\{-1,+1\}$ domains. Quality gates run on every pull request: a strict mypy
configuration with zero errors, a zero-warning Ruff lint profile, and an
enforced line-and-branch coverage floor over a suite of 3,800+ tests.

# State of the field

SageMath [@sagemath] provides Boolean-function and S-box functionality
inside a broad computer-algebra environment; it is strong for algebraic and
cryptographic workflows but requires the Sage environment and does not offer
BooFun's integrated notebook-oriented coverage of query complexity,
hypercontractivity, and learning algorithms. sboxU [@sboxu] is a Sage/Python
interface to a performance-oriented C++ library for vectorial Boolean
functions and S-box analysis; it is substantially deeper and faster than
BooFun in its cryptographic domain, and BooFun's S-box utilities are
complementary rather than competing. Avishay Tal's course and research
scripts motivated part of BooFun's API but are not a maintained, versioned
package; BooFun's migration guide documents correspondences and convention
differences. BoolForge [@boolforge] generates and analyzes Boolean functions
and networks with prescribed canalization structure for systems biology; the
projects are complementary, and their overlapping APIs are used for
cross-validation. Scott Aaronson's Boolean Function Wizard [@aaronson2000]
is an important historical tool in the query-complexity lineage.

The algorithms BooFun implements originate in the literature, including the
Kahn–Kalai–Linial theorem [@kkl1988], Friedgut's junta theorem
[@friedgut1998], the Goldreich–Levin algorithm [@goldreichlevin1989], and
the low-degree learning algorithm of Linial, Mansour, and Nisan [@lmn1993].

# Research use

<!-- TODO before submission: cite concrete evidence — the first BooFun-based
research note or preprint, externally documented workflows, and the Berkeley
course relationship described precisely and confirmed by the instructor.
This section must not be speculative. -->

BooFun was developed alongside a graduate course on analysis of Boolean
functions at UC Berkeley, where its notebooks accompany lecture topics.

# Availability

BooFun is available on PyPI (`pip install boofun`) for Python 3.10–3.13,
developed openly at <https://github.com/GabbyTab/boofun> under the MIT
license, with documentation at <https://gabbytab.github.io/boofun/>. The
test suite runs with `pytest tests/`.
<!-- TODO at acceptance: archived software DOI (Zenodo) and a reproducibility
script or notebook for the paper's claims. -->

# Acknowledgements

We thank Avishay Tal, whose course (CS 294-92, Spring 2025) and reference
scripts shaped part of BooFun's scope and API, and the course's scribes and
reviewers for the lecture materials the notebooks accompany.

<!-- TODO before submission: finalize the AI usage disclosure from
docs/AI_USAGE.md per JOSS policy, and add funding / conflict-of-interest
statements (including "none" where appropriate). -->

# References
