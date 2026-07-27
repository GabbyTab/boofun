BooFun
======

.. image:: ../logos/boo_horizontal.png
   :width: 600
   :align: center
   :alt: BooFun Logo

Boolean function analysis in Python.

Tools for Fourier analysis, property testing, and complexity measures of Boolean functions. Built while studying O'Donnell's *Analysis of Boolean Functions*.

`GitHub Repository <https://github.com/GabbyTab/boofun>`_ · `PyPI <https://pypi.org/project/boofun/>`_ · `Issues <https://github.com/GabbyTab/boofun/issues>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quickstart
   statement_of_need

.. toctree::
   :maxdepth: 2
   :caption: Guides:

   guides/spectral_analysis
   guides/query_complexity
   guides/hypercontractivity
   guides/cryptographic
   guides/learning
   guides/representations
   guides/operations
   guides/families
   guides/probabilistic
   guides/advanced
   guides/migration_from_tal

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   architecture
   comparison_guide
   prior_art_survey
   performance
   error_handling
   cross_validation

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   CONTRIBUTING
   STYLE_GUIDE
   TEST_GUIDELINES

Installation
------------

.. code-block:: bash

   pip install boofun

Usage
-----

.. code-block:: python

   import boofun as bf

   # Create
   xor_2 = bf.create([0, 1, 1, 0])
   maj_5 = bf.majority(5)

   # Evaluate (callable syntax)
   maj_5([1, 1, 0, 0, 1])  # True
   maj_5(7)                # True (7 = 00111)

   # Analyze
   maj_5.fourier()           # Fourier coefficients
   maj_5.influences()        # Variable influences
   maj_5.noise_stability(0.9)
   maj_5.is_monotone()

Convention
----------

O'Donnell standard: Boolean 0 → +1, Boolean 1 → −1.

This ensures ``f̂(∅) = E[f]``.

What's Here
-----------

**Core Analysis**

* **Fourier**: Walsh-Hadamard transform, influences, noise stability, spectral concentration
* **Property Testing**: BLR, junta, monotonicity, symmetry, balance
* **Query Complexity**: D(f), R(f), Q(f), sensitivity, certificates, adversary-method estimates
* **Representations**: Truth tables, ANF, BDD, circuits, DNF/CNF, Fourier expansion

**New in v1.1**

* **Hypercontractivity**: Noise operator, Bonami's Lemma, KKL theorem, Friedgut's junta theorem
* **Global Hypercontractivity**: p-biased analysis, threshold phenomena (Keevash et al.)
* **Invariance Principle**: Gaussian analysis, Berry-Esseen bounds, Majority is Stablest
* **Cryptographic Analysis**: Nonlinearity, bent functions, LAT/DDT, S-box analysis
* **Probabilistic View**: Monte Carlo estimation, p-biased measures, spectral sampling
* **Partial Functions**: Streaming specification, hex I/O, storage hints
* **Advanced Sensitivity**: Moments, histograms, p-biased sensitivity
* **Decision Trees**: DP algorithms, tree enumeration, randomized complexity

What's Unique
-------------

No located library combines all of the following in one package — see the
:doc:`comparison_guide` and the reproducible :doc:`prior_art_survey` for
what overlapping software exists:

* **Hypercontractivity and global hypercontractivity** analysis (Bonami,
  KKL, Friedgut; Keevash, Lifshitz, Long & Minzer)
* **Integrated query complexity suite** (D, R, Q, sensitivity,
  certificates, plus clearly labeled adversary-method estimates)
* **Invariance principle** with Gaussian analysis and Berry-Esseen bounds
* **Property testing** in the query model (BLR, monotonicity, symmetry)
* **Family tracking** for asymptotic analysis
* **Monte Carlo Fourier estimation** via sampling — scales beyond exact computation
* **Pseudorandomness connections** — spectral concentration, threshold phenomena
* **O'Donnell textbook alignment** with educational notebooks

Test Coverage
-------------

The suite has 3,800+ tests with an enforced line-and-branch coverage floor
in CI (~79%). If something breaks, please report it.

API Reference
=============

.. autosummary::
   :toctree: api/
   :recursive:

   boofun

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
