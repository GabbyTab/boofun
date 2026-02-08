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
* **Query Complexity**: D(f), R(f), Q(f), sensitivity, certificates, Ambainis bound
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

Features not found in other Boolean function libraries:

* **Global hypercontractivity** analysis (Keevash, Lifshitz, Long & Minzer)
* **Full query complexity suite** (D, R, Q, Ambainis, spectral adversary)
* **Invariance principle** with Gaussian analysis and Berry-Esseen bounds
* **Property testing** with probability bounds
* **Family tracking** for asymptotic analysis
* **Monte Carlo Fourier estimation** via sampling — scales beyond exact computation
* **Pseudorandomness connections** — spectral concentration, threshold phenomena
* **O'Donnell textbook alignment** with educational notebooks

Test Coverage
-------------

Test coverage is 72% with 3200+ tests. If something breaks, please report it.

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
