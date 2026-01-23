BooFun
======

.. image:: ../logos/boo_horizontal.png
   :width: 600
   :align: center
   :alt: BooFun Logo

Boolean function analysis in Python.

Tools for Fourier analysis, property testing, and complexity measures of Boolean functions. Built while studying O'Donnell's *Analysis of Boolean Functions*.

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
   guides/advanced

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   comparison_guide
   performance
   error_handling
   cross_validation

Installation
------------

.. code-block:: bash

   pip install boofun

Usage
-----

.. code-block:: python

   import boofun as bf

   # Create
   xor = bf.create([0, 1, 1, 0])
   maj = bf.majority(5)

   # Evaluate
   maj.evaluate([1, 1, 0, 0, 1])  # 1

   # Analyze
   maj.fourier()           # Fourier coefficients
   maj.influences()        # Variable influences
   maj.noise_stability(0.9)
   maj.is_monotone()

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
* **Cryptographic Analysis**: Nonlinearity, bent functions, LAT/DDT, S-box analysis
* **Partial Functions**: Streaming specification, hex I/O, storage hints
* **Advanced Sensitivity**: Moments, histograms, p-biased sensitivity
* **Decision Trees**: DP algorithms, tree enumeration, randomized complexity

What's Unique
-------------

Features not found in other Boolean function libraries:

* **Global hypercontractivity** analysis (Keevash, Lifshitz, Long & Minzer)
* **Full query complexity suite** (D, R, Q, Ambainis, spectral adversary)
* **Property testing** with probability bounds
* **Family tracking** for asymptotic analysis
* **Monte Carlo Fourier estimation** via sampling
* **O'Donnell textbook alignment** with educational notebooks

Test Coverage
-------------

Test coverage is 71% with 3056 tests. If something breaks, please report it.

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
