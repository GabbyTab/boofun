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
   :caption: Contents:

   quickstart
   performance
   comparison_guide
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

* **Fourier**: Walsh-Hadamard transform, influences, noise stability
* **Property Testing**: BLR, junta, monotonicity
* **Query Complexity**: D(f), R(f), Q(f), sensitivity, certificates
* **Representations**: Truth tables, ANF, BDD, circuits, Fourier expansion

Limitations
-----------

Test coverage is low (~38%). Edge cases may have bugs. If something breaks, please report it.

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
