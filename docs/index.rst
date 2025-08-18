BoolFunc Documentation
=====================

.. image:: ../logos/boo_horizontal.png
   :width: 600
   :align: center
   :alt: BoolFunc Logo

A Comprehensive Python Library for Boolean Function Analysis and Computation

Overview
--------

BoolFunc is a comprehensive Python library for the analysis and manipulation of Boolean functions, 
designed for researchers and practitioners in theoretical computer science, computational complexity, 
and quantum computing. The library provides a unified framework for working with Boolean functions 
across multiple mathematical representations, enabling efficient computation of spectral properties, 
influence measures, and complexity-theoretic characteristics.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api/index
   examples/index
   theory/index
   advanced/index

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install boolfunc

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import boolfunc as bf
   import numpy as np

   # Create Boolean functions
   xor = bf.create([0, 1, 1, 0])  # XOR function
   majority = bf.BooleanFunctionBuiltins.majority(3)  # 3-variable majority

   # Evaluate functions
   print(f"XOR(1,0) = {xor.evaluate([1, 0])}")  # True
   print(f"Majority(1,1,0) = {majority.evaluate([1, 1, 0])}")  # True

   # Spectral analysis
   analyzer = bf.SpectralAnalyzer(xor)
   influences = analyzer.influences()
   noise_stability = analyzer.noise_stability(0.9)

   print(f"Variable influences: {influences}")
   print(f"Noise stability (ρ=0.9): {noise_stability}")

Mathematical Foundation
-----------------------

BoolFunc operates on Boolean functions :math:`f: \{0,1\}^n \to \{0,1\}`, providing tools for:

* **Fourier Analysis**: Walsh-Hadamard transform and spectral properties
* **Influence Theory**: Variable influence :math:`I_i(f) = \Pr[f(x) \neq f(x \oplus e_i)]`
* **Noise Stability**: :math:`NS_\rho(f) = \mathbb{E}[f(x)f(N_\rho(x))]` for noise operator :math:`N_\rho`
* **Complexity Measures**: Certificate complexity, sensitivity, block sensitivity
* **Learning Theory**: PAC learning with membership and equivalence queries

Applications
------------

* **Computational Complexity**: Analysis of Boolean function complexity classes
* **Social Choice Theory**: Voting systems and preference aggregation  
* **Cryptography**: Security analysis of Boolean functions in stream ciphers
* **Quantum Computing**: Boolean function analysis in quantum algorithms
* **Machine Learning**: Feature selection and Boolean concept learning

API Reference
=============

.. autosummary::
   :toctree: api/
   :recursive:

   boolfunc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
