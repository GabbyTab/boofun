boofun.quantum\_complexity
==========================

.. note::

   **Classical computation of quantum query model bounds.**

   Everything in this module runs on a classical CPU. It computes
   closed-form complexity estimates from quantum query complexity theory
   (Grover iteration counts, quantum walk hitting times, etc.) but does
   **not** simulate quantum circuits or run quantum algorithms.

   For Fourier analysis, use :class:`~boofun.analysis.SpectralAnalyzer`.
   For property testing, use :class:`~boofun.analysis.PropertyTester`.
   For quantum query complexity lower bounds (Ambainis, spectral adversary,
   polynomial method), see :mod:`boofun.analysis.query_complexity`.

   Full quantum simulation support is planned for v2.0.0 — see
   :doc:`/ROADMAP`.

.. automodule:: boofun.quantum_complexity


   .. rubric:: Functions

   .. autosummary::

      create_complexity_analyzer
      element_distinctness_analysis
      grover_speedup
      quantum_walk_bounds
      quantum_walk_search_bounds

   .. rubric:: Classes

   .. autosummary::

      QuantumComplexityAnalyzer
