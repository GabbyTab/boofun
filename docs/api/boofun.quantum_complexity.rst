boofun.quantum\_complexity
==========================

.. warning::

   **Experimental playground — API will change.**

   This module is an exploratory sandbox for reasoning about quantum
   query complexity alongside the rest of BooFun's Boolean function
   analysis.  The API is unstable and the scope is still being figured
   out.

   **Everything here runs on a classical CPU.** The functions compute
   closed-form textbook formulas (Grover iteration counts, quantum walk
   hitting times, etc.) — useful for building intuition, but not a
   quantum simulator.

   We're actively thinking about what to build next: a lightweight
   statevector simulator, optional Qiskit/Cirq backends, genuine
   quantum property testers.  See :doc:`/ROADMAP` for the v2.0.0 plan.

   For mature, well-tested quantum complexity *lower bounds* (Ambainis,
   spectral adversary, polynomial method), see
   :mod:`boofun.analysis.query_complexity`.
   For Fourier analysis, use :class:`~boofun.analysis.SpectralAnalyzer`.
   For property testing, use :class:`~boofun.analysis.PropertyTester`.

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
