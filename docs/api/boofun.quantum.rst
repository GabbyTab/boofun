boofun.quantum
==============

.. warning::

   **Work in progress — classical fallbacks.**
   Most functions in this module currently execute classical algorithms and
   annotate the results with quantum complexity metadata (query counts,
   speedup factors, etc.).  No computation runs on quantum hardware or a
   full quantum simulator.  The module is useful for reasoning about quantum
   query complexity alongside classical measures, but should not be treated
   as a quantum computing toolkit.  Full quantum simulation support is
   planned for v2.0.0 — see :doc:`/ROADMAP`.

.. automodule:: boofun.quantum


   .. rubric:: Functions

   .. autosummary::

      create_quantum_boolean_function
      element_distinctness_analysis
      estimate_quantum_advantage
      grover_speedup
      quantum_walk_analysis
      quantum_walk_search

   .. rubric:: Classes

   .. autosummary::

      QuantumBooleanFunction
