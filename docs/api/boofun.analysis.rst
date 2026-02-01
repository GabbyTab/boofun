boofun.analysis
===============

.. automodule:: boofun.analysis

Classes
-------

PropertyTester
~~~~~~~~~~~~~~

.. autoclass:: boofun.analysis.PropertyTester
   :members:
   :undoc-members:
   :show-inheritance:

   Property testing and local correction for Boolean functions.

   **Key Methods:**

   - :meth:`blr_linearity_test` - BLR linearity test
   - :meth:`monotonicity_test` - Monotonicity test
   - :meth:`junta_test` - k-junta test
   - :meth:`local_correct` - Local correction for functions close to linear
   - :meth:`local_correct_all` - Apply local correction to all inputs

SpectralAnalyzer
~~~~~~~~~~~~~~~~

.. autoclass:: boofun.analysis.SpectralAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

   Spectral analysis tools for Boolean functions.

Modules
-------

.. autosummary::
   :toctree:
   :recursive:

   arrow
   basic_properties
   block_sensitivity
   canalization
   certificates
   communication_complexity
   complexity
   cryptographic
   decision_trees
   equivalence
   fkn
   fourier
   gaussian
   gf2
   global_hypercontractivity
   huang
   hypercontractivity
   invariance
   learning
   ltf_analysis
   p_biased
   pac_learning
   query_complexity
   restrictions
   sampling
   sensitivity
   sparsity
   symmetry
