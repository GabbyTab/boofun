Quick Start Guide
=================

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install boofun

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/GabbyTab/boofun.git
   cd boofun
   pip install -e ".[dev]"

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For visualization features
   pip install boofun[visualization]
   
   # For all features including development tools
   pip install boofun[all]

First Steps
-----------

Creating Boolean Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import boofun as bf

   # From truth table
   xor = bf.create([0, 1, 1, 0])
   
   # Built-in functions (simplified API!)
   majority = bf.majority(5)        # 5-variable majority
   parity = bf.parity(4)            # 4-variable parity
   dictator = bf.dictator(3, i=0)   # Dictator on x₀
   tribes = bf.tribes(2, 6)         # Tribes function
   ltf = bf.weighted_majority(5, [3, 2, 1, 1, 1])  # Weighted voting

Function Evaluation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Single input evaluation
   result = xor.evaluate([1, 0])  # True
   
   # Batch evaluation (optimized automatically)
   import numpy as np
   batch_inputs = np.random.randint(0, 4, 1000)
   results = xor.evaluate(batch_inputs)

Spectral Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   # NEW: Direct methods on BooleanFunction (no analyzer needed!)
   f = bf.majority(5)
   
   influences = f.influences()              # Per-variable influences
   total_inf = f.total_influence()          # I[f] = sum of influences
   max_inf = f.max_influence()              # max_i Inf_i[f]
   variance = f.variance()                  # Var[f]
   stability = f.noise_stability(0.9)       # Stab_ρ[f]
   fourier = f.fourier()                    # All Fourier coefficients
   degree = f.degree()                      # Fourier degree
   
   # Quick summary of all metrics
   print(f.analyze())
   
   # Find heavy Fourier coefficients
   heavy = f.heavy_coefficients(tau=0.1)    # |f̂(S)| ≥ 0.1
   weights = f.spectral_weight_by_degree()  # Weight at each degree

Property Testing
~~~~~~~~~~~~~~~

.. code-block:: python

   # Direct property methods (no tester needed!)
   f = bf.majority(5)
   
   print(f"Is linear: {f.is_linear()}")
   print(f"Is monotone: {f.is_monotone()}")
   print(f"Is balanced: {f.is_balanced()}")
   print(f"Is symmetric: {f.is_symmetric()}")
   print(f"Is 2-junta: {f.is_junta(2)}")

Multiple Representations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get different representations
   truth_table = xor.get_representation('truth_table')
   anf_data = xor.get_representation('anf')
   fourier_coeffs = xor.get_representation('fourier_expansion')
   
   # Check available representations
   print(f"Available: {list(xor.representations.keys())}")
   
   # Conversion cost analysis
   options = xor.get_conversion_options()
   cost = xor.estimate_conversion_cost('bdd')

Validation and Testing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quick validation
   is_valid = bf.quick_validate(xor)
   
   # Comprehensive validation
   validator = bf.BooleanFunctionValidator(xor)
   results = validator.validate_all()

Visualization
~~~~~~~~~~~~

.. code-block:: python

   # Create visualizer (requires matplotlib/plotly)
   viz = bf.BooleanFunctionVisualizer(xor)
   
   # Generate plots
   viz.plot_influences()
   viz.plot_fourier_spectrum()
   viz.plot_noise_stability_curve()
   
   # Interactive dashboard
   viz.create_dashboard()

Next Steps
----------

* Explore the :doc:`examples/index` for comprehensive usage examples
* Check the :doc:`api/index` for detailed API documentation  
* Read the :doc:`theory/index` for mathematical background
* See :doc:`advanced/index` for performance optimization and advanced features
