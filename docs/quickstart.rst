Quick Start Guide
=================

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install boolfunc

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/GabbyTab/boolfunc.git
   cd boolfunc
   pip install -e ".[dev]"

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For visualization features
   pip install boolfunc[visualization]
   
   # For all features including development tools
   pip install boolfunc[all]

First Steps
-----------

Creating Boolean Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import boolfunc as bf

   # From truth table
   xor = bf.create([0, 1, 1, 0])
   
   # Built-in functions
   majority = bf.BooleanFunctionBuiltins.majority(3)
   parity = bf.BooleanFunctionBuiltins.parity(4)
   tribes = bf.BooleanFunctionBuiltins.tribes(k=2, n=6)

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

   # Create analyzer
   analyzer = bf.SpectralAnalyzer(xor)
   
   # Compute spectral properties
   influences = analyzer.influences()
   total_influence = analyzer.total_influence()
   noise_stability = analyzer.noise_stability(0.9)
   fourier_coeffs = analyzer.fourier_expansion()

Property Testing
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create property tester
   tester = bf.PropertyTester(xor)
   
   # Test various properties
   is_linear = tester.test_linearity()
   is_monotone = tester.test_monotonicity()
   is_balanced = tester.test_balance()

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
