Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install boofun

Development:

.. code-block:: bash

   git clone https://github.com/GabbyTab/boofun.git
   cd boofun
   pip install -e ".[dev]"

Creating Functions
------------------

.. code-block:: python

   import boofun as bf

   # From truth table
   xor = bf.create([0, 1, 1, 0])
   
   # Built-in
   maj = bf.majority(5)
   par = bf.parity(4)
   dic = bf.dictator(3, i=0)
   tribes = bf.tribes(2, 6)
   ltf = bf.weighted_majority([3, 2, 1, 1, 1])

Evaluation
----------

.. code-block:: python

   xor.evaluate([1, 0])  # True
   maj.evaluate([1, 1, 0, 0, 1])  # 1

Analysis
--------

.. code-block:: python

   f = bf.majority(5)
   
   f.fourier()           # Fourier coefficients
   f.influences()        # Per-variable
   f.total_influence()   # I[f]
   f.noise_stability(0.9)
   f.degree()            # Fourier degree
   
   f.analyze()  # Dict with all metrics

Properties
----------

.. code-block:: python

   f.is_linear()
   f.is_monotone()
   f.is_balanced()
   f.is_junta(2)

Representations
---------------

.. code-block:: python

   f.get_representation('truth_table')
   f.get_representation('anf')
   f.get_representation('fourier_expansion')

Visualization
-------------

Requires matplotlib:

.. code-block:: python

   viz = bf.BooleanFunctionVisualizer(f)
   viz.plot_influences()
   viz.plot_fourier_spectrum()

Next
----

* ``examples/`` - tutorials
* :doc:`performance` - optimization
* :doc:`comparison_guide` - library comparison
