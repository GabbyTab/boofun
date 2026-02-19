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

   # From truth table (list or numpy array)
   xor_2 = bf.create([0, 1, 1, 0])

   # From callable (your own function)
   f = bf.create(lambda x: x[0] and x[1], n=3)

   # From file
   f = bf.load("function.json")
   f = bf.create("function.bf")  # Aaronson format

   # Built-in families (use descriptive names)
   maj_5 = bf.majority(5)
   par_4 = bf.parity(4)
   dic_3 = bf.dictator(3, i=0)
   tribes_2_6 = bf.tribes(2, 6)
   ltf = bf.weighted_majority([3, 2, 1, 1, 1])

Flexible Input Types
--------------------

``bf.create()`` auto-detects input type:

.. code-block:: python

   bf.create([0, 1, 1, 0])           # list → truth table
   bf.create(np.array([0, 1, 1, 0])) # numpy → truth table
   bf.create(lambda x: x[0] ^ x[1], n=2)  # callable
   bf.create({frozenset(): 1, frozenset({0}): 1})  # dict → polynomial
   bf.create("x0 & x1")              # string → symbolic
   bf.create({(0,1), (1,0)})         # set of tuples → which inputs are True

   # From files
   bf.load("func.json")   # JSON with metadata
   bf.load("func.bf")     # Aaronson .bf format
   bf.load("func.cnf")    # DIMACS CNF

Evaluation
----------

Functions are callable. Evaluation is flexible:

.. code-block:: python

   maj_5 = bf.majority(5)

   # Callable syntax (preferred)
   maj_5([1, 1, 0, 0, 1])  # → True (majority satisfied)
   maj_5(7)                # → True (7 = 00111, 3 ones)

   # Equivalent .evaluate() method
   maj_5.evaluate([1, 1, 0, 0, 1])

   # All input formats work
   maj_5(3)                    # Integer index (binary: 00011)
   maj_5([0, 1, 1, 0, 0])      # List of bits
   maj_5((0, 1, 1, 0, 0))      # Tuple
   maj_5(np.array([0,1,1,0,0]))  # NumPy array

Analysis
--------

.. code-block:: python

   maj_5 = bf.majority(5)

   # Fourier analysis
   maj_5.fourier()           # Fourier coefficients
   maj_5.influences()        # Per-variable
   maj_5.total_influence()   # I[f]
   maj_5.noise_stability(0.9)
   maj_5.degree()            # Fourier degree

   # Query complexity
   from boofun.analysis import complexity
   complexity.D(maj_5)       # Decision tree depth D(f)
   complexity.s(maj_5)       # Max sensitivity s(f)

   maj_5.analyze()  # Dict with all metrics

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
