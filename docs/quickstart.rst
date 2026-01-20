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
   xor = bf.create([0, 1, 1, 0])

   # From callable (your own function)
   f = bf.create(lambda x: x[0] and x[1], n=3)

   # From file
   f = bf.load("function.json")
   f = bf.create("function.bf")  # Aaronson format

   # Built-in families
   maj = bf.majority(5)
   par = bf.parity(4)
   dic = bf.dictator(3, i=0)
   tribes = bf.tribes(2, 6)
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

Evaluation is equally flexible:

.. code-block:: python

   f.evaluate(3)                   # Integer index (binary: 011)
   f.evaluate([0, 1, 1])           # List of bits
   f.evaluate((0, 1, 1))           # Tuple
   f.evaluate(np.array([0, 1, 1])) # NumPy array
   f.evaluate([[0,0], [0,1], [1,0]])  # Batch (2D array)

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
