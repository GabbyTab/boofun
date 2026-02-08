"""
Classical estimation of quantum query complexity bounds for Boolean functions.

.. warning:: **Experimental — NOT part of the stable public API**

   This module is **not exported** from the top-level ``boofun`` package.
   Import it explicitly if you want to use it::

       from boofun.quantum_complexity import QuantumComplexityAnalyzer

   It is an **exploratory sandbox** for thinking about quantum query
   complexity in the context of Boolean function analysis.  It is *not*
   a finished feature.  The API is unstable, the scope is still being
   figured out, and everything here may be reorganized, expanded, or
   removed in a future release.

   **Everything runs on a classical CPU.** There is no quantum hardware
   or quantum simulator involved.  The functions compute closed-form
   formulas from textbooks (Grover iteration counts, quantum walk hitting
   times, etc.) — useful for building intuition, but not a substitute
   for actual quantum simulation.

   What's here today:

   * **Grover complexity bounds** — closed-form ``O(√(N/M))`` formulas.
   * **Quantum walk complexity bounds** — analytical Szegedy walk
     estimates on the Boolean hypercube.
   * **Element distinctness analysis** — classical collision enumeration
     plus the known ``O(N^{2/3})`` quantum upper bound.
   * **Oracle circuit construction** — a Qiskit ``QuantumCircuit`` that
     implements the standard phase oracle (if Qiskit is installed).
     The circuit is built but **never executed**.

   What we're thinking about for v2.0.0 (see ROADMAP.md):

   * A lightweight statevector simulator so Grover / walk results come
     from actual simulated quantum dynamics, not just formulas.
   * Optional Qiskit/Cirq backends for larger simulations.
   * Genuine quantum property testers (BLR, monotonicity, junta).
   * Comparing simulated vs. analytical results side by side.

   If you have ideas about what would be most useful here, we'd love
   to hear them — see CONTRIBUTING.md.

   The ``boofun.analysis.query_complexity`` module provides the
   complementary quantum *lower bounds* (Ambainis adversary, spectral
   adversary, polynomial method).  Those are mature, well-tested, and
   correctly named.

   **What was removed (v1.3.0):** The previous ``quantum`` module
   contained classical algorithms with misleading ``quantum_`` prefixes
   (Fourier analysis, influence estimation, property testing) that
   duplicated ``SpectralAnalyzer`` and ``PropertyTester``.  Use those
   instead::

       analyzer = SpectralAnalyzer(f)
       analyzer.fourier_expansion()   # was quantum_fourier_analysis()
       analyzer.influences()          # was quantum_influence_estimation()

       tester = PropertyTester(f)
       tester.blr_linearity_test()    # was _quantum_linearity_test()
       tester.monotonicity_test()     # was _quantum_monotonicity_test()
       tester.junta_test(k)           # was _quantum_junta_test()
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cirq

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

from ..core.base import BooleanFunction


class QuantumComplexityAnalyzer:
    """
    Compute quantum complexity bounds for a Boolean function.

    .. warning:: **Experimental** — this class is part of BooFun's quantum
       complexity playground. The API may change in future releases.

    This is a *classical* analyzer that plugs numbers into closed-form
    formulas from quantum query complexity theory. It does **not**
    simulate quantum circuits or run quantum algorithms.

    Useful for building intuition about questions like:

    - "How many Grover iterations would be optimal for this function?"
    - "What is the quantum walk hitting time on the hypercube?"
    - "What does the Grover amplitude evolution look like analytically?"

    For Fourier analysis, influences, or property testing, use
    :class:`~boofun.analysis.SpectralAnalyzer` and
    :class:`~boofun.analysis.PropertyTester` — those are mature and
    well-tested.

    Example::

        import boofun as bf
        from boofun.quantum_complexity import QuantumComplexityAnalyzer

        f = bf.AND(6)
        qca = QuantumComplexityAnalyzer(f)
        result = qca.grover_analysis()
        print(f"Grover speedup: {result['speedup']:.1f}x")
    """

    def __init__(self, boolean_function: BooleanFunction):
        """
        Initialize the analyzer.

        Args:
            boolean_function: Classical Boolean function to analyze
        """
        self.function = boolean_function
        self.n_vars = boolean_function.n_vars
        if self.n_vars is None:
            raise ValueError("Function must have defined number of variables")

        # Cache for circuit construction
        self._quantum_circuit = None

    def create_quantum_oracle(self) -> Optional[Any]:
        """
        Create a Qiskit quantum oracle circuit for the Boolean function.

        This builds a valid ``QuantumCircuit`` implementing the standard
        phase-oracle construction (enumerate satisfying inputs, apply
        multi-controlled-X).  The circuit is **not executed** — it is
        returned as a data structure.

        Requires ``pip install qiskit``.

        Returns:
            Qiskit QuantumCircuit, or None if Qiskit is not installed.
        """
        assert self.n_vars is not None
        if not HAS_QISKIT:
            warnings.warn("Qiskit not available - cannot create quantum oracle")
            return None

        qreg = QuantumRegister(self.n_vars, "input")
        ancilla = QuantumRegister(1, "output")
        circuit = QuantumCircuit(qreg, ancilla)

        for x in range(2**self.n_vars):
            binary_x = [(x >> i) & 1 for i in range(self.n_vars)]
            f_x = self.function.evaluate(np.array(x))

            if f_x:
                for i, bit in enumerate(binary_x):
                    if bit == 0:
                        circuit.x(qreg[i])
                circuit.mcx(qreg, ancilla[0])
                for i, bit in enumerate(binary_x):
                    if bit == 0:
                        circuit.x(qreg[i])

        self._quantum_circuit = circuit
        return circuit

    def grover_analysis(self) -> Dict[str, Any]:
        """
        Compute Grover's algorithm complexity bounds (closed-form).

        Given a function f : {0,1}^n -> {0,1} with M satisfying
        assignments out of N = 2^n total inputs:

        - Classical expected queries: N / M
        - Grover queries: (pi/4) * sqrt(N / M)
        - Optimal iterations: floor(pi/4 * sqrt(N / M))

        These are standard textbook formulas (Grover 1996, Boyer et al. 1998).
        No quantum circuit is built or simulated.

        Returns:
            Dict with num_solutions, classical_queries, grover_queries,
            speedup, optimal_iterations, has_solutions, solution_density.
        """
        n = self.n_vars
        assert n is not None
        N = 2**n

        num_solutions = 0
        for x in range(N):
            if self.function.evaluate(x):
                num_solutions += 1

        M = num_solutions

        if M == 0:
            return {
                "num_solutions": 0,
                "classical_queries": N,
                "grover_queries": np.sqrt(N),
                "speedup": np.sqrt(N),
                "optimal_iterations": int(np.pi / 4 * np.sqrt(N)),
                "has_solutions": False,
            }

        classical_queries = N / M
        grover_queries = np.pi / 4 * np.sqrt(N / M)
        optimal_iterations = int(np.pi / 4 * np.sqrt(N / M))

        return {
            "num_solutions": M,
            "solution_density": M / N,
            "classical_queries": classical_queries,
            "grover_queries": grover_queries,
            "speedup": classical_queries / grover_queries,
            "optimal_iterations": optimal_iterations,
            "has_solutions": True,
        }

    def grover_amplitude_analysis(self) -> Dict[str, Any]:
        """
        Compute Grover amplitude evolution analytically (closed-form).

        Uses the exact formulas for amplitude amplification:

        - theta = arcsin(sqrt(M / N))
        - After k iterations: solution_amplitude = sin((2k+1) * theta)
        - Success probability = sin^2((2k+1) * theta)

        No statevector simulation — these are the known analytical results
        (Grover 1996, Boyer et al. 1998).

        Returns:
            Dict with num_solutions, theta, optimal_iterations,
            evolution (list of per-iteration amplitudes), max_success_prob.
        """
        n = self.n_vars
        assert n is not None
        N = 2**n

        solutions = []
        for x in range(N):
            if self.function.evaluate(x):
                solutions.append(x)

        M = len(solutions)
        if M == 0 or M == N:
            return {
                "num_solutions": M,
                "evolution": [],
                "message": "All or no solutions - Grover not applicable",
            }

        theta = np.arcsin(np.sqrt(M / N))
        optimal_k = int(np.pi / (4 * theta))

        evolution = []
        for k in range(min(optimal_k + 3, 20)):
            sol_amp = np.sin((2 * k + 1) * theta)
            non_sol_amp = np.cos((2 * k + 1) * theta) / np.sqrt(N - M)
            success_prob = sol_amp**2

            evolution.append(
                {
                    "iteration": k,
                    "solution_amplitude": sol_amp,
                    "success_probability": success_prob,
                    "non_solution_amplitude": non_sol_amp if M < N else 0,
                }
            )

        return {
            "num_solutions": M,
            "theta": theta,
            "optimal_iterations": optimal_k,
            "evolution": evolution,
            "max_success_prob": max(e["success_probability"] for e in evolution),
        }


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def create_complexity_analyzer(
    classical_function: BooleanFunction,
) -> QuantumComplexityAnalyzer:
    """
    Create a quantum complexity analyzer from a classical Boolean function.

    Args:
        classical_function: Classical Boolean function

    Returns:
        QuantumComplexityAnalyzer instance
    """
    return QuantumComplexityAnalyzer(classical_function)


def grover_speedup(f: BooleanFunction) -> Dict[str, Any]:
    """
    Convenience function: compute Grover speedup bounds for a Boolean function.

    Returns the same dict as ``QuantumComplexityAnalyzer(f).grover_analysis()``.
    """
    return QuantumComplexityAnalyzer(f).grover_analysis()


def quantum_walk_bounds(f: BooleanFunction) -> Dict[str, Any]:
    """
    Compute quantum walk complexity bounds on the Boolean hypercube (closed-form).

    Estimates the Szegedy quantum walk parameters for searching marked
    vertices (where f(x) = 1) on the n-dimensional hypercube {0,1}^n.

    Formulas used (Szegedy 2004, Magniez et al. 2011):
    - Spectral gap of hypercube walk: 2/n
    - Classical mixing time: n * log(N) / 2
    - Classical hitting time: N / M
    - Quantum hitting time: sqrt(classical_hitting * mixing)
    - Quantum walk complexity: sqrt(N/M) * sqrt(mixing_time)

    No walk is simulated — these are plugged-in analytical formulas.

    Args:
        f: Boolean function (marked vertices are where f(x) = 1)

    Returns:
        Dict with spectral_gap, mixing_time, hitting times, speedup, etc.
    """
    n = f.n_vars
    assert n is not None
    N = 2**n

    ones = sum(1 for x in range(N) if f.evaluate(x))
    zeros = N - ones

    spectral_gap = 2 / n if n > 0 else 1
    mixing_time = n * np.log(N) / 2
    classical_hitting_time = N / max(ones, 1)
    quantum_hitting_time = np.sqrt(classical_hitting_time * mixing_time)
    setup_cost = np.sqrt(mixing_time)
    checking_cost = 1

    if ones > 0:
        S = ones
        quantum_complexity = np.sqrt(N / S) * np.sqrt(mixing_time)
    else:
        quantum_complexity = np.sqrt(N * mixing_time)

    return {
        "n_vars": n,
        "state_space_size": N,
        "marked_states": ones,
        "unmarked_states": zeros,
        "hypercube_degree": n,
        "spectral_gap": spectral_gap,
        "mixing_time": mixing_time,
        "classical_hitting_time": classical_hitting_time,
        "quantum_hitting_time": quantum_hitting_time,
        "setup_cost": setup_cost,
        "checking_cost": checking_cost,
        "quantum_walk_complexity": quantum_complexity,
        "speedup_over_classical": (
            classical_hitting_time / quantum_hitting_time
            if quantum_hitting_time > 0
            else float("inf")
        ),
        "algorithm": "Szegedy quantum walk on hypercube (analytical bounds)",
    }


def element_distinctness_analysis(f: BooleanFunction) -> Dict[str, Any]:
    """
    Analyze element distinctness structure of a Boolean function.

    Performs classical enumeration to find collision pairs, then reports
    the known quantum upper bound O(N^{2/3}) from Ambainis (2007).

    The collision structure is computed classically by exhaustive evaluation.
    The quantum complexity is the known theoretical bound, not a simulated result.

    Args:
        f: Boolean function (viewed as function from [N] to some range)

    Returns:
        Dict with collision info, classical_complexity, quantum_complexity, speedup.
    """
    n = f.n_vars
    assert n is not None
    N = 2**n

    value_to_inputs: Dict[int, list[int]] = {}
    for x in range(N):
        val = int(f.evaluate(x))
        if val not in value_to_inputs:
            value_to_inputs[val] = []
        value_to_inputs[val].append(x)

    collisions: list[Dict[str, Any]] = []
    for val, inputs in value_to_inputs.items():
        if len(inputs) > 1:
            num_pairs = len(inputs) * (len(inputs) - 1) // 2
            collisions.append(
                {
                    "value": val,
                    "inputs": inputs,
                    "num_colliding": len(inputs),
                    "num_pairs": num_pairs,
                }
            )

    has_collision = len(collisions) > 0
    total_collision_pairs = sum(c["num_pairs"] for c in collisions)

    classical_complexity = N
    quantum_complexity = N ** (2 / 3)  # Ambainis 2007 (theoretical bound)

    return {
        "has_collision": has_collision,
        "num_distinct_values": len(value_to_inputs),
        "total_collision_pairs": total_collision_pairs,
        "collision_details": collisions[:5] if len(collisions) > 5 else collisions,
        "classical_complexity": classical_complexity,
        "quantum_complexity": quantum_complexity,
        "speedup": classical_complexity / quantum_complexity,
        "algorithm": "Ambainis element distinctness (theoretical bound)",
    }


def quantum_walk_search_bounds(
    f: BooleanFunction, num_iterations: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute quantum walk search success probabilities analytically.

    Uses the same closed-form Grover-like formulas:
    - theta = arcsin(sqrt(M / N))
    - Success probability after t steps: sin^2((2t+1) * theta)

    This is the analytical result, not a simulation.  The connection is
    mathematically valid: quantum walk search on the complete graph
    reduces to Grover's algorithm (Szegedy 2004).

    Args:
        f: Boolean function (marked vertices are where f(x) = 1)
        num_iterations: Number of walk iterations (default: optimal)

    Returns:
        Dict with marked_vertices, success probabilities, evolution, speedup.
    """
    n = f.n_vars
    assert n is not None
    N = 2**n

    marked = set(x for x in range(N) if f.evaluate(x))
    M = len(marked)

    if M == 0:
        return {
            "marked_vertices": 0,
            "optimal_iterations": 0,
            "success_probability": 0.0,
            "message": "No marked vertices",
        }

    if M == N:
        return {
            "marked_vertices": N,
            "optimal_iterations": 0,
            "success_probability": 1.0,
            "message": "All vertices marked",
        }

    if num_iterations is None:
        optimal = int(np.pi / 4 * np.sqrt(N / M))
        num_iterations = optimal

    theta = np.arcsin(np.sqrt(M / N))

    evolution = []
    for t in range(min(num_iterations + 3, 30)):
        prob = np.sin((2 * t + 1) * theta) ** 2
        evolution.append({"iteration": t, "success_probability": prob})

    max_prob = max(e["success_probability"] for e in evolution)
    optimal_t = max(range(len(evolution)), key=lambda t: evolution[t]["success_probability"])

    return {
        "marked_vertices": M,
        "total_vertices": N,
        "marked_fraction": M / N,
        "optimal_iterations": optimal_t,
        "iterations_used": num_iterations,
        "max_success_probability": max_prob,
        "final_success_probability": evolution[-1]["success_probability"] if evolution else 0,
        "evolution": evolution[:10],
        "speedup": np.sqrt(N / M) if M > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Backwards compatibility aliases (deprecated, will be removed in v2.0.0)
# ---------------------------------------------------------------------------

# Old name -> new name
QuantumBooleanFunction = QuantumComplexityAnalyzer
"""Deprecated alias. Use :class:`QuantumComplexityAnalyzer` instead."""

create_quantum_boolean_function = create_complexity_analyzer
"""Deprecated alias. Use :func:`create_complexity_analyzer` instead."""

quantum_walk_analysis = quantum_walk_bounds
"""Deprecated alias. Use :func:`quantum_walk_bounds` instead."""

quantum_walk_search = quantum_walk_search_bounds
"""Deprecated alias. Use :func:`quantum_walk_search_bounds` instead."""


__all__ = [
    # New names (preferred)
    "QuantumComplexityAnalyzer",
    "create_complexity_analyzer",
    "grover_speedup",
    "quantum_walk_bounds",
    "element_distinctness_analysis",
    "quantum_walk_search_bounds",
    # Deprecated aliases (backwards compatibility)
    "QuantumBooleanFunction",
    "create_quantum_boolean_function",
    "quantum_walk_analysis",
    "quantum_walk_search",
]
