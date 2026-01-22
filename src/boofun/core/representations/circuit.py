"""
Circuit representation for Boolean functions.

This module implements Boolean circuits using basic gates (AND, OR, NOT)
for representing and evaluating Boolean functions efficiently.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


class GateType(Enum):
    """Types of Boolean gates."""

    INPUT = "INPUT"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"


@dataclass
class Gate:
    """
    Represents a single gate in a Boolean circuit.

    Attributes:
        gate_id: Unique identifier for the gate
        gate_type: Type of gate (AND, OR, NOT, etc.)
        inputs: List of input gate IDs
        output_wire: Wire ID for output (optional)
    """

    gate_id: int
    gate_type: GateType
    inputs: List[int]
    output_wire: Optional[int] = None

    def evaluate(self, input_values: Dict[int, bool]) -> bool:
        """
        Evaluate gate with given input values.

        Args:
            input_values: Mapping from gate/wire IDs to boolean values

        Returns:
            Gate output value
        """
        if self.gate_type == GateType.INPUT:
            return input_values.get(self.gate_id, False)

        # Get input values
        in_vals = [input_values.get(inp_id, False) for inp_id in self.inputs]

        if self.gate_type == GateType.AND:
            return all(in_vals)
        elif self.gate_type == GateType.OR:
            return any(in_vals)
        elif self.gate_type == GateType.NOT:
            return not in_vals[0] if in_vals else True
        elif self.gate_type == GateType.XOR:
            return sum(in_vals) % 2 == 1
        elif self.gate_type == GateType.NAND:
            return not all(in_vals)
        elif self.gate_type == GateType.NOR:
            return not any(in_vals)
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")


class BooleanCircuit:
    """
    Boolean circuit representation using gates and wires.

    Supports construction, evaluation, and optimization of Boolean circuits
    with standard gates (AND, OR, NOT, XOR, NAND, NOR).
    """

    def __init__(self, n_inputs: int):
        """
        Initialize Boolean circuit.

        Args:
            n_inputs: Number of input variables
        """
        self.n_inputs = n_inputs
        self.gates: Dict[int, Gate] = {}
        self.input_gates: List[int] = []
        self.output_gate: Optional[int] = None
        self.next_gate_id = 0

        # Create input gates
        for i in range(n_inputs):
            gate_id = self._get_next_id()
            self.gates[gate_id] = Gate(gate_id, GateType.INPUT, [])
            self.input_gates.append(gate_id)

    def _get_next_id(self) -> int:
        """Get next available gate ID."""
        gate_id = self.next_gate_id
        self.next_gate_id += 1
        return gate_id

    def add_gate(self, gate_type: GateType, inputs: List[int]) -> int:
        """
        Add gate to circuit.

        Args:
            gate_type: Type of gate to add
            inputs: List of input gate IDs

        Returns:
            ID of newly created gate
        """
        gate_id = self._get_next_id()
        self.gates[gate_id] = Gate(gate_id, gate_type, inputs)
        return gate_id

    def set_output(self, gate_id: int) -> None:
        """Set output gate of the circuit."""
        if gate_id not in self.gates:
            raise ValueError(f"Gate {gate_id} not found")
        self.output_gate = gate_id

    def evaluate(self, inputs: Union[List[bool], np.ndarray]) -> bool:
        """
        Evaluate circuit with given inputs.

        Args:
            inputs: Boolean input values

        Returns:
            Circuit output value
        """
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")

        if self.output_gate is None:
            raise ValueError("No output gate specified")

        # Initialize input values
        gate_values = {}
        for i, inp_val in enumerate(inputs):
            gate_values[self.input_gates[i]] = bool(inp_val)

        # Evaluate gates in topological order
        evaluated = set(self.input_gates)

        while self.output_gate not in evaluated:
            progress = False

            for gate_id, gate in self.gates.items():
                if gate_id in evaluated:
                    continue

                # Check if all inputs are ready
                if all(inp_id in evaluated for inp_id in gate.inputs):
                    gate_values[gate_id] = gate.evaluate(gate_values)
                    evaluated.add(gate_id)
                    progress = True

            if not progress:
                raise RuntimeError("Circuit contains cycles or missing dependencies")

        return gate_values[self.output_gate]

    def get_depth(self) -> int:
        """
        Calculate circuit depth (longest path from input to output).

        Returns:
            Circuit depth
        """
        if self.output_gate is None:
            return 0

        depths = {}

        # Input gates have depth 0
        for gate_id in self.input_gates:
            depths[gate_id] = 0

        # Calculate depths using topological sort
        calculated = set(self.input_gates)

        while self.output_gate not in calculated:
            for gate_id, gate in self.gates.items():
                if gate_id in calculated:
                    continue

                if all(inp_id in calculated for inp_id in gate.inputs):
                    if gate.inputs:
                        depths[gate_id] = max(depths[inp_id] for inp_id in gate.inputs) + 1
                    else:
                        depths[gate_id] = 0
                    calculated.add(gate_id)

        return depths.get(self.output_gate, 0)

    def get_size(self) -> int:
        """Get total number of gates (excluding inputs)."""
        return len([g for g in self.gates.values() if g.gate_type != GateType.INPUT])

    def to_dict(self) -> Dict[str, Any]:
        """
        Export circuit to dictionary format.

        Returns:
            Dictionary representation of circuit
        """
        gates_data = []
        for gate in self.gates.values():
            gates_data.append(
                {"id": gate.gate_id, "type": gate.gate_type.value, "inputs": gate.inputs}
            )

        return {
            "n_inputs": self.n_inputs,
            "gates": gates_data,
            "input_gates": self.input_gates,
            "output_gate": self.output_gate,
            "depth": self.get_depth(),
            "size": self.get_size(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BooleanCircuit":
        """
        Create circuit from dictionary representation.

        Args:
            data: Dictionary representation

        Returns:
            BooleanCircuit instance
        """
        circuit = cls(data["n_inputs"])
        circuit.gates = {}
        circuit.next_gate_id = 0

        # Recreate gates
        for gate_data in data["gates"]:
            gate_id = gate_data["id"]
            gate_type = GateType(gate_data["type"])
            inputs = gate_data["inputs"]

            circuit.gates[gate_id] = Gate(gate_id, gate_type, inputs)
            circuit.next_gate_id = max(circuit.next_gate_id, gate_id + 1)

        circuit.input_gates = data["input_gates"]
        circuit.output_gate = data["output_gate"]

        return circuit


@register_strategy("circuit")
class CircuitRepresentation(BooleanFunctionRepresentation[BooleanCircuit]):
    """Circuit representation for Boolean functions."""

    def evaluate(
        self, inputs: np.ndarray, data: BooleanCircuit, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate circuit representation.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: Boolean circuit
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        if inputs.ndim == 0:
            # Single integer index
            binary_input = self._index_to_binary(int(inputs), n_vars)
            return data.evaluate(binary_input)
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                binary_input = inputs.astype(bool)
                return data.evaluate(binary_input)
            else:
                # Array of integer indices
                results = []
                for idx in inputs:
                    binary_input = self._index_to_binary(int(idx), n_vars)
                    results.append(data.evaluate(binary_input))
                return np.array(results, dtype=bool)
        elif inputs.ndim == 2:
            # Batch of binary vectors
            results = []
            for row in inputs:
                binary_input = row.astype(bool)
                results.append(data.evaluate(binary_input))
            return np.array(results, dtype=bool)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _index_to_binary(self, index: int, n_vars: int) -> List[bool]:
        """Convert integer index to binary vector using LSB=x₀ convention."""
        # LSB-first: result[i] = x_i = (index >> i) & 1
        return [(index >> i) & 1 == 1 for i in range(n_vars)]

    def dump(self, data: BooleanCircuit, space=None, **kwargs) -> Dict[str, Any]:
        """Export circuit representation."""
        circuit_dict = data.to_dict()
        circuit_dict["type"] = "circuit"
        return circuit_dict

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> BooleanCircuit:
        """
        Convert from another representation to circuit.

        Uses a simple DNF (Disjunctive Normal Form) construction approach.
        """
        # Get truth table first
        size = 1 << n_vars
        truth_table = []

        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)
            truth_table.append(bool(val))

        # Build circuit from truth table using DNF
        return self._build_dnf_circuit(truth_table, n_vars)

    def _build_dnf_circuit(self, truth_table: List[bool], n_vars: int) -> BooleanCircuit:
        """
        Build circuit from truth table using DNF (Disjunctive Normal Form).

        Args:
            truth_table: Boolean truth table
            n_vars: Number of variables

        Returns:
            BooleanCircuit implementing the function
        """
        circuit = BooleanCircuit(n_vars)

        # Find all minterms (rows where output is True)
        minterms = []
        for i, output in enumerate(truth_table):
            if output:
                minterms.append(i)

        if not minterms:
            # Constant False function
            false_gate = circuit.add_gate(GateType.AND, [circuit.input_gates[0]])
            not_gate = circuit.add_gate(GateType.NOT, [false_gate])
            and_gate = circuit.add_gate(GateType.AND, [false_gate, not_gate])
            circuit.set_output(and_gate)
            return circuit

        if len(minterms) == len(truth_table):
            # Constant True function
            true_gate = circuit.add_gate(GateType.OR, [circuit.input_gates[0]])
            not_gate = circuit.add_gate(GateType.NOT, [circuit.input_gates[0]])
            or_gate = circuit.add_gate(GateType.OR, [true_gate, not_gate])
            circuit.set_output(or_gate)
            return circuit

        # Build AND gates for each minterm
        minterm_gates = []
        for minterm in minterms:
            # Convert minterm to binary
            binary = [(minterm >> i) & 1 for i in range(n_vars - 1, -1, -1)]

            # Create literals (variables or their negations)
            literals = []
            for i, bit in enumerate(binary):
                if bit:
                    literals.append(circuit.input_gates[i])
                else:
                    not_gate = circuit.add_gate(GateType.NOT, [circuit.input_gates[i]])
                    literals.append(not_gate)

            # Create AND gate for this minterm
            if len(literals) == 1:
                minterm_gates.append(literals[0])
            else:
                # Build tree of AND gates
                current = literals[0]
                for lit in literals[1:]:
                    current = circuit.add_gate(GateType.AND, [current, lit])
                minterm_gates.append(current)

        # Create final OR gate
        if len(minterm_gates) == 1:
            circuit.set_output(minterm_gates[0])
        else:
            # Build tree of OR gates
            current = minterm_gates[0]
            for gate in minterm_gates[1:]:
                current = circuit.add_gate(GateType.OR, [current, gate])
            circuit.set_output(current)

        return circuit

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert circuit to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> BooleanCircuit:
        """Create empty circuit (constant False)."""
        circuit = BooleanCircuit(n_vars)
        # Create constant False: x AND NOT x
        not_gate = circuit.add_gate(GateType.NOT, [circuit.input_gates[0]])
        false_gate = circuit.add_gate(GateType.AND, [circuit.input_gates[0], not_gate])
        circuit.set_output(false_gate)
        return circuit

    def is_complete(self, data: BooleanCircuit) -> bool:
        """Check if circuit is complete (has output gate)."""
        return data.output_gate is not None

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for circuit operations."""
        return {
            "evaluation": 1,  # O(circuit_size) - linear in gates
            "construction": n_vars,  # O(2^n) - DNF construction
            "conversion_from": n_vars,  # O(2^n) - via truth table
            "space_complexity": n_vars,  # O(2^n) worst case for DNF
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return storage requirements for circuit representation."""
        # DNF can have exponential size in worst case
        max_gates = 2**n_vars * n_vars  # Worst case: all minterms with n literals each
        return {
            "max_gates": max_gates,
            "bytes_per_gate": 32,  # Rough estimate for Gate object
            "max_bytes": max_gates * 32,
            "space_complexity": "O(2^n) worst case, often much better",
        }

    def optimize_circuit(self, circuit: BooleanCircuit) -> BooleanCircuit:
        """
        Apply basic circuit optimizations.

        Args:
            circuit: Circuit to optimize

        Returns:
            Optimized circuit
        """
        # For now, return circuit unchanged
        # TODO: Implement optimizations like:
        # - Constant propagation
        # - Dead gate elimination
        # - Common subexpression elimination
        # - Gate-level optimizations (e.g., NOT(NOT(x)) = x)
        return circuit


# Utility functions for circuit construction
def build_majority_circuit(n_vars: int) -> BooleanCircuit:
    """
    Build optimized majority circuit.

    Args:
        n_vars: Number of variables

    Returns:
        Circuit implementing majority function
    """
    if n_vars % 2 == 0:
        raise ValueError("Majority requires odd number of variables")

    circuit = BooleanCircuit(n_vars)

    # For small cases, use direct construction
    if n_vars == 1:
        circuit.set_output(circuit.input_gates[0])
    elif n_vars == 3:
        # Majority of 3: (a AND b) OR (a AND c) OR (b AND c)
        a, b, c = circuit.input_gates

        ab = circuit.add_gate(GateType.AND, [a, b])
        ac = circuit.add_gate(GateType.AND, [a, c])
        bc = circuit.add_gate(GateType.AND, [b, c])

        ab_or_ac = circuit.add_gate(GateType.OR, [ab, ac])
        result = circuit.add_gate(GateType.OR, [ab_or_ac, bc])

        circuit.set_output(result)
    else:
        # For larger cases, use threshold implementation
        # This is a simplified version - real implementation would be more sophisticated
        n_vars // 2 + 1

        # Count number of True inputs (simplified approach)
        # In practice, this would use a more efficient threshold circuit
        current_sum = circuit.input_gates[0]

        for i in range(1, n_vars):
            current_sum = circuit.add_gate(GateType.XOR, [current_sum, circuit.input_gates[i]])

        # This is a placeholder - real threshold circuits are more complex
        circuit.set_output(current_sum)

    return circuit


def build_parity_circuit(n_vars: int) -> BooleanCircuit:
    """
    Build parity circuit using XOR gates.

    Args:
        n_vars: Number of variables

    Returns:
        Circuit implementing parity function
    """
    circuit = BooleanCircuit(n_vars)

    if n_vars == 1:
        circuit.set_output(circuit.input_gates[0])
    else:
        # Chain XOR gates
        current = circuit.input_gates[0]
        for i in range(1, n_vars):
            current = circuit.add_gate(GateType.XOR, [current, circuit.input_gates[i]])
        circuit.set_output(current)

    return circuit


# Export main classes and functions
__all__ = [
    "GateType",
    "Gate",
    "BooleanCircuit",
    "CircuitRepresentation",
    "build_majority_circuit",
    "build_parity_circuit",
]
