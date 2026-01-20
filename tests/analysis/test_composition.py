import sys

sys.path.insert(0, "src")
import numpy as np
import pytest

import boofun as bf


def manual_compose_truth_table(outer: bf.BooleanFunction, inner: bf.BooleanFunction) -> np.ndarray:
    outer_n = outer.n_vars
    inner_n = inner.n_vars
    total_n = outer_n * inner_n
    size = 1 << total_n
    table = np.zeros(size, dtype=bool)

    for idx in range(size):
        bits = [(idx >> shift) & 1 for shift in range(total_n - 1, -1, -1)]
        inner_outputs = []
        for j in range(outer_n):
            block = bits[j * inner_n : (j + 1) * inner_n]
            inner_outputs.append(int(inner.evaluate(block)))
        table[idx] = bool(outer.evaluate(inner_outputs))

    return table


def test_compose_matches_manual_xor_dictator():
    outer = bf.BooleanFunctionBuiltins.parity(2)  # XOR on two vars
    inner = bf.BooleanFunctionBuiltins.dictator(1, 0)

    composed = outer.compose(inner)
    assert composed.n_vars == outer.n_vars * inner.n_vars

    expected = manual_compose_truth_table(outer, inner)
    result = composed.get_representation("truth_table")
    assert np.array_equal(result, expected)
    assert composed.space == outer.space


def test_compose_majority_parity():
    outer = bf.BooleanFunctionBuiltins.majority(2)
    inner = bf.BooleanFunctionBuiltins.parity(1)

    composed = outer.compose(inner)
    expected = manual_compose_truth_table(outer, inner)
    assert np.array_equal(composed.get_representation("truth_table"), expected)


def test_compose_space_mismatch_raises():
    outer = bf.create([0, 1], space="plus_minus_cube")
    inner = bf.create([0, 1], space="boolean_cube")

    with pytest.raises(ValueError):
        outer.compose(inner)
