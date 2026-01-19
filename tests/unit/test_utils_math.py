import numpy as np

from boofun.utils import math as math_utils


def test_popcnt_and_parity():
    assert math_utils.popcnt(0b1011) == 3
    assert math_utils.poppar(0b1011) == 1
    assert math_utils.poppar(0b1001) == 0


def test_over_bounds():
    assert math_utils.over(5, 2) == 10
    assert math_utils.over(5, -1) == 0
    assert math_utils.over(5, 6) == 0


def test_subsets_counts():
    elems = [0, 1, 2]
    all_subs = list(math_utils.subsets(elems))
    assert len(all_subs) == 2 ** len(elems)
    two_subs = list(math_utils.subsets(elems, 2))
    assert sorted(two_subs) == [(0, 1), (0, 2), (1, 2)]


def test_tensor_product_matches_numpy():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[0, 5], [6, 7]])
    assert np.allclose(math_utils.tensor_product(a, b), np.kron(a, b))


def test_krawchouk_values():
    # K_1(x;3) = 3 - 2x for binary case
    values = [math_utils.krawchouk(3, 1, x) for x in range(4)]
    assert values == [3, 1, -1, -3]
    # Legacy variant agrees with closed form for simple inputs
    assert math_utils.krawchouk2(3, 0, 2) == 1

