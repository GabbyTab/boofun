import pytest

import boolfunc as bf
from boolfunc.analysis import SpectralAnalyzer
from boolfunc.analysis import sensitivity as sens
from boolfunc.analysis import block_sensitivity as bs
from boolfunc.analysis import certificates as cert
from boolfunc.analysis import symmetry as sym


def brute_force_sensitivity(f: bf.BooleanFunction, x: int) -> int:
    n = f.n_vars or 0
    base = bool(f.evaluate(x))
    return sum(bool(f.evaluate(x ^ (1 << i))) != base for i in range(n))


def test_sensitivity_profile_matches_bruteforce():
    func = bf.create([0, 1, 1, 0])  # XOR on 2 vars
    profile = sens.sensitivity_profile(func)
    manual = [brute_force_sensitivity(func, i) for i in range(1 << func.n_vars)]
    assert profile.tolist() == manual
    total = sens.total_influence_via_sensitivity(func)
    spectral = SpectralAnalyzer(func).total_influence()
    assert pytest.approx(total, rel=1e-9) == spectral


def test_block_sensitivity_known_functions():
    dictator = bf.BooleanFunctionBuiltins.dictator(2, 0)
    parity = bf.BooleanFunctionBuiltins.parity(3)

    for x in range(1 << dictator.n_vars):
        assert bs.block_sensitivity_at(dictator, x) == 1

    for x in range(1 << parity.n_vars):
        assert bs.block_sensitivity_at(parity, x) == parity.n_vars
    assert bs.max_block_sensitivity(parity) == parity.n_vars


def test_certificates_for_and_function():
    and_func = bf.create([0, 0, 0, 1])
    size_0, vars_0 = cert.certificate(and_func, 0)
    assert size_0 == 1
    assert len(vars_0) == 1

    size_1, vars_1 = cert.certificate(and_func, 3)
    assert size_1 == 2
    assert set(vars_1) == {0, 1}
    assert cert.max_certificate_size(and_func) == 2


def test_symmetry_counts_majority():
    majority = bf.BooleanFunctionBuiltins.majority(3)
    counts = sym.symmetrize(majority)
    assert counts.tolist() == [0, 0, 3, 1]
    assert sym.degree_sym(majority) == 3
    assert sym.sens_sym(majority) == pytest.approx((2 * 3 + 3 * 1) / 4)

