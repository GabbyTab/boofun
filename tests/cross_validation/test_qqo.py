"""
Cross-validation against QuantumQueryOptimizer's ADV+- SDP, via pinned fixtures.

The reference values in fixtures/qqo.json are optima of the general
(negative-weight) adversary semidefinite program, computed by
quantum-query-optimizer (Witter & Czekanski,
https://github.com/rtealwitter/QuantumQueryOptimizer) at the version
recorded in the fixture metadata. ADV+-(f) characterizes bounded-error
quantum query complexity: Q2(f) = Theta(ADV+-(f)) (Hoyer-Lee-Spalek 2007;
Reichardt 2011). Fixtures are regenerated with
scripts/generate_qqo_fixtures.py (see its docstring for the venv recipe).

What is validated:

- BooFun's adversary functions (``ambainis_complexity``,
  ``spectral_adversary_bound``, ``general_adversary_bound``) claim to be
  certified *lower bounds* on ADV+-(f). Against the SDP optimum this is a
  one-sided check: BooFun value <= ADV+-(f) + tolerance, for every fixture
  function. A violation would falsify the certification claim.
- On anchor families with closed-form optima (AND/OR -> sqrt(n),
  PARITY -> n, MAJ3 -> 2, dictators -> 1) the sensitive-edge witness is
  actually optimal, so equality with the SDP value is asserted, showing
  the lower bounds are not vacuous.

Conventions: truth_table[x] = f(x) with variable i in bit i of x
(variable 0 = LSB), identical to the SageMath fixtures. The adversary
value is basis-independent, so no further conversion is needed.

Tolerance: QQO's first-order SDP solver shows absolute deviations up to
~1.5e-3 from closed-form anchors (recorded in the fixture metadata), so
comparisons use ABS_TOL = 5e-3.
"""

import json
import sys
from math import isclose
from pathlib import Path

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.query_complexity import (
    ambainis_complexity,
    general_adversary_bound,
    spectral_adversary_bound,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qqo.json"

_DATA = json.loads(FIXTURE_PATH.read_text())
METADATA = _DATA["metadata"]
FUNCTIONS = _DATA["functions"]
IDS = [entry["name"] for entry in FUNCTIONS]

ABS_TOL = 5e-3

# Anchor families where the sensitive-edge adversary witness is known to
# achieve the full ADV+- optimum, so the lower bound must be *tight*.
TIGHT = {
    "AND3",
    "OR3",
    "PARITY3",
    "MAJ3",
    "DICT3",
    "AND4",
    "OR4",
    "PARITY4",
    "exhaustive2_0110",  # XOR2
    "exhaustive2_1001",  # XNOR2
}


def make_function(entry: dict) -> bf.BooleanFunction:
    return bf.create(entry["truth_table"])


class TestFixtureIntegrity:
    def test_metadata_pins_package_version(self):
        assert METADATA["package"] == "quantum-query-optimizer"
        assert METADATA["package_version"] == "0.1.4"
        assert METADATA["n_functions"] == len(FUNCTIONS)

    @pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
    def test_known_anchors_match_sdp(self, entry):
        """Fixture self-consistency: closed-form literature values."""
        if entry["known_adv"] is None:
            pytest.skip("no closed-form anchor for this function")
        assert isclose(entry["qqo"]["adv_pm"], entry["known_adv"], abs_tol=2e-3)


class TestAdversaryLowerBounds:
    """BooFun's certified lower bounds must never exceed the SDP optimum."""

    @pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
    def test_ambainis_below_adv_pm(self, entry):
        f = make_function(entry)
        assert ambainis_complexity(f) <= entry["qqo"]["adv_pm"] + ABS_TOL

    @pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
    def test_spectral_below_adv_pm(self, entry):
        f = make_function(entry)
        assert spectral_adversary_bound(f) <= entry["qqo"]["adv_pm"] + ABS_TOL

    @pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
    def test_general_below_adv_pm(self, entry):
        f = make_function(entry)
        assert general_adversary_bound(f) <= entry["qqo"]["adv_pm"] + ABS_TOL

    @pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
    def test_nontrivial_for_nonconstant(self, entry):
        """Every non-constant function has a sensitive edge, so the
        witness value is at least 1 -- the bounds are never vacuous."""
        f = make_function(entry)
        assert general_adversary_bound(f) >= 1.0


_TIGHT_ENTRIES = [e for e in FUNCTIONS if e["name"] in TIGHT]


class TestTightAnchors:
    """On anchor families the sensitive-edge witness achieves ADV+-."""

    @pytest.mark.parametrize("entry", _TIGHT_ENTRIES, ids=[e["name"] for e in _TIGHT_ENTRIES])
    def test_witness_is_tight(self, entry):
        f = make_function(entry)
        assert isclose(general_adversary_bound(f), entry["qqo"]["adv_pm"], abs_tol=ABS_TOL)
