"""
Cross-validation against SageMath's BooleanFunction, via pinned fixtures.

The reference values in fixtures/sagemath.json were computed by SageMath
itself (sage.crypto.boolean_function.BooleanFunction) running inside the
pinned Docker image recorded in the fixture metadata. They are regenerated
with scripts/generate_sage_fixtures.sh; the metadata records the Sage
version, image digest, generation date, and exact command.

Conventions (see also the fixture metadata):

- Truth tables: ``t[x] = f(x)`` with variable ``i`` in bit ``i`` of ``x``
  (variable 0 = least significant bit). SageMath and BooFun agree on this,
  which the generator asserts at generation time.
- Walsh transform: Sage's ``walsh_hadamard_transform()`` and BooFun's
  ``walsh_transform()`` both transform ``(-1)^f = 1 - 2f`` with the same
  mask indexing (verified by a runtime assertion in the generator), so
  spectra are compared entry-by-entry with exact signed equality — no
  absolute values, no dodges. See :func:`sage_walsh_to_boofun`.
- Algebraic degree: Sage reports the ANF degree, which is -1 for the zero
  function (degree of the zero polynomial); BooFun returns 0 for both
  constants. Converted by :func:`sage_degree_to_boofun`.

Tolerances: every property compared here is an exact integer or boolean,
so all comparisons are exact equality.

Reference:
https://doc.sagemath.org/html/en/reference/cryptography/sage/crypto/boolean_function.html
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.cryptographic import (
    algebraic_degree,
    correlation_immunity,
    is_balanced,
    is_bent,
    nonlinearity,
    walsh_transform,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sagemath.json"

_DATA = json.loads(FIXTURE_PATH.read_text())
METADATA = _DATA["metadata"]
FUNCTIONS = _DATA["functions"]
IDS = [entry["name"] for entry in FUNCTIONS]


# ---------------------------------------------------------------------------
# Convention conversion helpers (documented per property)
# ---------------------------------------------------------------------------


def sage_walsh_to_boofun(sage_wht: list[int]) -> list[int]:
    """Convert Sage's Walsh-Hadamard transform to BooFun's convention.

    The conversion is the identity: Sage 10.9 and BooFun both transform the
    ``(-1)^f(x) = 1 - 2f(x)`` encoding (0 -> +1, 1 -> -1) and both index
    the spectrum by the mask ``a`` with variable 0 in the least significant
    bit. This is not assumed — the fixture generator asserts it at
    generation time using the dictator function, whose transform is +4 at
    ``a = 1`` under this convention. The helper exists so that any future
    Sage convention change has exactly one place to be handled.
    """
    return list(sage_wht)


def sage_degree_to_boofun(sage_degree: int) -> int:
    """Convert Sage's ANF degree to BooFun's algebraic_degree convention.

    Sage returns -1 for the zero function (the degree of the zero
    polynomial over GF(2)); BooFun's algebraic_degree returns 0 for both
    constant functions. All other values agree.
    """
    return max(sage_degree, 0)


def textbook_ci_from_walsh(walsh: list[int], n: int) -> int:
    """Siegenthaler correlation-immunity order from a Walsh spectrum.

    f is correlation-immune of order k iff the Walsh transform vanishes on
    every mask ``a`` with ``1 <= popcount(a) <= k`` (the a = 0 coefficient,
    which only encodes balancedness, is ignored). This is BooFun's
    convention and the textbook one (Siegenthaler 1984; Xiao-Massey 1988).

    Sage's ``correlation_immunity()`` instead scans *all* nonzero Walsh
    coefficients including a = 0, so it returns -1 for every unbalanced
    function. The two conventions coincide exactly on balanced functions
    (checked empirically on all 303 fixture functions;
    :class:`TestFixtureIntegrity` keeps that relation executable). For
    unbalanced functions Sage's -1 carries no order information, so we
    validate BooFun against the textbook order derived from Sage's own
    recorded spectrum.
    """
    nonzero_weights = [bin(a).count("1") for a, v in enumerate(walsh) if v != 0 and a != 0]
    return min(nonzero_weights) - 1 if nonzero_weights else n


# ---------------------------------------------------------------------------
# Property-by-property cross-validation (exact equality)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
def test_walsh_spectrum(entry):
    """BooFun's full Walsh spectrum matches Sage's, entry by entry."""
    f = bf.create(entry["truth_table"])
    expected = sage_walsh_to_boofun(entry["sage"]["walsh_hadamard_transform"])
    actual = [int(v) for v in walsh_transform(f)]
    assert actual == expected


@pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
def test_nonlinearity(entry):
    f = bf.create(entry["truth_table"])
    assert nonlinearity(f) == entry["sage"]["nonlinearity"]


@pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
def test_algebraic_degree(entry):
    f = bf.create(entry["truth_table"])
    assert algebraic_degree(f) == sage_degree_to_boofun(entry["sage"]["algebraic_degree"])


@pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
def test_correlation_immunity(entry):
    """BooFun's CI equals the textbook order computed from Sage's spectrum.

    See :func:`textbook_ci_from_walsh` for why Sage's own
    ``correlation_immunity()`` value is only used directly on balanced
    functions.
    """
    f = bf.create(entry["truth_table"])
    ours = correlation_immunity(f)
    expected = textbook_ci_from_walsh(entry["sage"]["walsh_hadamard_transform"], entry["n"])
    assert ours == expected
    if entry["sage"]["is_balanced"]:
        # On balanced functions Sage's convention coincides with the
        # textbook one, so its reported value must match directly.
        assert ours == entry["sage"]["correlation_immunity"]


@pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
def test_is_balanced(entry):
    f = bf.create(entry["truth_table"])
    assert is_balanced(f) == entry["sage"]["is_balanced"]


@pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
def test_is_bent(entry):
    f = bf.create(entry["truth_table"])
    assert is_bent(f) == entry["sage"]["is_bent"]


# ---------------------------------------------------------------------------
# Corpus integrity
# ---------------------------------------------------------------------------


class TestFixtureIntegrity:
    """The fixture corpus is what it claims to be."""

    def test_metadata_provenance(self):
        """The fixture records its Sage version, image, and generator."""
        assert METADATA["sage_version"]
        assert METADATA["image"].startswith("sagemath/sagemath")
        assert METADATA["n_functions"] == len(FUNCTIONS)

    @pytest.mark.parametrize("entry", FUNCTIONS, ids=IDS)
    def test_sage_ci_convention(self, entry):
        """Sage's correlation_immunity() is the textbook order except that
        it also scans the a = 0 Walsh coefficient, giving -1 for every
        unbalanced function. This keeps that documented claim executable
        against the recorded spectra."""
        walsh = entry["sage"]["walsh_hadamard_transform"]
        nonzero_weights = [bin(a).count("1") for a, v in enumerate(walsh) if v != 0]
        sage_style = min(nonzero_weights) - 1 if nonzero_weights else entry["n"]
        assert sage_style == entry["sage"]["correlation_immunity"]

    def test_exhaustive_coverage(self):
        """All 16 two-variable and all 256 three-variable functions present."""
        by_n = {2: set(), 3: set()}
        for entry in FUNCTIONS:
            if entry["family"] == "exhaustive":
                code = sum(v << x for x, v in enumerate(entry["truth_table"]))
                by_n[entry["n"]].add(code)
        assert by_n[2] == set(range(16))
        assert by_n[3] == set(range(256))

    @pytest.mark.parametrize(
        "entry",
        [e for e in FUNCTIONS if e["family"] in ("parity", "majority", "tribes")],
        ids=[e["name"] for e in FUNCTIONS if e["family"] in ("parity", "majority", "tribes")],
    )
    def test_family_tables_match_boofun_constructors(self, entry):
        """Fixture family truth tables equal BooFun's own constructors,
        tying bf.parity/bf.majority/bf.tribes into the validated corpus."""
        name, n = entry["name"], entry["n"]
        if entry["family"] == "parity":
            f = bf.parity(n)
        elif entry["family"] == "majority":
            f = bf.majority(n)
        else:  # tribes{k}_{n}
            k = int(name.removeprefix("tribes").split("_")[0])
            f = bf.tribes(k, n)
        table = [int(f.evaluate(x)) for x in range(1 << n)]
        assert table == entry["truth_table"], f"{name}: constructor table != fixture table"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
