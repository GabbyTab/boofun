"""Generate SageMath reference fixtures for BooFun's cross-validation suite.

This script runs *inside* the pinned SageMath Docker container and has no
dependency on boofun. It evaluates SageMath's implementations of standard
Boolean-function properties on a fixed corpus and writes the results (with
full provenance metadata) to a JSON file. BooFun's test suite
(tests/cross_validation/test_sagemath.py) then compares its own output
against these recorded values.

Usage (from the repository root, via the wrapper):

    ./scripts/generate_sage_fixtures.sh

or directly:

    docker run --rm -v "$PWD":/work -w /work \
        -e SAGE_FIXTURE_IMAGE="sagemath/sagemath:<tag>@sha256:<digest>" \
        sagemath/sagemath:<tag> \
        sage -python scripts/generate_sage_fixtures.py \
        tests/cross_validation/fixtures/sagemath.json

Truth-table convention
----------------------
Truth tables are lists ``t`` of length ``2**n`` where ``t[x]`` is the value
of the function on the assignment encoded by the integer ``x``, with
variable ``i`` stored in bit ``i`` of ``x`` (variable 0 = least significant
bit). SageMath's ``BooleanFunction(list)`` uses the same convention; the
generator asserts this at runtime via a dictator function.

Recorded SageMath properties (per function)
-------------------------------------------
- ``walsh_hadamard_transform``: tuple of 2**n integers. Sage (verified on
  10.9 by a runtime assertion) transforms ``(-1)^f(x) = 1 - 2*f(x)``, the
  same convention as BooFun's walsh_transform.
- ``nonlinearity``: integer.
- ``algebraic_degree``: degree of the algebraic normal form over GF(2);
  Sage returns -1 for the zero function (degree of the zero polynomial).
- ``correlation_immunity``: integer order.
- ``is_balanced``, ``is_bent``: booleans.
"""

import datetime
import json
import os
import sys

from sage.crypto.boolean_function import BooleanFunction  # type: ignore[import-not-found]
from sage.env import SAGE_VERSION  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Corpus construction (pure Python, no sage, no boofun)
# ---------------------------------------------------------------------------


def popcount(x):
    return bin(x).count("1")


def parity_tt(n):
    """Parity: f(x) = x_0 XOR ... XOR x_{n-1}."""
    return [popcount(x) % 2 for x in range(1 << n)]


def majority_tt(n):
    """Majority (odd n): f(x) = 1 iff popcount(x) > n/2. Matches bf.majority."""
    return [1 if popcount(x) > n / 2 else 0 for x in range(1 << n)]


def threshold_tt(k, n):
    """Threshold: f(x) = 1 iff popcount(x) >= k."""
    return [1 if popcount(x) >= k else 0 for x in range(1 << n)]


def tribes_tt(k, n):
    """Tribes (AND-of-ORs convention, matching bf.tribes(k, n)).

    f(x) = AND over consecutive groups of k variables of (OR of the group).
    If n is not divisible by k the last group is smaller.
    """
    groups = [list(range(start, min(start + k, n))) for start in range(0, n, k)]

    def value(x):
        return int(all(any((x >> i) & 1 for i in group) for group in groups))

    return [value(x) for x in range(1 << n)]


def inner_product_tt(n):
    """Inner product bent function (even n): f(x) = XOR of x_{2i} AND x_{2i+1}."""
    assert n % 2 == 0

    def value(x):
        acc = 0
        for j in range(n // 2):
            acc ^= ((x >> (2 * j)) & 1) & ((x >> (2 * j + 1)) & 1)
        return acc

    return [value(x) for x in range(1 << n)]


def aes_sbox():
    """Compute the AES S-box from its definition (FIPS 197, section 5.1.1).

    S(x) = affine transform of the multiplicative inverse of x in
    GF(2^8) = GF(2)[t] / (t^8 + t^4 + t^3 + t + 1), with S(0) using inv(0)=0.
    Computed rather than hand-typed to rule out transcription errors.
    """

    def gf_mul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B  # reduction by t^8 + t^4 + t^3 + t + 1
            b >>= 1
        return p

    def gf_inv(a):
        if a == 0:
            return 0
        # a^(2^8 - 2) = a^254 by square-and-multiply
        result = 1
        power = a
        exponent = 254
        while exponent:
            if exponent & 1:
                result = gf_mul(result, power)
            power = gf_mul(power, power)
            exponent >>= 1
        return result

    def affine(b):
        c = 0x63
        out = 0
        for i in range(8):
            bit = (
                (b >> i)
                ^ (b >> ((i + 4) % 8))
                ^ (b >> ((i + 5) % 8))
                ^ (b >> ((i + 6) % 8))
                ^ (b >> ((i + 7) % 8))
                ^ (c >> i)
            ) & 1
            out |= bit << i
        return out

    box = [affine(gf_inv(x)) for x in range(256)]
    # Self-check against well-known values (FIPS 197, Figure 7).
    assert box[0x00] == 0x63
    assert box[0x01] == 0x7C
    assert box[0x53] == 0xED
    assert box[0xFF] == 0x16
    return box


def aes_component_tt(mask):
    """Component function of the AES S-box: f_b(x) = <b, S(x)> over GF(2)."""
    box = aes_sbox()
    return [popcount(box[x] & mask) % 2 for x in range(256)]


def build_corpus():
    """Return the list of (name, family, n, truth_table) entries."""
    corpus = []

    # Exhaustive: every 2-variable and 3-variable function.
    for n in (2, 3):
        for code in range(1 << (1 << n)):
            tt = [(code >> x) & 1 for x in range(1 << n)]
            corpus.append(("exhaustive{}_{:0{}b}".format(n, code, 1 << n), "exhaustive", n, tt))

    # Standard families up to n = 8.
    for n in range(2, 9):
        corpus.append((f"parity{n}", "parity", n, parity_tt(n)))
    for n in (3, 5, 7):
        corpus.append((f"majority{n}", "majority", n, majority_tt(n)))
    for k, n in ((2, 4), (3, 5), (3, 6), (4, 7), (5, 8)):
        corpus.append((f"threshold{k}_{n}", "threshold", n, threshold_tt(k, n)))
    for k, n in ((2, 4), (2, 6), (3, 6), (2, 8), (4, 8)):
        corpus.append((f"tribes{k}_{n}", "tribes", n, tribes_tt(k, n)))

    # Standard bent functions: inner product on 4, 6, 8 variables.
    for n in (4, 6, 8):
        corpus.append((f"inner_product{n}", "bent", n, inner_product_tt(n)))

    # AES S-box component functions (one per output bit).
    for bit in range(8):
        mask = 1 << bit
        corpus.append(
            (f"aes_sbox_component_{mask:02x}", "aes_component", 8, aes_component_tt(mask))
        )

    return corpus


# ---------------------------------------------------------------------------
# SageMath evaluation
# ---------------------------------------------------------------------------


def check_sage_conventions():
    """Assert Sage's truth-table indexing matches our convention.

    BooleanFunction([0, 1, 0, 1]) must be the dictator on variable 0
    (variable 0 = least significant bit of the truth-table index), so its
    Walsh-Hadamard transform is supported exactly on the mask a = 1.
    """
    dictator0 = BooleanFunction([0, 1, 0, 1])
    wht = [int(v) for v in dictator0.walsh_hadamard_transform()]
    support = [a for a, v in enumerate(wht) if v != 0]
    assert support == [1], f"Sage truth-table indexing changed: WHT support {support}"
    # Sage (verified on 10.9) transforms (-1)^f = 1 - 2f, same as BooFun,
    # so the dictator's coefficient at a=1 is +4.
    assert wht[1] == 4, f"Sage WHT sign convention changed: {wht}"


def sage_properties(tt):
    f = BooleanFunction(tt)
    anf_degree = int(f.algebraic_normal_form().degree())
    return {
        "walsh_hadamard_transform": [int(v) for v in f.walsh_hadamard_transform()],
        "nonlinearity": int(f.nonlinearity()),
        "algebraic_degree": anf_degree,
        "correlation_immunity": int(f.correlation_immunity()),
        "is_balanced": bool(f.is_balanced()),
        "is_bent": bool(f.is_bent()),
    }


def main():
    if len(sys.argv) != 2:
        print("usage: sage -python scripts/generate_sage_fixtures.py OUTPUT.json")
        return 1

    check_sage_conventions()

    corpus = build_corpus()
    functions = []
    for name, family, n, tt in corpus:
        functions.append(
            {
                "name": name,
                "family": family,
                "n": n,
                "truth_table": tt,
                "sage": sage_properties(tt),
            }
        )

    payload = {
        "metadata": {
            "generator": "scripts/generate_sage_fixtures.py",
            "sage_version": SAGE_VERSION,
            "image": os.environ.get("SAGE_FIXTURE_IMAGE", "unknown"),
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "truth_table_convention": (
                "t[x] = f(x) with variable i in bit i of x (variable 0 = LSB)"
            ),
            "walsh_convention": (
                "Sage and BooFun both transform (-1)^f = 1-2f (verified by a "
                "runtime assertion at generation), so boofun_walsh[a] == "
                "sage_wht[a] for every mask a"
            ),
            "n_functions": len(corpus),
        },
        "functions": functions,
    }

    with open(sys.argv[1], "w") as fh:
        json.dump(payload, fh, indent=None, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {len(functions)} functions to {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
