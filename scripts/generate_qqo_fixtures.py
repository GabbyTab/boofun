#!/usr/bin/env python3
"""
Generate pinned ADV+- fixtures from quantum-query-optimizer (QQO).

QQO (Witter & Czekanski, JOSS 2021, https://github.com/rtealwitter/
QuantumQueryOptimizer) solves the general adversary bound semidefinite
program of Reichardt (2009), whose optimum ADV+-(f) characterizes
bounded-error quantum query complexity: Q2(f) = Theta(ADV+-(f)).
BooFun deliberately does not ship an SDP solver, so these fixtures are
the exact external reference that BooFun's certified adversary *lower
bounds* (ambainis_complexity, spectral_adversary_bound,
general_adversary_bound) are validated against in
tests/cross_validation/test_qqo.py.

This script is intentionally independent of BooFun: it only needs a
Python environment with quantum-query-optimizer installed (plus its
dependencies numpy, scipy, cvxpy, matplotlib, termcolor).

Usage:
    python -m venv qqo-venv
    qqo-venv/bin/pip install quantum-query-optimizer==0.1.4 \
        numpy scipy cvxpy matplotlib termcolor
    qqo-venv/bin/python scripts/generate_qqo_fixtures.py \
        tests/cross_validation/fixtures/qqo.json

Conventions:
    truth_table[x] = f(x) with variable i in bit i of x (variable 0 =
    LSB), matching every other BooFun fixture. QQO input bitstrings are
    built so string position i holds variable i.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version


def _popcount(x: int) -> int:
    return bin(x).count("1")


def battery() -> list[dict]:
    """The pinned function set: name, family, n, truth table, and a
    closed-form ADV+- anchor where one is known from the literature."""
    functions: list[dict] = []

    # Exhaustive n = 2 (all 16 functions). Constants are skipped because
    # the adversary bound is 0 and QQO's SDP setup assumes both outputs
    # occur.
    for code in range(16):
        tt = [(code >> x) & 1 for x in range(4)]
        if len(set(tt)) < 2:
            continue
        functions.append(
            {
                "name": f"exhaustive2_{''.join(map(str, tt))}",
                "family": "exhaustive",
                "n": 2,
                "truth_table": tt,
                "known_adv": None,
            }
        )

    named: list[tuple[str, int, list[int], float | None]] = [
        # ADV+-(AND_n) = ADV+-(OR_n) = sqrt(n); ADV+-(PARITY_n) = n.
        ("AND3", 3, [int(x == 7) for x in range(8)], math.sqrt(3)),
        ("OR3", 3, [int(x != 0) for x in range(8)], math.sqrt(3)),
        ("PARITY3", 3, [_popcount(x) % 2 for x in range(8)], 3.0),
        ("MAJ3", 3, [int(_popcount(x) >= 2) for x in range(8)], 2.0),
        ("DICT3", 3, [x & 1 for x in range(8)], 1.0),
        # x0 AND (x1 OR x2): read-once AND-OR
        ("AND_OR3", 3, [int(bool(x & 1) and bool(x & 6)) for x in range(8)], None),
        # multiplexer: f = x1 if x0 = 0 else x2
        ("MUX3", 3, [(x >> 2) & 1 if x & 1 else (x >> 1) & 1 for x in range(8)], None),
        ("AND4", 4, [int(x == 15) for x in range(16)], 2.0),
        ("OR4", 4, [int(x != 0) for x in range(16)], 2.0),
        ("PARITY4", 4, [_popcount(x) % 2 for x in range(16)], 4.0),
        # threshold-2 (at least two ones)
        ("THR2_4", 4, [int(_popcount(x) >= 2) for x in range(16)], None),
        # 2x2 AND-OR tree: (x0 OR x1) AND (x2 OR x3)
        ("TREE2x2", 4, [int(bool(x & 3) and bool(x & 12)) for x in range(16)], None),
        # inner product / bent: (x0 AND x1) XOR (x2 AND x3)
        (
            "BENT4_IP",
            4,
            [((x & 1) & ((x >> 1) & 1)) ^ (((x >> 2) & 1) & ((x >> 3) & 1)) for x in range(16)],
            None,
        ),
    ]
    for name, n, tt, known in named:
        functions.append(
            {"name": name, "family": "named", "n": n, "truth_table": tt, "known_adv": known}
        )
    return functions


def main() -> None:
    import quantum_query_optimizer as qqo

    out_path = sys.argv[1] if len(sys.argv) > 1 else "tests/cross_validation/fixtures/qqo.json"

    entries = []
    for spec in battery():
        n, tt = spec["n"], spec["truth_table"]
        # String position i = variable i = bit i of the index.
        D = [format(x, f"0{n}b")[::-1] for x in range(1 << n)]
        E = [str(v) for v in tt]
        sol = qqo.runSDP(D=D, E=E, print_output=False, run_checks=False)
        adv = float(sol["query_complexity"])
        if spec["known_adv"] is not None and abs(adv - spec["known_adv"]) > 1e-3:
            raise SystemExit(
                f"anchor mismatch for {spec['name']}: QQO={adv}, literature={spec['known_adv']}"
            )
        entries.append(
            {
                "name": spec["name"],
                "family": spec["family"],
                "n": n,
                "truth_table": tt,
                "qqo": {"adv_pm": adv, "num_iterations": int(sol["num_iteration"])},
                "known_adv": spec["known_adv"],
            }
        )
        print(f"{spec['name']}: ADV+- = {adv:.6f}")

    fixture = {
        "metadata": {
            "generator": "scripts/generate_qqo_fixtures.py",
            "package": "quantum-query-optimizer",
            "package_version": pkg_version("quantum-query-optimizer"),
            "reference": (
                "Witter & Czekanski, 'QuantumQueryOptimizer', "
                "https://github.com/rtealwitter/QuantumQueryOptimizer; "
                "solves Reichardt's ADV+- SDP, Q2(f) = Theta(ADV+-(f))"
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "command": f"scripts/generate_qqo_fixtures.py {out_path}",
            "truth_table_convention": (
                "t[x] = f(x) with variable i in bit i of x (variable 0 = LSB); "
                "QQO bitstring position i = variable i"
            ),
            "value_semantics": (
                "adv_pm is the optimum of the general adversary SDP, ADV+-(f). "
                "Solver accuracy is ~1e-4 in observed anchor deviation; compare "
                "with tolerance 2e-3. Constant functions are excluded (ADV = 0)."
            ),
            "n_functions": len(entries),
        },
        "functions": entries,
    }
    with open(out_path, "w") as fh:
        json.dump(fixture, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out_path} ({len(entries)} functions)")


if __name__ == "__main__":
    main()
