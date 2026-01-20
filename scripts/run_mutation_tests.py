#!/usr/bin/env python3
"""
Mutation Testing Runner for BooFun

This script provides an easy interface for running mutation tests using mutmut.
Mutation testing helps identify weaknesses in the test suite by introducing
small changes (mutants) to the code and checking if tests catch them.

Usage:
    python scripts/run_mutation_tests.py [module]

Examples:
    python scripts/run_mutation_tests.py              # Run on default modules
    python scripts/run_mutation_tests.py core.base    # Run on specific module

After running:
    mutmut results           # Show summary
    mutmut show <id>         # Show specific mutant
    mutmut html              # Generate HTML report
"""

import os
import subprocess
import sys
from pathlib import Path

# Default modules to test (most critical code paths)
DEFAULT_MODULES = [
    "src/boofun/core/base.py",
    "src/boofun/core/spaces.py",
    "src/boofun/analysis/spectral.py",
]

# High-value targets for mutation testing
EXTENDED_MODULES = [
    "src/boofun/analysis/fourier.py",
    "src/boofun/analysis/hypercontractivity.py",
    "src/boofun/core/representations.py",
    "src/boofun/families/theoretical.py",
]


def run_mutation_tests(modules=None, quick=False):
    """Run mutation tests on specified modules."""
    if modules is None:
        modules = DEFAULT_MODULES

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    paths = ",".join(modules)

    cmd = ["mutmut", "run", "--paths-to-mutate", paths]

    if quick:
        # Quick mode: stop at first surviving mutant per file
        cmd.append("--simple-mode")

    print(f"Running mutation tests on: {paths}")
    print("This may take a while...\n")

    result = subprocess.run(cmd, capture_output=False)

    # Show results
    print("\n" + "=" * 60)
    subprocess.run(["mutmut", "results"])

    return result.returncode


def show_summary():
    """Show mutation testing summary."""
    subprocess.run(["mutmut", "results"])


def generate_report():
    """Generate HTML report."""
    subprocess.run(["mutmut", "html"])
    print("\nHTML report generated in htmlcov/")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    if "--summary" in args:
        show_summary()
        sys.exit(0)

    if "--report" in args:
        generate_report()
        sys.exit(0)

    if "--extended" in args:
        modules = DEFAULT_MODULES + EXTENDED_MODULES
        args.remove("--extended")
    elif args:
        # Convert module names to paths
        modules = []
        for arg in args:
            if arg.startswith("src/"):
                modules.append(arg)
            else:
                # Convert dotted module name to path
                path = f"src/boofun/{arg.replace('.', '/')}.py"
                modules.append(path)
    else:
        modules = DEFAULT_MODULES

    quick = "--quick" in args
    sys.exit(run_mutation_tests(modules, quick))
