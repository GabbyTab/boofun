#!/usr/bin/env bash
# Verify that requirements/*.txt lockfiles are consistent with their inputs
# (pyproject.toml extras and requirements/*.in).
#
# Why this exists: CI installs from the lockfiles with `--require-hashes` and
# then installs the package with `pip install --no-deps -e .`, so pip never
# cross-checks the lockfiles against pyproject.toml. Without this guard,
# Dependabot (which treats requirements/ as an independent manifest) or a
# forgotten recompile could silently move a lockfile past an exact pin in
# pyproject.toml — e.g. the mypy/ruff toolchain pins from issue #59.
#
# Mechanism: rerun every `uv pip compile` command from requirements/README.md
# WITHOUT --upgrade. uv keeps versions already pinned in the existing output
# file when they still satisfy the inputs, so new upstream releases do not
# change the output; only a genuine input/lockfile inconsistency does.
# A dirty `git diff` in requirements/ therefore means the lockfiles no longer
# match their inputs: recompile (see requirements/README.md) and commit.
#
# Usage: ./scripts/check_lockfiles.sh  (requires uv and a clean requirements/)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! git diff --quiet -- requirements/; then
    echo "error: requirements/ has uncommitted changes; commit or stash them first" >&2
    exit 1
fi

echo "Recompiling lockfiles with $(uv --version)"

uv pip compile pyproject.toml requirements/ci.in \
    --extra dev --extra visualization --extra performance --extra docs \
    --universal --python-version 3.10 --generate-hashes --quiet \
    -o requirements/ci.txt

uv pip compile pyproject.toml \
    --extra dev --extra visualization \
    --universal --python-version 3.10 --generate-hashes --quiet \
    -o requirements/typecheck.txt

uv pip compile requirements/lint.in \
    --universal --python-version 3.10 --generate-hashes --quiet \
    -o requirements/lint.txt

uv pip compile requirements/publish.in \
    --universal --python-version 3.10 --generate-hashes --quiet \
    -o requirements/publish.txt

uv pip compile pyproject.toml requirements/fuzz.in \
    --extra dev --extra visualization \
    --universal --python-version 3.12 --generate-hashes --quiet \
    -o requirements/fuzz.txt

uv pip compile pyproject.toml requirements/boolforge.in \
    --universal --python-version 3.10 --generate-hashes --quiet \
    -o requirements/boolforge.txt

if git diff --exit-code -- requirements/; then
    echo "Lockfiles are in sync with pyproject.toml and requirements/*.in."
else
    echo >&2
    echo "error: lockfiles are out of sync with their inputs (diff above)." >&2
    echo "Recompile them per requirements/README.md and commit the result." >&2
    exit 1
fi
