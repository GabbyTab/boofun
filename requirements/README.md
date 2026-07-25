# CI lockfiles

Hash-pinned requirements for GitHub Actions, addressing OSSF Scorecard's
`Pinned-Dependencies` check ([#58](https://github.com/GabbyTab/boofun/issues/58))
and keeping the CI toolchain (mypy, ruff, pytest) at known versions
([#59](https://github.com/GabbyTab/boofun/issues/59)).

## Policy

- CI installs dependencies with `pip install --require-hashes -r <lockfile>`,
  then installs the package itself with `pip install --no-deps -e .`.
- The `*.in` files (plus `pyproject.toml` extras) are the human-edited inputs;
  the `*.txt` lockfiles are compiled artifacts — never edit them by hand.
- Lockfiles are compiled with [uv](https://docs.astral.sh/uv/) in `--universal`
  mode so a single file resolves on all CI platforms (Linux/macOS/Windows,
  Python 3.10–3.13).
- Dependabot updates the lockfiles weekly. `pyproject.toml` keeps version
  *ranges* — end users are not pinned; only CI is.

## Files

| Lockfile      | Used by                                                    |
|---------------|------------------------------------------------------------|
| `ci.txt`      | typecheck, test, notebooks, mutation, benchmark, docs jobs |
| `lint.txt`    | lint job (pre-commit)                                      |
| `publish.txt` | publish job (build, twine, sigstore)                       |
| `fuzz.txt`    | fuzz workflow (atheris is Linux-only, hence separate)      |

## Regenerating

After changing an `.in` file or the extras in `pyproject.toml`:

```bash
uv pip compile pyproject.toml requirements/ci.in \
    --extra dev --extra visualization --extra performance --extra docs \
    --universal --python-version 3.10 --generate-hashes \
    -o requirements/ci.txt

uv pip compile requirements/lint.in \
    --universal --python-version 3.10 --generate-hashes \
    -o requirements/lint.txt

uv pip compile requirements/publish.in \
    --universal --python-version 3.10 --generate-hashes \
    -o requirements/publish.txt

uv pip compile pyproject.toml requirements/fuzz.in \
    --extra dev --extra visualization \
    --universal --python-version 3.10 --generate-hashes \
    -o requirements/fuzz.txt
```

Add `--upgrade` to any of the commands to refresh all pins.

**`--python-version 3.10` is required.** Without it, uv treats the Python
version of your local interpreter as the lower bound of the "universal"
resolution and silently drops packages (and marker bounds) that only apply
to older interpreters, producing a lockfile that fails to install on part
of the CI matrix.
