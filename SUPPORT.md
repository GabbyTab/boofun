# Support

## Where to ask

- **Bug reports and feature requests**: open an issue on the
  [issue tracker](https://github.com/GabbyTab/boofun/issues). For bugs, please
  include steps to reproduce, expected vs. actual behavior, and your
  environment (OS, Python version, boofun version) — see
  [CONTRIBUTING.md](CONTRIBUTING.md) for details.
- **Usage questions**: post in
  [Discussions → Q&A](https://github.com/GabbyTab/boofun/discussions/categories/q-a).
- **Release news**: watch
  [Discussions → Announcements](https://github.com/GabbyTab/boofun/discussions/categories/announcements)
  or the [releases page](https://github.com/GabbyTab/boofun/releases).

Please do not email the maintainer directly for support; public questions help
the next person with the same problem.

## What to expect

BooFun is maintained by one person on a best-effort basis. Issues and
questions are usually answered within days, not hours. Well-scoped bug
reports with a reproduction get attention first; test cases that demonstrate
a mathematical error are the fastest path to a fix.

## How decisions are made

The maintainer decides scope, conventions, and releases, in the open:
direction lives on the [issue tracker](https://github.com/GabbyTab/boofun/issues)
(see the pinned issues for the current roadmap), and changes land through
pull requests with CI gates (tests, coverage floor, strict mypy, Ruff). If
you want to influence direction, comment on an existing issue or open a new
one — substantial proposals are discussed there before any code is written.

## Stability expectations

- The core API (`boofun.create`, built-in families, `analysis` modules,
  representations) follows semantic versioning; breaking changes are listed
  in the [CHANGELOG](CHANGELOG.md).
- `boofun.quantum_complexity` is **experimental**: it computes classical
  estimates of quantum complexity bounds, is not re-exported from the
  top-level package, and may change without a major version bump.
- Optional extras (`performance`, `gpu`, `visualization`) are supported on a
  best-effort basis; the core library never requires them.
