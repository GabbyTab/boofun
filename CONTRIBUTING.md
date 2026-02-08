# Contributing

## Context

This is a solo-maintained project. I built it while studying Boolean function analysis and wanted tools that didn't exist. It's large, partially AI-assisted, and I haven't verified every edge case.

Your contributions (bug reports, test cases, corrections, improvements) directly improve reliability. I treat contributions seriously because they catch things I missed.

Response times vary. I'll review when I can.

## How to Help

### Report Bugs

Before opening an issue:
1. Search existing issues
2. Check [ROADMAP.md](ROADMAP.md) for known limitations

Include:
- Steps to reproduce
- Expected vs. actual behavior
- Error message (if any)
- Environment: OS, Python version, boofun version

### Suggest Features

Open an issue. Explain what problem it solves and why it matters for Boolean function analysis.

### Fix Documentation

Corrections, clarifications, and examples welcome. These are easy to merge.

### Submit Code

```bash
git clone https://github.com/YOUR_USERNAME/boofun.git
cd boofun
pip install -e ".[dev]"
pytest tests/
```

Then:
1. Create a branch: `git checkout -b fix/your-fix`
2. Make changes, add tests if applicable
3. Run `pytest tests/` and `black src/`
4. Submit PR

Keep PRs focused. One fix or feature per PR.

## Code Style

- **Format:** Black, 100-char lines
- **Imports:** isort
- **Docstrings:** Google style for public functions
- **Types:** Encouraged, not required

## Testing

```bash
pytest tests/                    # All tests
pytest --cov=boofun tests/       # With coverage
pytest tests/unit/               # Just unit tests
pytest tests/property/           # Property-based (Hypothesis)
```

Coverage is incomplete. If you add tests for uncovered paths, that's valuable.

## Where Help Matters Most

- **Test cases** for edge cases and untested paths
- **Bug reports** with reproducible examples
- **Corrections** to mathematical errors
- **Documentation** that clarifies the non-obvious

See [ROADMAP.md](ROADMAP.md) for specific areas.

## Intellectual Standards

We value:
- **Precision** over enthusiasm
- **Correctness** over completeness
- **Honesty** about what we don't know
- **Good-faith criticism** and being open to being wrong

If you find an error, say so directly. If I'm wrong, I want to know.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Treat each other as capable adults engaging in good faith.

## Questions

- [GitHub Discussions](https://github.com/GabbyTab/boofun/discussions) for questions
- [Issues](https://github.com/GabbyTab/boofun/issues) for bugs

For the full development guide (setup, architecture, code style, test guidelines), see the [full contributing docs](https://gabbytab.github.io/boofun/CONTRIBUTING.html).
