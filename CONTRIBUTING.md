# Contributing to BooFun

Thank you for considering contributing to BooFun! All contributions—code, documentation, bug reports, feature requests, and ideas—are welcome.

## Important Context

**This is a solo-maintained project.** While I welcome contributions, please understand:

- **Response times may vary.** I review issues and PRs as time permits.
- **This is a large library.** A significant portion of the implementation was developed with the assistance of generative AI tools. The design, architecture, testing strategy, and verification are human-led, but given the library's scope, not every edge case has been manually verified.
- **Your help is valuable.** Bug reports, test cases, and documentation improvements are especially appreciated as they help improve the library's reliability.

---

## How Can I Contribute?

### Reporting Bugs

Before opening an issue:
1. **Search existing issues** to avoid duplicates
2. **Check if it's a known limitation** in the ROADMAP.md

When reporting a bug, include:
- Clear steps to reproduce
- What you expected vs. what happened
- Error messages (if any)
- Your environment (OS, Python version, boofun version)

### Suggesting Enhancements

Use GitHub Issues to suggest new features. Please explain:
- What problem does this solve?
- Why is it useful for Boolean function analysis?
- Any ideas for implementation (optional)

### Improving Documentation

Found a typo, unclear explanation, or missing example? PRs for documentation are always welcome and easy to review.

### Code Contributions

1. **Fork** the repository and clone your fork
2. **Create a branch** for your change: `git checkout -b feature/your-feature`
3. **Make your changes** with tests if applicable
4. **Run the test suite**: `pytest tests/`
5. **Submit a pull request**

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/boofun.git
cd boofun

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting (optional but appreciated)
black src/ tests/
flake8 src/ tests/
```

---

## Code Style

- **Formatting**: We use [Black](https://black.readthedocs.io/) with 100-character line length
- **Imports**: Sorted with [isort](https://pycqa.github.io/isort/)
- **Type hints**: Encouraged but not strictly required
- **Docstrings**: Use Google-style docstrings for public functions

---

## Pull Request Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Include tests** for new functionality when possible
- **Update documentation** if you're changing public APIs
- **Reference related issues** in your PR description

---

## Testing

The test suite includes:
- **Unit tests**: `tests/unit/`
- **Integration tests**: `tests/integration/`
- **Property-based tests**: `tests/property/` (using Hypothesis)
- **Benchmarks**: `tests/benchmarks/`

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=boofun tests/

# Run specific category
pytest tests/unit/
pytest tests/property/
```

---

## Areas Where Help is Needed

See [ROADMAP.md](ROADMAP.md) for detailed tracking, but contributions in these areas are especially welcome:

- **Bug reports and test cases** - Help verify edge cases
- **Documentation improvements** - Examples, tutorials, clarifications
- **Performance optimizations** - Profiling and optimization of hot paths
- **New analysis algorithms** - Implementations of results from Boolean function literature

---

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

---

## Questions?

- Open a [GitHub Discussion](https://github.com/GabbyTab/boofun/discussions) for general questions
- Open an [Issue](https://github.com/GabbyTab/boofun/issues) for bugs or feature requests

Thank you for helping make BooFun better!
