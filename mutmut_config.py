"""
Mutmut configuration for mutation testing.

This configuration limits mutation testing to critical modules
to keep CI runtime reasonable.
"""

# Modules to mutate (critical paths only)
# Full mutation testing on the entire codebase would take too long
CRITICAL_PATHS = [
    "src/boofun/core/optimizations.py",
    "src/boofun/core/builtins.py",
    "src/boofun/analysis/fourier.py",
]


def pre_mutation(context):
    """Hook called before each mutation."""
    # Skip mutations in:
    # - Test files
    # - __init__.py (mostly imports)
    # - Visualization code (UI, not critical)
    # - Quantum code (optional feature)
    if "__pycache__" in context.filename:
        context.skip = True
    if "/tests/" in context.filename:
        context.skip = True
    if "visualization" in context.filename:
        context.skip = True
    if "quantum" in context.filename:
        context.skip = True
    if context.filename.endswith("__init__.py"):
        context.skip = True


def mutmut_config(arg, cwd, config_file):
    """Configure mutmut runtime options."""
    # Use pytest as the test runner
    return {
        "runner": "pytest -x --tb=no -q",
        "tests_dir": "tests/",
    }
