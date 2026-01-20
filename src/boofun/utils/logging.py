"""
Logging utilities for the BooFun library.

This module provides a consistent logging interface for the library,
enabling users to configure debug logging for troubleshooting.

Usage:
    import logging
    
    # Enable debug logging for BooFun
    logging.getLogger("boofun").setLevel(logging.DEBUG)
    logging.getLogger("boofun").addHandler(logging.StreamHandler())
    
    # Or use the convenience function
    from boofun.utils.logging import enable_debug_logging
    enable_debug_logging()
"""

import logging
from typing import Optional

# Library-wide logger
logger = logging.getLogger("boofun")

# Submodule loggers
core_logger = logging.getLogger("boofun.core")
analysis_logger = logging.getLogger("boofun.analysis")
visualization_logger = logging.getLogger("boofun.visualization")
optimization_logger = logging.getLogger("boofun.optimization")


def enable_debug_logging(
    level: int = logging.DEBUG,
    format_string: Optional[str] = None,
) -> None:
    """
    Enable debug logging for the BooFun library.
    
    This is a convenience function for users who want to see
    detailed logging output for debugging purposes.
    
    Args:
        level: Logging level (default: DEBUG)
        format_string: Custom format string (default: timestamp, level, name, message)
        
    Example:
        >>> from boofun.utils.logging import enable_debug_logging
        >>> enable_debug_logging()
        >>> # Now BooFun will log debug messages
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    
    logger.setLevel(level)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific BooFun submodule.
    
    Args:
        name: Submodule name (e.g., "core.base", "analysis.fourier")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"boofun.{name}")


# Null handler to avoid "No handler found" warnings
logger.addHandler(logging.NullHandler())
