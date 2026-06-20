"""Math Anything REPL - Unified Interactive Interface.

Inspired by CLI-Anything's ReplSkin, this module provides a consistent
interactive experience across all computational engines.

Features:
- Unified command interface for all harnesses
- Session state management
- Mathematical semantic extraction
- Cross-engine comparison
- Colored output and progress indicators
"""

from .commands import CommandRegistry
from .core import MathAnythingREPL, REPLSession
from .diff import DiffResult, MathDiff

__all__ = [
    "MathAnythingREPL",
    "REPLSession",
    "CommandRegistry",
    "MathDiff",
    "DiffResult",
]
