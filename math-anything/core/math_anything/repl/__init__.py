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

from .core import MathAnythingREPL, REPLSession
from .commands import CommandRegistry
from .diff import MathDiff, DiffResult

__all__ = [
    "MathAnythingREPL",
    "REPLSession",
    "CommandRegistry",
    "MathDiff",
    "DiffResult",
]
