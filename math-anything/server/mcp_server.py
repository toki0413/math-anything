"""Deprecated: Use math_anything.mcp_server instead."""
import warnings
warnings.warn(
    "server.mcp_server is deprecated. Use math_anything.mcp_server instead.",
    DeprecationWarning,
    stacklevel=2,
)
from math_anything.mcp_server import *  # noqa: F401,F403
from math_anything.mcp_server import main  # noqa: F401
