"""Additional commands for math-anything CLI."""

from .agent_cmd import cmd_agent
from .config_cmd import cmd_config
from .generate_cmd import cmd_generate
from .rag_cmd import cmd_rag
from .visualize_cmd import cmd_visualize
from .watch_cmd import cmd_watch

__all__ = [
    "cmd_config",
    "cmd_watch",
    "cmd_agent",
    "cmd_generate",
    "cmd_rag",
    "cmd_visualize",
]
