"""Command Registry for Math Anything REPL.

Manages available commands and provides help documentation.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class CommandInfo:
    """Information about a REPL command."""
    name: str
    description: str
    usage: str
    examples: List[str]
    category: str


class CommandRegistry:
    """Registry of available REPL commands."""
    
    COMMANDS = {
        'load': CommandInfo(
            name='load',
            description='Load input file(s) from a computational engine',
            usage='load <engine> <file1> [file2] ...',
            examples=[
                'load vasp INCAR POSCAR KPOINTS',
                'load lammps simulation.in',
                'load ansys beam_bending.inp',
            ],
            category='File Operations',
        ),
        'extract': CommandInfo(
            name='extract',
            description='Extract mathematical structures from loaded files',
            usage='extract [session_name]',
            examples=[
                'extract',
                'extract vasp_1',
            ],
            category='Extraction',
        ),
        'constraints': CommandInfo(
            name='constraints',
            description='Validate symbolic constraints',
            usage='constraints [session_name]',
            examples=[
                'constraints',
                'constraints vasp_1',
            ],
            category='Validation',
        ),
        'compare': CommandInfo(
            name='compare',
            description='Compare mathematical structures between sessions',
            usage='compare <session1> <session2>',
            examples=[
                'compare vasp_1 vasp_2',
                'compare lammps_1 ansys_1',
            ],
            category='Comparison',
        ),
        'list': CommandInfo(
            name='list',
            description='List all loaded sessions',
            usage='list',
            examples=['list'],
            category='Session Management',
        ),
        'save': CommandInfo(
            name='save',
            description='Save current session to file',
            usage='save <filepath>',
            examples=['save my_session.json'],
            category='Session Management',
        ),
        'export': CommandInfo(
            name='export',
            description='Export schema to JSON file',
            usage='export <session_name> <filepath>',
            examples=['export vasp_1 output.json'],
            category='Export',
        ),
        'exit': CommandInfo(
            name='exit',
            description='Exit the REPL',
            usage='exit',
            examples=['exit', 'quit'],
            category='System',
        ),
        'help': CommandInfo(
            name='help',
            description='Show help information',
            usage='help [command]',
            examples=['help', 'help load'],
            category='System',
        ),
    }
    
    @classmethod
    def get_command(cls, name: str) -> Optional[CommandInfo]:
        """Get command info by name."""
        return cls.COMMANDS.get(name)
    
    @classmethod
    def list_commands(cls) -> List[str]:
        """List all command names."""
        return list(cls.COMMANDS.keys())
    
    @classmethod
    def get_by_category(cls) -> Dict[str, List[CommandInfo]]:
        """Group commands by category."""
        by_category = {}
        for cmd in cls.COMMANDS.values():
            by_category.setdefault(cmd.category, []).append(cmd)
        return by_category
    
    @classmethod
    def format_help(cls, command: Optional[str] = None) -> str:
        """Format help text."""
        if command:
            cmd = cls.COMMANDS.get(command)
            if not cmd:
                return f"Unknown command: {command}"
            
            lines = [
                f"\n{cmd.name} - {cmd.description}",
                f"\nUsage: {cmd.usage}",
                "\nExamples:",
            ]
            for ex in cmd.examples:
                lines.append(f"  {ex}")
            return '\n'.join(lines)
        
        # General help
        lines = ["\nAvailable Commands:", "=" * 60]
        
        for category, cmds in cls.get_by_category().items():
            lines.append(f"\n{category}:")
            for cmd in cmds:
                lines.append(f"  {cmd.name:15} - {cmd.description}")
        
        lines.append("\nUse 'help <command>' for detailed information.")
        return '\n'.join(lines)
