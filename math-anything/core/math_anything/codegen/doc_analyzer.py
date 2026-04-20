"""Documentation Analyzer - Extract commands from documentation.

Parses PDF, HTML, Markdown documentation to extract:
- Command syntax definitions
- Parameter descriptions and types
- Usage examples
- Mathematical constraints mentioned in docs
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DocCommand:
    """Command extracted from documentation."""
    name: str
    syntax: str
    description: str
    parameters: List[Dict[str, Any]]
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class DocumentationAnalyzer:
    """Analyze documentation files to extract command structures.
    
    Supports multiple formats:
    - Markdown (.md)
    - HTML (.html, .htm)
    - Plain text (.txt)
    - PDF (basic text extraction)
    
    Example:
        ```python
        analyzer = DocumentationAnalyzer()
        result = analyzer.analyze("/path/to/manual.md")
        
        for cmd in result['commands']:
            print(f"Command: {cmd['name']}")
            print(f"  Syntax: {cmd['syntax']}")
        ```
    """
    
    # Common command syntax patterns in documentation
    SYNTAX_PATTERNS = [
        # LAMMPS-style: command arg1 arg2 ...
        re.compile(r'^(\w+)\s+([A-Z_][A-Z_0-9]*)\s+(.+)$', re.MULTILINE),
        # Markdown code block
        re.compile(r'```\s*\n?(\w+)[^`]*```', re.DOTALL),
        # Syntax: command
        re.compile(r'[Ss]yntax:\s*`?(\w+)[^`\n]*`?', re.MULTILINE),
        # Command: description
        re.compile(r'^###?\s*(\w+)\s*[-:]\s*(.+)$', re.MULTILINE),
    ]
    
    # Parameter patterns
    PARAM_PATTERNS = [
        # arg (type) - description
        re.compile(r'(\w+)\s*\((\w+)\)\s*[-:]\s*([^\n]+)'),
        # arg = value - description
        re.compile(r'(\w+)\s*=\s*(\S+)\s*[-:]\s*([^\n]+)'),
        # * arg - description
        re.compile(r'[*-]\s+(\w+)\s*[-:]\s*([^\n]+)'),
    ]
    
    # Constraint patterns
    CONSTRAINT_PATTERNS = [
        re.compile(r'(?:must be|requires?|constraint)[:\s]+([^\n.]+)', re.IGNORECASE),
        re.compile(r'([\w\s]+)\s*([<>=]+)\s*([\d.]+)', re.IGNORECASE),
        re.compile(r'range[:\s]+\[?([\d.]+)\s*,\s*([\d.]+)\]?', re.IGNORECASE),
        re.compile(r'(?:positive|negative|non-negative|non-zero)', re.IGNORECASE),
    ]
    
    def __init__(self):
        self.commands: List[DocCommand] = []
    
    def analyze(self, doc_path: str) -> Dict[str, Any]:
        """Analyze documentation file.
        
        Args:
            doc_path: Path to documentation file
            
        Returns:
            Dictionary with extracted commands, parameters, sections
        """
        path = Path(doc_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Documentation not found: {doc_path}")
        
        # Read content
        content = self._read_file(path)
        
        # Determine format and parse
        suffix = path.suffix.lower()
        
        if suffix in ['.md', '.markdown']:
            return self._parse_markdown(content, str(path))
        elif suffix in ['.html', '.htm']:
            return self._parse_html(content, str(path))
        elif suffix == '.pdf':
            return self._parse_pdf(content, str(path))
        else:
            return self._parse_text(content, str(path))
    
    def _read_file(self, path: Path) -> str:
        """Read file content."""
        if path.suffix.lower() == '.pdf':
            # For PDF, we'd use PyPDF2 or similar
            # For now, return placeholder
            return self._extract_pdf_text(path)
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF (placeholder implementation)."""
        # In real implementation, use PyPDF2 or pdfplumber
        # For now, return empty with note
        return "# PDF extraction requires PyPDF2 or pdfplumber\n"
    
    def _parse_markdown(self, content: str, source: str) -> Dict[str, Any]:
        """Parse Markdown documentation."""
        commands = []
        parameters = []
        sections = []
        
        lines = content.split('\n')
        current_section = None
        current_command = None
        
        for i, line in enumerate(lines):
            # Track sections
            if line.startswith('#'):
                section_level = len(line) - len(line.lstrip('#'))
                section_title = line.lstrip('#').strip()
                current_section = {
                    'level': section_level,
                    'title': section_title,
                    'content': '',
                }
                sections.append(current_section)
                continue
            
            if current_section:
                current_section['content'] += line + '\n'
            
            # Extract commands from headers
            if line.startswith('##') or line.startswith('###'):
                cmd_match = re.match(r'#+\s*(\w+)\s*[-:]\s*(.+)', line)
                if cmd_match:
                    cmd_name = cmd_match.group(1)
                    cmd_desc = cmd_match.group(2)
                    
                    # Look for syntax in next lines
                    syntax = self._find_syntax(lines, i)
                    
                    cmd = DocCommand(
                        name=cmd_name,
                        syntax=syntax or cmd_name,
                        description=cmd_desc,
                        parameters=[],
                    )
                    commands.append(cmd)
                    current_command = cmd
            
            # Extract parameters
            for pattern in self.PARAM_PATTERNS:
                match = pattern.search(line)
                if match:
                    param = {
                        'name': match.group(1),
                        'type': match.group(2) if len(match.groups()) > 2 else 'unknown',
                        'description': match.groups()[-1].strip(),
                    }
                    parameters.append(param)
                    
                    if current_command:
                        current_command.parameters.append(param)
            
            # Extract constraints
            for pattern in self.CONSTRAINT_PATTERNS:
                match = pattern.search(line)
                if match and current_command:
                    constraint = match.group(0).strip()
                    current_command.constraints.append(constraint)
        
        return {
            'commands': [
                {
                    'name': c.name,
                    'syntax': c.syntax,
                    'description': c.description,
                    'parameters': c.parameters,
                    'constraints': c.constraints,
                }
                for c in commands
            ],
            'parameters': parameters,
            'sections': [
                {
                    'title': s['title'],
                    'level': s['level'],
                    'preview': s['content'][:200],
                }
                for s in sections
            ],
            'source_path': source,
        }
    
    def _parse_html(self, content: str, source: str) -> Dict[str, Any]:
        """Parse HTML documentation."""
        # Simple HTML tag stripping
        text = re.sub(r'<[^>]+>', '', content)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        
        # Then parse as text
        return self._parse_text(text, source)
    
    def _parse_pdf(self, content: str, source: str) -> Dict[str, Any]:
        """Parse PDF text content."""
        # PDF text is already extracted, parse as text
        return self._parse_text(content, source)
    
    def _parse_text(self, content: str, source: str) -> Dict[str, Any]:
        """Parse plain text documentation."""
        commands = []
        parameters = []
        sections = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for command definitions
            for pattern in self.SYNTAX_PATTERNS:
                match = pattern.search(line)
                if match:
                    cmd_name = match.group(1)
                    cmd_syntax = match.group(0)
                    
                    # Get description from surrounding lines
                    description = ""
                    for j in range(max(0, i-2), min(len(lines), i+3)):
                        if j != i:
                            description += lines[j].strip() + " "
                    
                    cmd = {
                        'name': cmd_name,
                        'syntax': cmd_syntax,
                        'description': description.strip()[:200],
                        'parameters': [],
                    }
                    commands.append(cmd)
            
            # Extract parameters
            for pattern in self.PARAM_PATTERNS:
                match = pattern.search(line)
                if match:
                    param = {
                        'name': match.group(1),
                        'type': match.group(2) if len(match.groups()) > 2 else 'unknown',
                        'description': match.groups()[-1].strip(),
                    }
                    parameters.append(param)
                    
                    # Add to last command
                    if commands:
                        commands[-1]['parameters'].append(param)
        
        # Deduplicate commands
        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd['name'] not in seen:
                seen.add(cmd['name'])
                unique_commands.append(cmd)
        
        return {
            'commands': unique_commands,
            'parameters': parameters,
            'sections': sections,
            'source_path': source,
        }
    
    def _find_syntax(self, lines: List[str], start_idx: int) -> Optional[str]:
        """Find command syntax in lines following start_idx."""
        for i in range(start_idx + 1, min(len(lines), start_idx + 10)):
            line = lines[i].strip()
            
            # Check for code block
            if line.startswith('```'):
                syntax_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    syntax_lines.append(lines[i])
                    i += 1
                return ' '.join(syntax_lines).strip()
            
            # Check for syntax line
            if 'syntax' in line.lower() or line.startswith('`'):
                return line.strip('`').strip()
        
        return None
    
    def infer_from_examples(self, examples: List[str]) -> List[Dict[str, Any]]:
        """Infer command structure from usage examples.
        
        Examples:
            "fix 1 all nvt temp 300 300 100"
            -> {name: "nvt", params: ["temp", "300", "300", "100"]}
        """
        inferred = []
        
        for example in examples:
            parts = example.split()
            if len(parts) >= 2:
                # Try to identify command
                for i, part in enumerate(parts):
                    if part.isalpha() and len(part) > 2:
                        # Likely a command keyword
                        inferred.append({
                            'command': part,
                            'context': parts[:i],
                            'arguments': parts[i+1:],
                            'full_example': example,
                        })
                        break
        
        return inferred


# Convenience function
def quick_analyze_docs(doc_path: str) -> Dict[str, Any]:
    """Quick analysis of documentation."""
    analyzer = DocumentationAnalyzer()
    return analyzer.analyze(doc_path)
