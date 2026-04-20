"""Source Code Analyzer - Static analysis for Harness auto-generation.

Analyzes C++, Python, Java source code to extract:
- Command keywords and syntax patterns
- Parameter types and ranges
- Mathematical expressions in comments
- Variable relationships

IMPORTANT: This is a heuristic extractor based on pattern matching.
It does NOT perform full semantic analysis or compilation.
All extractions should be considered "best effort" and require validation.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ExtractionConfidence(Enum):
    """Confidence level for extracted items."""
    HIGH = "high"      # Clear pattern match with context
    MEDIUM = "medium"  # Pattern match but limited context
    LOW = "low"        # Weak pattern match, requires validation
    HEURISTIC = "heuristic"  # Best guess, definitely needs review


@dataclass
class ExtractedCommand:
    """A command extracted from source code."""
    name: str
    pattern: str
    parameters: List[Dict[str, Any]]
    description: str = ""
    source_file: str = ""
    line_number: int = 0
    language: str = ""
    confidence: ExtractionConfidence = ExtractionConfidence.HEURISTIC
    extraction_method: str = "pattern_match"  # How this was extracted


@dataclass
class ExtractedParameter:
    """A parameter with inferred constraints."""
    name: str
    param_type: str  # int, float, string, vector, etc.
    default_value: Optional[str] = None
    description: str = ""
    constraints: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0
    confidence: ExtractionConfidence = ExtractionConfidence.HEURISTIC


@dataclass
class SourceAnalysisResult:
    """Result of source code analysis with coverage metrics."""
    commands: List[ExtractedCommand]
    parameters: List[ExtractedParameter]
    equations: List[Dict[str, Any]]
    command_contexts: Dict[str, List[str]]
    source_path: str
    file_extensions: List[str]
    coverage_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class SourceCodeAnalyzer:
    """Static analyzer for simulation software source code.
    
    WARNING: This analyzer uses REGEX-BASED HEURISTICS, not full parsing.
    It is designed for "zero-intrusion" extraction without compilation,
    but has inherent limitations:
    
    1. Cannot handle complex C++ templates or macros
    2. May miss commands defined through metaprogramming
    3. Type inference is approximate (uses string matching)
    4. Confidence varies - see ExtractionConfidence enum
    
    ALL EXTRACTIONS REQUIRE HUMAN VALIDATION.
    
    Example:
        ```python
        analyzer = SourceCodeAnalyzer()
        result = analyzer.analyze(
            source_dir="/path/to/lammps/src",
            file_patterns=["*.cpp", "*.h"],
        )
        
        print(f"Coverage: {result.coverage_metrics}")
        print(f"Warnings: {result.warnings}")
        
        for cmd in result.commands:
            print(f"Command: {cmd.name} (confidence: {cmd.confidence.value})")
            print(f"  Pattern: {cmd.pattern}")
            print(f"  Extraction method: {cmd.extraction_method}")
        ```
    """
    
    # Coverage limitations - known patterns we CANNOT reliably extract
    COVERAGE_LIMITATIONS = {
        'cpp': [
            "Template metaprogramming (e.g., template <typename T> class...)",
            "Macro-generated code (e.g., #define FIX_CLASS(name) ...)",
            "Dynamic dispatch through function pointers",
            "Commands defined in separate compilation units",
        ],
        'python': [
            "Dynamically generated classes (e.g., type('DynamicClass', ...))",
            "Commands registered via decorators with complex logic",
            "Import-time code generation",
        ],
        'java': [
            "Reflection-based command registration",
            "Annotation processors",
        ],
    }
    
    # Common patterns for command extraction
    COMMAND_PATTERNS = {
        'cpp': {
            'class_command': re.compile(
                r'class\s+(\w+)\s*:\s*public\s+(?:Fix|Compute|Pair|Region|Dump)',
                re.IGNORECASE
            ),
            'arg_parsing': re.compile(
                r'(?:arg|args)\[(\d+)\]\s*[=)]',
                re.IGNORECASE
            ),
            'force_cast': re.compile(
                r'force->\w+\s*\(\s*["\']([^"\']+)["\']\s*\)',
                re.IGNORECASE
            ),
            'error_message': re.compile(
                r'error->\w+\([^)]*["\']([^"\']+(?: illegal| expected| must be))',
                re.IGNORECASE
            ),
            'variable_decl': re.compile(
                r'(\w+)\s+(\w+)\s*=\s*([^;]+);\s*//\s*(.+)',
                re.IGNORECASE
            ),
        },
        'python': {
            'class_command': re.compile(
                r'class\s+(\w+)\s*\([^)]*(?:Command|Parser|Handler)',
                re.IGNORECASE
            ),
            'arg_parsing': re.compile(
                r'self\.(\w+)\s*=\s*(?:args|argv|params)\[(\d+)\]',
                re.IGNORECASE
            ),
            'click_command': re.compile(
                r'@click\.(?:command|option|argument)[^\n]*\n+def\s+(\w+)',
                re.DOTALL | re.IGNORECASE
            ),
            'argparse': re.compile(
                r'add_argument\s*\(\s*["\']([^"\']+)["\']',
                re.IGNORECASE
            ),
        },
        'java': {
            'class_command': re.compile(
                r'class\s+(\w+)\s+(?:extends|implements)\s+(?:Command|Parser)',
                re.IGNORECASE
            ),
            'method_param': re.compile(
                r'public\s+\w+\s+(\w+)\s*\([^)]*String\s+(\w+)',
                re.IGNORECASE
            ),
        },
    }
    
    # Mathematical keywords for semantic detection
    MATH_KEYWORDS = {
        'equation': ['equation', 'formula', 'expression', 'governing'],
        'boundary': ['boundary', 'bc', 'condition', 'constraint'],
        'discretization': ['mesh', 'grid', 'element', 'node', 'finite'],
        'solver': ['solver', 'iteration', 'convergence', 'tolerance'],
        'tensor': ['tensor', 'stress', 'strain', 'vector', 'matrix'],
        'integral': ['integral', 'integration', 'quadrature'],
        'differential': ['differential', 'derivative', 'gradient', 'laplacian'],
    }
    
    def __init__(self):
        self.results: List[SourceAnalysisResult] = []
        self._files_analyzed: Set[str] = set()
        self._warnings: List[str] = []
    
    def analyze(
        self,
        source_dir: str,
        file_patterns: List[str] = None,
        entry_hints: List[str] = None,
    ) -> Dict[str, Any]:
        """Analyze source code directory.
        
        Args:
            source_dir: Path to source code
            file_patterns: File patterns to include (e.g., ['*.cpp', '*.h'])
            entry_hints: Keywords to identify command entry points
            
        Returns:
            Dictionary with extracted commands, parameters, equations, contexts
            AND coverage metrics and warnings about limitations.
        """
        self._warnings = []  # Reset warnings
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        file_patterns = file_patterns or ['*.cpp', '*.c', '*.h', '*.py', '*.java']
        entry_hints = entry_hints or ['command', 'fix', 'compute', 'pair', 'dump']
        
        commands: List[ExtractedCommand] = []
        parameters: List[ExtractedParameter] = []
        equations: List[Dict[str, Any]] = []
        command_contexts: Dict[str, List[str]] = {}
        
        # Collect all source files
        source_files = []
        for pattern in file_patterns:
            source_files.extend(source_path.rglob(pattern))
        
        print(f"   Analyzing {len(source_files)} files...")
        
        # Calculate coverage metrics
        files_by_language: Dict[str, int] = {}
        processed_files = 0
        
        # Analyze each file
        for file_path in source_files:
            try:
                file_cmds, file_params, file_eqs = self._analyze_file(
                    file_path, entry_hints
                )
                commands.extend(file_cmds)
                parameters.extend(file_params)
                equations.extend(file_eqs)
                processed_files += 1
                
                # Track language stats
                lang = self._detect_language(file_path)
                files_by_language[lang] = files_by_language.get(lang, 0) + 1
                
                # Build command contexts
                for cmd in file_cmds:
                    if cmd.name not in command_contexts:
                        command_contexts[cmd.name] = []
                    command_contexts[cmd.name].append(str(file_path))
                    
            except Exception as e:
                # Continue on error - zero-judgment principle
                self._warnings.append(f"Failed to analyze {file_path}: {str(e)[:50]}")
                continue
        
        # Deduplicate commands
        seen_commands = set()
        unique_commands = []
        for cmd in commands:
            if cmd.name not in seen_commands:
                seen_commands.add(cmd.name)
                unique_commands.append(cmd)
        
        # Get file extensions
        extensions = list(set(
            f.suffix for f in source_files if f.suffix
        ))
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage(
            total_files=len(source_files),
            processed_files=processed_files,
            commands_found=len(unique_commands),
            parameters_found=len(parameters),
            equations_found=len(equations),
            files_by_language=files_by_language,
        )
        
        # Add coverage warnings
        if coverage_metrics['estimated_command_coverage'] < 0.3:
            self._warnings.append(
                "LOW COMMAND COVERAGE: Found few commands. "
                "Source may use patterns not covered by this analyzer."
            )
        
        return {
            'commands': [
                {
                    'name': c.name,
                    'pattern': c.pattern,
                    'parameters': c.parameters,
                    'description': c.description,
                    'source_file': c.source_file,
                    'line_number': c.line_number,
                    'language': c.language,
                    'confidence': c.confidence.value,
                    'extraction_method': c.extraction_method,
                }
                for c in unique_commands
            ],
            'parameters': [
                {
                    'name': p.name,
                    'type': p.param_type,
                    'default': p.default_value,
                    'description': p.description,
                    'constraints': p.constraints,
                    'confidence': p.confidence.value,
                }
                for p in parameters
            ],
            'equations': equations,
            'command_contexts': command_contexts,
            'source_path': source_dir,
            'file_extensions': extensions,
            'coverage_metrics': coverage_metrics,
            'warnings': self._warnings,
            'limitations': self.COVERAGE_LIMITATIONS,
        }
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        return {
            '.cpp': 'cpp', '.c': 'cpp', '.h': 'cpp', '.hpp': 'cpp',
            '.py': 'python',
            '.java': 'java',
        }.get(suffix, 'unknown')
    
    def _calculate_coverage(
        self,
        total_files: int,
        processed_files: int,
        commands_found: int,
        parameters_found: int,
        equations_found: int,
        files_by_language: Dict[str, int],
    ) -> Dict[str, Any]:
        """Calculate extraction coverage metrics."""
        file_success_rate = processed_files / total_files if total_files > 0 else 0
        
        # Heuristic coverage scores
        # These are estimates - real coverage can only be determined by domain expert
        command_coverage = min(1.0, commands_found / 20)  # Assume ~20 major commands typical
        param_coverage = min(1.0, parameters_found / 50)  # Assume ~50 parameters typical
        
        return {
            'total_files_scanned': total_files,
            'files_successfully_processed': processed_files,
            'file_success_rate': round(file_success_rate, 2),
            'commands_found': commands_found,
            'parameters_found': parameters_found,
            'equations_found': equations_found,
            'estimated_command_coverage': round(command_coverage, 2),
            'estimated_parameter_coverage': round(param_coverage, 2),
            'files_by_language': files_by_language,
            'method': 'regex_heuristic',
            'reliability': 'medium',  # Overall reliability assessment
            'note': 'Coverage is estimated. Real coverage requires domain expert validation.',
        }
    
    def _analyze_file(
        self,
        file_path: Path,
        entry_hints: List[str],
    ) -> Tuple[List[ExtractedCommand], List[ExtractedParameter], List[Dict]]:
        """Analyze a single source file."""
        commands = []
        parameters = []
        equations = []
        
        language = self._detect_language(file_path)
        
        if language == 'unknown':
            return commands, parameters, equations
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception:
            return commands, parameters, equations
        
        patterns = self.COMMAND_PATTERNS.get(language, {})
        
        # Extract class-based commands (Fix, Compute, etc.)
        if 'class_command' in patterns:
            for match in patterns['class_command'].finditer(content):
                class_name = match.group(1)
                
                # Check if it matches entry hints
                is_command = any(
                    hint.lower() in class_name.lower()
                    for hint in entry_hints
                )
                
                if is_command:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Extract description from preceding comments
                    description = self._extract_description(lines, line_num)
                    
                    # Calculate confidence
                    confidence = ExtractionConfidence.HIGH if description else ExtractionConfidence.MEDIUM
                    
                    cmd = ExtractedCommand(
                        name=class_name,
                        pattern=class_name.lower(),
                        parameters=[],
                        description=description,
                        source_file=str(file_path),
                        line_number=line_num,
                        language=language,
                        confidence=confidence,
                        extraction_method='class_inheritance_pattern',
                    )
                    commands.append(cmd)
        
        # Extract parameters from argument parsing
        if 'arg_parsing' in patterns:
            for match in patterns['arg_parsing'].finditer(content):
                param_idx = match.group(1)
                
                # Find context
                line_num = content[:match.start()].count('\n') + 1
                context_line = lines[line_num - 1] if line_num <= len(lines) else ""
                
                # Try to infer parameter name from context
                param_name = self._infer_param_name(context_line, param_idx)
                param_type = self._infer_param_type(context_line)
                
                param = ExtractedParameter(
                    name=param_name or f"arg_{param_idx}",
                    param_type=param_type,
                    description=self._extract_description(lines, line_num),
                    source_file=str(file_path),
                    line_number=line_num,
                    confidence=ExtractionConfidence.HEURISTIC,  # Type inference is uncertain
                )
                parameters.append(param)
        
        # Extract mathematical equations from comments
        equations = self._extract_equations_from_comments(lines, str(file_path))
        
        # Enrich commands with parameters
        for cmd in commands:
            cmd.parameters = [
                {'name': p.name, 'type': p.param_type, 'confidence': p.confidence.value}
                for p in parameters
                if p.source_file == str(file_path)
            ]
        
        return commands, parameters, equations
    
    def _extract_description(self, lines: List[str], line_num: int) -> str:
        """Extract description from preceding comments."""
        description = ""
        
        # Look at previous lines for comments
        for i in range(max(0, line_num - 5), line_num):
            line = lines[i].strip()
            if line.startswith('//') or line.startswith('#') or line.startswith('/*') or line.startswith('*'):
                comment = line.lstrip('/#* ')
                if comment and not comment.startswith('Copyright'):
                    description = comment + " " + description
        
        return description.strip()[:200]
    
    def _infer_param_name(self, line: str, idx: str) -> Optional[str]:
        """Infer parameter name from code context."""
        # Look for variable assignments like: double x = arg[0];
        patterns = [
            r'(?:double|int|float|string)\s+(\w+)\s*=\s*arg',
            r'(\w+)\s*=\s*(?:atof|atoi)\s*\(\s*arg',
            r'self\.(\w+)\s*=',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _infer_param_type(self, line: str) -> str:
        """Infer parameter type from code context."""
        line_lower = line.lower()
        
        if 'atof' in line_lower or 'double' in line_lower:
            return 'float'
        elif 'atoi' in line_lower or 'int' in line_lower:
            return 'int'
        elif 'strcmp' in line_lower or 'string' in line_lower:
            return 'string'
        elif '[' in line and ']' in line:
            return 'vector'
        
        return 'unknown'
    
    def _extract_equations_from_comments(
        self,
        lines: List[str],
        source_file: str,
    ) -> List[Dict[str, Any]]:
        """Extract mathematical equations from comments."""
        equations = []
        
        equation_patterns = [
            # LaTeX-style equations
            r'\$\$([^$]+)\$\$',
            r'\\\[([^\\]+)\\\]',
            # Inline math
            r'\$([^$]+)\$',
            # Equation with = sign in comment
            r'(?:equation|formula):\s*([^\n]+=[^\n]+)',
        ]
        
        for line_num, line in enumerate(lines, 1):
            # Check if line contains math-related keywords
            has_math_keyword = any(
                keyword in line.lower()
                for keywords in self.MATH_KEYWORDS.values()
                for keyword in keywords
            )
            
            if has_math_keyword:
                for pattern in equation_patterns:
                    for match in re.finditer(pattern, line, re.IGNORECASE):
                        eq_text = match.group(1).strip()
                        if len(eq_text) > 3:  # Avoid short matches
                            equations.append({
                                'form': eq_text,
                                'source_file': source_file,
                                'line_number': line_num,
                                'context': line.strip()[:100],
                                'type': 'inferred_from_comment',
                                'confidence': 'medium',  # Comment-based is less reliable
                            })
        
        return equations
    
    def infer_constraints(
        self,
        parameter: ExtractedParameter,
        context_lines: List[str],
    ) -> List[str]:
        """Infer mathematical constraints from code context.
        
        Extracts patterns like:
        - "if (x > 0)" -> "x > 0"
        - "must be positive" -> "x > 0"
        - "range [0, 1]" -> "0 <= x <= 1"
        """
        constraints = []
        param_name = parameter.name
        
        constraint_patterns = [
            (rf'if\s*\(\s*{param_name}\s*([<>=]+)\s*([\d.]+)', lambda m: f"{param_name} {m.group(1)} {m.group(2)}"),
            (rf'{param_name}\s*must be (greater than|less than|positive|negative)', 
             lambda m: f"{param_name} > 0" if 'positive' in m.group(1) else f"{param_name} < 0"),
            (rf'range\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
             lambda m: f"{m.group(1)} <= {param_name} <= {m.group(2)}"),
        ]
        
        for line in context_lines:
            for pattern, extractor in constraint_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        constraint = extractor(match)
                        constraints.append(constraint)
                    except Exception:
                        continue
        
        return constraints


# Convenience function
def quick_analyze(source_dir: str) -> Dict[str, Any]:
    """Quick analysis of source directory with coverage warnings."""
    analyzer = SourceCodeAnalyzer()
    result = analyzer.analyze(source_dir)
    
    # Print warnings
    if result.get('warnings'):
        print("\n⚠️  Analysis Warnings:")
        for warning in result['warnings'][:5]:  # Limit to first 5
            print(f"   - {warning}")
    
    # Print coverage
    coverage = result.get('coverage_metrics', {})
    print(f"\n📊 Coverage Metrics:")
    print(f"   Files: {coverage.get('files_successfully_processed', 0)}/{coverage.get('total_files_scanned', 0)}")
    print(f"   Commands: {coverage.get('commands_found', 0)} (est. {coverage.get('estimated_command_coverage', 0)*100:.0f}% coverage)")
    print(f"   Reliability: {coverage.get('reliability', 'unknown')}")
    
    return result
