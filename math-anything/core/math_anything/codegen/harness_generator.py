"""Harness Auto-Generator - Main Entry Point.

Generates Math Anything Harness from software source code or documentation.
Follows zero-intrusion, zero-judgment principles.

Workflow:
1. Input: Software source directory or documentation
2. Static Analysis: Extract commands, parameters, types
3. Semantic Mapping: Identify mathematical concepts
4. Constraint Inference: Extract symbolic relationships
5. Output: Ready-to-review Harness skeleton
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import Jinja2, fall back to string template if not available
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .constraint_inference import ConstraintInference
from .doc_analyzer import DocumentationAnalyzer
from .semantic_mapper import SemanticMapper
from .source_analyzer import SourceCodeAnalyzer


@dataclass
class HarnessTemplate:
    """Generated Harness template."""

    engine_name: str
    engine_version: str
    source_path: str
    extracted_commands: List[Dict[str, Any]]
    mathematical_mappings: Dict[str, Any]
    constraint_expressions: List[str]
    generated_code: str
    review_checklist: List[str]
    extraction_confidence: Dict[str, float]


class HarnessAutoGenerator:
    """Auto-generate Harness from software source/documentation.

    Uses Jinja2 templates for clean code generation with proper escaping.
    Falls back to string formatting if Jinja2 not available.

    Example:
        ```python
        generator = HarnessAutoGenerator()

        # From source code
        template = generator.generate_from_source(
            source_dir="/path/to/lammps/src",
            engine_name="lammps",
            file_patterns=["*.cpp", "*.h"],
        )

        # From documentation
        template = generator.generate_from_docs(
            doc_path="/path/to/manual.pdf",
            engine_name="gromacs",
        )

        # Save generated Harness
        generator.save_harness(template, output_dir="./generated")
        ```
    """

    def __init__(self):
        self.source_analyzer = SourceCodeAnalyzer()
        self.doc_analyzer = DocumentationAnalyzer()
        self.semantic_mapper = SemanticMapper()
        self.constraint_inference = ConstraintInference()

        # Setup Jinja2 environment
        self.jinja_env = None
        if JINJA2_AVAILABLE:
            template_dir = Path(__file__).parent / "templates"
            if template_dir.exists():
                self.jinja_env = Environment(
                    loader=FileSystemLoader(str(template_dir)),
                    autoescape=select_autoescape(["html", "xml"]),
                    trim_blocks=True,
                    lstrip_blocks=True,
                )

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate extraction confidence metrics."""
        commands = analysis.get("commands", [])
        parameters = analysis.get("parameters", [])
        equations = analysis.get("equations", [])

        # Heuristic confidence calculation
        cmd_confidence = min(100, max(30, len(commands) * 10)) if commands else 0
        param_confidence = min(100, max(30, len(parameters) * 5)) if parameters else 0
        eq_confidence = min(100, max(20, len(equations) * 15)) if equations else 0

        overall = (cmd_confidence + param_confidence + eq_confidence) / 3

        return {
            "command_extraction": round(cmd_confidence, 1),
            "parameter_extraction": round(param_confidence, 1),
            "equation_mapping": round(eq_confidence, 1),
            "overall": round(overall, 1),
        }

    def generate_from_source(
        self,
        source_dir: str,
        engine_name: str,
        engine_version: str = "1.0.0",
        file_patterns: List[str] = None,
        entry_point_hints: List[str] = None,
    ) -> HarnessTemplate:
        """Generate Harness from software source code.

        Args:
            source_dir: Path to source code directory
            engine_name: Name of the engine
            engine_version: Version string
            file_patterns: File patterns to analyze (e.g., ['*.cpp', '*.h'])
            entry_point_hints: Hints for command entry points

        Returns:
            HarnessTemplate ready for review
        """
        print(f"🔍 Analyzing source: {source_dir}")

        # Step 1: Static analysis
        analysis = self.source_analyzer.analyze(
            source_dir=source_dir,
            file_patterns=file_patterns or ["*.cpp", "*.c", "*.h", "*.py", "*.java"],
            entry_hints=entry_point_hints,
        )

        print(f"   Found {len(analysis['commands'])} commands")
        print(f"   Found {len(analysis['parameters'])} parameters")

        # Step 2: Semantic mapping
        print("🧠 Mapping mathematical semantics...")
        mappings = self.semantic_mapper.map_commands(
            commands=analysis["commands"],
            parameters=analysis["parameters"],
            equations=analysis.get("equations", []),
        )

        # Step 3: Constraint inference
        print("📐 Inferring symbolic constraints...")
        constraints = self.constraint_inference.infer(
            parameters=analysis["parameters"],
            command_contexts=analysis["command_contexts"],
        )

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(analysis)
        print(f"   Overall extraction confidence: {confidence['overall']}%")

        # Step 5: Generate code
        print("📝 Generating Harness skeleton...")
        code = self._generate_harness_code(
            engine_name=engine_name,
            engine_version=engine_version,
            analysis=analysis,
            mappings=mappings,
            constraints=constraints,
            confidence=confidence,
        )

        # Step 6: Create review checklist
        checklist = self._create_review_checklist(analysis, mappings, confidence)

        return HarnessTemplate(
            engine_name=engine_name,
            engine_version=engine_version,
            source_path=source_dir,
            extracted_commands=analysis["commands"],
            mathematical_mappings=mappings,
            constraint_expressions=constraints,
            generated_code=code,
            review_checklist=checklist,
            extraction_confidence=confidence,
        )

    def generate_from_docs(
        self,
        doc_path: str,
        engine_name: str,
        engine_version: str = "1.0.0",
    ) -> HarnessTemplate:
        """Generate Harness from documentation.

        Args:
            doc_path: Path to documentation (PDF, HTML, Markdown)
            engine_name: Name of the engine
            engine_version: Version string

        Returns:
            HarnessTemplate ready for review
        """
        print(f"📚 Analyzing documentation: {doc_path}")

        # Analyze documentation
        analysis = self.doc_analyzer.analyze(doc_path)

        print(f"   Found {len(analysis['commands'])} commands")
        print(f"   Found {len(analysis['parameters'])} parameters")

        # Semantic mapping
        mappings = self.semantic_mapper.map_commands(
            commands=analysis["commands"],
            parameters=analysis["parameters"],
        )

        # Constraint inference
        constraints = self.constraint_inference.infer_from_docs(
            parameters=analysis["parameters"],
            doc_sections=analysis.get("sections", []),
        )

        # Calculate confidence (lower for doc-based extraction)
        confidence = self._calculate_confidence(analysis)
        confidence["overall"] = round(
            confidence["overall"] * 0.8, 1
        )  # Doc extraction is less reliable

        # Generate code
        code = self._generate_harness_code(
            engine_name=engine_name,
            engine_version=engine_version,
            analysis=analysis,
            mappings=mappings,
            constraints=constraints,
            confidence=confidence,
        )

        checklist = self._create_review_checklist(analysis, mappings, confidence)

        return HarnessTemplate(
            engine_name=engine_name,
            engine_version=engine_version,
            source_path=doc_path,
            extracted_commands=analysis["commands"],
            mathematical_mappings=mappings,
            constraint_expressions=constraints,
            generated_code=code,
            review_checklist=checklist,
            extraction_confidence=confidence,
        )

    def _generate_harness_code(
        self,
        engine_name: str,
        engine_version: str,
        analysis: Dict[str, Any],
        mappings: Dict[str, Any],
        constraints: List[str],
        confidence: Dict[str, float],
    ) -> str:
        """Generate Harness Python code using Jinja2 or fallback."""

        if self.jinja_env:
            return self._generate_with_jinja(
                engine_name, engine_version, analysis, mappings, constraints, confidence
            )
        else:
            return self._generate_with_fallback(
                engine_name, engine_version, analysis, mappings, constraints, confidence
            )

    def _generate_with_jinja(
        self,
        engine_name: str,
        engine_version: str,
        analysis: Dict[str, Any],
        mappings: Dict[str, Any],
        constraints: List[str],
        confidence: Dict[str, float],
    ) -> str:
        """Generate code using Jinja2 template."""
        template = self.jinja_env.get_template("harness_template.py.j2")

        # Prepare constraint variables
        constraint_vars = []
        for constraint in constraints[:5]:
            vars_list = self._extract_variables(constraint)
            constraint_vars.append(vars_list)

        # Prepare confidence metrics
        confidence_metrics = [
            {
                "name": "Command Extraction",
                "value": confidence.get("command_extraction", 0),
            },
            {
                "name": "Parameter Extraction",
                "value": confidence.get("parameter_extraction", 0),
            },
            {
                "name": "Equation Mapping",
                "value": confidence.get("equation_mapping", 0),
            },
            {"name": "Overall", "value": confidence.get("overall", 0)},
        ]

        return template.render(
            engine_name=engine_name,
            engine_version=engine_version,
            source_path=analysis.get("source_path", "unknown"),
            generated_at=datetime.now().isoformat(),
            overall_confidence=confidence.get("overall", 0),
            class_name=self._to_class_name(engine_name),
            file_extensions=analysis.get("file_extensions", [".inp", ".dat"]),
            commands=analysis.get("commands", []),
            parameters=analysis.get("parameters", []),
            equations=mappings.get("equations", []),
            constraints=constraints,
            constraint_vars=constraint_vars,
            constraint_confidence=0.6,  # Default confidence for constraints
            confidence_metrics=confidence_metrics,
        )

    def _generate_with_fallback(
        self,
        engine_name: str,
        engine_version: str,
        analysis: Dict[str, Any],
        mappings: Dict[str, Any],
        constraints: List[str],
        confidence: Dict[str, float],
    ) -> str:
        """Fallback code generation without Jinja2."""
        # Simplified fallback - just return a warning comment with basic structure
        code_lines = [
            f'"""Auto-generated Harness for {engine_name} (FALLBACK MODE).',
            "",
            "WARNING: Jinja2 not available. Install with: pip install jinja2",
            f"Engine: {engine_name}",
            f"Version: {engine_version}",
            f'Confidence: {confidence.get("overall", 0)}%',
            '"""',
            "",
            "# TODO: Implement harness manually or install jinja2 and regenerate",
            "",
            "class PlaceholderHarness:",
            f'    ENGINE_NAME = "{engine_name}"',
            f"    pass",
        ]
        return "\n".join(code_lines)

    def _create_review_checklist(
        self,
        analysis: Dict[str, Any],
        mappings: Dict[str, Any],
        confidence: Dict[str, float],
    ) -> List[str]:
        """Create review checklist for generated Harness."""

        checklist = [
            f"[ ] Review extraction confidence: {confidence.get('overall', 0)}%",
            f"[ ] Verify {len(analysis.get('commands', []))} extracted command patterns match actual syntax",
            f"[ ] Validate {len(mappings.get('equations', []))} mathematical equation forms",
            "[ ] Check symbolic constraint expressions for correctness",
            "[ ] Review boundary condition identification logic",
            "[ ] Verify numerical method detection",
            "[ ] Implement TODO sections in harness.py",
            "[ ] Test with real input files",
            "[ ] Uncomment HarnessRegistry.register() after validation",
            "[ ] Add proper error handling",
            "[ ] Write unit tests for parser",
            "[ ] Document any manual adjustments made",
        ]

        # Add specific items based on analysis
        for cmd in analysis.get("commands", [])[:5]:
            checklist.append(
                f"[ ] Verify command '{cmd.get('name', 'unknown')}' extraction: {cmd.get('pattern', 'N/A')[:50]}"
            )

        for eq in mappings.get("equations", [])[:3]:
            checklist.append(f"[ ] Validate equation: {eq.get('name', 'Unknown')}")

        return checklist

    def save_harness(self, template: HarnessTemplate, output_dir: str):
        """Save generated Harness to directory.

        Args:
            template: HarnessTemplate to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create harness directory structure
        harness_dir = output_path / f"{template.engine_name}-harness"
        harness_dir.mkdir(exist_ok=True)

        # Create package structure
        pkg_dir = harness_dir / "math_anything" / template.engine_name.lower()
        pkg_dir.mkdir(parents=True, exist_ok=True)

        core_dir = pkg_dir / "core"
        core_dir.mkdir(exist_ok=True)

        # Write harness.py
        harness_file = core_dir / "harness.py"
        with open(harness_file, "w", encoding="utf-8") as f:
            f.write(template.generated_code)

        # Write __init__.py
        init_file = pkg_dir / "__init__.py"
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(f'''"""{template.engine_name.title()} Harness for Math Anything.

AUTO-GENERATED - REVIEW REQUIRED BEFORE USE
"""

from .core.harness import {self._to_class_name(template.engine_name)}Harness

__version__ = "{template.engine_version}"

__all__ = ["{self._to_class_name(template.engine_name)}Harness"]
''')

        # Write analysis summary
        summary_file = harness_dir / "ANALYSIS_SUMMARY.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(self._generate_summary(template))

        # Write review checklist
        review_file = harness_dir / "REVIEW_CHECKLIST.md"
        with open(review_file, "w", encoding="utf-8") as f:
            f.write("# Review Checklist\n\n")
            for item in template.review_checklist:
                f.write(f"- {item}\n")

        print(f"\n✅ Generated Harness saved to: {harness_dir}")
        print(f"   Files created:")
        print(f"   - {harness_file}")
        print(f"   - {init_file}")
        print(f"   - {summary_file}")
        print(f"   - {review_file}")

    def _generate_summary(self, template: HarnessTemplate) -> str:
        """Generate analysis summary markdown."""
        conf = template.extraction_confidence
        summary = f"""# {template.engine_name.title()} Harness - Analysis Summary

## Generation Info
- **Engine**: {template.engine_name}
- **Version**: {template.engine_version}
- **Source**: {template.source_path}
- **Status**: ⚠️  AUTO-GENERATED - REVIEW REQUIRED
- **Generated**: {datetime.now().isoformat()}

## Extraction Confidence
- **Overall**: {conf.get('overall', 0)}%
- **Command Extraction**: {conf.get('command_extraction', 0)}%
- **Parameter Extraction**: {conf.get('parameter_extraction', 0)}%
- **Equation Mapping**: {conf.get('equation_mapping', 0)}%

## Extracted Commands ({len(template.extracted_commands)})
"""

        for cmd in template.extracted_commands[:20]:
            summary += f"\n### {cmd.get('name', 'Unknown')}\n"
            summary += f"- Description: {cmd.get('description', 'N/A')}\n"
            summary += f"- Pattern: `{cmd.get('pattern', 'N/A')}`\n"
            summary += f"- Parameters: {cmd.get('parameters', [])}\n"

        summary += f"""

## Mathematical Mappings

### Equations ({len(template.mathematical_mappings.get('equations', []))})
"""

        for eq in template.mathematical_mappings.get("equations", [])[:10]:
            summary += f"\n- **{eq.get('name', 'Unknown')}**\n"
            summary += f"  - Form: `{eq.get('form', 'N/A')[:80]}`\n"
            summary += f"  - Type: {eq.get('type', 'unknown')}\n"

        summary += f"""

## Symbolic Constraints ({len(template.constraint_expressions)})
"""

        for constraint in template.constraint_expressions[:10]:
            summary += f"\n- `{constraint[:100]}`\n"

        summary += """

## Next Steps
1. Review all extracted commands and patterns
2. Validate mathematical equation forms
3. Verify symbolic constraints
4. Implement TODO sections in harness.py
5. Test with real input files
6. Register harness after validation

## Notes
- This harness was auto-generated from source code analysis
- All extractions are heuristic and require validation
- The confidence scores indicate estimated reliability
"""

        return summary

    def _to_class_name(self, engine_name: str) -> str:
        """Convert engine name to class name."""
        return "".join(word.capitalize() for word in engine_name.split("_"))

    def _extract_variables(self, constraint: str) -> List[str]:
        """Extract variable names from constraint expression."""
        # Simple regex to find potential variable names
        import re

        variables = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", constraint)
        # Filter out common math words
        math_words = {"sin", "cos", "tan", "exp", "log", "sqrt", "and", "or", "not"}
        return list(set(v for v in variables if v not in math_words and len(v) > 1))[:5]
