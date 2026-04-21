"""Code Generation Module for Math Anything.

Harness Auto-Generator: Generate harness skeletons from source code.
Uses static analysis + LLM assistance to reduce community extension cost.

Key Features:
- SourceCodeAnalyzer: Static analysis of C++/Python/Java source
- DocumentationAnalyzer: Parse PDF/HTML/Markdown docs
- SemanticMapper: Map commands to mathematical concepts
- ConstraintInference: Extract symbolic relationships
- HarnessAutoGenerator: Main entry point

Example:
    ```python
    from math_anything.codegen import HarnessAutoGenerator

    generator = HarnessAutoGenerator()
    template = generator.generate_from_source(
        source_dir="/path/to/lammps/src",
        engine_name="lammps",
    )

    # Review and save
    generator.save_harness(template, output_dir="./generated")
    ```
"""

# Lazy imports to avoid circular dependencies
# These are loaded only when accessed


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "HarnessAutoGenerator":
        from .harness_generator import HarnessAutoGenerator

        return HarnessAutoGenerator
    elif name == "HarnessTemplate":
        from .harness_generator import HarnessTemplate

        return HarnessTemplate
    elif name == "SourceCodeAnalyzer":
        from .source_analyzer import SourceCodeAnalyzer

        return SourceCodeAnalyzer
    elif name == "DocumentationAnalyzer":
        from .doc_analyzer import DocumentationAnalyzer

        return DocumentationAnalyzer
    elif name == "SemanticMapper":
        from .semantic_mapper import SemanticMapper

        return SemanticMapper
    elif name == "ConstraintInference":
        from .constraint_inference import ConstraintInference

        return ConstraintInference
    elif name == "InferredConstraint":
        from .constraint_inference import InferredConstraint

        return InferredConstraint
    elif name == "quick_analyze":
        from .source_analyzer import quick_analyze

        return quick_analyze
    elif name == "quick_analyze_docs":
        from .doc_analyzer import quick_analyze_docs

        return quick_analyze_docs
    elif name == "quick_map":
        from .semantic_mapper import quick_map

        return quick_map
    elif name == "quick_infer":
        from .constraint_inference import quick_infer

        return quick_infer
    elif name == "extract_symbolic_constraints":
        from .constraint_inference import extract_symbolic_constraints

        return extract_symbolic_constraints

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# __all__ defines what gets imported with "from module import *"
__all__ = [
    # Main generator
    "HarnessAutoGenerator",
    "HarnessTemplate",
    # Analyzers
    "SourceCodeAnalyzer",
    "DocumentationAnalyzer",
    "SemanticMapper",
    "ConstraintInference",
    # Data classes
    "InferredConstraint",
    # Convenience functions
    "quick_analyze",
    "quick_analyze_docs",
    "quick_map",
    "quick_infer",
    "extract_symbolic_constraints",
]
