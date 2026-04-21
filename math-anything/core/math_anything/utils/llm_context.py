"""LLM Context Protocol - Standardized interaction with language models.

This module defines a protocol for LLMs to consume Math Schema data
in a structured way, enabling consistent integration with AI assistants.
"""

import json
from typing import Any, Dict, List, Optional


class LLMContextProtocol:
    """Protocol for standardizing LLM consumption of Math Schema.

    Instead of just dumping JSON to an LLM, this protocol provides:
    - Structured context sections
    - Query templates for common operations
    - Consumption hints for the LLM

    Example:
        ```python
        protocol = LLMContextProtocol()
        context = protocol.generate_context(schema)

        # Use with LLM
        response = llm.chat(context + "\n\nUser question: " + user_query)
        ```
    """

    def __init__(self):
        self.version = "1.0.0"

    def generate_context(
        self, schema_data: Dict[str, Any], include_sections: Optional[List[str]] = None
    ) -> str:
        """Generate LLM-optimized context from schema.

        Args:
            schema_data: Math Schema dictionary
            include_sections: Specific sections to include (default: all)

        Returns:
            Formatted context string for LLM consumption
        """
        sections = include_sections or [
            "overview",
            "governing_equations",
            "boundary_conditions",
            "numerical_method",
            "conservation_properties",
            "query_hints",
        ]

        parts = []

        if "overview" in sections:
            parts.append(self._format_overview(schema_data))

        if "governing_equations" in sections:
            parts.append(self._format_equations(schema_data))

        if "boundary_conditions" in sections:
            parts.append(self._format_boundary_conditions(schema_data))

        if "numerical_method" in sections:
            parts.append(self._format_numerical_method(schema_data))

        if "conservation_properties" in sections:
            parts.append(self._format_conservation(schema_data))

        if "query_hints" in sections:
            parts.append(self._format_query_hints())

        return "\n\n".join(parts)

    def _format_overview(self, data: Dict[str, Any]) -> str:
        """Format overview section."""
        meta = data.get("meta", {})

        lines = [
            "=" * 60,
            "MATHEMATICAL MODEL OVERVIEW",
            "=" * 60,
            f"Extracted by: {meta.get('extracted_by', 'unknown')}",
            f"Schema version: {data.get('schema_version', 'unknown')}",
            f"Extraction date: {meta.get('extracted_at', 'unknown')}",
            "",
            "This is a structured mathematical model extracted from computational",
            "simulation software. The model contains governing equations, boundary",
            "conditions, numerical methods, and computational graph information.",
        ]

        return "\n".join(lines)

    def _format_equations(self, data: Dict[str, Any]) -> str:
        """Format governing equations section."""
        model = data.get("mathematical_model", {})
        equations = model.get("governing_equations", [])

        lines = [
            "-" * 60,
            "GOVERNING EQUATIONS",
            "-" * 60,
        ]

        for eq in equations:
            lines.extend(
                [
                    f"\n[{eq.get('id', 'unknown')}]: {eq.get('name', 'Unnamed')}",
                    f"Type: {eq.get('type', 'unknown')}",
                    f"Form: {eq.get('mathematical_form', 'N/A')}",
                ]
            )
            if eq.get("description"):
                lines.append(f"Description: {eq['description']}")

        return "\n".join(lines)

    def _format_boundary_conditions(self, data: Dict[str, Any]) -> str:
        """Format boundary conditions section."""
        model = data.get("mathematical_model", {})
        bcs = model.get("boundary_conditions", [])

        lines = [
            "-" * 60,
            "BOUNDARY CONDITIONS",
            "-" * 60,
        ]

        for bc in bcs:
            mo = bc.get("mathematical_object", {})
            lines.extend(
                [
                    f"\n[{bc.get('id', 'unknown')}]: {bc.get('type', 'unknown')}",
                ]
            )

            if mo.get("tensor_rank") is not None:
                lines.append(f"Tensor Rank: {mo['tensor_rank']}")
            if mo.get("tensor_form"):
                lines.append(f"Tensor Form: {mo['tensor_form']}")
            if mo.get("symmetry"):
                lines.append(f"Symmetry: {mo['symmetry']}")

            if bc.get("software_implementation"):
                impl = bc["software_implementation"]
                lines.append(f"Implementation: {impl.get('command', 'N/A')}")

        return "\n".join(lines)

    def _format_numerical_method(self, data: Dict[str, Any]) -> str:
        """Format numerical method section."""
        nm = data.get("numerical_method", {})
        disc = nm.get("discretization", {})
        solver = nm.get("solver", {})

        lines = [
            "-" * 60,
            "NUMERICAL METHOD",
            "-" * 60,
            f"Time Integrator: {disc.get('time_integrator', 'N/A')}",
            f"Order: {disc.get('order', 'N/A')}",
            f"Time Step: {disc.get('time_step', 'N/A')}",
            f"Space Discretization: {disc.get('space_discretization', 'N/A')}",
            f"Solver: {solver.get('algorithm', 'N/A')}",
        ]

        return "\n".join(lines)

    def _format_conservation(self, data: Dict[str, Any]) -> str:
        """Format conservation properties section."""
        cp = data.get("conservation_properties", {})

        lines = [
            "-" * 60,
            "CONSERVATION PROPERTIES",
            "-" * 60,
        ]

        if not cp:
            lines.append("No conservation properties specified.")
        else:
            for prop_name, prop_data in cp.items():
                preserved = prop_data.get("preserved", False)
                status = "✓ Preserved" if preserved else "✗ Not Preserved"
                lines.append(f"\n{prop_name}: {status}")
                if prop_data.get("mechanism"):
                    lines.append(f"  Mechanism: {prop_data['mechanism']}")

        return "\n".join(lines)

    def _format_query_hints(self) -> str:
        """Format query hints for LLM."""
        return """
-" * 60,
QUERY GUIDELINES
-" * 60,

When answering questions about this model:
1. Reference specific equation IDs and boundary condition IDs when relevant
2. Note the tensor rank of boundary conditions when discussing constraints
3. Consider conservation properties when discussing physical validity
4. Be aware of the numerical method's limitations (order, stability)
5. Distinguish between mathematical structure and material-specific parameters

Common query patterns:
- "What are the governing equations?" → Reference governing_equations section
- "What boundary conditions are applied?" → Reference boundary_conditions section  
- "Is energy conserved?" → Check conservation_properties.energy.preserved
- "What numerical method is used?" → Reference numerical_method section
- "Explain the computational graph" → Reference computational_graph nodes and edges
"""

    def generate_prompt_template(self, task: str) -> str:
        """Generate a prompt template for common tasks.

        Args:
            task: Task type (e.g., 'explain', 'validate', 'compare')

        Returns:
            Prompt template string
        """
        templates = {
            "explain": """Explain the following mathematical model in clear, accessible terms.
Focus on:
1. The physical system being modeled
2. The governing equations and their meaning
3. The boundary conditions and their physical interpretation
4. The numerical approach and its implications

{context}

Provide a structured explanation suitable for someone with undergraduate
physics/engineering background.""",
            "validate": """Analyze the following mathematical model for potential issues:

{context}

Check for:
1. Dimensional consistency in equations
2. Compatibility of boundary conditions with governing equations
3. Conservation law violations
4. Numerical stability concerns
5. Physical plausibility

Report any concerns or confirm the model appears consistent.""",
            "compare": """Compare the following two mathematical models:

Model A:
{context_a}

Model B:
{context_b}

Identify:
1. Structural differences (equations, boundary conditions)
2. Numerical method differences
3. Conservation property changes
4. Which model is more appropriate for different scenarios""",
        }

        return templates.get(task, "{context}")


def create_llm_message(
    schema_data: Dict[str, Any],
    user_query: str,
    context_sections: Optional[List[str]] = None,
) -> str:
    """Create a complete message for LLM consumption.

    Args:
        schema_data: Math Schema dictionary
        user_query: User's question or request
        context_sections: Sections to include in context

    Returns:
        Complete formatted message
    """
    protocol = LLMContextProtocol()
    context = protocol.generate_context(schema_data, context_sections)

    return f"""{context}

User Query:
{user_query}
"""
