"""Proposition tool - generate mathematical propositions and proof tasks."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import PropositionInput, PropositionOutput


async def _proposition_call(inp: PropositionInput, ctx: ToolContext) -> ToolResult[PropositionOutput]:
    from math_anything import PropositionGenerator

    schema = inp.math_schema
    if not isinstance(schema, dict):
        schema = schema.to_dict() if hasattr(schema, "to_dict") else {}

    gen = PropositionGenerator()
    propositions = gen.translate(schema)

    output = PropositionOutput(
        core_problem=propositions.core_problem,
        proof_tasks=[t.to_dict() for t in propositions.proof_tasks],
        validation_tasks=[t.to_dict() for t in propositions.validation_tasks],
        consistency_checks=[t.to_dict() for t in propositions.consistency_checks],
        total_tasks=len(propositions.all_tasks()),
    )
    return ToolResult(
        success=True,
        data=output,
        display=f"Generated {output.total_tasks} mathematical tasks: core problem = {output.core_problem[:80]}",
    )


PropositionTool = build_math_tool(
    name="proposition",
    description="Generate mathematical propositions, proof tasks, validation tasks, and consistency checks from an extracted schema.",
    input_schema=PropositionInput,
    call=_proposition_call,
)
