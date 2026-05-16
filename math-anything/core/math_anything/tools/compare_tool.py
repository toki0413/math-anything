"""Compare tool - cross-engine and cross-model comparison."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import CompareInput, CompareOutput


async def _compare_call(inp: CompareInput, ctx: ToolContext) -> ToolResult[CompareOutput]:
    from math_anything.agents.compare_agent import CompareAgent

    agent = CompareAgent()
    result = agent.safe_run({"schema_a": inp.schema_a, "schema_b": inp.schema_b})

    data = result.data
    output = CompareOutput(
        equations_changed=data.get("equations_changed", False),
        problem_type_a=data.get("problem_type_a", ""),
        problem_type_b=data.get("problem_type_b", ""),
        shared_structure=data.get("shared_structure", {}),
        unique_to_a=data.get("unique_to_a", {}),
        unique_to_b=data.get("unique_to_b", {}),
    )
    return ToolResult(
        success=result.success,
        data=output,
        display=f"Comparison: equations {'differ' if output.equations_changed else 'match'}, {len(output.shared_structure)} shared structures",
    )


CompareTool = build_math_tool(
    name="compare",
    description="Compare two mathematical schemas to find shared structures, unique features, and parameter mappings across engines or models.",
    input_schema=CompareInput,
    call=_compare_call,
)
