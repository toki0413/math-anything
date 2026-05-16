"""Validate tool - validate parameter consistency and constraints."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import ValidateInput, ValidateOutput


async def _validate_call(inp: ValidateInput, ctx: ToolContext) -> ToolResult[ValidateOutput]:
    from math_anything.agents.validate_agent import ValidateAgent

    agent = ValidateAgent()
    result = agent.safe_run({"schema": inp.math_schema})

    data = result.data
    output = ValidateOutput(
        valid=data.get("valid", False),
        constraint_results=data.get("constraint_results", []),
        dimensional_issues=data.get("dimensional_issues", []),
        warnings=data.get("warnings", []),
        total_issues=data.get("total_issues", 0),
    )
    return ToolResult(
        success=result.success,
        data=output,
        display=f"Validation: {'PASS' if output.valid else f'{output.total_issues} issues found'}",
    )


ValidateTool = build_math_tool(
    name="validate",
    description="Validate mathematical consistency of an extracted schema: constraint satisfaction, dimensional consistency, parameter ranges, cross-parameter dependencies.",
    input_schema=ValidateInput,
    call=_validate_call,
)
