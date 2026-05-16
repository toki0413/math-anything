"""Cross-validation tool - method × conclusion validation matrix."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import CrossValidateInput, CrossValidateOutput


async def _crossval_call(inp: CrossValidateInput, ctx: ToolContext) -> ToolResult[CrossValidateOutput]:
    try:
        from math_anything.validation_toolkit import CrossValidationMatrix
    except ImportError:
        from math_anything import CrossValidationMatrix

    matrix = CrossValidationMatrix(methods=inp.methods, conclusions=inp.conclusions)
    output = CrossValidateOutput(
        matrix=matrix.to_dict(),
        report=matrix.report(),
    )
    return ToolResult(
        success=True,
        data=output,
        display=f"Cross-validation matrix: {len(inp.methods)} methods × {len(inp.conclusions)} conclusions",
    )


CrossValidateTool = build_math_tool(
    name="cross_validate",
    description="Build a cross-validation matrix mapping multiple methods against conclusions. Each cell tracks whether a method confirms, partially confirms, or contradicts a conclusion.",
    input_schema=CrossValidateInput,
    call=_crossval_call,
)
