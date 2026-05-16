"""Verify tool - formal multi-layer verification of mathematical statements."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import VerifyInput, VerifyOutput


async def _verify_call(inp: VerifyInput, ctx: ToolContext) -> ToolResult[VerifyOutput]:
    from math_anything import FormalVerifier, MathematicalTask, TaskType as TT
    from math_anything import DifferentialGeometryLayer

    task_type = TT.PROOF
    for t in TT:
        if t.value == inp.task_type:
            task_type = t
            break

    task = MathematicalTask(
        id="tool-verify",
        type=task_type,
        name="",
        statement=inp.statement,
        assumptions=inp.assumptions,
        goals=inp.goals,
    )

    geo_ctx = None
    if inp.with_geometry and inp.engine:
        try:
            geo = DifferentialGeometryLayer()
            geo_ctx = geo.extract(inp.engine, {})
        except Exception:
            pass

    fv = FormalVerifier()
    if ctx.llm_config:
        fv = FormalVerifier(api_config=ctx.llm_config)

    result = fv.verify(task, inp.proof_text, geometric_context=geo_ctx)

    output = VerifyOutput(
        formal_status=result.formal_status.value,
        overall_confidence=result.overall_confidence,
        layer_results=[lr.to_dict() if hasattr(lr, "to_dict") else lr for lr in result.layer_results],
        errors=result.errors if hasattr(result, "errors") else [],
    )
    return ToolResult(
        success=result.formal_status.value in ("verified", "inconclusive"),
        data=output,
        display=f"Verification: {output.formal_status} (confidence: {output.overall_confidence:.2f})",
    )


VerifyTool = build_math_tool(
    name="verify",
    description="Perform multi-layer formal verification of a mathematical statement: symbolic verification, type system check, logic verification, LLM semantic analysis, and optional Lean4 formal proof.",
    input_schema=VerifyInput,
    call=_verify_call,
)
