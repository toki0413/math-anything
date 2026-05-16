"""Dual perspective tool - geometric and analytic analysis."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import DualPerspectiveInput, DualPerspectiveOutput


async def _dual_call(
    inp: DualPerspectiveInput, ctx: ToolContext
) -> ToolResult[DualPerspectiveOutput]:
    try:
        from math_anything.validation_toolkit import DualPerspectiveAnalyzer
    except ImportError:
        from math_anything import DualPerspectiveAnalyzer

    analyzer = DualPerspectiveAnalyzer(conclusion=inp.conclusion)
    analyzer.set_geometric_checklist(inp.geometric_checks)
    analyzer.set_analytic_checklist(inp.analytic_checks)
    result = analyzer.evaluate()

    output = DualPerspectiveOutput(
        result={
            "conclusion": result.conclusion,
            "geometric_verdict": result.geometric_verdict,
            "analytic_verdict": result.analytic_verdict,
            "agreement": result.agreement,
        },
        report=analyzer.report(),
    )
    return ToolResult(
        success=True,
        data=output,
        display=f"Dual perspective: geometric={result.geometric_verdict}, analytic={result.analytic_verdict}, agreement={result.agreement}",
    )


DualPerspectiveTool = build_math_tool(
    name="dual_perspective",
    description="Analyze a conclusion from dual perspectives: geometric (differential geometry, curvature, manifolds) and analytic (probability, harmonic analysis, signal processing).",
    input_schema=DualPerspectiveInput,
    call=_dual_call,
)
