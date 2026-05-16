"""Emergence tool - phase transition and emergence analysis."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import EmergenceInput, EmergenceOutput


async def _emergence_call(inp: EmergenceInput, ctx: ToolContext) -> ToolResult[EmergenceOutput]:
    from math_anything import MathAnything
    from math_anything.emergence import EmergenceLayer

    ma = MathAnything()
    em = EmergenceLayer()

    schema = None
    if inp.filepath:
        result = ma.extract_file(inp.engine, inp.filepath)
        schema = result.schema if result.success else None
        params = result.files.get("params", {}) if hasattr(result, "files") else inp.params
    else:
        result = ma.extract(inp.engine, inp.params)
        schema = result.schema if result.success else None
        params = inp.params

    emergence = em.extract(inp.engine, params, schema=schema)

    output = EmergenceOutput(
        emergence=emergence.to_dict(),
        math_schema=schema if isinstance(schema, dict) else (schema.to_dict() if hasattr(schema, "to_dict") else {}),
        warnings=emergence.warnings if hasattr(emergence, "warnings") else [],
    )
    return ToolResult(
        success=True,
        data=output,
        display=f"Emergence analysis: {len(emergence.warnings) if hasattr(emergence, 'warnings') else 0} warnings",
    )


EmergenceTool = build_math_tool(
    name="emergence",
    description="Analyze phase transitions and emergent phenomena in computational science simulations. Detects spectral gaps, scale separation, and structural transitions.",
    input_schema=EmergenceInput,
    call=_emergence_call,
    is_concurrency_safe=lambda _: False,
)
