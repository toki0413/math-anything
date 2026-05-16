"""Extract tool - extract mathematical structures from simulation inputs."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import ExtractInput, ExtractOutput


async def _extract_call(
    inp: ExtractInput, ctx: ToolContext
) -> ToolResult[ExtractOutput]:
    from math_anything import MathAnything

    ma = MathAnything()
    if inp.filepath:
        result = ma.extract_file(inp.engine, inp.filepath)
    else:
        result = ma.extract(inp.engine, inp.params)

    schema = result.schema if hasattr(result, "schema") else {}
    if not isinstance(schema, dict):
        schema = schema.to_dict() if hasattr(schema, "to_dict") else {}

    output = ExtractOutput(
        math_schema=schema,
        constraints=schema.get("constraints", []) if isinstance(schema, dict) else [],
        approximations=(
            schema.get("approximations", []) if isinstance(schema, dict) else []
        ),
    )
    return ToolResult(
        success=result.success if hasattr(result, "success") else True,
        data=output,
        display=f"Extracted {inp.engine} schema with {len(output.constraints)} constraints",
    )


ExtractTool = build_math_tool(
    name="extract",
    description="Extract mathematical structures (governing equations, boundary conditions, constraints) from computational science engine inputs like VASP, LAMMPS, Abaqus, etc.",
    input_schema=ExtractInput,
    call=_extract_call,
)
