"""Geometry tool - differential geometry structure extraction."""

from __future__ import annotations

from ..tool_system import ToolContext, ToolResult, build_math_tool
from .schemas import GeometryInput, GeometryOutput


async def _geometry_call(inp: GeometryInput, ctx: ToolContext) -> ToolResult[GeometryOutput]:
    from math_anything import DifferentialGeometryLayer

    geo = DifferentialGeometryLayer()
    structure = geo.extract(
        inp.engine,
        inp.params,
        lattice_vectors=inp.lattice_vectors,
        space_group=inp.space_group,
    )

    output = GeometryOutput(
        manifold=structure.manifold.to_dict() if hasattr(structure, "manifold") and hasattr(structure.manifold, "to_dict") else {},
        metric_tensor=structure.metric_tensor.to_dict() if hasattr(structure, "metric_tensor") and hasattr(structure.metric_tensor, "to_dict") else {},
        curvature=structure.curvature.to_dict() if hasattr(structure, "curvature") and hasattr(structure.curvature, "to_dict") else {},
        fiber_bundle=structure.fiber_bundle.to_dict() if hasattr(structure, "fiber_bundle") and structure.fiber_bundle and hasattr(structure.fiber_bundle, "to_dict") else None,
    )
    return ToolResult(
        success=True,
        data=output,
        display=f"Geometry: manifold extracted for {inp.engine}",
    )


GeometryTool = build_math_tool(
    name="geometry",
    description="Extract differential geometry structures from computational models: manifolds, metric tensors, curvature, fiber bundles, and symmetry groups.",
    input_schema=GeometryInput,
    call=_geometry_call,
    is_concurrency_safe=lambda _: False,
)
