"""Visualization utilities for morphism graphs and topology loops."""

from __future__ import annotations

from math_anything.topology.loop import Loop


def _escape(node: str) -> str:
    return node.replace(" ", "_").replace("-", "_")


def to_mermaid(
    category_engine,
    loops: list[Loop] | None = None,
    curvature_map: dict[str, float] | None = None,
) -> str:
    """Render a CategoryEngine graph as a Mermaid flowchart string."""
    loops = loops or []
    curvature_map = curvature_map or {}
    lines = ["graph LR"]

    for link in category_engine.morphism_links:
        src = _escape(link.source_structure)
        dst = _escape(link.target_structure)
        name = _escape(link.morphism.name)
        lines.append(f"    {src} -->|{name}| {dst}")

    if loops:
        lines.append("    subgraph Loops")
        for loop in loops:
            label = loop.canonical_form
            curvature = curvature_map.get(loop.canonical_form, 0.0)
            lines.append(f"    note[{label} | curvature={curvature:.3f}]")
        lines.append("    end")

    return "\n".join(lines) + "\n"


def to_graphviz(
    category_engine,
    loops: list[Loop] | None = None,
    curvature_map: dict[str, float] | None = None,
) -> str:
    """Render a CategoryEngine graph as a Graphviz DOT string."""
    loops = loops or []
    curvature_map = curvature_map or {}
    lines = ["digraph G {"]

    for link in category_engine.morphism_links:
        src = _escape(link.source_structure)
        dst = _escape(link.target_structure)
        name = _escape(link.morphism.name)
        lines.append(f'    {src} -> {dst} [label="{name}"];')

    if loops:
        lines.append('    subgraph cluster_loops {')
        lines.append('        label="Loops";')
        for loop in loops:
            label = loop.canonical_form
            curvature = curvature_map.get(loop.canonical_form, 0.0)
            node_id = _escape(f"loop_{label}")
            lines.append(
                f'        {node_id} [shape=note, label="{label}\\ncurvature={curvature:.3f}"];'
            )
        lines.append("    }")

    lines.append("}")
    return "\n".join(lines) + "\n"
