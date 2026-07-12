"""Visualization utilities for morphism graphs and topology loops."""

from __future__ import annotations

import re

from math_anything.categories.engine import CategoryEngine
from math_anything.topology.loop import Loop

_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def _escape_id(node: str) -> str:
    """Sanitize a string so it is a valid Mermaid/Graphviz node ID.

    Only alphanumeric characters and underscores are preserved; everything
    else is collapsed to an underscore.
    """
    return _ID_RE.sub("_", node)


# Backwards-compatible alias used by earlier callers.
_escape = _escape_id


def _escape_label_mermaid(label: str) -> str:
    """Escape a label for safe use inside Mermaid quoted text.

    Mermaid node text and edge labels can be wrapped in double quotes.  We
    escape characters that would otherwise terminate the quoted region or be
    interpreted as syntax.
    """
    escaped = label.replace("\\", "\\\\").replace('"', '\\"').replace("]", "\\]").replace("|", "\\|")
    return f'"{escaped}"'


def _escape_label_graphviz(label: str) -> str:
    """Escape a label for safe use inside a Graphviz double-quoted string.

    Backslashes that are not part of a recognized escape such as ``\\n`` are
    doubled, and internal double quotes are escaped.
    """
    escaped = re.sub(r"\\(?!n)", r"\\\\", label)
    escaped = escaped.replace('"', '\\"')
    return f'"{escaped}"'


def to_mermaid(
    category_engine: CategoryEngine,
    loops: list[Loop] | None = None,
    curvature_map: dict[str, float] | None = None,
) -> str:
    """Render a CategoryEngine graph as a Mermaid flowchart string."""
    loops = loops or []
    curvature_map = curvature_map or {}
    lines = ["graph LR"]

    for link in category_engine.morphism_links:
        src = _escape_id(link.source_structure)
        dst = _escape_id(link.target_structure)
        name = _escape_label_mermaid(link.morphism.name)
        lines.append(f"    {src} -->|{name}| {dst}")

    if loops:
        lines.append("    subgraph Loops")
        for idx, loop in enumerate(loops):
            label = loop.canonical_form
            curvature = curvature_map.get(loop.canonical_form, 0.0)
            text = _escape_label_mermaid(f"{label} | curvature={curvature:.3f}")
            lines.append(f"    note{idx}[{text}]")
        lines.append("    end")

    return "\n".join(lines) + "\n"


def to_graphviz(
    category_engine: CategoryEngine,
    loops: list[Loop] | None = None,
    curvature_map: dict[str, float] | None = None,
) -> str:
    """Render a CategoryEngine graph as a Graphviz DOT string."""
    loops = loops or []
    curvature_map = curvature_map or {}
    lines = ["digraph G {"]

    for link in category_engine.morphism_links:
        src = _escape_id(link.source_structure)
        dst = _escape_id(link.target_structure)
        name = _escape_label_graphviz(link.morphism.name)
        lines.append(f"    {src} -> {dst} [label={name}];")

    if loops:
        lines.append("    subgraph cluster_loops {")
        lines.append('        label="Loops";')
        for loop in loops:
            label = loop.canonical_form
            curvature = curvature_map.get(loop.canonical_form, 0.0)
            node_id = _escape_id(f"loop_{label}")
            node_label = _escape_label_graphviz(f"{label}\ncurvature={curvature:.3f}")
            lines.append(f"        {node_id} [shape=note, label={node_label}];")
        lines.append("    }")

    lines.append("}")
    return "\n".join(lines) + "\n"
