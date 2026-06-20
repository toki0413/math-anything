"""Base insight engine for mathematical explanation generation.

Core design principle: every explanation must connect
parameter → mathematical concept → physical consequence.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..schemas import MathSchema


class InsightBlock:
    """A single block of insight output."""

    def __init__(
        self,
        title: str,
        content: str,
        level: str = "info",  # info, warning, tip, math
        params: Optional[List[str]] = None,
    ):
        self.title = title
        self.content = content
        self.level = level
        self.params = params or []

    def to_text(self, width: int = 72) -> str:
        """Format as terminal-friendly text."""
        icon = {"info": "[*]", "warning": "[WARN]", "tip": "[TIP]", "math": "[MATH]"}.get(self.level, "[*]")
        lines = [f"\n{icon} {self.title}", "-" * width, self.content]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "params": self.params,
        }


class InsightEngine(ABC):
    """Base class for engine-specific mathematical insight generation."""

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @abstractmethod
    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        """Generate insight blocks from a MathSchema."""
        ...

    def explain(self, schema: MathSchema, fmt: str = "text") -> str:
        """Generate full explanation."""
        blocks = self.generate(schema)
        if fmt == "json":
            import json

            return json.dumps([b.to_dict() for b in blocks], indent=2, ensure_ascii=False)
        return self._format_text(blocks)

    def _format_text(self, blocks: List[InsightBlock]) -> str:
        """Format blocks as human-readable text."""
        lines = [f"\n{'=' * 72}", "Mathematical Insight Report", f"{'=' * 72}"]
        for block in blocks:
            lines.append(block.to_text())
        lines.append(f"\n{'=' * 72}\n")
        return "\n".join(lines)


class GenericInsightEngine(InsightEngine):
    """Fallback engine for any MathSchema."""

    @property
    def engine_name(self) -> str:
        return "generic"

    def generate(self, schema: MathSchema) -> List[InsightBlock]:
        blocks: List[InsightBlock] = []
        model = schema.mathematical_model

        if model.governing_equations:
            eqs = "\n".join(f"  [{e.type}] {e.name}\n    {e.mathematical_form}" for e in model.governing_equations)
            blocks.append(
                InsightBlock(
                    title="Governing Equations",
                    content=f"The simulation solves:\n{eqs}",
                    level="math",
                )
            )

        if schema.symbolic_constraints:
            constraints = "\n".join(
                f"  {'✓' if 'SATISFIED' in (c.description or '') else '○'} {c.expression}"
                for c in schema.symbolic_constraints
            )
            blocks.append(
                InsightBlock(
                    title="Symbolic Constraints",
                    content=f"Parameter relationships:\n{constraints}",
                    level="info",
                )
            )

        if model.boundary_conditions:
            blocks.append(
                InsightBlock(
                    title="Boundary Conditions",
                    content=f"{len(model.boundary_conditions)} boundary condition(s) detected.",
                    level="info",
                )
            )

        return blocks


# Registry of insight engines
_INSIGHT_ENGINES: Dict[str, InsightEngine] = {}


def register_insight_engine(engine: InsightEngine) -> None:
    _INSIGHT_ENGINES[engine.engine_name] = engine


def get_insight_engine(engine_name: str) -> InsightEngine:
    return _INSIGHT_ENGINES.get(engine_name, GenericInsightEngine())


def explain_schema(schema: MathSchema, engine_name: str, fmt: str = "text") -> str:
    """Convenience function: explain a schema with the appropriate engine."""
    engine = get_insight_engine(engine_name)
    return engine.explain(schema, fmt=fmt)
