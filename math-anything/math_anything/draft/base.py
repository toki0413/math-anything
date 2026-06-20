"""Base draft engine for publication methodology generation.

Design principle: every sentence must be mathematically precise
and defensible in peer review.
"""

from abc import ABC, abstractmethod
from typing import Dict

from ..schemas import MathSchema


class DraftEngine(ABC):
    """Base class for engine-specific methodology draft generation."""

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @abstractmethod
    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        """Generate methodology section."""
        ...

    def _section(self, title: str, body: str, fmt: str) -> str:
        if fmt == "latex":
            return f"\\subsection{{{title}}}\n{body}\n"
        return f"## {title}\n\n{body}\n"

    def _inline_math(self, text: str, fmt: str) -> str:
        if fmt == "latex":
            return f"${text}$"
        return f"${text}$"

    def _display_math(self, text: str, fmt: str) -> str:
        if fmt == "latex":
            return f"\\begin{{equation}}\n{text}\n\\end{{equation}}\n"
        return f"$${text}$$\n"


class GenericDraftEngine(DraftEngine):
    """Fallback engine."""

    @property
    def engine_name(self) -> str:
        return "generic"

    def generate(self, schema: MathSchema, fmt: str = "markdown") -> str:
        lines = [
            "# Computational Methodology",
            "",
            f"The calculations were performed using the {self.engine_name} engine.",
            "",
            "## Mathematical Model",
        ]
        for eq in schema.mathematical_model.governing_equations:
            lines.append(f"- **{eq.name}**: {eq.mathematical_form}")
        return "\n".join(lines) + "\n"


_DRAFT_ENGINES: Dict[str, DraftEngine] = {}


def register_draft_engine(engine: DraftEngine) -> None:
    _DRAFT_ENGINES[engine.engine_name] = engine


def get_draft_engine(engine_name: str) -> DraftEngine:
    return _DRAFT_ENGINES.get(engine_name, GenericDraftEngine())


def draft_schema(schema: MathSchema, engine_name: str, fmt: str = "markdown") -> str:
    engine = get_draft_engine(engine_name)
    return engine.generate(schema, fmt=fmt)
