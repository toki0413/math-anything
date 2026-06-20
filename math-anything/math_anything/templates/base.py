"""Base template for mathematical narrative generation.

Design principle: every domain (DFT, FEM, MD, CFD) shares ~80% of its
mathematical narrative structure. New engines only need to provide:
  1. A parameter dictionary extracted from input files
  2. Engine-specific overrides for terminology and warnings
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class NarrativeSection:
    """A reusable section of mathematical narrative."""

    title: str
    body: str
    level: str = "info"  # info, warning, tip, math, error
    params: List[str] = field(default_factory=list)

    def to_insight_block(self) -> Any:
        """Convert to InsightBlock (imported lazily to avoid circular deps)."""
        from ..insight.base import InsightBlock

        return InsightBlock(title=self.title, content=self.body, level=self.level, params=self.params)

    def to_check_result(self, rule_name: str = "") -> Any:
        """Convert to CheckResult."""
        from ..check.base import CheckResult

        severity = {"error": "error", "warning": "warning"}.get(self.level, "info")
        return CheckResult(
            rule=rule_name or self.title,
            severity=severity,
            message=self.body,
            suggestion="",
        )


class MathNarrativeTemplate(ABC):
    """Base template for domain-specific mathematical narratives.

    Subclasses override section generators. Engines instantiate the domain
    template and optionally override engine-specific terms.
    """

    domain_name: str = "generic"
    software_name: str = "Generic Software"

    # ── Parameter map (filled by engine extractor) ──
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    # ── Section registry ──
    _INSIGHT_SECTIONS: List[Callable[["MathNarrativeTemplate"], Optional[NarrativeSection]]] = []
    _DRAFT_SECTIONS: List[Callable[["MathNarrativeTemplate"], Optional[NarrativeSection]]] = []
    _CHECK_SECTIONS: List[Callable[["MathNarrativeTemplate"], List[NarrativeSection]]] = []

    @classmethod
    def register_insight_section(cls, fn: Callable) -> Callable:
        cls._INSIGHT_SECTIONS.append(fn)
        return fn

    @classmethod
    def register_draft_section(cls, fn: Callable) -> Callable:
        cls._DRAFT_SECTIONS.append(fn)
        return fn

    @classmethod
    def register_check_section(cls, fn: Callable) -> Callable:
        cls._CHECK_SECTIONS.append(fn)
        return fn

    # ── Core generation API ──
    def generate_insight(self) -> List[NarrativeSection]:
        """Generate all insight sections."""
        sections: List[NarrativeSection] = []
        for fn in self._INSIGHT_SECTIONS:
            sec = fn(self)
            if sec is not None:
                sections.append(sec)
        return sections

    def generate_draft(self) -> List[NarrativeSection]:
        """Generate all draft sections."""
        sections: List[NarrativeSection] = []
        for fn in self._DRAFT_SECTIONS:
            sec = fn(self)
            if sec is not None:
                sections.append(sec)
        return sections

    def generate_check(self) -> List[NarrativeSection]:
        """Generate all check sections."""
        sections: List[NarrativeSection] = []
        for fn in self._CHECK_SECTIONS:
            sections.extend(fn(self))
        return sections

    # ── Helper: format math strings ──
    def _fmt(self, text: str, fmt: str = "text") -> str:
        if fmt == "latex":
            return text.replace("**", "\\mathbf{").replace("**", "}")
        return text

    def _param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def _has(self, key: str) -> bool:
        return key in self.params and self.params[key] is not None


class InsightTemplate(MathNarrativeTemplate):
    """Template specialized for insight (explain) generation."""

    def to_insight_blocks(self) -> List[Any]:
        """Generate InsightBlock list for the insight engine."""
        return [s.to_insight_block() for s in self.generate_insight()]


class DraftTemplate(MathNarrativeTemplate):
    """Template specialized for draft (methodology) generation."""

    def to_draft_text(self, fmt: str = "markdown") -> str:
        """Generate draft text in markdown or LaTeX."""
        sections = self.generate_draft()
        lines: List[str] = []

        if fmt == "markdown":
            lines.append("# Computational Details")
        else:
            lines.append("\\section{Computational Details}")
        lines.append("")

        for sec in sections:
            if fmt == "markdown":
                lines.append(f"## {sec.title}\n\n{sec.body}\n")
            else:
                lines.append(f"\\subsection{{{sec.title}}}\n{sec.body}\n")

        return "\n".join(lines)


class CheckTemplate(MathNarrativeTemplate):
    """Template specialized for check (validation) generation."""

    def to_check_results(self) -> List[Any]:
        """Generate CheckResult list for the check engine."""
        return [s.to_check_result() for s in self.generate_check()]
