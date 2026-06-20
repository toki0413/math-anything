"""Base check engine for pre-flight validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from ..schemas import MathSchema


@dataclass
class CheckResult:
    """Single validation result."""

    rule: str
    severity: str  # error, warning, info
    message: str
    suggestion: str = ""

    def to_text(self) -> str:
        icon = {"error": "[FAIL]", "warning": "[WARN]", "info": "[INFO]"}.get(self.severity, "[?]")
        lines = [f"  {icon} {self.rule}", f"      {self.message}"]
        if self.suggestion:
            lines.append(f"      -> {self.suggestion}")
        return "\n".join(lines)


class CheckEngine(ABC):
    """Base class for engine-specific pre-flight checks."""

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @abstractmethod
    def check(self, schema: MathSchema) -> List[CheckResult]:
        """Run all validation rules and return results."""
        ...

    def run(self, schema: MathSchema) -> tuple[int, str]:
        """Run checks and return (exit_code, report_text).

        Returns:
            exit_code: 0 if no errors/warnings, 1 otherwise
            report_text: Human-readable report
        """
        results = self.check(schema)
        errors = [r for r in results if r.severity == "error"]
        warnings = [r for r in results if r.severity == "warning"]
        infos = [r for r in results if r.severity == "info"]

        lines = [
            "",
            "=" * 72,
            "Pre-flight Check Report",
            "=" * 72,
            "",
            f"  Engine: {self.engine_name}",
            f"  Errors:   {len(errors)}",
            f"  Warnings: {len(warnings)}",
            f"  Info:     {len(infos)}",
            "",
        ]

        if errors:
            lines.append("Errors (will likely cause incorrect results or crashes):")
            for r in errors:
                lines.append(r.to_text())
            lines.append("")

        if warnings:
            lines.append("Warnings (may affect accuracy or convergence):")
            for r in warnings:
                lines.append(r.to_text())
            lines.append("")

        if infos:
            lines.append("Notes:")
            for r in infos:
                lines.append(r.to_text())
            lines.append("")

        if not errors and not warnings:
            lines.append("  [PASS] All checks passed. Ready to submit.")
            lines.append("")

        lines.append("=" * 72)
        lines.append("")

        exit_code = 1 if (errors or warnings) else 0
        return exit_code, "\n".join(lines)


class GenericCheckEngine(CheckEngine):
    """Fallback check engine that performs no checks."""

    @property
    def engine_name(self) -> str:
        return "generic"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        return []


# Registry
_CHECK_ENGINES: Dict[str, CheckEngine] = {}


def register_check_engine(engine: CheckEngine) -> None:
    _CHECK_ENGINES[engine.engine_name] = engine


def get_check_engine(engine_name: str) -> CheckEngine:
    return _CHECK_ENGINES.get(engine_name, GenericCheckEngine())


def check_schema(schema: MathSchema, engine_name: str) -> tuple[int, str]:
    engine = get_check_engine(engine_name)
    return engine.run(schema)
