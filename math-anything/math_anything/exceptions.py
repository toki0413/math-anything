"""错误和异常体系。

层次化异常，每个子模块定义自己的异常子类。
调用方可以通过 except 子句精确捕获特定错误类型。

原则：
  - code: 机器可读错误码（UPPER_SNAKE_CASE）
  - detail: 人类可读描述
  - suggestion: 可操作的修复建议
"""

from __future__ import annotations

from typing import Any


class MathAnythingError(Exception):
    """所有异常的基类.

    Attributes:
        code: 机器可读错误码
        detail: 详细信息
        suggestion: 修复建议
    """

    code: str = "UNKNOWN_ERROR"
    default_detail: str = "An unexpected error occurred."

    def __init__(
        self,
        detail: str | None = None,
        suggestion: str | None = None,
        **context: Any,
    ) -> None:
        self.detail = detail or self.default_detail
        self.suggestion = suggestion or ""
        self.context = context
        super().__init__(self.detail)

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.detail}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


# ── 结构相关 ──


class StructureError(MathAnythingError):
    code = "STRUCTURE_ERROR"
    default_detail = "Invalid mathematical structure definition."


class InvalidStructurePropertyError(StructureError):
    code = "INVALID_STRUCTURE_PROPERTY"
    default_detail = "Structure property has an invalid value."


class MissingInvariantError(StructureError):
    code = "MISSING_INVARIANT"
    default_detail = "Required structural invariant is missing."


class ContradictoryInvariantsError(StructureError):
    code = "CONTRADICTORY_INVARIANTS"
    default_detail = "Two or more structural invariants contradict each other."


# ── 态射相关 ──


class MorphismError(MathAnythingError):
    code = "MORPHISM_ERROR"
    default_detail = "Invalid morphism definition or application."


class MorphismNotApplicableError(MorphismError):
    code = "MORPHISM_NOT_APPLICABLE"
    default_detail = "The morphism's condition is not satisfied for the given parameters."


class MorphismCompositionError(MorphismError):
    code = "MORPHISM_COMPOSITION_ERROR"
    default_detail = "Cannot compose these morphisms: target of first != source of second."


class MorphismNotFoundError(MorphismError):
    code = "MORPHISM_NOT_FOUND"
    default_detail = "The requested morphism is not registered."


# ── 提取相关 ──


class ExtractionError(MathAnythingError):
    code = "EXTRACTION_ERROR"
    default_detail = "Failed to extract mathematical structure."


class EngineNotFoundError(ExtractionError):
    code = "ENGINE_NOT_FOUND"
    default_detail = "The specified simulation engine is not available."


class ParseError(ExtractionError):
    code = "PARSE_ERROR"
    default_detail = "Failed to parse input file."


class UnsupportedEngineError(ExtractionError):
    code = "UNSUPPORTED_ENGINE"
    default_detail = "Engine is not supported for this operation."


class ExtractionFileNotFoundError(ExtractionError):
    code = "EXTRACTION_FILE_NOT_FOUND"
    default_detail = "Input file not found for extraction."


# ── 验证相关 ──


class ValidationError(MathAnythingError):
    code = "VALIDATION_ERROR"
    default_detail = "Validation check failed."


class DimensionalInconsistencyError(ValidationError):
    code = "DIMENSIONAL_INCONSISTENCY"
    default_detail = "Equation dimensions are inconsistent."


class ConstraintViolationError(ValidationError):
    code = "CONSTRAINT_VIOLATION"
    default_detail = "A mathematical constraint is violated."


class ConservationLawViolationError(ValidationError):
    code = "CONSERVATION_LAW_VIOLATED"
    default_detail = "A conservation law is violated in the simulation setup."


# ── 知识图谱相关 ──


class KnowledgeGraphError(MathAnythingError):
    code = "KG_ERROR"
    default_detail = "Knowledge graph operation failed."


class EntityNotFoundError(KnowledgeGraphError):
    code = "ENTITY_NOT_FOUND"
    default_detail = "The requested entity was not found in the knowledge graph."


class GraphQueryError(KnowledgeGraphError):
    code = "GRAPH_QUERY_ERROR"
    default_detail = "Graph query failed."


# ── 维度分析相关 ──


class DimensionalAnalysisError(MathAnythingError):
    code = "DIMENSIONAL_ANALYSIS_ERROR"
    default_detail = "Dimensional analysis failed."


class UnknownUnitError(DimensionalAnalysisError):
    code = "UNKNOWN_UNIT"
    default_detail = "The specified unit is not recognized."


# ── 配置相关 ──


class ConfigurationError(MathAnythingError):
    code = "CONFIG_ERROR"
    default_detail = "Configuration is invalid."


class MissingAPIKeyError(ConfigurationError):
    code = "MISSING_API_KEY"
    default_detail = "API key is not configured."
    default_suggestion = "Set it via: math-anything config set llm.api_key YOUR_KEY"


# ── 插件相关 ──


class PluginError(MathAnythingError):
    code = "PLUGIN_ERROR"
    default_detail = "Plugin system error."


class PluginNotFoundError(PluginError):
    code = "PLUGIN_NOT_FOUND"
    default_detail = "The requested plugin is not available."


class PluginLoadError(PluginError):
    code = "PLUGIN_LOAD_ERROR"
    default_detail = "Failed to load plugin."
