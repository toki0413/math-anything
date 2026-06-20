"""全局类型定义。

提供 math-anything 所有模块共享的精确类型标注。
目标是 mypy --strict 零错误。
"""

from __future__ import annotations

from typing import (
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

# ── 泛型类型变量 ──

StructureType = TypeVar("StructureType")
SourceStructure = TypeVar("SourceStructure")
TargetStructure = TypeVar("TargetStructure")

# ── 基础别名 ──

JsonPrimitive: TypeAlias = Union[str, int, float, bool, None]
JsonValue: TypeAlias = Union[JsonPrimitive, list["JsonValue"], dict[str, "JsonValue"]]
DimensionVector: TypeAlias = tuple[float, float, float, float, float, float, float]
# [M, L, T, Theta, I, N, J]


# ── 图谱类型 ──


class GraphNode(TypedDict, total=False):
    id: str
    label: str
    type: str
    source: str
    confidence: float
    created_at: str
    last_seen: str
    mentions: int
    canonical_form: str | None
    expression: str | None
    theorem: str | None
    family: str | None
    function_space: str | None


class GraphEdge(TypedDict, total=False):
    source: str
    target: str
    relation: str
    confidence: float
    meaning: str | None


class GraphQueryResult(TypedDict, total=False):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    error: str | None


# ── 结构导出类型 ──


class StructureDict(TypedDict, total=False):
    family: str
    name: str
    canonical_form: str
    function_space: str
    dimensional_rank: int
    structural_invariants: list[dict[str, str]]


class InvariantDict(TypedDict, total=False):
    name: str
    expression: str
    theorem: str
    condition: str | None
    severity: str
    affected_quantities: list[str]


class MorphismDict(TypedDict, total=False):
    name: str
    source_type: str
    target_type: str
    category: str
    mathematical_form: str
    invariants_kept: list[str]
    invariants_lost: list[str]
    invariants_introduced: list[str]
    kernel: str
    is_injective: bool
    is_surjective: bool
    is_isomorphism: bool
    condition: str


class BuckinghamPiDict(TypedDict, total=False):
    pi_id: int
    name: str
    expression: str
    variables: dict[str, float]
    physical_meaning: str


class DimensionalCheckDict(TypedDict, total=False):
    equation: str
    consistent: bool
    notes: list[str]


class InvariantAnalysisDict(TypedDict, total=False):
    engine: str
    invariants: list[InvariantDict]
    parameter_checks: list[dict[str, Any]]
    conditional_chains: list[str]
    contradictions: list[str]
    missing: list[str]


# ── 协议 ──


class HasToDict(Protocol):
    """可序列化为 dict 的对象协议."""

    def to_dict(self) -> dict[str, Any]: ...


class HasStructuralInvariants(Protocol):
    """具有结构不变量的对象协议."""

    @property
    def structural_invariants(self) -> list[dict[str, str]]: ...


class HasMathematicalForm(Protocol):
    """具有数学表达式的对象协议."""

    @property
    def mathematical_form(self) -> str: ...


# ── 域分类 ──

DomainScope: TypeAlias = Literal[
    "dft",
    "md",
    "fem",
    "cfd",
    "phonon",
    "quantum_chemistry",
    "electromagnetics",
    "thermodynamics",
    "phase_field",
    "topology_optimization",
    "uncertainty_quantification",
    "universal",
]

InvariantSeverity: TypeAlias = Literal[
    "theorem",
    "consistency",
    "conservation",
    "stability",
    "convergence",
]

FamilyLabel: TypeAlias = Literal[
    "spectral",
    "evolution",
    "equilibrium",
    "coupled",
    "ensemble",
]
