"""类型论三层架构.

Layer 1 — MLTT (Martin-Löf 类型理论):
- terms: 核心项语言（Π, Σ, Id, Transport 等）
- checker: 双向类型检查器（infer/check/def_eq）
- bridge: MLTT 与现有结构系统的桥接
- metamath: 元数学（Gödel 不完备性、可判定性边界）

Layer 2 — CIC (归纳构造演算):
- cic: Prop/Type 分层、CoInductive、Quotient、Fixpoint

Layer 3 — HoTT (同伦类型论):
- hott: Univalence、Higher Inductive Types、h-levels、Fiber

设计原则：
1. 结构不变量 → Identity 类型（命题相等性）
2. 态射 → Pi 类型 + 证明义务
3. 约束传播 → Transport（沿等式证明传输）
4. 数学结构 → Inductive 类型族
5. 等价结构 → Univalence（等价 = 同一）
6. 态射链 → HIT 路径构造子
"""

from .bridge import (
    MorphismType,
    TypeTheoryBridge,
    invariant_to_identity,
    invariant_to_prop_type,
    morphism_to_type,
    propagation_to_transport,
    structure_to_inductive,
)
from .checker import TypeChecker, TypeCheckError, TypeCheckResult
from .cic import (
    PROP,
    TYPE0_SORT,
    TYPE1_SORT,
    CICBridge,
    CICTypeChecker,
    CoConstructor,
    CoFix,
    CoInductiveType,
    CoMatch,
    Fixpoint,
    InductiveFamily,
    PropTypeRule,
    QuotientIntro,
    QuotientLift,
    QuotientType,
    Sort,
    SortKind,
    StructuralRecursionChecker,
    TerminationCheck,
    check_termination,
)
from .hott import (
    CIRCLE,
    INTERVAL,
    TORUS,
    Equivalence,
    Fiber,
    HigherInductiveType,
    HITElim,
    HLevel,
    HoTTBridge,
    HoTTTypeChecker,
    IdToEquiv,
    IsHLevel,
    PathConstructor,
    Univalence,
    UnivalenceVerifier,
)
from .metamath import (
    GODEL_BOUNDARIES,
    UNDECIDABILITY_RESULTS,
    DecidabilityBoundary,
    DecidabilityClass,
    FormalSystemStrength,
    GodelBoundary,
    MetamathAnalyzer,
)
from .terms import (
    BOOL_TYPE,
    NAT_TYPE,
    TYPE0,
    TYPE1,
    Annotation,
    App,
    Cong,
    Construct,
    Constructor,
    Context,
    Identity,
    IndElim,
    InductiveType,
    Judgment,
    Lam,
    Pair,
    Pi,
    Proj1,
    Proj2,
    Refl,
    Sigma,
    Sym,
    Term,
    TermKind,
    Trans,
    Transport,
    Universe,
    Var,
    arrow,
    free_vars,
    product,
    substitute,
    term_to_str,
    whnf,
)

__all__ = [
    # 项语言
    "Term",
    "TermKind",
    "Var",
    "Universe",
    "Pi",
    "Lam",
    "App",
    "Sigma",
    "Pair",
    "Proj1",
    "Proj2",
    "Identity",
    "Refl",
    "Sym",
    "Trans",
    "Cong",
    "Transport",
    "Annotation",
    "InductiveType",
    "Constructor",
    "Construct",
    "IndElim",
    "Context",
    "Judgment",
    "free_vars",
    "substitute",
    "whnf",
    "term_to_str",
    "arrow",
    "product",
    "TYPE0",
    "TYPE1",
    "BOOL_TYPE",
    "NAT_TYPE",
    # 类型检查器
    "TypeChecker",
    "TypeCheckResult",
    "TypeCheckError",
    # 桥接
    "TypeTheoryBridge",
    "MorphismType",
    "invariant_to_identity",
    "invariant_to_prop_type",
    "morphism_to_type",
    "propagation_to_transport",
    "structure_to_inductive",
    # 元数学
    "FormalSystemStrength",
    "DecidabilityClass",
    "GodelBoundary",
    "DecidabilityBoundary",
    "MetamathAnalyzer",
    "UNDECIDABILITY_RESULTS",
    "GODEL_BOUNDARIES",
    # CIC (Layer 2)
    "SortKind",
    "Sort",
    "PROP",
    "TYPE0_SORT",
    "TYPE1_SORT",
    "PropTypeRule",
    "CoConstructor",
    "CoInductiveType",
    "CoFix",
    "CoMatch",
    "QuotientType",
    "QuotientIntro",
    "QuotientLift",
    "Fixpoint",
    "TerminationCheck",
    "check_termination",
    "InductiveFamily",
    "CICTypeChecker",
    "CICBridge",
    "StructuralRecursionChecker",
    # HoTT (Layer 3)
    "HLevel",
    "IsHLevel",
    "Equivalence",
    "Univalence",
    "IdToEquiv",
    "PathConstructor",
    "HigherInductiveType",
    "HITElim",
    "Fiber",
    "HoTTTypeChecker",
    "HoTTBridge",
    "INTERVAL",
    "CIRCLE",
    "TORUS",
    "UnivalenceVerifier",
]
