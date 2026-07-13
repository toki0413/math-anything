"""元数学模块.

建模形式化系统本身的能力边界：
- Gödel 不完备性定理
- 一致性强度层次
- 可判定性边界
- 与项目约束系统的映射

核心思想：项目的 LearnedInvariant.UNKNOWN 状态
和 BoundaryEvolution 的操作边界概念，
都有深刻的元数学根源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class FormalSystemStrength(Enum):
    """形式系统的证明强度层次.

    按一致性强度递增排列。
    每个层级能证明前一层级的一致性，但不能证明自身的。
    """

    # 弱算术
    Q = auto()  # Robinson 算术
    PA = auto()  # Peano 算术
    # 二阶算术片段
    RCA0 = auto()  # 递归理解
    WKL0 = auto()  # 弱 König 引理
    ACA0 = auto()  # 算术理解
    ATR0 = auto()  # 自反理解
    PI11_CA0 = auto()  # Π¹₁-理解
    # 类型论
    MLTT0 = auto()  # Martin-Löf 类型论 (无宇宙)
    MLTT1 = auto()  # MLTT + 1 个宇宙
    MLTT_N = auto()  # MLTT + N 个宇宙
    CIC = auto()  # 归纳构造演算 (Coq/Lean)
    # 集合论
    ZFC = auto()  # Zermelo-Fraenkel + 选择
    ZFC_I = auto()  # ZFC + 不可达基数
    TG = auto()  # Tarski-Grothendieck

    def consistency_strength_order(self) -> int:
        """一致性强度排序（越大越强）."""
        orders = {
            FormalSystemStrength.Q: 1,
            FormalSystemStrength.PA: 2,
            FormalSystemStrength.RCA0: 3,
            FormalSystemStrength.WKL0: 4,
            FormalSystemStrength.ACA0: 5,
            FormalSystemStrength.ATR0: 6,
            FormalSystemStrength.PI11_CA0: 7,
            FormalSystemStrength.MLTT0: 3,
            FormalSystemStrength.MLTT1: 5,
            FormalSystemStrength.MLTT_N: 7,
            FormalSystemStrength.CIC: 8,
            FormalSystemStrength.ZFC: 9,
            FormalSystemStrength.ZFC_I: 10,
            FormalSystemStrength.TG: 11,
        }
        return orders.get(self, 0)

    def can_prove_consistency_of(self, other: "FormalSystemStrength") -> bool:
        """判断此系统能否证明另一系统的一致性.

        由 Gödel 第二不完备性定理：
        系统不能证明自身的一致性。
        但可以证明更弱系统的一致性。
        """
        return self.consistency_strength_order() > other.consistency_strength_order()


class DecidabilityClass(Enum):
    """可判定性分类."""

    DECIDABLE = auto()  # 存在算法判定
    SEMI_DECIDABLE = auto()  # 存在算法验证"是"的情况
    CO_SEMI_DECIDABLE = auto()  # 存在算法验证"否"的情况
    UNDECIDABLE = auto()  # 不存在算法判定
    INDEPENDENT = auto()  # 独立于给定公理系统


@dataclass
class GodelBoundary:
    """Gödel 不完备性边界.

    对任何足够强的一致递归可枚举理论 T：
    1. (第一不完备性) 存在命题 G_T 使得 T ⊬ G_T 且 T ⊬ ¬G_T
    2. (第二不完备性) T ⊬ Con(T)

    在项目中的映射：
    - G_T ↔ LearnedInvariant 的 UNKNOWN 状态
    - Con(T) ↔ 系统自身一致性的不可证性
    """

    theory_name: str
    theory_strength: FormalSystemStrength
    godel_sentence: str  # G_T 的描述
    is_provable: bool | None = None  # None = 独立
    consistency_provable_in: list[FormalSystemStrength] = field(default_factory=list)

    def __post_init__(self):
        # 计算哪些系统能证明此理论的一致性
        self.consistency_provable_in = [
            s for s in FormalSystemStrength if s.can_prove_consistency_of(self.theory_strength)
        ]


@dataclass
class DecidabilityBoundary:
    """可判定性边界.

    记录数学问题的可判定性分类。
    """

    problem_name: str
    decidability: DecidabilityClass
    reduction_from: str | None = None  # 归约来源（如 "halting_problem"）
    description: str = ""

    # 与项目约束的映射
    maps_to_invariant_state: str = ""  # 对应的 LearnedInvariant 状态


# ── 预定义的可判定性边界 ──

UNDECIDABILITY_RESULTS: list[DecidabilityBoundary] = [
    DecidabilityBoundary(
        problem_name="halting_problem",
        decidability=DecidabilityClass.UNDECIDABLE,
        description="不存在算法判定程序是否停机",
        maps_to_invariant_state="UNKNOWN",
    ),
    DecidabilityBoundary(
        problem_name="hilbert_tenth",
        decidability=DecidabilityClass.UNDECIDABLE,
        reduction_from="halting_problem",
        description="不存在算法判定丢番图方程是否有整数解",
        maps_to_invariant_state="UNKNOWN",
    ),
    DecidabilityBoundary(
        problem_name="word_problem_groups",
        decidability=DecidabilityClass.UNDECIDABLE,
        reduction_from="halting_problem",
        description="群的字问题不可判定 (Novikov-Boone)",
        maps_to_invariant_state="UNKNOWN",
    ),
    DecidabilityBoundary(
        problem_name="homeomorphism_4manifolds",
        decidability=DecidabilityClass.UNDECIDABLE,
        reduction_from="halting_problem",
        description="4-流形的同胚问题不可判定 (Markov)",
        maps_to_invariant_state="UNKNOWN",
    ),
    DecidabilityBoundary(
        problem_name="navier_stokes_regularity",
        decidability=DecidabilityClass.SEMI_DECIDABLE,
        description="Navier-Stokes 方程整体光滑性是千禧年问题，可判定性未知",
        maps_to_invariant_state="CONDITIONAL",
    ),
    DecidabilityBoundary(
        problem_name="scf_convergence",
        decidability=DecidabilityClass.SEMI_DECIDABLE,
        description="SCF 迭代收敛性：能验证收敛，但不能判定是否终将收敛",
        maps_to_invariant_state="WEAKENED",
    ),
    DecidabilityBoundary(
        problem_name="eigenvalue_computation",
        decidability=DecidabilityClass.SEMI_DECIDABLE,
        description="特征值计算：能验证给定值是否特征值，但不能精确判定所有特征值",
        maps_to_invariant_state="CONDITIONAL",
    ),
    DecidabilityBoundary(
        problem_name="type_checking_MLTT",
        decidability=DecidabilityClass.DECIDABLE,
        description="MLTT 的类型检查是可判定的（判断相等性可判定）",
        maps_to_invariant_state="SATISFIED",
    ),
    DecidabilityBoundary(
        problem_name="type_inference_MLTT",
        decidability=DecidabilityClass.SEMI_DECIDABLE,
        description="MLTT 的类型推断是半可判定的（可能不终止）",
        maps_to_invariant_state="CONDITIONAL",
    ),
]


# ── Gödel 边界注册表 ──

GODEL_BOUNDARIES: list[GodelBoundary] = [
    GodelBoundary(
        theory_name="PA",
        theory_strength=FormalSystemStrength.PA,
        godel_sentence="G_PA: 'G_PA is not provable in PA'",
    ),
    GodelBoundary(
        theory_name="ZFC",
        theory_strength=FormalSystemStrength.ZFC,
        godel_sentence="G_ZFC: 'G_ZFC is not provable in ZFC'",
    ),
    GodelBoundary(
        theory_name="CIC (Coq/Lean)",
        theory_strength=FormalSystemStrength.CIC,
        godel_sentence="G_CIC: 'G_CIC is not provable in CIC'",
    ),
    GodelBoundary(
        theory_name="MLTT",
        theory_strength=FormalSystemStrength.MLTT1,
        godel_sentence="G_MLTT: 'G_MLTT is not provable in MLTT'",
    ),
]


# ── 元数学分析器 ──


@dataclass
class MetamathAnalyzer:
    """元数学分析器.

    分析形式化系统的能力边界，
    并将结果映射到项目的约束系统。
    """

    boundaries: list[DecidabilityBoundary] = field(default_factory=lambda: list(UNDECIDABILITY_RESULTS))
    godel_boundaries: list[GodelBoundary] = field(default_factory=lambda: list(GODEL_BOUNDARIES))

    def analyze_invariant_decidability(
        self,
        invariant_name: str,
        invariant_expression: str = "",
    ) -> DecidabilityBoundary | None:
        """分析不变量的可判定性.

        根据不变量的表达式特征，判断其可判定性分类。
        """
        # 关键词匹配
        expr = invariant_expression.lower()

        # 不可判定模式
        undecidable_patterns = [
            ("convergence", "all", "arbitrary"),  # 任意系统收敛性
            ("exists", "integer", "solution"),  # 丢番图
            ("homeomorphism", "4"),  # 4-流形
        ]

        # 半可判定模式
        semi_decidable_patterns = [
            ("convergence", "iterat"),  # 迭代收敛
            ("eigenvalue", "all"),  # 全部特征值
            ("regularity", "global"),  # 整体正则性
        ]

        # 可判定模式
        decidable_patterns = [
            ("type", "check"),  # 类型检查
            ("equality", "judgmental"),  # 判断相等
            ("dimensional", "consistency"),  # 量纲一致性
        ]

        for boundary in self.boundaries:
            if invariant_name in boundary.problem_name:
                return boundary

        # 启发式判断
        for patterns, decidability in [
            (undecidable_patterns, DecidabilityClass.UNDECIDABLE),
            (semi_decidable_patterns, DecidabilityClass.SEMI_DECIDABLE),
            (decidable_patterns, DecidabilityClass.DECIDABLE),
        ]:
            for pattern in patterns:  # type: ignore[attr-defined]
                if all(p in expr for p in pattern):
                    return DecidabilityBoundary(
                        problem_name=invariant_name,
                        decidability=decidability,
                        description=f"Heuristic: pattern {pattern} matched in '{invariant_expression}'",
                        maps_to_invariant_state={
                            DecidabilityClass.DECIDABLE: "SATISFIED",
                            DecidabilityClass.SEMI_DECIDABLE: "CONDITIONAL",
                            DecidabilityClass.UNDECIDABLE: "UNKNOWN",
                        }.get(decidability, "UNKNOWN"),
                    )

        return None

    def godel_limitation(
        self,
        system: FormalSystemStrength = FormalSystemStrength.CIC,
    ) -> GodelBoundary:
        """获取形式化系统的 Gödel 限制."""
        for gb in self.godel_boundaries:
            if gb.theory_strength == system:
                return gb
        return GodelBoundary(
            theory_name=system.name,
            theory_strength=system,
            godel_sentence=f"G_{system.name}: 'This sentence is not provable in {system.name}'",
        )

    def consistency_strength_comparison(
        self,
        system_a: FormalSystemStrength,
        system_b: FormalSystemStrength,
    ) -> str:
        """比较两个形式系统的一致性强度."""
        a_order = system_a.consistency_strength_order()
        b_order = system_b.consistency_strength_order()

        if a_order == b_order:
            return f"{system_a.name} 和 {system_b.name} 具有相同的一致性强度 (equi-consistent)"
        elif a_order > b_order:
            return f"{system_a.name} 严格强于 {system_b.name}，可以证明 {system_b.name} 的一致性"
        else:
            return f"{system_a.name} 严格弱于 {system_b.name}，不能证明 {system_b.name} 的一致性"

    def invariant_state_from_decidability(
        self,
        decidability: DecidabilityClass,
    ) -> str:
        """将可判定性分类映射到 LearnedInvariant 状态.

        这是元数学与项目约束系统的核心接口。
        """
        mapping = {
            DecidabilityClass.DECIDABLE: "SATISFIED",
            DecidabilityClass.SEMI_DECIDABLE: "CONDITIONAL",
            DecidabilityClass.CO_SEMI_DECIDABLE: "CONDITIONAL",
            DecidabilityClass.UNDECIDABLE: "UNKNOWN",
            DecidabilityClass.INDEPENDENT: "UNKNOWN",
        }
        return mapping.get(decidability, "UNKNOWN")

    def full_analysis(
        self,
        invariant_name: str,
        invariant_expression: str = "",
        system: FormalSystemStrength = FormalSystemStrength.CIC,
    ) -> dict[str, Any]:
        """完整元数学分析."""
        decidability = self.analyze_invariant_decidability(invariant_name, invariant_expression)
        godel = self.godel_limitation(system)

        return {
            "invariant": invariant_name,
            "decidability": decidability.decidability.name if decidability else "UNKNOWN",
            "invariant_state": decidability.maps_to_invariant_state if decidability else "UNKNOWN",
            "godel_limitation": {
                "theory": godel.theory_name,
                "godel_sentence": godel.godel_sentence,
                "consistency_provable_in": [s.name for s in godel.consistency_provable_in],
            },
            "system_strength": system.name,
            "strength_order": system.consistency_strength_order(),
        }
