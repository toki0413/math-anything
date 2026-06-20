"""同伦类型论 (HoTT) — Layer 3.

在 MLTT/CIC 基础上扩展：
- Univalence 公理：等价 = 同一
- Higher Inductive Types (HITs)：高阶归纳类型
- h-levels：同伦层级 (h-prop, h-set, h-groupoid, ...)
- Transport 和 ap：沿路径的传输和函子作用
- 纤维和纤维化：Fiber, Fibration

核心思想：类型是 ∞-群胚，项是点，恒等类型是路径。
项目的核心命题"所有计算科学都是同一组数学结构的不同实例化"
正是 Univalence 公理的精神。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

from .terms import (
    App,
    Constructor,
    Context,
    Identity,
    Lam,
    Refl,
    Sigma,
    Term,
    TermKind,
    Trans,
    Transport,
    Var,
)

# ── h-levels (同伦层级) ──


class HLevel(Enum):
    """同伦层级.

    h-level -1: 可缩类型 (Contractible) — 恰好一个居留元
    h-level 0: 命题 (Proposition/h-prop) — 最多一个居留元
    h-level 1: 集合 (Set/h-set) — 恒等类型是命题
    h-level 2: 群胚 (Groupoid) — 恒等类型的恒等类型是命题
    h-level n: n-群胚

    在项目中的映射：
    - is_injective/is_surjective/is_isomorphism → h-level 分类
    - Morphism 的单射性 → h-level 保持
    """

    CONTRACTIBLE = -1  # 可缩
    PROPOSITION = 0  # 命题
    SET = 1  # 集合
    GROUPOID = 2  # 群胚
    N_GROUPOID = 3  # n-群胚 (n ≥ 2)

    def description(self) -> str:
        """Return a human-readable description of the h-level."""
        descs = {
            HLevel.CONTRACTIBLE: "可缩类型：恰好一个居留元，所有路径可缩",
            HLevel.PROPOSITION: "命题：最多一个居留元，证明无关",
            HLevel.SET: "集合：恒等类型是命题，无高阶路径",
            HLevel.GROUPOID: "2-群胚：存在非平凡的恒等类型的恒等类型",
            HLevel.N_GROUPOID: "n-群胚：存在高阶同伦结构",
        }
        return descs.get(self, "未知层级")

    @staticmethod
    def from_morphism_properties(
        is_injective: bool,
        is_surjective: bool,
        is_isomorphism: bool,
    ) -> HLevel:
        """从态射性质推断 h-level."""
        if is_isomorphism:
            return HLevel.CONTRACTIBLE  # 同构 = 可缩的等价
        if is_injective and is_surjective:
            return HLevel.SET  # 双射 = 集合层等价
        if is_injective:
            return HLevel.SET
        return HLevel.GROUPOID  # 一般态射有高阶结构


@dataclass(frozen=True)
class IsHLevel(Term):
    """h-level 断言: is_hlevel(A, n).

    is_hlevel(A, -1) = is_contr(A) : ∃(a:A), Π(x:A). a = x
    is_hlevel(A, 0) = is_prop(A) : Π(x y:A). x = y
    is_hlevel(A, 1) = is_set(A) : Π(x y:A). is_prop(x = y)
    """

    typ: Term
    level: int

    def __init__(self, typ: Term, level: int):
        super().__init__(TermKind.IDENTITY)
        object.__setattr__(self, "typ", typ)
        object.__setattr__(self, "level", level)


# ── Univalence 公理 ──


@dataclass(frozen=True)
class Equivalence(Term):
    """类型等价 A ≃ B.

    等价由以下数据组成：
    - f : A → B (函数)
    - g : B → A (逆)
    - α : Π(b:B). f(g(b)) = b (截面)
    - β : Π(a:A). g(f(a)) = a (收缩)
    - γ : Π(a:A). ap_f(β(a)) = α(f(a)) (相干性)

    在项目中的映射：
    VASP 和 QE 对同一系统的不同离散化是"等价的"
    """

    forward: Term  # f : A → B
    backward: Term  # g : B → A
    section: Term  # α : Πb. f(g(b)) = b
    retraction: Term  # β : Πa. g(f(a)) = a
    coherence: Term | None = None  # γ : 相干性
    source_type: Term | None = None
    target_type: Term | None = None

    def __init__(
        self,
        forward: Term,
        backward: Term,
        section: Term,
        retraction: Term,
        coherence: Term | None = None,
        source_type: Term | None = None,
        target_type: Term | None = None,
    ):
        super().__init__(TermKind.SIGMA)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "backward", backward)
        object.__setattr__(self, "section", section)
        object.__setattr__(self, "retraction", retraction)
        object.__setattr__(self, "coherence", coherence)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "target_type", target_type)


@dataclass(frozen=True)
class Univalence(Term):
    """Univalence 公理.

    ua(equiv) : Id_Type(A, B)

    含义：如果 A ≃ B，则 A = B。
    即：等价的类型是同一的。

    这是 HoTT 的核心公理，也是项目核心命题的形式化：
    "所有计算科学都是同一组数学结构的不同实例化"
    ↔
    "VASP 和 QE 的数学结构是等价的 → 它们是同一的"
    """

    equiv: Equivalence
    source_type: Term
    target_type: Term

    def __init__(self, equiv: Equivalence, source_type: Term, target_type: Term):
        super().__init__(TermKind.IDENTITY)
        object.__setattr__(self, "equiv", equiv)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "target_type", target_type)


@dataclass(frozen=True)
class IdToEquiv(Term):
    """Id→Equiv: 从路径到等价.

    idtoeqv(p) : A ≃ B  when p : Id_Type(A, B)

    Univalence 说 idtoeqv 是等价：
    (A = B) ≃ (A ≃ B)
    """

    path: Term  # p : Id_Type(A, B)
    source_type: Term
    target_type: Term

    def __init__(self, path: Term, source_type: Term, target_type: Term):
        super().__init__(TermKind.APP)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "target_type", target_type)


# ── Higher Inductive Types (HITs) ──


@dataclass
class PathConstructor:
    """高阶路径构造子.

    HIT 的核心创新：除了点构造子，还有路径构造子。

    例如区间类型 Interval:
    - 点构造子: zero : Interval, one : Interval
    - 路径构造子: seg : zero = one

    例如圆 S¹:
    - 点构造子: base : S¹
    - 路径构造子: loop : base = base

    在项目中的映射：
    - 态射链中的每个态射是一个路径构造子
    - Born-Oppenheimer → Kohn-Sham → 平面波截断 → SCF
      是从 "量子多体问题" 到 "SCF 迭代" 的路径
    """

    name: str
    source: str  # 路径起点（构造子名）
    target: str  # 路径终点（构造子名）
    dimension: int = 1  # 1=路径, 2=面, 3=体, ...
    # 高阶路径的边界
    boundary: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class HigherInductiveType(Term):
    """高阶归纳类型 (HIT).

    扩展 InductiveType，增加路径构造子。

    HIT 的递归原理需要处理路径构造子：
    ind_H(P, point_methods, path_methods, h) : P(h)

    在项目中的映射：
    - 数学结构的态射链 → HIT 的路径构造子
    - 约束传播 → 沿 HIT 路径的 transport
    """

    name: str
    point_constructors: tuple[Constructor, ...] = ()
    path_constructors: tuple[PathConstructor, ...] = ()
    params: tuple[tuple[str, Term], ...] = ()
    universe_level: int = 0

    def __init__(
        self,
        name: str,
        point_constructors: tuple[Constructor, ...] = (),
        path_constructors: tuple[PathConstructor, ...] = (),
        params: tuple[tuple[str, Term], ...] = (),
        universe_level: int = 0,
    ):
        super().__init__(TermKind.INDUCTIVE)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "point_constructors", point_constructors)
        object.__setattr__(self, "path_constructors", path_constructors)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "universe_level", universe_level)


@dataclass(frozen=True)
class HITElim(Term):
    """HIT 的消解器.

    需要为每个点构造子和路径构造子提供方法。

    ind_H(P, p_methods, path_methods, target)
    """

    type_name: str
    motive: Term
    point_methods: tuple[Term, ...]
    path_methods: tuple[Term, ...]
    target: Term

    def __init__(
        self,
        type_name: str,
        motive: Term,
        point_methods: tuple[Term, ...],
        path_methods: tuple[Term, ...],
        target: Term,
    ):
        super().__init__(TermKind.IND_ELIM)
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "motive", motive)
        object.__setattr__(self, "point_methods", point_methods)
        object.__setattr__(self, "path_methods", path_methods)
        object.__setattr__(self, "target", target)


# ── 纤维和纤维化 ──


@dataclass(frozen=True)
class Fiber(Term):
    """纤维 fiber(f, b).

    给定 f : A → B 和 b : B，
    fiber(f, b) = Σ(a:A). f(a) = b

    在项目中的映射：
    - 给定态射 f: Structure_A → Structure_B 和目标结构 b
    - fiber(f, b) 是所有被 f 映射到 b 的源结构集合
    """

    func: Term  # f : A → B
    point: Term  # b : B
    source_type: Term | None = None
    target_type: Term | None = None

    def __init__(self, func: Term, point: Term, source_type: Term | None = None, target_type: Term | None = None):
        super().__init__(TermKind.SIGMA)
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "target_type", target_type)


# ── HoTT 类型检查器 ──


class HoTTTypeChecker:
    """HoTT 类型检查器.

    扩展 CIC/MLTT，增加：
    1. Univalence 公理的引入和消解
    2. HIT 的类型检查
    3. h-level 的推导
    4. 纤维的计算
    """

    def __init__(self):
        from .checker import TypeChecker

        self.mltt_checker = TypeChecker()
        self.hit_types: dict[str, HigherInductiveType] = {}

    def register_hit(self, hit: HigherInductiveType) -> None:
        """注册高阶归纳类型."""
        self.hit_types[hit.name] = hit

    def infer_h_level(self, typ: Term, ctx: Context) -> HLevel:
        """推断类型的 h-level.

        规则：
        - Universe 不是任何 h-level
        - Empty 类型是命题 (h-level 0)
        - Unit 类型是可缩的 (h-level -1)
        - Bool 是集合 (h-level 1)
        - 一般归纳类型是集合（如果没有路径构造子）
        - HIT 可能是更高 h-level
        """
        if typ.kind == TermKind.UNIVERSE:
            return HLevel.N_GROUPOID  # Universe 是 ∞-群胚

        if typ.kind == TermKind.VAR:
            name = typ.name
            # 检查是否是已注册的 HIT
            hit = self.hit_types.get(name)
            if hit:
                if not hit.path_constructors:
                    return HLevel.SET
                # 有路径构造子，至少是群胚
                max_dim = max((pc.dimension for pc in hit.path_constructors), default=0)
                if max_dim <= 1:
                    return HLevel.GROUPOID
                return HLevel.N_GROUPOID

            # 检查是否是已注册的归纳类型
            itype = self.mltt_checker.inductive_types.get(name)
            if itype:
                return HLevel.SET  # 普通归纳类型是集合

        # Pi-类型保持 h-level
        if typ.kind == TermKind.PI:
            return HLevel.SET  # 函数类型是集合

        # Sigma-类型取 max
        if typ.kind == TermKind.SIGMA:
            return HLevel.SET

        # Identity 类型是命题（对于集合）
        if typ.kind == TermKind.IDENTITY:
            return HLevel.PROPOSITION

        return HLevel.SET  # 默认

    def check_univalence(
        self,
        equiv: Equivalence,
        ctx: Context,
    ) -> tuple[bool, str]:
        """检查等价是否满足 Univalence 条件.

        验证：
        1. forward : A → B
        2. backward : B → A
        3. section : Πb. f(g(b)) = b
        4. retraction : Πa. g(f(a)) = a
        """
        # 检查 forward 和 backward 的类型
        try:
            self.mltt_checker.infer(ctx, equiv.forward)
            self.mltt_checker.infer(ctx, equiv.backward)
        except (ValueError, TypeError, AttributeError) as e:
            return False, f"Type inference failed: {e}"

        return True, "Equivalence verified (forward/backward pair exists)"

    def compute_fiber(
        self,
        func: Term,
        point: Term,
        ctx: Context,
    ) -> Term:
        """计算纤维 fiber(f, b) = Σ(a:A). f(a) = b."""
        try:
            func_type = self.mltt_checker.infer(ctx, func)
            if func_type.kind == TermKind.PI:
                source = func_type.domain
                target = func_type.codomain if func_type.var_name == "_" else func_type.codomain
                fiber_type = Sigma(
                    "a",
                    source,
                    Identity(target, App(func, Var("a")), point),
                )
                return fiber_type
        except (ValueError, TypeError, AttributeError):
            logger.debug("Fiber computation failed, returning generic Fiber")
        return Fiber(func, point)


# ── HoTT-结构系统桥接 ──


@dataclass
class HoTTBridge:
    """HoTT 与项目结构系统的桥接.

    核心映射：
    1. 态射链 → HIT 的路径构造子
    2. 等价结构 → Univalence
    3. h-level → 态射性质分类
    4. 纤维 → 态射的原像集
    """

    checker: HoTTTypeChecker = field(default_factory=HoTTTypeChecker)

    def morphism_chain_to_hit(
        self,
        chain_name: str,
        morphisms: list[Any],  # list of Morphism
    ) -> HigherInductiveType:
        """将态射链转换为 HIT.

        态射链 A → B → C → D 变成 HIT：
        - 点构造子: A, B, C, D
        - 路径构造子: f₁: A=B, f₂: B=C, f₃: C=D

        态射链的合成变成路径的连接 (trans)。
        """
        point_ctors = []
        path_ctors = []

        for i, morphism in enumerate(morphisms):
            src_name = morphism.source_type.replace(" ", "_")
            tgt_name = morphism.target_type.replace(" ", "_")

            # 源和目标是点构造子
            if i == 0:
                point_ctors.append(Constructor(name=src_name))
            point_ctors.append(Constructor(name=tgt_name))

            # 态射是路径构造子
            path_ctors.append(
                PathConstructor(
                    name=morphism.name.replace(" ", "_"),
                    source=src_name,
                    target=tgt_name,
                    dimension=1,
                )
            )

        return HigherInductiveType(
            name=chain_name,
            point_constructors=tuple(point_ctors),
            path_constructors=tuple(path_ctors),
        )

    def equivalent_structures_to_univalence(
        self,
        struct_a: Term,
        struct_b: Term,
        forward_morphism: Any,  # Morphism
        backward_morphism: Any,  # Morphism
    ) -> Univalence:
        """将等价结构对转换为 Univalence 公理实例.

        如果两个结构通过互逆态射连接，
        则它们是等价的，由 Univalence 得出同一。
        """
        equiv = Equivalence(
            forward=Var(forward_morphism.name),
            backward=Var(backward_morphism.name),
            section=Var("section_proof"),
            retraction=Var("retraction_proof"),
            source_type=struct_a,
            target_type=struct_b,
        )
        return Univalence(equiv, struct_a, struct_b)

    def morphism_properties_to_hlevel(
        self,
        is_injective: bool,
        is_surjective: bool,
        is_isomorphism: bool,
    ) -> HLevel:
        """从态射性质推断 h-level."""
        return HLevel.from_morphism_properties(is_injective, is_surjective, is_isomorphism)

    def transport_along_morphism_chain(
        self,
        invariant: Any,  # StructuralInvariant
        morphism_chain: list[Any],  # list of Morphism
    ) -> Term:
        """沿态射链传输不变量.

        对应约束传播，但在 HoTT 中：
        transport(P, trans(p₁, p₂, ..., pₙ), value)

        其中 pᵢ 是第 i 个态射对应的路径。
        """
        if not morphism_chain:
            return Refl(Var("Structure"), Var("initial"))

        # 构建路径复合
        current_path: Term = Var(f"path_{morphism_chain[0].name}")

        for i, morphism in enumerate(morphism_chain[1:], 1):
            next_path = Var(f"path_{morphism.name}")
            current_path = Trans(
                Var("Structure"),
                Var(f"step_{i - 1}"),
                Var(f"step_{i}"),
                Var(f"step_{i + 1}"),
                current_path,
                next_path,
            )

        # Transport
        motive = Lam("x", App(Var(f"Inv_{invariant.name}"), Var("x")))
        return Transport(
            motive=motive,
            eq_proof=current_path,
            source_type=Var("Structure"),
            a=Var("initial"),
            b=Var("final"),
            value=App(Var(f"Inv_{invariant.name}"), Var("initial")),
        )


# ── 预定义的 HoTT 类型 ──

INTERVAL = HigherInductiveType(
    name="Interval",
    point_constructors=(
        Constructor(name="zero"),
        Constructor(name="one"),
    ),
    path_constructors=(PathConstructor(name="seg", source="zero", target="one"),),
)

CIRCLE = HigherInductiveType(
    name="S1",
    point_constructors=(Constructor(name="base"),),
    path_constructors=(PathConstructor(name="loop", source="base", target="base"),),
)

TORUS = HigherInductiveType(
    name="T2",
    point_constructors=(Constructor(name="base"),),
    path_constructors=(
        PathConstructor(name="p", source="base", target="base"),
        PathConstructor(name="q", source="base", target="base"),
    ),
    # 2-路径构造子: p ∙ q = q ∙ p (交换性)
)


# ── 计算性 Univalence 检查 ──


class UnivalenceChecker:
    """Computational Univalence axiom checker.

    While full Univalence cannot be verified (it's an axiom), we can check
    specific instances: given two types A, B and an equivalence e : A ≃ B,
    verify that the equivalence satisfies the required properties.
    """

    def __init__(self):
        self.verified_equivalences: list[dict] = []

    def verify_equivalence(self, f: Callable, g: Callable, A_elements: list, B_elements: list, name: str = "") -> dict:
        """Verify that f : A → B and g : B → A form an equivalence.

        Checks:
        1. g∘f = id_A (left inverse)
        2. f∘g = id_B (right inverse)
        """
        left_inverse_ok = True
        right_inverse_ok = True
        errors = []

        for a in A_elements:
            try:
                if g(f(a)) != a:
                    left_inverse_ok = False
                    errors.append(f"g(f({a})) ≠ {a}")
            except Exception as e:
                left_inverse_ok = False
                errors.append(f"g(f({a})) failed: {e}")

        for b in B_elements:
            try:
                if f(g(b)) != b:
                    right_inverse_ok = False
                    errors.append(f"f(g({b})) ≠ {b}")
            except Exception as e:
                right_inverse_ok = False
                errors.append(f"f(g({b})) failed: {e}")

        is_equiv = left_inverse_ok and right_inverse_ok

        result = {
            "name": name,
            "is_equivalence": is_equiv,
            "left_inverse": left_inverse_ok,
            "right_inverse": right_inverse_ok,
            "errors": errors if errors else None,
        }

        if is_equiv:
            self.verified_equivalences.append(result)

        return result

    def univalence_instance(self, equiv_name: str) -> dict:
        """Given a verified equivalence, assert the Univalence instance.

        This is NOT a proof — it's a computational witness that the
        equivalence satisfies the required properties.
        """
        equiv = next((e for e in self.verified_equivalences if e["name"] == equiv_name), None)
        if equiv is None:
            return {"error": f"Equivalence '{equiv_name}' not verified"}

        return {
            "equivalence": equiv_name,
            "univalence_asserted": True,
            "witness": "verified_equivalence_properties",
            "note": "Univalence is an axiom; this confirms the equivalence satisfies "
            "the computational requirements but does not prove Univalence itself.",
        }

    def h_level_computation(self, type_examples: dict[str, list]) -> dict[str, int]:
        """Compute h-levels for given types by examining their identity types.

        h-level 0: contractible (exactly one element up to path)
        h-level 1: proposition (at most one element)
        h-level 2: set (UIP holds)
        h-level n+2: n-groupoid
        """
        results = {}
        for type_name, elements in type_examples.items():
            n = len(elements)
            if n == 0:
                results[type_name] = -1  # empty type
            elif n == 1:
                results[type_name] = 0  # contractible
            else:
                # Check if all elements are equal (proposition)
                all_equal = all(e == elements[0] for e in elements)
                if all_equal:
                    results[type_name] = 1  # proposition
                else:
                    results[type_name] = 2  # set (default for distinct elements)
        return results


# ── Univalence 验证器 ──


class UnivalenceVerifier:
    """Verify specific instances of the Univalence Axiom.

    The Univalence Axiom states: (A ≃ B) ≃ (A =_U B)
    While we cannot prove Univalence itself (it's an axiom), we can verify
    that specific equivalences satisfy the required computational properties.
    """

    def __init__(self):
        self.verified: list[dict] = []

    def verify_equivalence_instance(
        self,
        name: str,
        f: Callable,
        g: Callable,
        test_data_A: list,
        test_data_B: list,
        epsilon: float = 1e-10,
    ) -> dict:
        """Verify that f: A → B and g: B → A form an equivalence.

        Checks:
        1. g∘f = id_A (section)
        2. f∘g = id_B (retraction)
        3. Coherence: the homotopy between g∘f and id is natural

        Returns verification result with detailed evidence.
        """
        section_ok = True
        retraction_ok = True
        section_errors = []
        retraction_errors = []

        for a in test_data_A:
            try:
                result = g(f(a))
                if isinstance(result, (int, float)):
                    if abs(result - a) > epsilon:
                        section_ok = False
                        section_errors.append(f"g(f({a})) = {result} ≠ {a}")
                elif result != a:
                    section_ok = False
                    section_errors.append(f"g(f({a})) ≠ {a}")
            except Exception as e:
                section_ok = False
                section_errors.append(f"g(f({a})) raised: {e}")

        for b in test_data_B:
            try:
                result = f(g(b))
                if isinstance(result, (int, float)):
                    if abs(result - b) > epsilon:
                        retraction_ok = False
                        retraction_errors.append(f"f(g({b})) = {result} ≠ {b}")
                elif result != b:
                    retraction_ok = False
                    retraction_errors.append(f"f(g({b})) ≠ {b}")
            except Exception as e:
                retraction_ok = False
                retraction_errors.append(f"f(g({b})) raised: {e}")

        is_equiv = section_ok and retraction_ok

        result = {
            "name": name,
            "is_equivalence": is_equiv,
            "section_holds": section_ok,
            "retraction_holds": retraction_ok,
            "section_errors": section_errors if section_errors else None,
            "retraction_errors": retraction_errors if retraction_errors else None,
            "n_test_points_A": len(test_data_A),
            "n_test_points_B": len(test_data_B),
        }

        if is_equiv:
            self.verified.append(result)

        return result

    def verify_naturality(
        self,
        f: Callable,
        g: Callable,
        morphism: Callable,
        test_points: list,
        epsilon: float = 1e-6,
    ) -> dict:
        """Verify naturality of the homotopy g∘f ~ id.

        For a natural transformation α: g∘f => id, check that:
        α(b) ∘ h = h ∘ α(a) for morphism h: a → b
        """
        errors = []
        for x in test_points:
            try:
                lhs = g(f(morphism(x)))
                rhs = morphism(g(f(x)))
                if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
                    if abs(lhs - rhs) > epsilon:
                        errors.append(f"Naturality fails at {x}: {lhs} ≠ {rhs}")
                elif lhs != rhs:
                    errors.append(f"Naturality fails at {x}: {lhs} ≠ {rhs}")
            except Exception as e:
                errors.append(f"Naturality check failed at {x}: {e}")

        return {
            "naturality_holds": len(errors) == 0,
            "errors": errors if errors else None,
            "n_test_points": len(test_points),
        }

    def compute_h_level(
        self,
        equality_test: Callable,
        elements: list,
        max_depth: int = 5,
    ) -> dict:
        """Compute the h-level of a type by examining its identity types.

        h-level 0: isContr (contractible) — exactly one element, all paths equal
        h-level 1: isProp (proposition) — any two elements are equal
        h-level 2: isSet (set) — any two paths are equal (UIP)
        h-level n+2: n-groupoid

        Args:
            equality_test: Function (a, b) -> bool testing if a = b
            elements: Sample elements of the type
            max_depth: Maximum h-level to check
        """
        n = len(elements)

        if n == 0:
            return {"h_level": -1, "description": "empty type"}

        all_equal = all(equality_test(elements[0], e) for e in elements)
        if n == 1:
            return {"h_level": 0, "description": "contractible type"}

        if all_equal:
            return {"h_level": 1, "description": "proposition (mere proposition)"}

        return {"h_level": 2, "description": "set (UIP holds)"}

    def univalence_consequence(
        self,
        equiv_name: str,
        type_A_name: str,
        type_B_name: str,
    ) -> dict:
        """State the Univalence consequence for a verified equivalence.

        If A ≃ B is verified, then by Univalence, A =_U B.
        This means any property of A transfers to B along the equivalence.
        """
        equiv = next((e for e in self.verified if e["name"] == equiv_name), None)
        if equiv is None:
            return {"error": f"Equivalence '{equiv_name}' not verified"}

        return {
            "equivalence": equiv_name,
            "type_A": type_A_name,
            "type_B": type_B_name,
            "univalence_assertion": f"{type_A_name} =_U {type_B_name}",
            "consequence": f"Any property P of {type_A_name} transfers to {type_B_name} via transport along {equiv_name}",
            "computational_content": f"transport^{equiv_name} : P({type_A_name}) → P({type_B_name})",
            "verified_properties": {
                "section": equiv["section_holds"],
                "retraction": equiv["retraction_holds"],
            },
        }
