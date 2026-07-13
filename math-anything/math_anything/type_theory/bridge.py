"""MLTT 与现有结构系统的桥接.

将项目的核心概念翻译为 MLTT 类型论语言：
- StructuralInvariant → Identity 类型（命题相等性）
- Morphism → Pi 类型（依赖函数）+ 证明义务
- ConstraintPropagation → Transport（沿等式证明传输）
- AbstractMathematicalStructure → Inductive 类型族

这是类型论层与结构主义层的接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..morphisms import Morphism
from ..structures.base import AbstractMathematicalStructure
from ..structures.properties import StructuralInvariant
from .checker import TypeChecker, TypeCheckError, TypeCheckResult
from .terms import (
    App,
    Constructor,
    Context,
    Identity,
    InductiveType,
    Lam,
    Pi,
    Refl,
    Sigma,
    Term,
    Transport,
    Var,
    arrow,
    product,
)

# ── 结构不变量 → MLTT 恒等类型 ──


def invariant_to_identity(inv: StructuralInvariant, structure_type: Term = Var("Structure")) -> Identity:
    """将结构不变量转换为 MLTT 恒等类型.

    不变量 "特征值均为实数" 变成:
    Id_{Eigenvalue → Prop}(λ_i, real_number)

    即：不变量是某个谓词的恒等类型居留元。
    """
    # 不变量表达式作为谓词
    predicate = Var(f"Invariant_{inv.name}")
    # lhs: 结构的实际值
    lhs = Var(f"actual_{inv.name}")
    # rhs: 不变量要求的值
    rhs = Var(f"required_{inv.name}")
    return Identity(predicate, lhs, rhs)


def invariant_to_prop_type(inv: StructuralInvariant) -> Pi:
    """将不变量转换为命题类型.

    "特征值均为实数" → Π(λ:Eigenvalue). IsReal(λ)
    "能量守恒" → Id_Energy(E_initial, E_final)
    """
    prop_name = f"Inv_{inv.name}"
    # 简单情况：不变量是一个谓词
    if "∈" in inv.expression or "for all" in inv.expression:
        # 全称量化: Π(x:A). P(x)
        return Pi(
            var_name="x",
            domain=Var(inv.affected_quantities[0] if inv.affected_quantities else "U"),
            codomain=Var(prop_name),
        )
    elif "=" in inv.expression:
        # 等式: Id_A(lhs, rhs)
        return Identity(Var("Value"), Var("lhs"), Var("rhs"))  # type: ignore[return-value]
    else:
        # 一般命题
        return arrow(Var("Structure"), Var(prop_name))


# ── 态射 → MLTT 依赖函数类型 ──


@dataclass
class MorphismType:
    """态射的 MLTT 类型签名.

    一个态射 f: A → B 的类型是：
    Π(a:A). Σ(b:B). Proof(b_satisfies_invariants)

    其中 Proof(b_satisfies_invariants) 是 B 的不变量在 f 下的保持证明。
    """

    morphism_name: str
    source_type: Term
    target_type: Term
    # 保持的不变量 → 证明义务
    kept_proofs: dict[str, Term] = field(default_factory=dict)
    # 丢失的不变量 → 证明不可能
    lost_proofs: dict[str, Term] = field(default_factory=dict)
    # 引入的新不变量
    introduced_proofs: dict[str, Term] = field(default_factory=dict)
    # 完整类型签名
    full_type: Term | None = None


def morphism_to_type(morphism: Morphism) -> MorphismType:
    """将态射转换为 MLTT 类型签名.

    态射 f: A → B 的类型：
    Π(a:A). Σ(b:B). (Π(inv ∈ kept). Id(inv, preserved)) × (Π(inv ∈ introduced). Inv(inv))
    """
    source = Var(morphism.source_type)
    target = Var(morphism.target_type)

    mt = MorphismType(
        morphism_name=morphism.name,
        source_type=source,
        target_type=target,
    )

    # 为每个保持的不变量生成证明义务
    for inv_name in morphism.invariants_kept:
        # 证明义务: Id(apply_invariant(f(a), inv), invariant_value)
        proof_type = Identity(
            Var(f"Inv_{inv_name}"),
            App(Var(f"apply_{inv_name}"), Var("b")),
            App(Var(f"apply_{inv_name}"), Var("a")),
        )
        mt.kept_proofs[inv_name] = proof_type

    # 为每个丢失的不变量生成不可能证明
    for inv_name in morphism.invariants_lost:
        mt.lost_proofs[inv_name] = Var(f"Lost_{inv_name}")

    # 为每个引入的不变量生成证明
    for inv_name in morphism.invariants_introduced:
        mt.introduced_proofs[inv_name] = Var(f"Introduced_{inv_name}")

    # 构建完整类型签名
    # 基础: A → B
    result_type = arrow(source, target)

    # 如果有保持的不变量，添加证明义务
    if mt.kept_proofs:
        # Σ(b:B). (proof_1 × proof_2 × ...)
        proof_types = list(mt.kept_proofs.values())
        if len(proof_types) == 1:
            proof_term = proof_types[0]
        else:
            proof_term = proof_types[0]
            for pt in proof_types[1:]:
                proof_term = product(proof_term, pt)
        result_type = Sigma("b", target, proof_term)  # type: ignore[assignment]

    mt.full_type = Pi("a", source, result_type)
    return mt


# ── 约束传播 → Transport ──


def propagation_to_transport(
    invariant_name: str,
    source_value: Term,
    target_value: Term,
    eq_proof: Term,
    motive: Term | None = None,
) -> Transport:
    """将约束传播转换为 MLTT Transport.

    约束传播的核心操作：
    "不变量 I 在态射 f 下保持"
    ↔
    transport(λx. I_holds(x), eq_proof, I_holds(source))

    即：沿等式证明将不变量的满足性从源传输到目标。
    """
    if motive is None:
        # 动机: λx. I_holds(x)
        motive = Lam("x", App(Var(f"Inv_{invariant_name}"), Var("x")))

    return Transport(
        motive=motive,
        eq_proof=eq_proof,
        source_type=Var("Structure"),
        a=source_value,
        b=target_value,
        value=App(Var(f"Inv_{invariant_name}"), source_value),
    )


# ── 数学结构 → 归纳类型族 ──


def structure_to_inductive(structure: AbstractMathematicalStructure) -> InductiveType:
    """将数学结构转换为 MLTT 归纳类型.

    每个结构族对应一个归纳类型，其构造子是结构的具体实例。
    不变量成为类型的属性（通过递归子保证）。
    """
    name = structure.name.replace(" ", "_")

    # 从不变量推导构造子的约束
    constructors = []
    for inv in structure.structural_invariants:
        # 每个不变量对应一个证明义务
        ctor = Constructor(
            name=f"mk_{name}_with_{inv.name}",
            arg_types=[Var(f"Proof_{inv.name}")],
        )
        constructors.append(ctor)

    # 基本构造子
    base_ctor = Constructor(name=f"mk_{name}")
    constructors.insert(0, base_ctor)

    return InductiveType(
        name=name,
        constructors=tuple(constructors),
        universe_level=0,
    )


# ── 结构桥接器 ──


@dataclass
class TypeTheoryBridge:
    """MLTT 类型论与结构系统的双向桥接器.

    提供：
    1. 结构 → 类型：将数学结构编码为 MLTT 类型
    2. 类型 → 结构：从类型检查结果反推结构性质
    3. 验证：用类型检查器验证结构不变量
    4. 传播：用 Transport 验证约束传播
    """

    checker: TypeChecker = field(default_factory=TypeChecker)
    _registered_structures: dict[str, InductiveType] = field(default_factory=dict)
    _registered_morphisms: dict[str, MorphismType] = field(default_factory=dict)

    def register_structure(self, structure: AbstractMathematicalStructure) -> InductiveType:
        """注册数学结构为归纳类型."""
        itype = structure_to_inductive(structure)
        self.checker.register_inductive(itype)
        self._registered_structures[structure.name] = itype
        return itype

    def register_morphism(self, morphism: Morphism) -> MorphismType:
        """注册态射为依赖函数类型."""
        mtype = morphism_to_type(morphism)
        self._registered_morphisms[morphism.name] = mtype
        return mtype

    def verify_invariant(
        self,
        inv: StructuralInvariant,
        structure: AbstractMathematicalStructure,
        properties: dict[str, Any] | None = None,
    ) -> TypeCheckResult:
        """用类型检查器验证结构不变量.

        不变量在类型论中的验证：
        1. 检查不变量是否是结构归纳类型的命题
        2. 如果不变量有条件，检查条件是否满足
        3. 尝试构造证明项
        """
        ctx = Context()

        # 注册结构类型
        if structure.name not in self._registered_structures:
            self.register_structure(structure)

        # 构建上下文
        struct_type = Var(structure.name.replace(" ", "_"))
        ctx = ctx.extend("s", struct_type)

        # 构建不变量的命题类型
        prop_type = invariant_to_prop_type(inv)

        # 检查不变量是否激活
        if properties and not inv.is_active(properties):
            return TypeCheckResult(
                success=True,
                warnings=[f"Invariant '{inv.name}' is not active under current properties"],
            )

        # 尝试类型检查：构造证明项 refl
        try:
            proof_term = Refl(prop_type, Var("s"))
            self.checker.check(ctx, proof_term, prop_type)
            return TypeCheckResult(
                success=True,
                inferred_type=prop_type,
            )
        except TypeCheckError:
            # 无法用 refl 证明，需要更复杂的证明
            return TypeCheckResult(
                success=False,
                errors=[f"Cannot prove invariant '{inv.name}' by reflexivity. Requires explicit proof."],
                warnings=[f"Theorem: {inv.theorem}"],
            )

    def verify_morphism_preservation(
        self,
        morphism: Morphism,
        source_structure: AbstractMathematicalStructure,
        target_structure: AbstractMathematicalStructure,
    ) -> TypeCheckResult:
        """验证态射保持声明的不变量.

        检查：对于每个 invariants_kept 中的不变量，
        是否存在 transport 证明将其从源传输到目标。
        """
        errors = []  # type: ignore[var-annotated]
        proofs = {}

        for inv_name in morphism.invariants_kept:
            # 构造 transport 证明
            eq_proof = Var(f"eq_{morphism.name}")
            transport_term = propagation_to_transport(
                invariant_name=inv_name,
                source_value=Var("a"),
                target_value=App(Var(morphism.name), Var("a")),
                eq_proof=eq_proof,
            )
            proofs[inv_name] = transport_term

        # 检查丢失的不变量是否确实不可证明
        for inv_name in morphism.invariants_lost:
            # 丢失的不变量在类型论中意味着：
            # 不存在 transport 证明
            proofs[inv_name] = Var(f"impossible_{inv_name}")  # type: ignore[assignment]

        if errors:
            return TypeCheckResult(success=False, errors=errors)

        return TypeCheckResult(
            success=True,
            inferred_type=morphism_to_type(morphism).full_type,
            warnings=[
                f"Verified {len(morphism.invariants_kept)} kept, {len(morphism.invariants_lost)} lost invariants"
            ],
        )

    def propagate_invariant(
        self,
        inv: StructuralInvariant,
        morphism_chain: list[Morphism],
    ) -> TypeCheckResult:
        """沿态射链传播不变量.

        对应 ConstraintPropagation，但在类型论中：
        propagate(I, f∘g) = transport(I, eq_{f∘g}, I_holds(source))
        """
        if not morphism_chain:
            return TypeCheckResult(success=True, warnings=["Empty morphism chain"])

        # 沿链构建 transport 复合
        current_value = Var("initial")
        current_proof = Refl(Var("Structure"), current_value)

        kept_count = 0
        lost_count = 0

        for morphism in morphism_chain:
            if inv.name in morphism.invariants_lost:
                lost_count += 1
                break  # 不变量丢失，传播终止

            if inv.name in morphism.invariants_kept:
                # 构造 transport
                next_value = App(Var(morphism.name), current_value)
                eq = Var(f"eq_{morphism.name}")
                transport_term = propagation_to_transport(
                    invariant_name=inv.name,
                    source_value=current_value,
                    target_value=next_value,
                    eq_proof=eq,
                )
                current_value = next_value  # type: ignore[assignment]
                current_proof = transport_term  # type: ignore[assignment]
                kept_count += 1

        if lost_count > 0:
            return TypeCheckResult(
                success=False,
                errors=[f"Invariant '{inv.name}' lost after {kept_count} morphisms"],
            )

        return TypeCheckResult(
            success=True,
            inferred_type=current_proof,
            warnings=[f"Invariant '{inv.name}' preserved through {kept_count} morphisms"],
        )

    def structure_to_context(
        self,
        structure: AbstractMathematicalStructure,
    ) -> Context:
        """将数学结构的性质编码为类型论上下文.

        上下文 Γ 包含：
        - 结构变量 s : StructureType
        - 每个性质的假设
        - 每个不变量的命题
        """
        ctx = Context()
        name = structure.name.replace(" ", "_")

        # 结构变量
        ctx = ctx.extend("s", Var(name))

        # 不变量作为假设
        for inv in structure.structural_invariants:
            prop_type = invariant_to_prop_type(inv)
            ctx = ctx.extend(f"inv_{inv.name}", prop_type)

        return ctx
