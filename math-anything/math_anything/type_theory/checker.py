"""MLTT 双向类型检查器.

实现了 Martin-Löf 类型理论的判断性类型检查：
- infer(Γ, t): 从项推断类型
- check(Γ, t, A): 检查项是否具有给定类型
- def_eq(Γ, t1, t2): 判断相等性检查

双向类型检查的核心思想：
- 构造子（introduction rules）用 check 模式
- 消去子（elimination rules）用 infer 模式
- 注解 (t : A) 连接两个方向
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from math_anything.rust_bridge import EMLAccelerator

from .terms import (
    TYPE0,
    App,
    Context,
    Identity,
    InductiveType,
    Judgment,
    Proj1,
    Term,
    TermKind,
    Universe,
    Var,
    substitute,
    term_to_str,
    whnf,
)

_accel = EMLAccelerator()
_logger = logging.getLogger(__name__)
_logger.info(f"Type checker WHNF backend: {'Rust' if _accel.using_rust else 'Python'}")


class TypeCheckError(Exception):
    """类型检查失败."""

    def __init__(self, msg: str, context: Context | None = None, term: Term | None = None):
        self.ctx = context
        self.term = term
        super().__init__(msg)


@dataclass
class TypeCheckResult:
    """类型检查结果."""

    success: bool
    inferred_type: Term | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


class TypeChecker:
    """MLTT 双向类型检查器.

    支持的类型规则：
    - Universe: Type_i : Type_{i+1}
    - Pi: Π(x:A).B(x) : Type_{max(i,j)+1} when A : Type_i, B(x) : Type_j
    - Lambda: λx.b : Π(x:A).B(x) when Γ,x:A ⊢ b : B(x)
    - App: f(a) : B[a/x] when f : Π(x:A).B(x), a : A
    - Sigma: Σ(x:A).B(x) : Type_{max(i,j)+1}
    - Pair: (a,b) : Σ(x:A).B(x) when a:A, b:B[a/x]
    - Identity: Id_A(a,b) : Type_0 when A : Type_i, a,b : A
    - Refl: refl_A(a) : Id_A(a,a) when a : A
    - Transport: transport(P,p,va) : P(b) when p:Id_A(a,b), va:P(a)
    """

    # 已注册的归纳类型
    inductive_types: dict[str, InductiveType]

    def __init__(self):
        self.inductive_types = {}
        # 注册预定义类型
        from .terms import BOOL_TYPE, NAT_TYPE

        self.inductive_types["Bool"] = BOOL_TYPE
        self.inductive_types["Nat"] = NAT_TYPE

    def register_inductive(self, itype: InductiveType) -> None:
        """注册归纳类型."""
        self.inductive_types[itype.name] = itype

    # ── 推断模式 ──

    def infer(self, ctx: Context, term: Term) -> Term:
        """从项推断类型: Γ ⊢ t ⇒ A.

        Raises TypeCheckError if type cannot be inferred.
        """
        kind = term.kind

        if kind == TermKind.VAR:
            typ = ctx.lookup(term.name)
            if typ is None:
                raise TypeCheckError(f"Unbound variable: {term.name}", context=ctx, term=term)
            return typ

        elif kind == TermKind.UNIVERSE:
            # Type_i : Type_{i+1}
            return Universe(term.level + 1)

        elif kind == TermKind.PI:
            # Π(x:A).B : Type_{max(i,j)+1}
            a_type = self.infer(ctx, term.domain)
            if a_type.kind != TermKind.UNIVERSE:
                raise TypeCheckError(f"Pi domain must be a type, got {term_to_str(a_type)}", context=ctx, term=term)
            i = a_type.level
            ctx_ext = ctx.extend(term.var_name, term.domain)
            b_type = self.infer(ctx_ext, term.codomain)
            if b_type.kind != TermKind.UNIVERSE:
                raise TypeCheckError(f"Pi codomain must be a type, got {term_to_str(b_type)}", context=ctx, term=term)
            j = b_type.level
            return Universe(max(i, j) + 1)

        elif kind == TermKind.LAMBDA:
            # λ-抽象只能用 check 模式，不能推断
            raise TypeCheckError(
                "Cannot infer type of lambda without annotation. Use (λx. t : Π(x:A).B) or check mode.",
                context=ctx,
                term=term,
            )

        elif kind == TermKind.APP:
            # f(a): 先推断 f 的类型，检查是 Π-类型，再检查参数
            func_type = self.infer(ctx, term.func)
            func_type = whnf(func_type)
            if func_type.kind != TermKind.PI:
                raise TypeCheckError(f"Cannot apply non-function type {term_to_str(func_type)}", context=ctx, term=term)
            self.check(ctx, term.arg, func_type.domain)
            return substitute(func_type.codomain, func_type.var_name, term.arg)

        elif kind == TermKind.SIGMA:
            a_type = self.infer(ctx, term.fst_type)
            if a_type.kind != TermKind.UNIVERSE:
                raise TypeCheckError(
                    f"Sigma first component must be a type, got {term_to_str(a_type)}", context=ctx, term=term
                )
            i = a_type.level
            ctx_ext = ctx.extend(term.var_name, term.fst_type)
            b_type = self.infer(ctx_ext, term.snd_type)
            if b_type.kind != TermKind.UNIVERSE:
                raise TypeCheckError(
                    f"Sigma second component must be a type, got {term_to_str(b_type)}", context=ctx, term=term
                )
            j = b_type.level
            return Universe(max(i, j) + 1)

        elif kind == TermKind.PAIR:
            raise TypeCheckError(
                "Cannot infer type of pair without annotation. Use ((a,b) : Σ(x:A).B) or check mode.",
                context=ctx,
                term=term,
            )

        elif kind == TermKind.PROJ1:
            pair_type = self.infer(ctx, term.pair)
            pair_type = whnf(pair_type)
            if pair_type.kind != TermKind.SIGMA:
                raise TypeCheckError(
                    f"Cannot project from non-Sigma type {term_to_str(pair_type)}", context=ctx, term=term
                )
            return pair_type.fst_type

        elif kind == TermKind.PROJ2:
            pair_type = self.infer(ctx, term.pair)
            pair_type = whnf(pair_type)
            if pair_type.kind != TermKind.SIGMA:
                raise TypeCheckError(
                    f"Cannot project from non-Sigma type {term_to_str(pair_type)}", context=ctx, term=term
                )
            return substitute(pair_type.snd_type, pair_type.var_name, Proj1(term.pair))

        elif kind == TermKind.IDENTITY:
            a_type = self.infer(ctx, term.typ)
            if a_type.kind != TermKind.UNIVERSE:
                raise TypeCheckError(
                    f"Identity type requires a type, got {term_to_str(a_type)}", context=ctx, term=term
                )
            self.check(ctx, term.lhs, term.typ)
            self.check(ctx, term.rhs, term.typ)
            return TYPE0

        elif kind == TermKind.REFL:
            a_type = self.infer(ctx, term.typ)
            self.check(ctx, term.term, term.typ)
            return Identity(term.typ, term.term, term.term)

        elif kind == TermKind.SYM:
            p_type = self.infer(ctx, term.proof)
            if p_type.kind != TermKind.IDENTITY:
                raise TypeCheckError(
                    f"sym requires identity type proof, got {term_to_str(p_type)}", context=ctx, term=term
                )
            return Identity(p_type.typ, p_type.rhs, p_type.lhs)

        elif kind == TermKind.TRANS:
            p_type = self.infer(ctx, term.p)
            q_type = self.infer(ctx, term.q)
            if p_type.kind != TermKind.IDENTITY:
                raise TypeCheckError(
                    f"trans first arg requires identity proof, got {term_to_str(p_type)}", context=ctx, term=term
                )
            if q_type.kind != TermKind.IDENTITY:
                raise TypeCheckError(
                    f"trans second arg requires identity proof, got {term_to_str(q_type)}", context=ctx, term=term
                )
            # 检查 p 的右端等于 q 的左端
            if not self.def_eq(ctx, p_type.rhs, q_type.lhs):
                raise TypeCheckError(
                    f"trans: endpoints don't match: {term_to_str(p_type.rhs)} ≠ {term_to_str(q_type.lhs)}",
                    context=ctx,
                    term=term,
                )
            if not self.def_eq(ctx, p_type.typ, q_type.typ):
                raise TypeCheckError(
                    f"trans: base types don't match: {term_to_str(p_type.typ)} ≠ {term_to_str(q_type.typ)}",
                    context=ctx,
                    term=term,
                )
            return Identity(p_type.typ, p_type.lhs, q_type.rhs)

        elif kind == TermKind.CONG:
            p_type = self.infer(ctx, term.proof)
            if p_type.kind != TermKind.IDENTITY:
                raise TypeCheckError(f"cong requires identity proof, got {term_to_str(p_type)}", context=ctx, term=term)
            # 检查 func : dom_type → cod_type
            func_type = self.infer(ctx, term.func)
            func_type_whnf = whnf(func_type)
            if func_type_whnf.kind != TermKind.PI:
                raise TypeCheckError(f"cong requires function, got {term_to_str(func_type)}", context=ctx, term=term)
            return Identity(term.cod_type, App(term.func, p_type.lhs), App(term.func, p_type.rhs))

        elif kind == TermKind.TRANSPORT:
            # transport(P, p, va) : P(b)
            # p : Id_A(a, b), va : P(a)
            p_type = self.infer(ctx, term.eq_proof)
            if p_type.kind != TermKind.IDENTITY:
                raise TypeCheckError(
                    f"transport requires identity proof, got {term_to_str(p_type)}", context=ctx, term=term
                )
            # 检查 motive : A → Type
            motive_type = self.infer(ctx, term.motive)
            motive_whnf = whnf(motive_type)
            if motive_whnf.kind != TermKind.PI:
                raise TypeCheckError(
                    f"transport motive must be a dependent function A → Type, got {term_to_str(motive_type)}",
                    context=ctx,
                    term=term,
                )
            # P(a) 的类型
            pa_type = App(term.motive, term.a)
            self.check(ctx, term.value, pa_type)
            # 结果类型 P(b)
            return App(term.motive, term.b)

        elif kind == TermKind.ANNOTATION:
            # (t : A): 先检查 A 是类型，再用 check 模式
            self.infer(ctx, term.typ)
            self.check(ctx, term.term, term.typ)
            return term.typ

        elif kind == TermKind.INDUCTIVE:
            # 归纳类型名本身是一个类型
            return Universe(term.universe_level + 1)

        elif kind == TermKind.CONSTRUCT:
            itype = self.inductive_types.get(term.type_name)
            if itype is None:
                raise TypeCheckError(f"Unknown inductive type: {term.type_name}", context=ctx, term=term)
            # 找到构造子
            for ctor in itype.constructors:
                if ctor.name == term.ctor_name:
                    if len(term.args) != len(ctor.arg_types):
                        raise TypeCheckError(
                            f"Constructor {term.ctor_name} expects {len(ctor.arg_types)} args, got {len(term.args)}",
                            context=ctx,
                            term=term,
                        )
                    for arg, expected in zip(term.args, ctor.arg_types):
                        self.check(ctx, arg, expected)
                    return Var(term.type_name)
            raise TypeCheckError(f"Unknown constructor {term.ctor_name} of {term.type_name}", context=ctx, term=term)

        elif kind == TermKind.IND_ELIM:
            itype = self.inductive_types.get(term.type_name)
            if itype is None:
                raise TypeCheckError(f"Unknown inductive type: {term.type_name}", context=ctx, term=term)
            # 检查 target 的类型
            self.check(ctx, term.target, Var(term.type_name))
            # 结果类型: motive(target)
            return App(term.motive, term.target)

        raise TypeCheckError(f"Cannot infer type of {term.kind.name}", context=ctx, term=term)

    # ── 检查模式 ──

    def check(self, ctx: Context, term: Term, expected: Term) -> None:
        """检查项是否具有给定类型: Γ ⊢ t ⇐ A.

        Raises TypeCheckError if check fails.
        """
        kind = term.kind
        expected_whnf = whnf(expected)

        if kind == TermKind.LAMBDA and expected_whnf.kind == TermKind.PI:
            # λx. body ⇐ Π(x:A).B(x)
            # 检查: Γ, x:A ⊢ body ⇐ B(x)
            ctx_ext = ctx.extend(expected_whnf.var_name, expected_whnf.domain)
            body_expected = expected_whnf.codomain
            # 如果 λ 的变量名和 Π 的不同，需要重命名
            if term.var_name != expected_whnf.var_name:
                body_expected = substitute(body_expected, expected_whnf.var_name, Var(term.var_name))
                ctx_ext = ctx.extend(term.var_name, expected_whnf.domain)
            self.check(ctx_ext, term.body, body_expected)
            return

        elif kind == TermKind.PAIR and expected_whnf.kind == TermKind.SIGMA:
            # (a, b) ⇐ Σ(x:A).B(x)
            self.check(ctx, term.fst, expected_whnf.fst_type)
            snd_expected = substitute(expected_whnf.snd_type, expected_whnf.var_name, term.fst)
            self.check(ctx, term.snd, snd_expected)
            return

        elif kind == TermKind.REFL:
            # refl ⇐ Id_A(a, b)  当 a ≡ b
            if expected_whnf.kind == TermKind.IDENTITY:
                self.check(ctx, term.term, expected_whnf.typ)
                if not self.def_eq(ctx, term.term, expected_whnf.lhs):
                    raise TypeCheckError(
                        f"refl: term {term_to_str(term.term)} ≠ expected lhs {term_to_str(expected_whnf.lhs)}",
                        context=ctx,
                        term=term,
                    )
                if not self.def_eq(ctx, term.term, expected_whnf.rhs):
                    raise TypeCheckError(
                        f"refl: term {term_to_str(term.term)} ≠ expected rhs {term_to_str(expected_whnf.rhs)}",
                        context=ctx,
                        term=term,
                    )
                return
            # 如果期望类型不是 Id，尝试推断
            inferred = self.infer(ctx, term)
            if not self.def_eq(ctx, inferred, expected):
                raise TypeCheckError(
                    f"Type mismatch: inferred {term_to_str(inferred)}, expected {term_to_str(expected)}",
                    context=ctx,
                    term=term,
                )
            return

        # 默认: 先推断，再比较
        inferred = self.infer(ctx, term)
        if not self.def_eq(ctx, inferred, expected):
            raise TypeCheckError(
                f"Type mismatch: {term_to_str(term)} has type {term_to_str(inferred)}, expected {term_to_str(expected)}",
                context=ctx,
                term=term,
            )

    # ── 判断相等性 ──

    def def_eq(self, ctx: Context, t1: Term, t2: Term) -> bool:
        """判断相等性: Γ ⊢ t₁ ≡ t₂.

        使用 WHNF + 结构递归的方法。

        Note: Python WHNF 直接操作 Term 对象，是主路径。
        对于大型项，Rust WHNF 可通过序列化 Term → JSON 后调用
        _accel 加速，但当前保持 Python WHNF 为默认，
        因为 Term 对象与 JSON 之间的转换开销可能抵消加速收益。
        """
        # Rust WHNF 路径: 仅在 Rust 可用且项较大时尝试
        if _accel.using_rust and hasattr(_accel, "whnf"):
            try:
                t1n = _accel.whnf(t1)
                t2n = _accel.whnf(t2)
            except (ValueError, TypeError, RuntimeError):
                _logger.debug("Rust WHNF failed, falling back to Python")
                t1n = whnf(t1)
                t2n = whnf(t2)
        else:
            t1n = whnf(t1)
            t2n = whnf(t2)
        return self._def_eq_whnf(ctx, t1n, t2n)

    def _def_eq_whnf(self, ctx: Context, t1: Term, t2: Term) -> bool:
        """比较两个 WHNF 项的判断相等性."""
        # 相同对象
        if t1 is t2:
            return True

        k1, k2 = t1.kind, t2.kind

        # 不同种类直接不等
        if k1 != k2:
            # 宇宙累积性: Type_i ≡ Type_j only if i == j
            return False

        if k1 == TermKind.VAR:
            return t1.name == t2.name

        elif k1 == TermKind.UNIVERSE:
            return t1.level == t2.level

        elif k1 == TermKind.PI:
            # Π(x:A).B ≡ Π(x:A').B' when A ≡ A' and B ≡ B'
            if not self.def_eq(ctx, t1.domain, t2.domain):
                return False
            # 在扩展上下文中比较 codomain
            ctx_ext = ctx.extend(t1.var_name, t1.domain)
            # 重命名 t2 的变量
            cod2 = substitute(t2.codomain, t2.var_name, Var(t1.var_name))
            return self.def_eq(ctx_ext, t1.codomain, cod2)

        elif k1 == TermKind.LAMBDA:
            if t1.var_name == t2.var_name:
                ctx_ext = ctx.extend(t1.var_name, ctx.lookup(t1.var_name) or TYPE0)
                return self.def_eq(ctx_ext, t1.body, t2.body)
            # 重命名
            body2 = substitute(t2.body, t2.var_name, Var(t1.var_name))
            ctx_ext = ctx.extend(t1.var_name, ctx.lookup(t1.var_name) or TYPE0)
            return self.def_eq(ctx_ext, t1.body, body2)

        elif k1 == TermKind.APP:
            return self.def_eq(ctx, t1.func, t2.func) and self.def_eq(ctx, t1.arg, t2.arg)

        elif k1 == TermKind.SIGMA:
            if not self.def_eq(ctx, t1.fst_type, t2.fst_type):
                return False
            ctx_ext = ctx.extend(t1.var_name, t1.fst_type)
            snd2 = substitute(t2.snd_type, t2.var_name, Var(t1.var_name))
            return self.def_eq(ctx_ext, t1.snd_type, snd2)

        elif k1 == TermKind.PAIR:
            return self.def_eq(ctx, t1.fst, t2.fst) and self.def_eq(ctx, t1.snd, t2.snd)

        elif k1 in (TermKind.PROJ1, TermKind.PROJ2):
            return self.def_eq(ctx, t1.pair, t2.pair)

        elif k1 == TermKind.IDENTITY:
            return (
                self.def_eq(ctx, t1.typ, t2.typ)
                and self.def_eq(ctx, t1.lhs, t2.lhs)
                and self.def_eq(ctx, t1.rhs, t2.rhs)
            )

        elif k1 == TermKind.REFL:
            return self.def_eq(ctx, t1.typ, t2.typ) and self.def_eq(ctx, t1.term, t2.term)

        elif k1 == TermKind.INDUCTIVE:
            return t1.name == t2.name

        elif k1 == TermKind.CONSTRUCT:
            if t1.type_name != t2.type_name or t1.ctor_name != t2.ctor_name:
                return False
            if len(t1.args) != len(t2.args):
                return False
            return all(self.def_eq(ctx, a1, a2) for a1, a2 in zip(t1.args, t2.args))

        return False

    # ── 高层 API ──

    def type_check(self, ctx: Context, term: Term, expected: Term | None = None) -> TypeCheckResult:
        """类型检查入口.

        如果提供 expected，使用 check 模式；否则使用 infer 模式。
        """
        try:
            if expected is not None:
                self.check(ctx, term, expected)
                return TypeCheckResult(success=True, inferred_type=expected)
            else:
                typ = self.infer(ctx, term)
                return TypeCheckResult(success=True, inferred_type=typ)
        except TypeCheckError as e:
            return TypeCheckResult(success=False, errors=[str(e)])

    def verify_judgment(self, judgment: Judgment) -> TypeCheckResult:
        """验证一个类型论判断."""
        if judgment.judgment_type == "typing":
            return self.type_check(judgment.context, judgment.subject, judgment.predicate)
        elif judgment.judgment_type == "type_formation":
            try:
                typ = self.infer(judgment.context, judgment.subject)
                if typ.kind == TermKind.UNIVERSE:
                    return TypeCheckResult(success=True, inferred_type=typ)
                return TypeCheckResult(success=False, errors=[f"Not a type: {term_to_str(typ)}"])
            except TypeCheckError as e:
                return TypeCheckResult(success=False, errors=[str(e)])
        elif judgment.judgment_type in ("equality_type", "equality_term"):
            if judgment.predicate is None:
                return TypeCheckResult(success=False, errors=["Equality judgment requires predicate"])
            result = self.def_eq(judgment.context, judgment.subject, judgment.predicate)
            return TypeCheckResult(success=result)
        return TypeCheckResult(success=False, errors=[f"Unknown judgment type: {judgment.judgment_type}"])
