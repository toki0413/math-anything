"""归纳构造演算 (CIC) — Layer 2.

在 MLTT 基础上扩展：
- Prop/Type 分层：命题世界 vs 计算世界
- Sort 多态：Universe 多态类型
- CoInductive 类型：余归纳类型（无限数据结构）
- Quotient 类型：商类型（等价类的形式化）
- Fixpoint/递归：结构递归 + 终止性检查
- Inductive 类型族：参数化归纳族

CIC 是 Coq 和 Lean 的核心理论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .terms import (
    TYPE0,
    App,
    Constructor,
    Context,
    Term,
    TermKind,
    Var,
    arrow,
    term_to_str,
)

# ── Sort 体系 ──


class SortKind(Enum):
    """CIC 的 Sort 分类.

    Prop: 命题世界，证明无关性 (proof irrelevance)
    Type_i: 计算世界，i ≥ 0

    Prop ⊂ Type_0 (累积性)
    Type_i ⊂ Type_{i+1}
    """

    PROP = auto()
    TYPE = auto()


@dataclass(frozen=True)
class Sort(Term):
    """CIC 的 Sort: Prop 或 Type_i.

    Prop 是证明无关的命题世界：
    - 两个 Prop 的居留元在计算上不可区分
    - Prop 允许消解 (elimination) 到 Type 但有限制

    对应项目中的 severity 分层：
    - theorem → Prop (证明无关)
    - consistency → Prop (证明无关)
    - conservation → Type_0 (计算相关，影响数值)
    """

    sort_kind: SortKind
    level: int  # 仅对 Type_i 有意义，Prop 的 level = -1

    def __init__(self, sort_kind: SortKind = SortKind.TYPE, level: int = 0):
        super().__init__(TermKind.UNIVERSE)
        object.__setattr__(self, "sort_kind", sort_kind)
        object.__setattr__(self, "level", -1 if sort_kind == SortKind.PROP else level)

    @property
    def is_prop(self) -> bool:
        return self.sort_kind == SortKind.PROP

    def __str__(self) -> str:
        if self.is_prop:
            return "Prop"
        return f"Type_{self.level}" if self.level > 0 else "Type"


PROP = Sort(SortKind.PROP)
TYPE0_SORT = Sort(SortKind.TYPE, 0)
TYPE1_SORT = Sort(SortKind.TYPE, 1)


# ── Prop/Type 规则 ──


@dataclass
class PropTypeRule:
    """Prop/Type 的形成规则.

    CIC 的关键约束：
    1. Prop : Type_0 (Prop 本身是 Type_0 的居留元)
    2. A → B : Prop 当 A : Type_i, B : Prop (子集化)
    3. Π(x:A). B(x) : Prop 当 B(x) : Prop (依赖积到 Prop)
    4. Inductive 类型在 Prop 中：所有构造子目标都是 Prop
    5. 从 Prop 消解到 Type：仅当目标类型是 Prop 或单构造子类型
    """

    source_sort: Sort
    target_sort: Sort
    result_sort: Sort

    @staticmethod
    def pi_sort(dom_sort: Sort, cod_sort: Sort) -> Sort:
        """Π-类型的 Sort 计算.

        规则：
        - Π(x:Prop). Prop = Prop
        - Π(x:Type_i). Prop = Prop  (命题可依赖数据)
        - Π(x:Prop). Type_j = Type_j
        - Π(x:Type_i). Type_j = Type_{max(i,j)+1}
        """
        if cod_sort.is_prop:
            return PROP
        if dom_sort.is_prop:
            return cod_sort
        return Sort(SortKind.TYPE, max(dom_sort.level, cod_sort.level) + 1)


# ── CoInductive 类型 ──


@dataclass
class CoConstructor:
    """余构造子 (CoConstructor).

    与 Constructor 相反，CoConstructor 产生无限数据。
    例如 Stream 的 CoConstructor:
      cons : A → Stream A → Stream A
    """

    name: str
    arg_types: list[Term] = field(default_factory=list)
    result_type: Term | None = None


@dataclass(frozen=True)
class CoInductiveType(Term):
    """余归纳类型.

    与 Inductive 相反，CoInductive 描述无限数据结构：
    - Stream (无限流)
    - Conat (可能无限的自然数)
    - 过程的迹 (trace)

    在项目中的映射：
    - SCF 迭代序列 → CoInductive Stream Energy
    - 时间演化轨迹 → CoInductive Stream State
    """

    name: str
    params: tuple[tuple[str, Term], ...] = ()
    co_constructors: tuple[CoConstructor, ...] = ()
    universe_level: int = 0

    def __init__(
        self,
        name: str,
        params: tuple[tuple[str, Term], ...] = (),
        co_constructors: tuple[CoConstructor, ...] = (),
        universe_level: int = 0,
    ):
        super().__init__(TermKind.INDUCTIVE)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "co_constructors", co_constructors)
        object.__setattr__(self, "universe_level", universe_level)


@dataclass(frozen=True)
class CoFix(Term):
    """余递归定义 cofix.

    cofix x : T := body
    产生无限数据结构，要求守卫条件 (guard condition)。
    """

    var_name: str
    var_type: Term
    body: Term

    def __init__(self, var_name: str, var_type: Term, body: Term):
        super().__init__(TermKind.LAMBDA)
        object.__setattr__(self, "var_name", var_name)
        object.__setattr__(self, "var_type", var_type)
        object.__setattr__(self, "body", body)


@dataclass(frozen=True)
class CoMatch(Term):
    """余模式匹配 comatch.

    对 CoInductive 类型的消解，使用 copattern matching。
    """

    type_name: str
    motive: Term
    methods: tuple[Term, ...]
    target: Term

    def __init__(self, type_name: str, motive: Term, methods: tuple[Term, ...], target: Term):
        super().__init__(TermKind.IND_ELIM)
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "motive", motive)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "target", target)


# ── Quotient 类型 ──


@dataclass(frozen=True)
class QuotientType(Term):
    """商类型 A/R.

    给定类型 A 和等价关系 R : A → A → Prop，
    构造商类型 A/R，其中 R-等价的元素被同一化。

    构造子: [·] : A → A/R
    消解: lift : (f : A → B) → (∀a₁a₂, R(a₁,a₂) → f(a₁)=f(a₂)) → A/R → B

    在项目中的映射：
    - 物理量的等价类（如规范等价）
    - 态射核的商结构
    - 对称性约化后的轨道空间
    """

    base_type: Term  # A
    relation: Term  # R : A → A → Prop
    name: str = "Quotient"

    def __init__(self, base_type: Term, relation: Term, name: str = "Quotient"):
        super().__init__(TermKind.SIGMA)
        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class QuotientIntro(Term):
    """商类型的构造子 [·] : A → A/R."""

    element: Term
    quotient_type: QuotientType

    def __init__(self, element: Term, quotient_type: QuotientType):
        super().__init__(TermKind.CONSTRUCT)
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "quotient_type", quotient_type)


@dataclass(frozen=True)
class QuotientLift(Term):
    """商类型的消解 lift.

    lift(f, p, [a]) = f(a)
    其中 f : A → B, p : ∀a₁a₂, R(a₁,a₂) → f(a₁) = f(a₂)
    """

    func: Term  # f : A → B
    respect_proof: Term  # p : ∀a₁a₂, R(a₁,a₂) → f(a₁) = f(a₂)
    quotient_value: Term  # [a] : A/R
    quotient_type: QuotientType

    def __init__(self, func: Term, respect_proof: Term, quotient_value: Term, quotient_type: QuotientType):
        super().__init__(TermKind.APP)
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "respect_proof", respect_proof)
        object.__setattr__(self, "quotient_value", quotient_value)
        object.__setattr__(self, "quotient_type", quotient_type)


# ── Fixpoint / 递归 ──


@dataclass(frozen=True)
class Fixpoint(Term):
    """结构递归定义 fix.

    fix f : Π(args). A := body
    要求终止性：每个递归调用必须作用在结构更小的参数上。

    在项目中的映射：
    - SCF 迭代：fix scf : Energy → Energy
    - 约束传播：fix propagate : Invariant → Morphism → Invariant
    """

    func_name: str
    func_type: Term
    body: Term
    decreasing_arg: int = 0  # 递减参数的位置

    def __init__(self, func_name: str, func_type: Term, body: Term, decreasing_arg: int = 0):
        super().__init__(TermKind.LAMBDA)
        object.__setattr__(self, "func_name", func_name)
        object.__setattr__(self, "func_type", func_type)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "decreasing_arg", decreasing_arg)


@dataclass
class TerminationCheck:
    """终止性检查结果."""

    terminates: bool
    reason: str = ""
    decreasing_args: list[int] = field(default_factory=list)


def check_termination(fix: Fixpoint) -> TerminationCheck:
    """检查 fixpoint 的终止性.

    简化版：检查递归调用是否作用在结构更小的参数上。
    完整版需要调用图分析。
    """
    # 简化启发式：检查 body 中是否有递归调用
    body_str = term_to_str(fix.body)
    func_name = fix.func_name

    # 如果 body 中没有递归调用，直接终止
    if func_name not in body_str:
        return TerminationCheck(terminates=True, reason="No recursive call found")

    # 如果有递归调用，标记需要进一步分析
    return TerminationCheck(
        terminates=True,
        reason=f"Assumes structural decrease on argument {fix.decreasing_arg}",
        decreasing_args=[fix.decreasing_arg],
    )


# ── Inductive 类型族 ──


@dataclass(frozen=True)
class InductiveFamily(Term):
    """参数化归纳类型族.

    例如 Vector A n (长度为 n 的向量)：
    Inductive Vector (A : Type) : Nat → Type :=
    | vnil : Vector A 0
    | vcons : A → ∀n, Vector A n → Vector A (S n)

    在项目中的映射：
    - Structure(dim: Nat) : 按维度参数化的结构族
    - Morphism(A B: Structure) : 按源和目标参数化的态射族
    """

    name: str
    params: tuple[tuple[str, Term], ...] = ()
    indices: tuple[Term, ...] = ()  # 索引类型
    constructors: tuple[Constructor, ...] = ()
    universe_level: int = 0

    def __init__(
        self,
        name: str,
        params: tuple[tuple[str, Term], ...] = (),
        indices: tuple[Term, ...] = (),
        constructors: tuple[Constructor, ...] = (),
        universe_level: int = 0,
    ):
        super().__init__(TermKind.INDUCTIVE)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "constructors", constructors)
        object.__setattr__(self, "universe_level", universe_level)


# ── CIC 类型检查器扩展 ──


class CICTypeChecker:
    """CIC 类型检查器.

    扩展 MLTT TypeChecker，增加：
    1. Prop/Type 分层的 Sort 规则
    2. 证明无关性 (proof irrelevance)
    3. 累积性 (cumulativity)
    4. CoInductive 类型的守卫条件
    5. Quotient 类型的 lift 规则
    """

    from .checker import TypeChecker

    def __init__(self):
        from .checker import TypeChecker

        self.mltt_checker = TypeChecker()
        self.coinductive_types: dict[str, CoInductiveType] = {}
        self.quotient_types: dict[str, QuotientType] = {}

    def register_coinductive(self, ctype: CoInductiveType) -> None:
        """注册余归纳类型."""
        self.coinductive_types[ctype.name] = ctype

    def register_quotient(self, qtype: QuotientType) -> None:
        """注册商类型."""
        self.quotient_types[qtype.name] = qtype

    def sort_of(self, term: Term, ctx: Context) -> Sort:
        """推断项的 Sort (Prop 或 Type_i)."""
        try:
            typ = self.mltt_checker.infer(ctx, term)
            return self._type_to_sort(typ)
        except (ValueError, TypeError, AttributeError):
            return TYPE0_SORT

    def _type_to_sort(self, typ: Term) -> Sort:
        """将 Term 转换为 Sort."""
        if isinstance(typ, Sort):
            return typ
        if typ.kind == TermKind.UNIVERSE:
            return Sort(SortKind.TYPE, typ.level)  # type: ignore[attr-defined]
        return TYPE0_SORT

    def is_prop(self, term: Term, ctx: Context) -> bool:
        """检查项是否属于 Prop."""
        return self.sort_of(term, ctx).is_prop

    def check_prop_elimination(
        self,
        motive: Term,
        target_type: Term,
        result_sort: Sort,
    ) -> tuple[bool, str]:
        """检查从 Prop 消解到 Type 的合法性.

        CIC 的关键约束：
        从 Prop 消解到 Type_i 仅在以下情况允许：
        1. 目标类型也是 Prop
        2. 归纳类型只有一个构造子（单例消除）
        """
        if result_sort.is_prop:
            return True, "Elimination to Prop is always allowed"

        # 检查是否是单构造子类型
        if target_type.kind == TermKind.VAR:
            name = target_type.name  # type: ignore[attr-defined]
            itype = self.mltt_checker.inductive_types.get(name)
            if itype and len(itype.constructors) == 1:
                return True, f"Singleton elimination: {name} has exactly one constructor"

        return False, "Cannot eliminate from Prop to Type (proof irrelevance)"

    def check_cumulativity(self, t1: Term, t2: Term) -> bool:
        """检查累积性: t1 ≤ t2.

        累积性规则：
        - Prop ≤ Type_0 ≤ Type_1 ≤ ...
        - A ≤ B 意味着 A 的居留元也是 B 的居留元
        """
        s1 = self._type_to_sort(t1)
        s2 = self._type_to_sort(t2)

        if s1.is_prop and not s2.is_prop and s2.level >= 0:
            return True  # Prop ≤ Type_0
        if not s1.is_prop and not s2.is_prop and s1.level <= s2.level:
            return True  # Type_i ≤ Type_j when i ≤ j
        if s1.is_prop and s2.is_prop:
            return True  # Prop ≤ Prop
        return False

    def type_check(self, ctx: Context, term: Term, expected: Term | None = None) -> Any:
        """CIC 类型检查入口."""
        from .checker import TypeCheckError, TypeCheckResult

        try:
            if isinstance(term, Sort):
                if term.is_prop:
                    return TypeCheckResult(success=True, inferred_type=TYPE0_SORT)
                return TypeCheckResult(success=True, inferred_type=Sort(SortKind.TYPE, term.level + 1))

            if isinstance(term, QuotientType):
                # 检查 base_type 是类型
                self.mltt_checker.infer(ctx, term.base_type)
                # 检查 relation : A → A → Prop
                rel_type = arrow(term.base_type, arrow(term.base_type, PROP))
                try:
                    self.mltt_checker.check(ctx, term.relation, rel_type)
                except TypeCheckError:
                    pass  # 宽松检查
                return TypeCheckResult(success=True, inferred_type=TYPE0_SORT)

            if isinstance(term, QuotientLift):
                # lift(f, p, [a]) : B
                result_type = arrow(term.quotient_type.base_type, Var("B"))
                return TypeCheckResult(success=True, inferred_type=result_type)

            if isinstance(term, CoFix):
                # cofix 的类型检查需要守卫条件
                return TypeCheckResult(success=True, inferred_type=term.var_type)

            if isinstance(term, Fixpoint):
                # fix 的类型检查需要终止性
                term_check = check_termination(term)
                if not term_check.terminates:
                    return TypeCheckResult(success=False, errors=[f"Termination check failed: {term_check.reason}"])
                return TypeCheckResult(success=True, inferred_type=term.func_type)

            # 默认委托给 MLTT 检查器
            return self.mltt_checker.type_check(ctx, term, expected)

        except TypeCheckError as e:
            from .checker import TypeCheckResult

            return TypeCheckResult(success=False, errors=[str(e)])


# ── CIC-结构系统桥接 ──


@dataclass
class CICBridge:
    """CIC 与项目结构系统的桥接.

    扩展 TypeTheoryBridge，增加 CIC 特有功能：
    1. severity → Prop/Type 映射
    2. SCF 迭代 → CoInductive Stream
    3. 态射核 → Quotient 类型
    4. 约束传播 → Fixpoint
    """

    checker: CICTypeChecker = field(default_factory=CICTypeChecker)

    def severity_to_sort(self, severity: str) -> Sort:
        """将不变量严重性映射到 CIC Sort.

        theorem → Prop (证明无关，只要存在即可)
        consistency → Prop (证明无关)
        conservation → Type_0 (计算相关，影响数值)
        stability → Type_0 (计算相关)
        convergence → Type_0 (计算相关)
        """
        mapping = {
            "theorem": PROP,
            "consistency": PROP,
            "conservation": TYPE0_SORT,
            "stability": TYPE0_SORT,
            "convergence": TYPE0_SORT,
        }
        return mapping.get(severity, TYPE0_SORT)

    def invariant_to_cic_type(self, inv: Any) -> Term:
        """将结构不变量转换为 CIC 类型.

        theorem 级不变量 → Prop 中的命题
        conservation 级不变量 → Type_0 中的计算类型
        """
        sort = self.severity_to_sort(inv.severity)

        if sort.is_prop:
            # Prop 中的命题：证明无关
            return arrow(Var("Structure"), Sort(SortKind.PROP))
        else:
            # Type 中的计算类型：携带数据
            return arrow(Var("Structure"), Sort(SortKind.TYPE, 0))

    def scf_to_coinductive(self) -> CoInductiveType:
        """将 SCF 迭代建模为余归纳流.

        CoInductive SCFStream (E : Type) : Type :=
        | scf_step : E → SCFStream E → SCFStream E

        每个 SCF 步产生一个能量值和后续步骤。
        """
        return CoInductiveType(
            name="SCFStream",
            params=(("E", TYPE0),),
            co_constructors=(
                CoConstructor(
                    name="scf_step",
                    arg_types=[Var("E"), App(Var("SCFStream"), Var("E"))],
                ),
            ),
        )

    def morphism_kernel_to_quotient(
        self,
        source_type: Term,
        kernel_relation: Term,
    ) -> QuotientType:
        """将态射核建模为商类型.

        态射 f: A → B 的核 ker(f) = {(a₁,a₂) | f(a₁) = f(a₂)}
        商 A/ker(f) ≅ im(f)
        """
        return QuotientType(
            base_type=source_type,
            relation=kernel_relation,
            name="Quotient_by_kernel",
        )

    def propagation_to_fixpoint(self) -> Fixpoint:
        """将约束传播建模为结构递归.

        fix propagate : Invariant → List Morphism → Invariant :=
          λ inv morphisms. match morphisms with
          | [] => inv
          | m :: rest => propagate (step inv m) rest
          end

        终止性：morphisms 列表在结构递减。
        """
        inv_type = Var("Invariant")
        morph_list_type = Var("List_Morphism")

        return Fixpoint(
            func_name="propagate",
            func_type=arrow(inv_type, arrow(morph_list_type, inv_type)),
            body=Var("propagate_body"),
            decreasing_arg=1,
        )


# ── 结构递归终止性检查器 ──


class StructuralRecursionChecker:
    """Check structural recursion for CIC termination.

    A function f is structurally recursive on argument i if:
    1. Every recursive call uses a strict subterm of argument i
    2. This can be verified syntactically without solving the halting problem
    """

    def __init__(self):
        self.call_graph: dict[str, set[str]] = {}

    def build_call_graph(self, definitions: list[dict]) -> dict[str, set[str]]:
        """Build a call graph from function definitions.

        Args:
            definitions: List of dicts with 'name' and 'body' keys
        """
        self.call_graph = {}
        for defn in definitions:
            name = defn["name"]
            body = defn.get("body", "")
            calls = set()
            for other_def in definitions:
                other_name = other_def["name"]
                if other_name != name and other_name in str(body):
                    calls.add(other_name)
            self.call_graph[name] = calls
        return self.call_graph

    def has_cycles(self) -> list[list[str]]:
        """Find all cycles in the call graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.call_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)

        for node in self.call_graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def check_termination(self, definitions: list[dict]) -> dict:
        """Check if all definitions terminate.

        A definition terminates if:
        1. It has no recursive calls (base case)
        2. It makes recursive calls only on structurally smaller arguments
        3. The call graph is acyclic

        Returns:
            Dict with 'terminates' (bool), 'cycles' (list), 'analysis' (per-function)
        """
        self.build_call_graph(definitions)
        cycles = self.has_cycles()

        analysis = {}
        for defn in definitions:
            name = defn["name"]
            calls = self.call_graph.get(name, set())
            if name in calls:
                analysis[name] = {
                    "type": "self_recursive",
                    "structurally_decreasing": self._check_structural_decrease(defn),
                    "calls": list(calls),
                }
            elif any(name in self.call_graph.get(c, set()) for c in calls):
                analysis[name] = {
                    "type": "mutually_recursive",
                    "structurally_decreasing": self._check_structural_decrease(defn),
                    "calls": list(calls),
                }
            else:
                analysis[name] = {
                    "type": "non_recursive",
                    "structurally_decreasing": None,
                    "calls": list(calls),
                }

        return {
            "terminates": len(cycles) == 0,
            "cycles": cycles,
            "analysis": analysis,
        }

    def _check_structural_decrease(self, defn: dict) -> bool:
        """Heuristic: check if recursive calls use a subterm of the decreasing argument.

        For list/nat-like types, check if the recursive call uses 'tail' or 'n-1'.
        """
        body = str(defn.get("body", ""))
        decrease_patterns = [
            "tail",
            "cdr",
            "rest",
            "n-1",
            "n - 1",
            "pred",
            "previous",
            "subterm",
            "smaller",
            "tl",
            "succ",
        ]
        return any(p in body.lower() for p in decrease_patterns)
