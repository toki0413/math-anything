"""Martin-Löf 类型理论的核心项语言.

实现了 MLTT 的语法层，包括：
- 项 (Term): 表达式的基本构造
- 类型 (Type): 包括依赖类型 Π, Σ, 恒等类型 Id
- 宇宙 (Universe): 累积层级 Type_0, Type_1, ...
- 上下文 (Context): 类型判断的环境
- 值 (Value): 规约后的范式

设计原则：
1. 项是语法对象（可序列化），值是语义对象（求值结果）
2. 支持双向类型检查的 infer/check 模式
3. 与项目现有 StructuralInvariant/Morphism 系统无缝桥接
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# ── 项的基类 ──


class TermKind(Enum):
    """项的分类标签."""

    VAR = auto()
    UNIVERSE = auto()
    PI = auto()
    LAMBDA = auto()
    APP = auto()
    SIGMA = auto()
    PAIR = auto()
    PROJ1 = auto()
    PROJ2 = auto()
    IDENTITY = auto()
    REFL = auto()
    SYM = auto()
    TRANS = auto()
    CONG = auto()
    TRANSPORT = auto()
    BOOL = auto()
    TRUE = auto()
    FALSE = auto()
    BOOL_ELIM = auto()
    NAT = auto()
    ZERO = auto()
    SUCC = auto()
    NAT_REC = auto()
    INDUCTIVE = auto()
    CONSTRUCT = auto()
    IND_ELIM = auto()
    ANNOTATION = auto()


@dataclass(frozen=True)
class Term:
    """MLTT 项的基类.

    所有项都是不可变的（frozen=True），支持哈希和比较。
    """

    kind: TermKind

    def __str__(self) -> str:
        return term_to_str(self)


@dataclass(frozen=True)
class Var(Term):
    """变量引用 x."""

    name: str

    def __init__(self, name: str):
        super().__init__(TermKind.VAR)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class Universe(Term):
    """宇宙类型 Type_i.

    累积性: Type_i : Type_{i+1}
    """

    level: int

    def __init__(self, level: int = 0):
        super().__init__(TermKind.UNIVERSE)
        object.__setattr__(self, "level", level)


@dataclass(frozen=True)
class Pi(Term):
    """依赖函数类型 Π(x:A). B(x).

    当 x 不在 B 中自由出现时退化为 A → B。
    """

    var_name: str
    domain: Term
    codomain: Term

    def __init__(self, var_name: str, domain: Term, codomain: Term):
        super().__init__(TermKind.PI)
        object.__setattr__(self, "var_name", var_name)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codomain", codomain)


@dataclass(frozen=True)
class Lam(Term):
    """λ-抽象 λx. body."""

    var_name: str
    body: Term

    def __init__(self, var_name: str, body: Term):
        super().__init__(TermKind.LAMBDA)
        object.__setattr__(self, "var_name", var_name)
        object.__setattr__(self, "body", body)


@dataclass(frozen=True)
class App(Term):
    """函数应用 f(a)."""

    func: Term
    arg: Term

    def __init__(self, func: Term, arg: Term):
        super().__init__(TermKind.APP)
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "arg", arg)


@dataclass(frozen=True)
class Sigma(Term):
    """依赖对类型 Σ(x:A). B(x).

    当 x 不在 B 中自由出现时退化为 A × B。
    """

    var_name: str
    fst_type: Term
    snd_type: Term

    def __init__(self, var_name: str, fst_type: Term, snd_type: Term):
        super().__init__(TermKind.SIGMA)
        object.__setattr__(self, "var_name", var_name)
        object.__setattr__(self, "fst_type", fst_type)
        object.__setattr__(self, "snd_type", snd_type)


@dataclass(frozen=True)
class Pair(Term):
    """依赖对构造 (a, b)."""

    fst: Term
    snd: Term

    def __init__(self, fst: Term, snd: Term):
        super().__init__(TermKind.PAIR)
        object.__setattr__(self, "fst", fst)
        object.__setattr__(self, "snd", snd)


@dataclass(frozen=True)
class Proj1(Term):
    """第一投影 π₁."""

    pair: Term

    def __init__(self, pair: Term):
        super().__init__(TermKind.PROJ1)
        object.__setattr__(self, "pair", pair)


@dataclass(frozen=True)
class Proj2(Term):
    """第二投影 π₂."""

    pair: Term

    def __init__(self, pair: Term):
        super().__init__(TermKind.PROJ2)
        object.__setattr__(self, "pair", pair)


@dataclass(frozen=True)
class Identity(Term):
    """恒等类型 Id_A(a, b) — 命题相等性.

    Id_A(a, b) 的居留元是 a 和 b 相等的证明。
    """

    typ: Term
    lhs: Term
    rhs: Term

    def __init__(self, typ: Term, lhs: Term, rhs: Term):
        super().__init__(TermKind.IDENTITY)
        object.__setattr__(self, "typ", typ)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)


@dataclass(frozen=True)
class Refl(Term):
    """自反性 refl_a : Id_A(a, a).

    判断相等的项具有自反证明。
    """

    typ: Term
    term: Term

    def __init__(self, typ: Term, term: Term):
        super().__init__(TermKind.REFL)
        object.__setattr__(self, "typ", typ)
        object.__setattr__(self, "term", term)


@dataclass(frozen=True)
class Sym(Term):
    """对称性 sym(p) : Id_A(b, a) from p : Id_A(a, b)."""

    typ: Term
    lhs: Term
    rhs: Term
    proof: Term

    def __init__(self, typ: Term, lhs: Term, rhs: Term, proof: Term):
        super().__init__(TermKind.SYM)
        object.__setattr__(self, "typ", typ)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "proof", proof)


@dataclass(frozen=True)
class Trans(Term):
    """传递性 trans(p, q) : Id_A(a, c) from p : Id_A(a, b), q : Id_A(b, c)."""

    typ: Term
    a: Term
    b: Term
    c: Term
    p: Term
    q: Term

    def __init__(self, typ: Term, a: Term, b: Term, c: Term, p: Term, q: Term):
        super().__init__(TermKind.TRANS)
        object.__setattr__(self, "typ", typ)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)


@dataclass(frozen=True)
class Cong(Term):
    """合同性 cong(f, p) : Id_B(f(a), f(b)) from p : Id_A(a, b)."""

    func: Term
    dom_type: Term
    cod_type: Term
    a: Term
    b: Term
    proof: Term

    def __init__(self, func: Term, dom_type: Term, cod_type: Term, a: Term, b: Term, proof: Term):
        super().__init__(TermKind.CONG)
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "dom_type", dom_type)
        object.__setattr__(self, "cod_type", cod_type)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "proof", proof)


@dataclass(frozen=True)
class Transport(Term):
    """传输 transport(P, p, a) : P(b) from p : Id_A(a, b), a : P(a).

    这是约束传播在类型论中的对应物：
    沿等式证明 p 将 P(a) 中的居留元传输到 P(b)。
    """

    motive: Term  # P : A → Type
    eq_proof: Term  # p : Id_A(a, b)
    source_type: Term  # A
    a: Term  # source
    b: Term  # target
    value: Term  # 居留元 of P(a)

    def __init__(self, motive: Term, eq_proof: Term, source_type: Term, a: Term, b: Term, value: Term):
        super().__init__(TermKind.TRANSPORT)
        object.__setattr__(self, "motive", motive)
        object.__setattr__(self, "eq_proof", eq_proof)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class Annotation(Term):
    """类型标注 (t : A)."""

    term: Term
    typ: Term

    def __init__(self, term: Term, typ: Term):
        super().__init__(TermKind.ANNOTATION)
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "typ", typ)


# ── 归纳类型 ──


@dataclass
class Constructor:
    """归纳类型的构造子.

    例如 Bool 有两个构造子: true : Bool, false : Bool
    Nat 有两个构造子: zero : Nat, succ : Nat → Nat
    """

    name: str
    arg_types: list[Term] = field(default_factory=list)
    # 构造子的完整类型签名
    full_type: Term | None = None


@dataclass(frozen=True)
class InductiveType(Term):
    """归纳类型定义.

    MLTT 的核心构造：通过构造子和消去规则定义新类型。
    对应项目中的 AbstractMathematicalStructure 层次。
    """

    name: str
    params: tuple[tuple[str, Term], ...] = ()
    constructors: tuple[Constructor, ...] = ()
    universe_level: int = 0

    def __init__(
        self,
        name: str,
        params: tuple[tuple[str, Term], ...] = (),
        constructors: tuple[Constructor, ...] = (),
        universe_level: int = 0,
    ):
        super().__init__(TermKind.INDUCTIVE)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "constructors", constructors)
        object.__setattr__(self, "universe_level", universe_level)


@dataclass(frozen=True)
class Construct(Term):
    """归纳类型的构造子应用."""

    type_name: str
    ctor_name: str
    args: tuple[Term, ...] = ()

    def __init__(self, type_name: str, ctor_name: str, args: tuple[Term, ...] = ()):
        super().__init__(TermKind.CONSTRUCT)
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "ctor_name", ctor_name)
        object.__setattr__(self, "args", args)


@dataclass(frozen=True)
class IndElim(Term):
    """归纳类型的消去器（递归子/归纳原理）.

    ind_C(P, base, step, n) : P(n)
    """

    type_name: str
    motive: Term  # P : C → Type
    methods: tuple[Term, ...]  # 每个构造子对应一个方法
    target: Term  # 被消去的项

    def __init__(self, type_name: str, motive: Term, methods: tuple[Term, ...], target: Term):
        super().__init__(TermKind.IND_ELIM)
        object.__setattr__(self, "type_name", type_name)
        object.__setattr__(self, "motive", motive)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "target", target)


# ── 上下文 ──


@dataclass
class Context:
    """类型判断上下文 Γ = x₁:A₁, x₂:A₂, ...

    上下文是有序的变量绑定序列，支持扩展和查找。
    """

    bindings: list[tuple[str, Term]] = field(default_factory=list)

    def extend(self, name: str, typ: Term) -> Context:
        """扩展上下文 Γ, x:A."""
        return Context(bindings=self.bindings + [(name, typ)])

    def lookup(self, name: str) -> Term | None:
        """查找变量的类型（最近的绑定优先）."""
        for n, t in reversed(self.bindings):
            if n == name:
                return t
        return None

    def names(self) -> set[str]:
        """上下文中所有变量名."""
        return {n for n, _ in self.bindings}

    def __len__(self) -> int:
        return len(self.bindings)

    def __repr__(self) -> str:
        if not self.bindings:
            return "·"
        return ", ".join(f"{n}:{term_to_str(t)}" for n, t in self.bindings)


# ── 判断形式 ──


@dataclass
class Judgment:
    """类型论判断.

    MLTT 的四种基本判断：
    - Γ ⊢ A : Type        (A 是一个类型)
    - Γ ⊢ a : A           (a 是类型 A 的项)
    - Γ ⊢ A ≡ B : Type    (A 和 B 是判断相等的类型)
    - Γ ⊢ a ≡ b : A       (a 和 b 是判断相等的项)
    """

    context: Context
    subject: Term
    judgment_type: str  # "typing", "type_formation", "equality_type", "equality_term"
    predicate: Term | None = None  # 对于 typing 判断是类型，对于 equality 是比较对象

    def __str__(self) -> str:
        ctx = repr(self.context)
        subj = term_to_str(self.subject)
        if self.judgment_type == "typing":
            return f"{ctx} ⊢ {subj} : {term_to_str(self.predicate)}"  # type: ignore[arg-type]
        elif self.judgment_type == "type_formation":
            return f"{ctx} ⊢ {subj} : Type"
        elif self.judgment_type == "equality_type":
            return f"{ctx} ⊢ {subj} ≡ {term_to_str(self.predicate)} : Type"  # type: ignore[arg-type]
        elif self.judgment_type == "equality_term":
            return f"{ctx} ⊢ {subj} ≡ {term_to_str(self.predicate)}"  # type: ignore[arg-type]
        return f"{ctx} ⊢ {subj}"


# ── 自由变量和替换 ──


def free_vars(term: Term) -> set[str]:
    """收集项中的自由变量."""
    result: set[str] = set()
    _collect_free(term, result)
    return result


def _collect_free(term: Term, acc: set[str]) -> None:
    kind = term.kind
    if kind == TermKind.VAR:
        acc.add(term.name)  # type: ignore[attr-defined]
    elif kind == TermKind.UNIVERSE:
        pass
    elif kind == TermKind.PI:
        _collect_free(term.domain, acc)  # type: ignore[attr-defined]
        bound = {term.var_name}  # type: ignore[attr-defined]
        child_free = free_vars(term.codomain) - bound  # type: ignore[attr-defined]
        acc.update(child_free)
    elif kind == TermKind.LAMBDA:
        bound = {term.var_name}  # type: ignore[attr-defined]
        child_free = free_vars(term.body) - bound  # type: ignore[attr-defined]
        acc.update(child_free)
    elif kind == TermKind.APP:
        _collect_free(term.func, acc)  # type: ignore[attr-defined]
        _collect_free(term.arg, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.SIGMA:
        _collect_free(term.fst_type, acc)  # type: ignore[attr-defined]
        bound = {term.var_name}  # type: ignore[attr-defined]
        child_free = free_vars(term.snd_type) - bound  # type: ignore[attr-defined]
        acc.update(child_free)
    elif kind == TermKind.PAIR:
        _collect_free(term.fst, acc)  # type: ignore[attr-defined]
        _collect_free(term.snd, acc)  # type: ignore[attr-defined]
    elif kind in (TermKind.PROJ1, TermKind.PROJ2):
        _collect_free(term.pair, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.IDENTITY:
        _collect_free(term.typ, acc)  # type: ignore[attr-defined]
        _collect_free(term.lhs, acc)  # type: ignore[attr-defined]
        _collect_free(term.rhs, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.REFL:
        _collect_free(term.typ, acc)  # type: ignore[attr-defined]
        _collect_free(term.term, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.SYM:
        _collect_free(term.typ, acc)  # type: ignore[attr-defined]
        _collect_free(term.proof, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.TRANS:
        _collect_free(term.typ, acc)  # type: ignore[attr-defined]
        _collect_free(term.p, acc)  # type: ignore[attr-defined]
        _collect_free(term.q, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.CONG:
        _collect_free(term.func, acc)  # type: ignore[attr-defined]
        _collect_free(term.proof, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.TRANSPORT:
        _collect_free(term.motive, acc)  # type: ignore[attr-defined]
        _collect_free(term.eq_proof, acc)  # type: ignore[attr-defined]
        _collect_free(term.value, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.ANNOTATION:
        _collect_free(term.term, acc)  # type: ignore[attr-defined]
        _collect_free(term.typ, acc)  # type: ignore[attr-defined]
    elif kind == TermKind.INDUCTIVE:
        pass  # 归纳类型名不是变量
    elif kind == TermKind.CONSTRUCT:
        for a in term.args:  # type: ignore[attr-defined]
            _collect_free(a, acc)
    elif kind == TermKind.IND_ELIM:
        _collect_free(term.motive, acc)  # type: ignore[attr-defined]
        for m in term.methods:  # type: ignore[attr-defined]
            _collect_free(m, acc)
        _collect_free(term.target, acc)  # type: ignore[attr-defined]


def substitute(term: Term, var_name: str, replacement: Term) -> Term:
    """捕获避免替换 term[var_name := replacement]."""
    return _subst(term, var_name, replacement, set())


def _subst(term: Term, var_name: str, repl: Term, bound: set[str]) -> Term:
    kind = term.kind

    if kind == TermKind.VAR:
        return repl if term.name == var_name and term.name not in bound else term  # type: ignore[attr-defined]

    elif kind == TermKind.UNIVERSE:
        return term

    elif kind == TermKind.PI:
        new_bound = bound | {term.var_name}  # type: ignore[attr-defined]
        new_dom = _subst(term.domain, var_name, repl, bound)  # type: ignore[attr-defined]
        new_cod = _subst(term.codomain, var_name, repl, new_bound)  # type: ignore[attr-defined]
        if term.var_name == var_name:  # type: ignore[attr-defined]
            # 变量被遮蔽，不替换 codomain 中的同名变量
            return Pi(term.var_name, new_dom, term.codomain)  # type: ignore[attr-defined]
        return Pi(term.var_name, new_dom, new_cod)  # type: ignore[attr-defined]

    elif kind == TermKind.LAMBDA:
        new_bound = bound | {term.var_name}  # type: ignore[attr-defined]
        new_body = _subst(term.body, var_name, repl, new_bound)  # type: ignore[attr-defined]
        if term.var_name == var_name:  # type: ignore[attr-defined]
            return Lam(term.var_name, term.body)  # type: ignore[attr-defined]
        return Lam(term.var_name, new_body)  # type: ignore[attr-defined]

    elif kind == TermKind.APP:
        return App(_subst(term.func, var_name, repl, bound), _subst(term.arg, var_name, repl, bound))  # type: ignore[attr-defined]

    elif kind == TermKind.SIGMA:
        new_bound = bound | {term.var_name}  # type: ignore[attr-defined]
        new_fst = _subst(term.fst_type, var_name, repl, bound)  # type: ignore[attr-defined]
        new_snd = _subst(term.snd_type, var_name, repl, new_bound)  # type: ignore[attr-defined]
        if term.var_name == var_name:  # type: ignore[attr-defined]
            return Sigma(term.var_name, new_fst, term.snd_type)  # type: ignore[attr-defined]
        return Sigma(term.var_name, new_fst, new_snd)  # type: ignore[attr-defined]

    elif kind == TermKind.PAIR:
        return Pair(_subst(term.fst, var_name, repl, bound), _subst(term.snd, var_name, repl, bound))  # type: ignore[attr-defined]

    elif kind == TermKind.PROJ1:
        return Proj1(_subst(term.pair, var_name, repl, bound))  # type: ignore[attr-defined]
    elif kind == TermKind.PROJ2:
        return Proj2(_subst(term.pair, var_name, repl, bound))  # type: ignore[attr-defined]

    elif kind == TermKind.IDENTITY:
        return Identity(
            _subst(term.typ, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.lhs, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.rhs, var_name, repl, bound),  # type: ignore[attr-defined]
        )

    elif kind == TermKind.REFL:
        return Refl(_subst(term.typ, var_name, repl, bound), _subst(term.term, var_name, repl, bound))  # type: ignore[attr-defined]

    elif kind == TermKind.SYM:
        return Sym(
            _subst(term.typ, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.lhs, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.rhs, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.proof, var_name, repl, bound),  # type: ignore[attr-defined]
        )

    elif kind == TermKind.TRANSPORT:
        return Transport(
            _subst(term.motive, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.eq_proof, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.source_type, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.a, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.b, var_name, repl, bound),  # type: ignore[attr-defined]
            _subst(term.value, var_name, repl, bound),  # type: ignore[attr-defined]
        )

    elif kind == TermKind.ANNOTATION:
        return Annotation(_subst(term.term, var_name, repl, bound), _subst(term.typ, var_name, repl, bound))  # type: ignore[attr-defined]

    elif kind == TermKind.INDUCTIVE:
        return term

    elif kind == TermKind.CONSTRUCT:
        return Construct(term.type_name, term.ctor_name, tuple(_subst(a, var_name, repl, bound) for a in term.args))  # type: ignore[attr-defined]

    elif kind == TermKind.IND_ELIM:
        return IndElim(
            term.type_name,  # type: ignore[attr-defined]
            _subst(term.motive, var_name, repl, bound),  # type: ignore[attr-defined]
            tuple(_subst(m, var_name, repl, bound) for m in term.methods),  # type: ignore[attr-defined]
            _subst(term.target, var_name, repl, bound),  # type: ignore[attr-defined]
        )

    return term


# ── 弱头范式 (WHNF) ──


def whnf(term: Term) -> Term:
    """将项规约到弱头范式.

    WHNF 规则：
    - (λx. body)(a) → body[x := a]   (β-归约)
    - π₁(a, b) → a                   (Σ-消去)
    - π₂(a, b) → b                   (Σ-消去)
    """
    if term.kind == TermKind.APP:
        func = whnf(term.func)  # type: ignore[attr-defined]
        if func.kind == TermKind.LAMBDA:
            # β-归约
            return whnf(substitute(func.body, func.var_name, term.arg))  # type: ignore[attr-defined]
        return App(func, term.arg)  # type: ignore[attr-defined]

    elif term.kind == TermKind.PROJ1:
        pair = whnf(term.pair)  # type: ignore[attr-defined]
        if pair.kind == TermKind.PAIR:
            return pair.fst  # type: ignore[attr-defined, no-any-return]
        return Proj1(pair)

    elif term.kind == TermKind.PROJ2:
        pair = whnf(term.pair)  # type: ignore[attr-defined]
        if pair.kind == TermKind.PAIR:
            return pair.snd  # type: ignore[attr-defined, no-any-return]
        return Proj2(pair)

    return term


# ── 项的字符串表示 ──


def term_to_str(term: Term, _prec: int = 0) -> str:
    """将项转换为可读字符串."""
    kind = term.kind

    if kind == TermKind.VAR:
        return term.name  # type: ignore[attr-defined, no-any-return]
    elif kind == TermKind.UNIVERSE:
        return f"Type_{term.level}" if term.level > 0 else "Type"  # type: ignore[attr-defined]
    elif kind == TermKind.PI:
        fv = free_vars(term.codomain)  # type: ignore[attr-defined]
        if term.var_name not in fv:  # type: ignore[attr-defined]
            # 非依赖: A → B
            s = f"{term_to_str(term.domain, 2)} → {term_to_str(term.codomain, 1)}"  # type: ignore[attr-defined]
        else:
            s = f"Π({term.var_name}:{term_to_str(term.domain)}). {term_to_str(term.codomain)}"  # type: ignore[attr-defined]
        return f"({s})" if _prec > 1 else s
    elif kind == TermKind.LAMBDA:
        return f"λ{term.var_name}. {term_to_str(term.body)}"  # type: ignore[attr-defined]
    elif kind == TermKind.APP:
        return f"{term_to_str(term.func, 2)}({term_to_str(term.arg, 2)})"  # type: ignore[attr-defined]
    elif kind == TermKind.SIGMA:
        fv = free_vars(term.snd_type)  # type: ignore[attr-defined]
        if term.var_name not in fv:  # type: ignore[attr-defined]
            s = f"{term_to_str(term.fst_type, 2)} × {term_to_str(term.snd_type, 1)}"  # type: ignore[attr-defined]
        else:
            s = f"Σ({term.var_name}:{term_to_str(term.fst_type)}). {term_to_str(term.snd_type)}"  # type: ignore[attr-defined]
        return f"({s})" if _prec > 1 else s
    elif kind == TermKind.PAIR:
        return f"({term_to_str(term.fst)}, {term_to_str(term.snd)})"  # type: ignore[attr-defined]
    elif kind == TermKind.PROJ1:
        return f"π₁({term_to_str(term.pair)})"  # type: ignore[attr-defined]
    elif kind == TermKind.PROJ2:
        return f"π₂({term_to_str(term.pair)})"  # type: ignore[attr-defined]
    elif kind == TermKind.IDENTITY:
        return f"Id_{{{term_to_str(term.typ)}}}({term_to_str(term.lhs)}, {term_to_str(term.rhs)})"  # type: ignore[attr-defined]
    elif kind == TermKind.REFL:
        return f"refl_{{{term_to_str(term.typ)}}}({term_to_str(term.term)})"  # type: ignore[attr-defined]
    elif kind == TermKind.SYM:
        return f"sym({term_to_str(term.proof)})"  # type: ignore[attr-defined]
    elif kind == TermKind.TRANS:
        return f"trans({term_to_str(term.p)}, {term_to_str(term.q)})"  # type: ignore[attr-defined]
    elif kind == TermKind.CONG:
        return f"cong({term_to_str(term.func)}, {term_to_str(term.proof)})"  # type: ignore[attr-defined]
    elif kind == TermKind.TRANSPORT:
        return f"transport({term_to_str(term.motive)}, {term_to_str(term.eq_proof)}, {term_to_str(term.value)})"  # type: ignore[attr-defined]
    elif kind == TermKind.ANNOTATION:
        return f"({term_to_str(term.term)} : {term_to_str(term.typ)})"  # type: ignore[attr-defined]
    elif kind == TermKind.INDUCTIVE:
        return term.name  # type: ignore[attr-defined, no-any-return]
    elif kind == TermKind.CONSTRUCT:
        args = ", ".join(term_to_str(a) for a in term.args)  # type: ignore[attr-defined]
        return f"{term.ctor_name}" + (f"({args})" if args else "")  # type: ignore[attr-defined]
    elif kind == TermKind.IND_ELIM:
        return f"ind_{term.type_name}(...)"  # type: ignore[attr-defined]

    return str(kind)


# ── 便利构造函数 ──


def arrow(a: Term, b: Term) -> Pi:
    """非依赖函数类型 A → B."""
    return Pi("_", a, b)


def product(a: Term, b: Term) -> Sigma:
    """非依赖积类型 A × B."""
    return Sigma("_", a, b)


TYPE0 = Universe(0)
TYPE1 = Universe(1)

# 预定义的简单类型
BOOL_TYPE = InductiveType(
    name="Bool",
    constructors=(
        Constructor(name="true"),
        Constructor(name="false"),
    ),
)

NAT_TYPE = InductiveType(
    name="Nat",
    constructors=(
        Constructor(name="zero"),
        Constructor(name="succ", arg_types=[Var("Nat")]),
    ),
)
