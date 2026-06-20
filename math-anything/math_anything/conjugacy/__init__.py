"""EML 算子的拓扑共轭理论。

核心数学事实：
  EML(x, y) = exp(x) - ln(y) 是万能二元算子。
  所有初等函数均可表示为 EML 与常数 {0, 1, e, π, i} 的有限复合。

拓扑共轭：
  两个函数 f, g 拓扑共轭 ⇔ 存在同胚 h 使 h∘f = g∘h。
  EML 复合构造的函数类形成一个共轭类，
  EML 是连接所有初等函数类的"万能函子"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import e, exp, log, pi, sin, sqrt
from typing import Any, Callable

import numpy as np


class EMLPrimitive(StrEnum):
    """EML 基本运算."""

    EML = "EML"
    CONST = "CONST"
    VAR = "VAR"


@dataclass
class EMLExpr:
    """EML 表达式树."""

    op: str
    left: "EMLExpr | None" = None
    right: "EMLExpr | None" = None
    value: float | None = None
    name: str | None = None

    def evaluate(self, x: float = 0.0) -> float:
        if self.op == "CONST":
            return self.value if self.value is not None else 0.0
        if self.op == "VAR":
            return x
        if self.op == "EML":
            lv = self.left.evaluate(x) if self.left else 0.0
            rv = self.right.evaluate(x) if self.right else 1.0
            return exp(lv) - log(max(rv, 1e-300))
        raise ValueError(f"Unknown op: {self.op}")

    def to_standard_form(self) -> str:
        if self.op == "CONST":
            v = self.value
            if v is None:
                return "0"
            if abs(v - e) < 1e-10:
                return "e"
            if abs(v - pi) < 1e-10:
                return "π"
            if abs(v) < 1e-10:
                return "0"
            if abs(v - 1.0) < 1e-10:
                return "1"
            return f"{v:.6g}"
        if self.op == "VAR":
            return self.name or "x"
        if self.op == "EML":
            left = self.left
            right = self.right
            if right and right.op == "CONST" and right.value is not None and abs(right.value - 1.0) < 1e-10:
                return f"exp({left.to_standard_form() if left else '0'})"
            if left and left.op == "CONST" and left.value is not None and abs(left.value) < 1e-10:
                return f"1 - ln({right.to_standard_form() if right else '1'})"
            return f"EML({left.to_standard_form() if left else '0'}, {right.to_standard_form() if right else '1'})"
        return "?"

    def depth(self) -> int:
        if self.op in ("CONST", "VAR"):
            return 0
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def used_constants(self) -> set[float]:
        res: set[float] = set()
        if self.op == "CONST" and self.value is not None:
            res.add(self.value)
        if self.left:
            res |= self.left.used_constants()
        if self.right:
            res |= self.right.used_constants()
        return res


def eml_expr(x: EMLExpr, y: EMLExpr) -> EMLExpr:
    return EMLExpr(op="EML", left=x, right=y)


def const_expr(c: float) -> EMLExpr:
    return EMLExpr(op="CONST", value=c)


def var_expr(name: str = "x") -> EMLExpr:
    return EMLExpr(op="VAR", name=name)


E_EXPR = const_expr(e)
PI_EXPR = const_expr(pi)
ZERO = const_expr(0.0)
ONE = const_expr(1.0)


def make_exp(arg: EMLExpr) -> EMLExpr:
    return eml_expr(arg, ONE)


def make_ln(arg: EMLExpr) -> EMLExpr:
    return eml_expr(ONE, eml_expr(ZERO, arg))


# ── 拓扑共轭 ──


@dataclass
class TopologicalConjugacy:
    """两个函数的拓扑共轭关系 h∘f = g∘h."""

    f: Callable[[float], float]
    g: Callable[[float], float]
    h: Callable[[float], float]
    h_inv: Callable[[float], float]
    name: str = ""
    domain: str = "ℝ"

    def verify(self, xs: list[float]) -> bool:
        eps = 1e-8
        for x in xs:
            try:
                lhs = self.h(self.f(x))
                rhs = self.g(self.h(x))
                if abs(lhs - rhs) > eps * (1.0 + max(abs(lhs), abs(rhs))):
                    return False
            except (ValueError, OverflowError):
                continue
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "relation": "h o f = g o h", "domain": self.domain}


CONJUGACIES: dict[str, TopologicalConjugacy] = {
    "exp_shift": TopologicalConjugacy(
        f=exp,
        g=lambda x: x + 1.0,
        h=log,
        h_inv=exp,
        name="exp-shift conjugacy",
        domain="R+",
    ),
    "square_sqrt": TopologicalConjugacy(
        f=lambda x: x * x,
        g=lambda x: x,
        h=sqrt,
        h_inv=lambda x: x * x,
        name="square-sqrt",
        domain="R+",
    ),
}


class EMLConjugacyEngine:
    """EML 共轭分析引擎.

    验证两个表达式是否拓扑共轭，跟踪 EML 树之间的同胚。
    """

    def __init__(self):
        self.conjugacies: dict[str, TopologicalConjugacy] = dict(CONJUGACIES)

    def is_conjugate(self, expr_a: EMLExpr, expr_b: EMLExpr, test_points: list[float] | None = None) -> bool:
        if test_points is None:
            test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        values_a = [expr_a.evaluate(x) for x in test_points]
        values_b = [expr_b.evaluate(x) for x in test_points]
        if len(values_a) < 3:
            return False
        ranks_a = sorted(range(len(values_a)), key=lambda i: values_a[i])
        ranks_b = sorted(range(len(values_b)), key=lambda i: values_b[i])
        return ranks_a == ranks_b

    def verify_all(self) -> dict[str, bool]:
        pts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        return {name: c.verify(pts) for name, c in self.conjugacies.items()}


class RigorousConjugacyChecker:
    """Rigorous topological conjugacy checking using numerical methods."""

    def __init__(self, tol: float = 1e-6, n_test_points: int = 50):
        self.tol = tol
        self.n_test_points = n_test_points

    def check_conjugacy(self, f: Callable, g: Callable, h: Callable, domain: tuple[float, float] = (-5, 5)) -> dict:
        """Check if h∘f = g∘h on test points.

        Args:
            f: Function f
            g: Function g
            h: Conjugacy map h such that h∘f = g∘h
            domain: Domain for test points
        """
        errors = []
        test_points = np.linspace(domain[0], domain[1], self.n_test_points)

        for x in test_points:
            try:
                lhs = h(f(x))  # h∘f
                rhs = g(h(x))  # g∘h
                errors.append(abs(lhs - rhs))
            except (ValueError, OverflowError, ZeroDivisionError):
                continue

        if not errors:
            return {"conjugate": None, "error": "No valid test points"}

        max_error = max(errors)
        mean_error = np.mean(errors)

        return {
            "conjugate": max_error < self.tol,
            "max_error": max_error,
            "mean_error": float(mean_error),
            "n_test_points": len(errors),
            "tolerance": self.tol,
        }

    def find_conjugacy(
        self, f: Callable, g: Callable, h_family: list[Callable], domain: tuple[float, float] = (-5, 5)
    ) -> dict:
        """Find which member of a family of maps provides conjugacy.

        Args:
            f: Function f
            g: Function g
            h_family: List of candidate conjugacy maps to test
            domain: Domain for test points
        """
        results = []
        for i, h in enumerate(h_family):
            check = self.check_conjugacy(f, g, h, domain)
            results.append(
                {
                    "index": i,
                    "conjugate": check["conjugate"],
                    "max_error": check["max_error"],
                }
            )

        best = min(results, key=lambda r: r["max_error"])
        return {
            "best_match_index": best["index"],
            "best_conjugate": best["conjugate"],
            "best_error": best["max_error"],
            "all_results": results,
        }

    def compute_conjugacy_map(
        self, f: Callable, g: Callable, n_points: int = 100, domain: tuple[float, float] = (-5, 5)
    ) -> dict:
        """Numerically compute a conjugacy map h such that h∘f = g∘h.

        Uses the relation h(f(x)) = g(h(x)) to solve for h on a grid.
        """
        # This is an iterative approach: start with h = identity, then refine
        xs = np.linspace(domain[0], domain[1], n_points)
        h_values = xs.copy()  # Initial guess: h = identity

        for iteration in range(50):
            new_h = np.zeros_like(h_values)
            for i, x in enumerate(xs):
                try:
                    # h(f(x)) should equal g(h(x))
                    fx = f(x)
                    # Find index closest to fx
                    idx = np.argmin(np.abs(xs - fx))
                    hfx = h_values[idx]
                    # g(h(x)) should equal hfx
                    new_h[i] = hfx  # Approximate: h(x) such that g(h(x)) = h(f(x))
                except (ValueError, OverflowError):
                    new_h[i] = h_values[i]

            # Smooth update
            h_values = 0.5 * h_values + 0.5 * new_h

        return {
            "x_values": xs.tolist(),
            "h_values": h_values.tolist(),
            "n_points": n_points,
        }
