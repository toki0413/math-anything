"""Rust 加速桥接层。

优先使用 Rust 实现（math_anything_rs），
不可用时自动降级到纯 Python 实现。

使用:
    from math_anything.rust_bridge import EMLAccelerator
    acc = EMLAccelerator()
    closure = acc.eml_closure([0.0, 1.0, math.e, math.pi], 5)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class _EMLPyFallback:
    """EML 闭包计算的纯 Python 回退."""

    @staticmethod
    def eml_closure(base: list[float], max_depth: int, max_size: int = 100000) -> list[float]:
        closure: set[int] = set()
        current: list[float] = list(base)

        for v in base:
            closure.add(_float_to_key(v))

        for _ in range(max_depth):
            nxt: list[float] = []
            for i in range(min(len(current), 100)):
                for j in range(min(len(current), 100)):
                    try:
                        y = max(current[j], 1e-300)
                        v = math.exp(current[i]) - math.log(y)
                        if math.isfinite(v):
                            key = _float_to_key(v)
                            if key not in closure:
                                closure.add(key)
                                nxt.append(v)
                    except (ValueError, OverflowError):
                        continue
            if not nxt or len(closure) >= max_size:
                break
            current = nxt

        return [v for v in sorted(closure)][:max_size]

    @staticmethod
    def buckingham_pi(matrix: np.ndarray) -> list[list[float]]:
        """使用 SymPy 计算零空间."""
        import sympy as sp

        M = sp.Matrix(matrix.tolist())
        nullspace = M.nullspace()
        groups = []
        for vec in nullspace:
            coeffs = [float(v.evalf()) if hasattr(v, "evalf") else float(v) for v in vec]
            max_abs = max(abs(c) for c in coeffs) if coeffs else 1.0
            if max_abs < 1e-10:
                continue
            groups.append([c / max_abs for c in coeffs])
        return groups

    @staticmethod
    def shortest_path(n_nodes: int, edges: list[tuple[int, int]], start: int, end: int) -> list[int]:
        """BFS 最短路径."""
        from collections import deque

        adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))

        queue = deque([start])
        parent: list[tuple[int, int] | None] = [None] * n_nodes
        visited = [False] * n_nodes
        visited[start] = True

        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                cur = end
                while cur != start:
                    p = parent[cur]
                    if p is None:
                        break
                    path.append(p[1])
                    cur = p[0]
                path.reverse()
                return path
            for v, ei in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = (u, ei)
                    queue.append(v)
        return []

    @staticmethod
    def propagate_constraints(
        invariant_names: list[str],
        morphism_data: list[dict],
    ) -> list[list[str]]:
        """批量约束传播 (Python 回退).

        Args:
            invariant_names: 初始不变量名称列表
            morphism_data: 态射数据列表，每个元素为 dict:
                {name, kept: list[str], lost: list[str],
                 introduced: list[str], kernel_desc: str}

        Returns:
            二维列表，results[i][j] = 第 j 个不变量穿过第 i 个态射的结果
        """
        results: list[list[str]] = []
        current = list(invariant_names)

        for m in morphism_data:
            kept = set(m.get("kept", []))
            lost = set(m.get("lost", []))
            introduced = set(m.get("introduced", []))

            step: list[str] = []
            for name in current:
                if name in lost:
                    step.append("LOST")
                elif name in introduced:
                    step.append("EMERGED")
                elif name in kept:
                    step.append("PRESERVED")
                else:
                    step.append("PRESERVED")
            results.append(step)

            current = [n for n in current if n not in lost] + list(introduced)

        return results


def _float_to_key(v: float) -> int:
    """将浮点数量化为整数键，用于去重."""
    return int(abs(v + 1e-10) * 1e8)


def _json_to_term(data: dict) -> Any:
    """将 JSON 字典反序列化为 MLTT Term 对象."""
    from math_anything.type_theory.terms import (
        App,
        Identity,
        Lam,
        Pair,
        Pi,
        Proj1,
        Proj2,
        Refl,
        Sigma,
        Universe,
        Var,
    )

    kind = data.get("kind", "")
    if kind == "Var":
        return Var(data["name"])
    elif kind == "Universe":
        return Universe(data.get("level", 0))
    elif kind == "Pi":
        return Pi(data["var_name"], _json_to_term(data["domain"]), _json_to_term(data["codomain"]))
    elif kind == "Lam":
        return Lam(data["var_name"], _json_to_term(data["body"]))
    elif kind == "App":
        return App(_json_to_term(data["func"]), _json_to_term(data["arg"]))
    elif kind == "Sigma":
        return Sigma(data["var_name"], _json_to_term(data["fst_type"]), _json_to_term(data["snd_type"]))
    elif kind == "Pair":
        return Pair(_json_to_term(data["fst"]), _json_to_term(data["snd"]))
    elif kind == "Proj1":
        return Proj1(_json_to_term(data["pair"]))
    elif kind == "Proj2":
        return Proj2(_json_to_term(data["pair"]))
    elif kind == "Identity":
        return Identity(_json_to_term(data["typ"]), _json_to_term(data["lhs"]), _json_to_term(data["rhs"]))
    elif kind == "Refl":
        return Refl(_json_to_term(data["typ"]), _json_to_term(data["term"]))
    # 未知类型原样返回，避免崩溃
    return data


def _term_to_json(term: Any) -> dict:
    """将 MLTT Term 对象序列化为 JSON 字典."""
    from math_anything.type_theory.terms import (
        TermKind,
    )

    kind = term.kind
    if kind == TermKind.VAR:
        return {"kind": "Var", "name": term.name}
    elif kind == TermKind.UNIVERSE:
        return {"kind": "Universe", "level": term.level}
    elif kind == TermKind.PI:
        return {
            "kind": "Pi",
            "var_name": term.var_name,
            "domain": _term_to_json(term.domain),
            "codomain": _term_to_json(term.codomain),
        }
    elif kind == TermKind.LAMBDA:
        return {"kind": "Lam", "var_name": term.var_name, "body": _term_to_json(term.body)}
    elif kind == TermKind.APP:
        return {"kind": "App", "func": _term_to_json(term.func), "arg": _term_to_json(term.arg)}
    elif kind == TermKind.SIGMA:
        return {
            "kind": "Sigma",
            "var_name": term.var_name,
            "fst_type": _term_to_json(term.fst_type),
            "snd_type": _term_to_json(term.snd_type),
        }
    elif kind == TermKind.PAIR:
        return {"kind": "Pair", "fst": _term_to_json(term.fst), "snd": _term_to_json(term.snd)}
    elif kind == TermKind.PROJ1:
        return {"kind": "Proj1", "pair": _term_to_json(term.pair)}
    elif kind == TermKind.PROJ2:
        return {"kind": "Proj2", "pair": _term_to_json(term.pair)}
    elif kind == TermKind.IDENTITY:
        return {
            "kind": "Identity",
            "typ": _term_to_json(term.typ),
            "lhs": _term_to_json(term.lhs),
            "rhs": _term_to_json(term.rhs),
        }
    elif kind == TermKind.REFL:
        return {"kind": "Refl", "typ": _term_to_json(term.typ), "term": _term_to_json(term.term)}
    # 其他类型兜底
    return {"kind": str(kind)}


# ── Rust 优先 / Python 回退 ──

_use_rust = False
_rust_module: Any = None

try:
    import math_anything_rs as _rust_module

    _use_rust = True
except ImportError:
    pass


class EMLAccelerator:
    """EML 加速器，自动选择最快实现."""

    def eml(self, x: float, y: float) -> float:
        """EML 万能算子: exp(x) - ln(y)."""
        if _use_rust and hasattr(_rust_module, "eml"):
            try:
                return _rust_module.eml(x, y)
            except Exception:
                logger.debug("Rust eml() failed, using Python fallback")
        y_safe = max(y, 1e-300)
        return math.exp(x) - math.log(y_safe)

    def eml_closure(self, base: list[float], max_depth: int, max_size: int = 100000) -> list[float]:
        if _use_rust and hasattr(_rust_module, "eml_closure"):
            try:
                return _rust_module.eml_closure(base, max_depth, max_size)
            except Exception:
                logger.debug("Rust eml_closure() failed, using Python fallback")
        return _EMLPyFallback.eml_closure(base, max_depth, max_size)

    def buckingham_pi(self, matrix: np.ndarray) -> list[list[float]]:
        if _use_rust and hasattr(_rust_module, "buckingham_pi"):
            try:
                rows, cols = matrix.shape
                data = matrix.ravel().tolist()
                return _rust_module.buckingham_pi(rows, cols, data)
            except Exception:
                logger.debug("Rust buckingham_pi() failed, using Python fallback")
        return _EMLPyFallback.buckingham_pi(matrix)

    def parallel_buckingham_pi(self, matrices: list[tuple[int, int, list[float]]]) -> list[list[list[float]]]:
        """并行计算多个矩阵的 Buckingham Pi 群 (Rust Rayon)."""
        if _use_rust and hasattr(_rust_module, "parallel_buckingham_pi"):
            try:
                return _rust_module.parallel_buckingham_pi(matrices)
            except Exception:
                pass
        # Python 回退: 逐个计算
        return [self.buckingham_pi(np.array(d).reshape(r, c)) for r, c, d in matrices]

    def shortest_path(self, n_nodes: int, edges: list[tuple[int, int]], start: int, end: int) -> list[int]:
        if _use_rust and hasattr(_rust_module, "shortest_path"):
            try:
                return _rust_module.shortest_path(n_nodes, edges, start, end)
            except Exception:
                logger.debug("Rust shortest_path() failed, using Python fallback")
        return _EMLPyFallback.shortest_path(n_nodes, edges, start, end)

    def whnf_normalize(self, term_json: str) -> str:
        """WHNF normalization via Rust (fast) or Python (fallback)."""
        if _use_rust and hasattr(_rust_module, "whnf_normalize"):
            try:
                return _rust_module.whnf_normalize(term_json)
            except Exception:
                logger.debug("Rust whnf_normalize() failed, using Python fallback")
        # Python fallback: use the type_theory module
        import json

        from math_anything.type_theory.terms import whnf

        data = json.loads(term_json)
        term = _json_to_term(data)
        result = whnf(term)
        return json.dumps(_term_to_json(result))

    def propagate_constraints(self, invariant_names, morphism_kept, morphism_lost, morphism_introduced):
        """Batch constraint propagation via Rust (fast) or Python (fallback)."""
        if _use_rust and hasattr(_rust_module, "propagate_constraints"):
            try:
                return _rust_module.propagate_constraints(
                    invariant_names, morphism_kept, morphism_lost, morphism_introduced
                )
            except Exception:
                logger.debug("Rust propagate_constraints() failed, using Python fallback")
        # Python fallback
        results = []
        for inv_name in invariant_names:
            status = "preserved"
            lost_at = None
            found_introduced = False
            for step, (kept, lost, introduced) in enumerate(zip(morphism_kept, morphism_lost, morphism_introduced)):
                if inv_name in lost:
                    status = "lost"
                    lost_at = step
                    break
                if inv_name in introduced:
                    found_introduced = True
            if found_introduced and status == "preserved":
                status = "emerged"
            results.append((inv_name, status, lost_at))
        return results

    def batch_eval_expressions(
        self, expressions: list[str], var_names: list[str], data_rows: int, data_flat: list[float]
    ) -> list[list[float]]:
        """Batch evaluate expressions via Rust (fast) or Python (fallback)."""
        if _use_rust and hasattr(_rust_module, "batch_eval_expressions"):
            try:
                return _rust_module.batch_eval_expressions(expressions, var_names, data_rows, data_flat)
            except Exception:
                logger.debug("Rust batch_eval_expressions() failed, using Python fallback")
        # Python fallback using numpy
        from .utils.safe_eval import SafeEvalError, safe_eval

        data = np.array(data_flat).reshape(data_rows, len(var_names))
        results = []
        for expr in expressions:
            try:
                local_vars = {name: data[:, i] for i, name in enumerate(var_names)}
                local_vars.update(
                    {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}
                )
                result = safe_eval(expr, local_vars)
                results.append(result.tolist() if hasattr(result, "tolist") else [float(result)] * data_rows)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, SafeEvalError):
                results.append([])
        return results

    def check_def_eq(self, term_a_json: str, term_b_json: str) -> bool:
        """Check definitional equality via Rust (fast) or Python (fallback)."""
        if _use_rust and hasattr(_rust_module, "check_def_eq"):
            try:
                return _rust_module.check_def_eq(term_a_json, term_b_json)
            except Exception:
                logger.debug("Rust check_def_eq() failed, using Python fallback")
        # Python fallback: use type_theory checker
        import json

        from math_anything.type_theory.terms import def_eq as py_def_eq

        a = _json_to_term(json.loads(term_a_json))
        b = _json_to_term(json.loads(term_b_json))
        return py_def_eq(a, b)

    def compute_riemann_tensor(
        self,
        christoffel: list[float],
        d_christoffel: list[float],
        dim: int,
    ) -> list[float]:
        """Riemann 曲率张量计算 (Rust 加速 / Python 回退).

        Args:
            christoffel: 展平的 Christoffel 符号 (dim^3,)
                索引 [i][j][k] → i*dim^2 + j*dim + k
            d_christoffel: 展平的 Christoffel 偏导 (dim^4,)
                索引 [i][l][j][k] → i*dim^3 + l*dim^2 + j*dim + k
            dim: 流形维数

        Returns:
            展平的 Riemann 张量 (dim^4,)
        """
        if _use_rust and hasattr(_rust_module, "compute_riemann_tensor"):
            try:
                return _rust_module.compute_riemann_tensor(christoffel, d_christoffel, dim)
            except Exception:
                logger.debug("Rust compute_riemann_tensor() failed, using Python fallback")
        # Python fallback: O(n^5) nested loops
        if dim == 0 or len(christoffel) != dim**3 or len(d_christoffel) != dim**4:
            return []
        d2 = dim * dim
        d3 = dim * dim * dim
        riemann = [0.0] * d3 * dim
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for idx in range(dim):
                        d1 = d_christoffel[i * d3 + idx * d2 + j * dim + k]
                        d2v = d_christoffel[i * d3 + k * d2 + j * dim + idx]
                        sm = 0.0
                        for m in range(dim):
                            g1 = christoffel[i * d2 + k * dim + m]
                            g2 = christoffel[m * d2 + idx * dim + j]
                            g3 = christoffel[i * d2 + idx * dim + m]
                            g4 = christoffel[m * d2 + k * dim + j]
                            sm += g1 * g2 - g3 * g4
                        riemann[i * d3 + j * d2 + k * dim + idx] = d1 - d2v + sm
        return riemann

    def compute_ricci_tensor(self, riemann: list[float], dim: int) -> list[float]:
        """Ricci 张量计算 (Rust 加速 / Python 回退).

        R_{jk} = Σ_i R^i_{jik}

        Args:
            riemann: 展平的 Riemann 张量 (dim^4,)
            dim: 流形维数

        Returns:
            展平的 Ricci 张量 (dim^2,)
        """
        if _use_rust and hasattr(_rust_module, "compute_ricci_tensor"):
            try:
                return _rust_module.compute_ricci_tensor(riemann, dim)
            except Exception:
                logger.debug("Rust compute_ricci_tensor() failed, using Python fallback")
        # Python fallback
        if dim == 0 or len(riemann) != dim**4:
            return []
        d2 = dim * dim
        d3 = dim * dim * dim
        ricci = [0.0] * d2
        for j in range(dim):
            for k in range(dim):
                s = 0.0
                for i in range(dim):
                    s += riemann[i * d3 + j * d2 + i * dim + k]
                ricci[j * dim + k] = s
        return ricci

    def compute_scalar_curvature(self, ricci: list[float], inv_metric: list[float], dim: int) -> float:
        """标量曲率计算 (Rust 加速 / Python 回退).

        R = g^{jk} R_{jk}

        Args:
            ricci: 展平的 Ricci 张量 (dim^2,)
            inv_metric: 展平的逆度量 (dim^2,)
            dim: 流形维数

        Returns:
            标量曲率
        """
        if _use_rust and hasattr(_rust_module, "compute_scalar_curvature"):
            try:
                return _rust_module.compute_scalar_curvature(ricci, inv_metric, dim)
            except Exception:
                logger.debug("Rust compute_scalar_curvature() failed, using Python fallback")
        # Python fallback
        if dim == 0 or len(ricci) != dim * dim or len(inv_metric) != dim * dim:
            return float("nan")
        scalar = 0.0
        for j in range(dim):
            for k in range(dim):
                scalar += inv_metric[j * dim + k] * ricci[j * dim + k]
        return scalar

    @property
    def using_rust(self) -> bool:
        return _use_rust


def is_rust_available() -> bool:
    return _use_rust
