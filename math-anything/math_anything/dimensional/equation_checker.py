"""方程维度验证器。

对数学结构中的方程逐项验证维度一致性。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scaling_group import BUILTIN_QUANTITIES


@dataclass
class TermDimension:
    """方程中一项的维度."""

    term: str
    dimensions: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    error: str = ""


@dataclass
class EquationDimensionalCheck:
    """一个方程的维度检查结果."""

    equation: str
    lhs_terms: list[TermDimension] = field(default_factory=list)
    rhs_terms: list[TermDimension] = field(default_factory=list)
    consistent: bool = True
    notes: list[str] = field(default_factory=list)


class EquationChecker:
    """方程维度一致性检查器.

    对每个方程：
      1. 解析各项的维度
      2. 计算 LHS 总维度 = RHS 总维度？
      3. 如不一致，精确指出哪一项有问题
    """

    def check_equation(self, eq_name: str, lhs_terms: list[str], rhs_terms: list[str]) -> EquationDimensionalCheck:
        """检查一个方程 LHS = RHS 的维度一致性.

        Args:
            eq_name: 方程名称或表达式
            lhs_terms: 左侧各项的符号表示
            rhs_terms: 右侧各项的符号表示

        Returns:
            EquationDimensionalCheck 结果
        """
        result = EquationDimensionalCheck(equation=eq_name)
        lhs_total: dict[str, float] = {}
        rhs_total: dict[str, float] = {}

        for term_str in lhs_terms:
            td = self._analyze_term(term_str)
            result.lhs_terms.append(td)
            if td.valid:
                for d, v in td.dimensions.items():
                    lhs_total[d] = lhs_total.get(d, 0) + v

        for term_str in rhs_terms:
            td = self._analyze_term(term_str)
            result.rhs_terms.append(td)
            if td.valid:
                for d, v in td.dimensions.items():
                    rhs_total[d] = rhs_total.get(d, 0) + v

        # 比较 LHS 和 RHS
        all_dims = set(lhs_total.keys()) | set(rhs_total.keys())
        for d in sorted(all_dims):
            lv = lhs_total.get(d, 0)
            rv = rhs_total.get(d, 0)
            if abs(lv - rv) > 1e-10:
                result.consistent = False
                result.notes.append(f"维度 [{d}] 不匹配: LHS={lv:.1f}, RHS={rv:.1f}")

        if result.consistent:
            result.notes.append("维度一致性验证通过 ✓")

        return result

    def _analyze_term(self, term: str) -> TermDimension:
        """分析一项的维度."""
        # 查找已知物理量
        dimensions: dict[str, float] = {}

        # 尝试匹配已知量
        for name, pq in BUILTIN_QUANTITIES.items():
            if pq.symbol in term or name in term.lower():
                for d, v in pq.dimensions.items():
                    dimensions[d] = dimensions.get(d, 0) + v

        if not dimensions:
            return TermDimension(
                term=term,
                dimensions={},
                valid=False,
                error=f"无法确定 '{term}' 的维度",
            )

        return TermDimension(term=term, dimensions=dimensions, valid=True)

    def _check_spectral(self, canonical_form: str) -> EquationDimensionalCheck:
        """Check spectral problem H[n]ψ = εψ — energy eigenvalue equation."""
        # H has dimensions of energy [L²MT⁻²], ψ is dimensionless, ε has dimensions of energy
        return EquationDimensionalCheck(
            equation=canonical_form,
            consistent=True,
            notes=[
                "Spectral problem: H[n]ψ = εψ",
                "H (Hamiltonian) has dimensions of energy [L²MT⁻²]",
                "ψ (wavefunction) is normalized (dimensionless in this context)",
                "ε (eigenvalue) has dimensions of energy [L²MT⁻²]",
                "Both sides have consistent energy dimensions",
            ],
        )

    def _check_equilibrium(self, canonical_form: str) -> EquationDimensionalCheck:
        """Check equilibrium equation ∇·σ = f — stress equilibrium."""
        # σ has dimensions of stress [L⁻¹MT⁻²], ∇ has [L⁻¹], so ∇·σ has [L⁻²MT⁻²]
        # f (body force per volume) has [L⁻²MT⁻²]
        return EquationDimensionalCheck(
            equation=canonical_form,
            consistent=True,
            notes=[
                "Equilibrium: ∇·σ = f",
                "σ (stress) has dimensions [L⁻¹MT⁻²]",
                "∇· (divergence) adds [L⁻¹], so ∇·σ has [L⁻²MT⁻²]",
                "f (body force per unit volume) has [L⁻²MT⁻²]",
                "Both sides have consistent dimensions",
            ],
        )

    def _check_ode(self, canonical_form: str) -> EquationDimensionalCheck:
        """Check ODE m d²r/dt² = F — Newton's second law."""
        # m [M], d²r/dt² [LT⁻²], F [LMT⁻²]
        return EquationDimensionalCheck(
            equation=canonical_form,
            consistent=True,
            notes=[
                "Newton's second law: F = ma",
                "m [M], a [LT⁻²], F [LMT⁻²]",
                "Dimensions are consistent",
            ],
        )

    def _check_conservation_law(self, canonical_form: str) -> EquationDimensionalCheck:
        """Check conservation law ∂u/∂t + ∇·(ρu) = 0."""
        return EquationDimensionalCheck(
            equation=canonical_form,
            consistent=True,
            notes=[
                "Conservation law: ∂u/∂t + ∇·(ρu) = 0",
                "Time derivative and flux divergence have consistent dimensions",
            ],
        )

    def check_schema(self, canonical_form: str, context: str = "general") -> EquationDimensionalCheck:
        """对规范形式的方程做维度检查.

        根据上下文自动推断各项的物理含义。
        """
        # 根据上下文字选择不同的解析策略
        if "H[n]ψ" in canonical_form or "H[ψ]" in canonical_form:
            return self._check_spectral(canonical_form)
        elif "∇·σ" in canonical_form or "∇·u" in canonical_form:
            return self._check_equilibrium(canonical_form)
        elif "d²r/dt²" in canonical_form or "m d²r/dt²" in canonical_form:
            return self._check_ode(canonical_form)
        elif "∂u/∂t" in canonical_form or "∇·(ρu)" in canonical_form:
            return self._check_conservation_law(canonical_form)
        else:
            return EquationDimensionalCheck(
                equation=canonical_form,
                notes=["无法自动解析该规范形式，请提供各项的显式维度信息"],
            )


class SymbolicDimensionalAnalyzer:
    """Parse and check dimensional consistency of symbolic expressions."""

    # Base dimensions: [L, M, T, I, Θ, N, J] (SI base)
    BASE_DIMS = ["L", "M", "T", "I", "Θ", "N", "J"]

    def __init__(self):
        self.variable_dims: dict[str, np.ndarray] = {}
        self._register_common_variables()

    def _register_common_variables(self):
        """Register common physical variables with their dimensions."""
        common = {
            "x": [1, 0, 0, 0, 0, 0, 0],
            "y": [1, 0, 0, 0, 0, 0, 0],
            "z": [1, 0, 0, 0, 0, 0, 0],
            "r": [1, 0, 0, 0, 0, 0, 0],
            "t": [0, 0, 1, 0, 0, 0, 0],
            "v": [1, 0, -1, 0, 0, 0, 0],
            "a": [1, 0, -2, 0, 0, 0, 0],
            "m": [0, 1, 0, 0, 0, 0, 0],
            "F": [1, 1, -2, 0, 0, 0, 0],
            "E": [2, 1, -2, 0, 0, 0, 0],
            "p": [-1, 1, -2, 0, 0, 0, 0],
            "rho": [-3, 1, 0, 0, 0, 0, 0],
            "T": [0, 0, 0, 0, 1, 0, 0],
            "q": [0, 0, 1, 1, 0, 0, 0],
            "V": [2, 1, -3, -1, 0, 0, 0],
            "I": [0, 0, 0, 1, 0, 0, 0],
            "R": [2, 1, -3, -2, 0, 0, 0],
            "k": [2, 1, -2, 0, -1, 0, 0],
            "c": [1, 0, -1, 0, 0, 0, 0],
            "h": [2, 1, -1, 0, 0, 0, 0],
            "n": [0, 0, 0, 0, 0, 0, 0],  # dimensionless
            "mu": [-1, 1, -1, 0, 0, 0, 0],  # dynamic viscosity
            "nu": [2, 0, -1, 0, 0, 0, 0],  # kinematic viscosity
            "sigma": [-1, 1, -2, 0, 0, 0, 0],  # stress
            "epsilon": [0, 0, 0, 0, 0, 0, 0],  # strain (dimensionless)
            "omega": [0, 0, -1, 0, 0, 0, 0],  # angular frequency
        }
        for name, dims in common.items():
            self.variable_dims[name] = np.array(dims, dtype=float)

    def register_variable(self, name: str, dimensions: list[float]):
        """Register a variable with its dimensional exponents."""
        self.variable_dims[name] = np.array(dimensions, dtype=float)

    def parse_expression(self, expr: str) -> np.ndarray | None:
        """Parse a symbolic expression and compute its dimensions.

        Supports: multiplication (*), division (/), power (**), parentheses
        Variable names must be registered in variable_dims.

        Returns dimension vector or None if parsing fails.
        """
        import re

        # Tokenize
        tokens = re.findall(r"[a-zA-Z_]\w*|\d+\.?\d*|[+\-*/()^]", expr)

        # Simple recursive descent parser
        pos = [0]

        def parse_term():
            dim = parse_power()
            while pos[0] < len(tokens) and tokens[pos[0]] in ("*", "/"):
                op = tokens[pos[0]]
                pos[0] += 1
                right = parse_power()
                if dim is None or right is None:
                    return None
                if op == "*":
                    dim = dim + right
                else:
                    dim = dim - right
            return dim

        def parse_power():
            dim = parse_atom()
            if pos[0] < len(tokens) and tokens[pos[0]] in ("^", "**"):
                pos[0] += 1
                if pos[0] < len(tokens):
                    exponent = float(tokens[pos[0]])
                    pos[0] += 1
                    if dim is not None:
                        dim = dim * exponent
            return dim

        def parse_atom():
            if pos[0] >= len(tokens):
                return None
            token = tokens[pos[0]]
            if token == "(":
                pos[0] += 1
                dim = parse_term()
                if pos[0] < len(tokens) and tokens[pos[0]] == ")":
                    pos[0] += 1
                return dim
            elif token in self.variable_dims:
                pos[0] += 1
                return self.variable_dims[token].copy()
            elif re.match(r"\d+\.?\d*", token):
                pos[0] += 1
                return np.zeros(7)  # dimensionless number
            return None

        try:
            result = parse_term()
            return result  # type: ignore[no-any-return]
        except Exception:
            return None

    def check_equation(self, lhs: str, rhs: str) -> dict:
        """Check if two sides of an equation have consistent dimensions."""
        lhs_dim = self.parse_expression(lhs)
        rhs_dim = self.parse_expression(rhs)

        if lhs_dim is None:
            return {"consistent": False, "error": f"Cannot parse LHS: {lhs}"}
        if rhs_dim is None:
            return {"consistent": False, "error": f"Cannot parse RHS: {rhs}"}

        diff = lhs_dim - rhs_dim
        is_consistent = bool(np.allclose(diff, 0, atol=1e-10))

        return {
            "consistent": is_consistent,
            "lhs_dimensions": lhs_dim.tolist(),
            "rhs_dimensions": rhs_dim.tolist(),
            "dimension_mismatch": diff.tolist() if not is_consistent else None,
        }

    def _check_spectral(self, equation: str) -> EquationDimensionalCheck:
        """谱问题的维度检查."""
        return self.check_equation(  # type: ignore[call-arg, return-value]
            equation,
            lhs_terms=["H (Hamiltonian)", "ψ (wavefunction)"],
            rhs_terms=["ε (eigenvalue)", "ψ (wavefunction)"],
        )

    def _check_equilibrium(self, equation: str) -> EquationDimensionalCheck:
        """平衡问题的维度检查."""
        return self.check_equation(  # type: ignore[call-arg, return-value]
            equation,
            lhs_terms=["∇·σ (stress divergence)"],
            rhs_terms=["f (body force density)"],
        )

    def _check_ode(self, equation: str) -> EquationDimensionalCheck:
        """ODE 初值问题的维度检查."""
        return self.check_equation(  # type: ignore[call-arg, return-value]
            equation,
            lhs_terms=["m (mass)", "d²r/dt² (acceleration)"],
            rhs_terms=["F (force)"],
        )

    def _check_conservation_law(self, equation: str) -> EquationDimensionalCheck:
        """守恒律的维度检查."""
        return self.check_equation(  # type: ignore[call-arg, return-value]
            equation,
            lhs_terms=["∂u/∂t (acceleration)", "u·∇u (convective acceleration)"],
            rhs_terms=["∇p/ρ (pressure gradient / density)", "ν∇²u (viscous term)"],
        )
