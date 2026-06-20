"""维度分析 — 尺度变换群的不变量理论。

核心思想：
  一个物理系统的维度自洽性等价于它在尺度变换群（scaling group）
  作用下形式不变。Buckingham π 定理是该群作用的不变多项式环的生成元。

scale_group.py: 尺度变换群 + Buckingham π 定理引擎
equation_checker.py: 方程维度验证
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from math_anything.rust_bridge import EMLAccelerator

_accel = EMLAccelerator()
_logger = logging.getLogger(__name__)
_logger.info(f"Buckingham Pi backend: {'Rust' if _accel.using_rust else 'SymPy (Python)'}")


# ── 基础维度系统 ──

# 七种 SI 基础维度
BASE_DIMENSIONS = ("M", "L", "T", "Theta", "I", "N", "J")


@dataclass
class PhysicalQuantity:
    """一个物理量的完整维度表示."""

    name: str  # 名称，如 "energy_cutoff"
    symbol: str  # 符号，如 "E_cut"
    dimensions: dict[str, float] = field(default_factory=dict)  # {M:1, L:2, T:-2}
    canonical_unit: str = ""  # 规范单位，如 "eV"
    physical_role: str = ""  # 在方程中的角色: "state_variable", "parameter", "source"
    description: str = ""

    @property
    def dim_vector(self) -> np.ndarray:
        """维度向量 [M, L, T, Theta, I, N, J]."""
        return np.array([self.dimensions.get(d, 0.0) for d in BASE_DIMENSIONS])


@dataclass
class BuckinghamPiGroup:
    """一个 Buckingham π 无量纲群."""

    pi_id: int
    name: str  # 如 "Reynolds_number"
    expression: str  # 如 "ρ · U · L / μ"
    variables: dict[str, float]  # {var_name: exponent}
    physical_meaning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Buckingham π group to a dictionary."""
        return {
            "pi_id": self.pi_id,
            "name": self.name,
            "expression": self.expression,
            "variables": self.variables,
            "physical_meaning": self.physical_meaning,
        }


# ── 内建物理量数据库 ──

BUILTIN_QUANTITIES: dict[str, PhysicalQuantity] = {
    # 力学量
    "length": PhysicalQuantity("length", "L", {"L": 1}, "m", "state_variable", "特征长度"),
    "mass": PhysicalQuantity("mass", "m", {"M": 1}, "kg", "state_variable", "质量"),
    "time": PhysicalQuantity("time", "t", {"T": 1}, "s", "state_variable", "时间"),
    "velocity": PhysicalQuantity("velocity", "U", {"L": 1, "T": -1}, "m/s", "state_variable", "流速/速度"),
    "acceleration": PhysicalQuantity("acceleration", "a", {"L": 1, "T": -2}, "m/s²", "state_variable", "加速度"),
    "force": PhysicalQuantity("force", "F", {"M": 1, "L": 1, "T": -2}, "N", "state_variable", "力"),
    "pressure": PhysicalQuantity("pressure", "p", {"M": 1, "L": -1, "T": -2}, "Pa", "state_variable", "压力"),
    "stress": PhysicalQuantity("stress", "σ", {"M": 1, "L": -1, "T": -2}, "Pa", "state_variable", "应力"),
    "energy": PhysicalQuantity("energy", "E", {"M": 1, "L": 2, "T": -2}, "J", "state_variable", "能量"),
    "density": PhysicalQuantity("density", "ρ", {"M": 1, "L": -3}, "kg/m³", "state_variable", "密度"),
    "dynamic_viscosity": PhysicalQuantity(
        "dynamic_viscosity", "μ", {"M": 1, "L": -1, "T": -1}, "Pa·s", "parameter", "动力黏度"
    ),
    "kinematic_viscosity": PhysicalQuantity(
        "kinematic_viscosity", "ν", {"L": 2, "T": -1}, "m²/s", "parameter", "运动黏度"
    ),
    "frequency": PhysicalQuantity("frequency", "f", {"T": -1}, "Hz", "state_variable", "频率"),
    "surface_tension": PhysicalQuantity("surface_tension", "σ_s", {"M": 1, "T": -2}, "N/m", "parameter", "表面张力"),
    "gravity": PhysicalQuantity("gravity", "g", {"L": 1, "T": -2}, "m/s²", "parameter", "重力加速度"),
    "strain_rate": PhysicalQuantity("strain_rate", "ε̇", {"T": -1}, "1/s", "state_variable", "应变率"),
    # 热力学量
    "temperature": PhysicalQuantity("temperature", "T", {"Theta": 1}, "K", "state_variable", "温度"),
    "heat_capacity": PhysicalQuantity(
        "heat_capacity", "c_p", {"M": 1, "L": 2, "T": -2, "Theta": -1}, "J/(kg·K)", "parameter", "比热容"
    ),
    "thermal_conductivity": PhysicalQuantity(
        "thermal_conductivity", "k", {"M": 1, "L": 1, "T": -3, "Theta": -1}, "W/(m·K)", "parameter", "导热系数"
    ),
    "thermal_diffusivity": PhysicalQuantity(
        "thermal_diffusivity", "α", {"L": 2, "T": -1}, "m²/s", "parameter", "热扩散率"
    ),
    "boltzmann_constant": PhysicalQuantity(
        "boltzmann_constant", "k_B", {"M": 1, "L": 2, "T": -2, "Theta": -1}, "J/K", "parameter", "Boltzmann 常数"
    ),
    # 电磁学量
    "charge": PhysicalQuantity("charge", "q", {"T": 1, "I": 1}, "C", "state_variable", "电荷"),
    "electric_field": PhysicalQuantity(
        "electric_field", "E", {"M": 1, "L": 1, "T": -3, "I": -1}, "V/m", "state_variable", "电场"
    ),
    "magnetic_field": PhysicalQuantity(
        "magnetic_field", "B", {"M": 1, "T": -2, "I": -1}, "T", "state_variable", "磁感应强度"
    ),
    "permittivity": PhysicalQuantity(
        "permittivity", "ε", {"M": -1, "L": -3, "T": 4, "I": 2}, "F/m", "parameter", "介电常数"
    ),
    "permeability": PhysicalQuantity(
        "permeability", "μ_0", {"M": 1, "L": 1, "T": -2, "I": -2}, "H/m", "parameter", "磁导率"
    ),
    # 量子力学量
    "planck_constant": PhysicalQuantity(
        "planck_constant", "ħ", {"M": 1, "L": 2, "T": -1}, "J·s", "parameter", "约化 Planck 常数"
    ),
    "wavefunction": PhysicalQuantity(
        "wavefunction", "ψ", {"L": -1.5}, "m^(-3/2)", "state_variable", "波函数（归一化）"
    ),
    "electron_density": PhysicalQuantity("electron_density", "n", {"L": -3}, "m^(-3)", "state_variable", "电子密度"),
}

# 命名无量纲数 → 变量映射
NAMED_PI_GROUPS: dict[str, BuckinghamPiGroup] = {
    "Re": BuckinghamPiGroup(1, "Re", "ρUL/μ", {"ρ": 1, "U": 1, "L": 1, "μ": -1}, "惯性力/黏性力"),
    "Ma": BuckinghamPiGroup(2, "Ma", "U/c", {"U": 1, "c": -1}, "流速/声速"),
    "Fr": BuckinghamPiGroup(3, "Fr", "U/√(gL)", {"U": 1, "g": -0.5, "L": -0.5}, "惯性力/重力"),
    "St": BuckinghamPiGroup(4, "St", "fL/U", {"f": 1, "L": 1, "U": -1}, "非定常惯性力/对流惯性力"),
    "We": BuckinghamPiGroup(5, "We", "ρU²L/σ", {"ρ": 1, "U": 2, "L": 1, "σ_s": -1}, "惯性力/表面张力"),
    "Ca": BuckinghamPiGroup(6, "Ca", "μU/σ", {"μ": 1, "U": 1, "σ_s": -1}, "黏性力/表面张力"),
    "Pr": BuckinghamPiGroup(7, "Pr", "ν/α", {"ν": 1, "α": -1}, "动量扩散/热扩散"),
    "Pe": BuckinghamPiGroup(8, "Pe", "UL/α", {"U": 1, "L": 1, "α": -1}, "对流/扩散"),
    "Ec": BuckinghamPiGroup(9, "Ec", "U²/(c_p ΔT)", {"U": 2, "c_p": -1, "ΔT": -1}, "动能/焓差"),
    "Gr": BuckinghamPiGroup(10, "Gr", "g β ΔT L³/ν²", {"g": 1, "L": 3, "ν": -2, "ΔT": 1}, "浮力/黏性力"),
}


# ── π 定理引擎 ──


class BuckinghamPiEngine:
    """Buckingham π 定理实现。

    输入：N 个有量纲的物理量
    输出：N - rank(D) 个独立无量纲群

    算法：
      1. 构建维度矩阵 D_{ij} = 第j个量在第i个维度上的指数
      2. 计算 D 的核空间（零空间）
      3. 每个零空间向量对应一个 π 群
      4. 尝试与已知命名 π 群匹配
    """

    def compute(
        self,
        quantities: list[PhysicalQuantity],
    ) -> list[BuckinghamPiGroup]:
        """从物理量列表计算完整 π 群集."""
        if not quantities:
            return []

        # 构建维度矩阵
        n_vars = len(quantities)
        n_dims = len(BASE_DIMENSIONS)

        D = np.zeros((n_dims, n_vars))
        for j, q in enumerate(quantities):
            D[:, j] = q.dim_vector

        # 只保留非零行
        nonzero_rows = [i for i in range(n_dims) if np.any(np.abs(D[i, :]) > 1e-10)]
        if not nonzero_rows:
            return []
        D_reduced = D[nonzero_rows, :]

        # 优先使用 Rust 加速的零空间计算，回退到 SymPy
        pi_groups_raw: list[list[float]] = []
        try:
            pi_groups_raw = _accel.buckingham_pi(D_reduced)
            if pi_groups_raw:
                _logger.debug("Buckingham Pi: using Rust-accelerated nullspace")
            else:
                raise ValueError("Rust nullspace returned empty, falling back to SymPy")
        except (ValueError, TypeError, RuntimeError):
            M = sp.Matrix(D_reduced.tolist())
            nullspace = M.nullspace()
            pi_groups_raw = []
            for vec in nullspace:
                coeffs = [float(v.evalf()) if hasattr(v, "evalf") else float(v) for v in vec]
                max_abs = max(abs(c) for c in coeffs) if coeffs else 1.0
                if max_abs < 1e-10:
                    continue
                pi_groups_raw.append([c / max_abs for c in coeffs])
            _logger.debug("Buckingham Pi: using SymPy fallback nullspace")

        pi_groups = []
        for i, coeffs in enumerate(pi_groups_raw):
            # 归一化
            max_abs = max(abs(c) for c in coeffs) if coeffs else 1.0
            if max_abs < 1e-10:
                continue
            coeffs = [c / max_abs for c in coeffs]

            # 组成表达式
            var_dict: dict[str, float] = {}
            terms = []
            for j, c in enumerate(coeffs):
                if abs(c) > 1e-10 and j < len(quantities):
                    q = quantities[j]
                    var_dict[q.symbol] = round(c, 4)
                    if abs(c - 1.0) < 1e-10:
                        terms.append(q.symbol)
                    elif abs(c - (-1.0)) < 1e-10:
                        terms.append(f"1/{q.symbol}")
                    else:
                        terms.append(f"{q.symbol}^{round(c, 2)}")

            expression = " · ".join(terms) if terms else "1"

            # 尝试命名匹配
            match = self._match_named(var_dict)

            pi_groups.append(
                BuckinghamPiGroup(
                    pi_id=i + 1,
                    name=match if match else f"π_{i + 1}",
                    expression=expression,
                    variables=var_dict,
                    physical_meaning=NAMED_PI_GROUPS.get(match, BuckinghamPiGroup(0, "", "", {})).physical_meaning
                    if match
                    else "",
                )
            )

        return pi_groups

    def _match_named(self, var_dict: dict[str, float]) -> str | None:
        """尝试匹配已知的命名 π 群."""
        for name, pi in NAMED_PI_GROUPS.items():
            if set(var_dict.keys()) == set(pi.variables.keys()):
                # 检查指数是否一致
                match = True
                for var, exp in pi.variables.items():
                    if abs(var_dict.get(var, 0) - exp) > 0.1:
                        match = False
                        break
                if match:
                    return name
        return None

    def identify_fixed(
        self, pi_groups: list[BuckinghamPiGroup], user_params: dict[str, float]
    ) -> list[BuckinghamPiGroup]:
        """找出用户参数选择已经隐式固定的 π 群."""
        fixed = []
        for pg in pi_groups:
            if all(v in user_params for v in pg.variables):
                fixed.append(pg)
        return fixed

    def suggest_variations(self, pi_groups: list[BuckinghamPiGroup], user_params: dict[str, float]) -> list[str]:
        """建议用户应该主动变化的 π 群（科学探索的建议）."""
        suggestions = []
        free_groups = [pg for pg in pi_groups if not all(v in user_params for v in pg.variables)]
        for pg in free_groups:
            suggestions.append(
                f"π 群 '{pg.name}' ({pg.expression}) 未被用户参数完全确定。\n"
                f"  含义：{pg.physical_meaning or '未知'}\n"
                f"  建议：变化该 π 群对应的物理量以探索该系统的自由度。"
            )
        if not suggestions:
            suggestions.append("所有 π 群均被用户参数固定。系统没有可调自由度。")
        return suggestions


# ── 特定领域的 π 群生成 ──


class FluidDimensionAnalyzer:
    """流体力学的维度分析器。

    从 NS 方程的参数自动生成完整的无量纲数集合。
    """

    def analyze_ns(self, params: dict[str, Any]) -> list[BuckinghamPiGroup]:
        """从 NS 设置中分析无量纲数."""
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]

        # 根据参数添加额外物理量
        if params.get("include_energy") or params.get("temperature"):
            quantities.append(BUILTIN_QUANTITIES["thermal_diffusivity"])
            quantities.append(BUILTIN_QUANTITIES["temperature"])
        if params.get("include_gravity"):
            quantities.append(BUILTIN_QUANTITIES["gravity"])
        if params.get("include_surface_tension"):
            quantities.append(BUILTIN_QUANTITIES["surface_tension"])
        if params.get("regime", "").startswith("compressible"):
            # 可压 → 加声速（即频率×长度）
            quantities.append(BUILTIN_QUANTITIES["frequency"])

        engine = BuckinghamPiEngine()
        return engine.compute(quantities)


class QMDimensionAnalyzer:
    """量子力学的维度分析器."""

    def analyze_dft(self, params: dict[str, Any]) -> list[BuckinghamPiGroup]:
        """DFT 特有的无量纲群."""
        quantities = [
            BUILTIN_QUANTITIES["energy"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["electron_density"],
            BUILTIN_QUANTITIES["planck_constant"],
            BUILTIN_QUANTITIES["mass"],
        ]
        if params.get("temperature"):
            quantities.append(BUILTIN_QUANTITIES["boltzmann_constant"])
            quantities.append(BUILTIN_QUANTITIES["temperature"])

        engine = BuckinghamPiEngine()
        return engine.compute(quantities)
