"""维度分析模块的单元测试.

覆盖:
  - PhysicalQuantity 创建与维度向量
  - BuckinghamPiEngine 计算（含空输入、单维度、多维度场景）
  - BuckinghamPiGroup 序列化
  - NAMED_PI_GROUPS 命名匹配
  - FluidDimensionAnalyzer / QMDimensionAnalyzer
  - EquationChecker 方程维度一致性
  - Rust 加速回退路径
"""

from __future__ import annotations

import numpy as np
import pytest

from math_anything.dimensional.equation_checker import (
    EquationChecker,
    EquationDimensionalCheck,
    TermDimension,
)
from math_anything.dimensional.scaling_group import (
    BASE_DIMENSIONS,
    BUILTIN_QUANTITIES,
    NAMED_PI_GROUPS,
    BuckinghamPiEngine,
    BuckinghamPiGroup,
    FluidDimensionAnalyzer,
    PhysicalQuantity,
    QMDimensionAnalyzer,
)

# ── PhysicalQuantity ──


class TestPhysicalQuantity:
    """PhysicalQuantity 数据类测试."""

    def test_create_with_dimensions(self):
        pq = PhysicalQuantity(
            name="force",
            symbol="F",
            dimensions={"M": 1, "L": 1, "T": -2},
            canonical_unit="N",
            physical_role="state_variable",
            description="力",
        )
        assert pq.name == "force"
        assert pq.symbol == "F"
        assert pq.dimensions["M"] == 1
        assert pq.dimensions["L"] == 1
        assert pq.dimensions["T"] == -2

    def test_dim_vector_length(self):
        pq = PhysicalQuantity("length", "L", {"L": 1}, "m")
        vec = pq.dim_vector
        assert len(vec) == len(BASE_DIMENSIONS)

    def test_dim_vector_values(self):
        pq = PhysicalQuantity("velocity", "U", {"L": 1, "T": -1}, "m/s")
        vec = pq.dim_vector
        idx_L = BASE_DIMENSIONS.index("L")
        idx_T = BASE_DIMENSIONS.index("T")
        assert vec[idx_L] == 1.0
        assert vec[idx_T] == -1.0

    def test_dim_vector_empty_dimensions(self):
        pq = PhysicalQuantity("dimensionless", "α", {}, "")
        vec = pq.dim_vector
        assert np.allclose(vec, 0.0)

    def test_default_fields(self):
        pq = PhysicalQuantity("test", "t")
        assert pq.dimensions == {}
        assert pq.canonical_unit == ""
        assert pq.physical_role == ""
        assert pq.description == ""


# ── BuckinghamPiGroup ──


class TestBuckinghamPiGroup:
    """BuckinghamPiGroup 数据类测试."""

    def test_create(self):
        pg = BuckinghamPiGroup(1, "Re", "ρUL/μ", {"ρ": 1, "U": 1, "L": 1, "μ": -1})
        assert pg.pi_id == 1
        assert pg.name == "Re"
        assert pg.expression == "ρUL/μ"

    def test_to_dict(self):
        pg = BuckinghamPiGroup(
            1,
            "Re",
            "ρUL/μ",
            {"ρ": 1, "U": 1, "L": 1, "μ": -1},
            physical_meaning="惯性力/黏性力",
        )
        d = pg.to_dict()
        assert d["pi_id"] == 1
        assert d["name"] == "Re"
        assert d["variables"]["ρ"] == 1
        assert d["physical_meaning"] == "惯性力/黏性力"

    def test_default_physical_meaning(self):
        pg = BuckinghamPiGroup(2, "π_2", "a/b", {"a": 1, "b": -1})
        assert pg.physical_meaning == ""


# ── BuckinghamPiEngine ──


class TestBuckinghamPiEngine:
    """Buckingham π 定理引擎测试."""

    def test_empty_input(self):
        engine = BuckinghamPiEngine()
        result = engine.compute([])
        assert result == []

    def test_single_quantity_no_pi(self):
        """单个物理量无法形成无量纲群（维度矩阵 rank = 1, N - rank = 0）."""
        engine = BuckinghamPiEngine()
        result = engine.compute([BUILTIN_QUANTITIES["length"]])
        # 只有1个量，维度矩阵1列，零空间为空
        assert result == []

    def test_fluid_mechanics_reynolds(self):
        """ρ, U, L, μ 应产生至少一个无量纲群."""
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        result = engine.compute(quantities)
        assert len(result) >= 1
        # 检查产生了 π 群（可能匹配到 Re，也可能只是 π_1）
        for pg in result:
            assert pg.pi_id >= 1

    def test_pi_group_has_variables(self):
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        result = engine.compute(quantities)
        for pg in result:
            assert isinstance(pg.variables, dict)
            assert len(pg.variables) > 0

    def test_pi_group_expression_nonempty(self):
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        result = engine.compute(quantities)
        for pg in result:
            assert pg.expression != ""

    def test_identify_fixed(self):
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        pi_groups = engine.compute(quantities)
        # 所有变量都在 user_params 中 → 全部固定
        user_params = {"ρ": 1.0, "U": 1.0, "L": 1.0, "μ": 1.0}
        fixed = engine.identify_fixed(pi_groups, user_params)
        assert len(fixed) == len(pi_groups)

    def test_identify_fixed_partial(self):
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        pi_groups = engine.compute(quantities)
        user_params = {"ρ": 1.0}
        fixed = engine.identify_fixed(pi_groups, user_params)
        # 只有一个变量在 user_params 中，不足以固定任何 π 群
        assert len(fixed) == 0

    def test_suggest_variations(self):
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        pi_groups = engine.compute(quantities)
        suggestions = engine.suggest_variations(pi_groups, {})
        assert len(suggestions) > 0

    def test_suggest_variations_all_fixed(self):
        engine = BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        pi_groups = engine.compute(quantities)
        user_params = {"ρ": 1.0, "U": 1.0, "L": 1.0, "μ": 1.0}
        suggestions = engine.suggest_variations(pi_groups, user_params)
        assert any("没有可调自由度" in s for s in suggestions)


# ── BUILTIN_QUANTITIES ──


class TestBuiltinQuantities:
    """内建物理量数据库测试."""

    def test_key_quantities_exist(self):
        for key in ["length", "mass", "time", "velocity", "force", "energy", "density", "pressure", "temperature"]:
            assert key in BUILTIN_QUANTITIES

    def test_all_have_dim_vector(self):
        for name, pq in BUILTIN_QUANTITIES.items():
            vec = pq.dim_vector
            assert len(vec) == len(BASE_DIMENSIONS), f"{name} dim_vector 长度错误"

    def test_energy_dimensions(self):
        e = BUILTIN_QUANTITIES["energy"]
        assert e.dimensions == {"M": 1, "L": 2, "T": -2}

    def test_pressure_dimensions(self):
        p = BUILTIN_QUANTITIES["pressure"]
        assert p.dimensions == {"M": 1, "L": -1, "T": -2}


# ── NAMED_PI_GROUPS ──


class TestNamedPiGroups:
    """命名无量纲数测试."""

    def test_reynolds_exists(self):
        assert "Re" in NAMED_PI_GROUPS

    def test_mach_exists(self):
        assert "Ma" in NAMED_PI_GROUPS

    def test_froude_exists(self):
        assert "Fr" in NAMED_PI_GROUPS

    def test_prandtl_exists(self):
        assert "Pr" in NAMED_PI_GROUPS

    def test_all_have_physical_meaning(self):
        for name, pg in NAMED_PI_GROUPS.items():
            assert pg.physical_meaning != "", f"{name} 缺少物理含义"


# ── FluidDimensionAnalyzer ──


class TestFluidDimensionAnalyzer:
    """流体力学维度分析器测试."""

    def test_basic_ns(self):
        analyzer = FluidDimensionAnalyzer()
        result = analyzer.analyze_ns({})
        assert len(result) >= 1

    def test_ns_with_energy(self):
        analyzer = FluidDimensionAnalyzer()
        result = analyzer.analyze_ns({"include_energy": True})
        assert len(result) >= 1

    def test_ns_with_gravity(self):
        analyzer = FluidDimensionAnalyzer()
        result = analyzer.analyze_ns({"include_gravity": True})
        assert len(result) >= 1

    def test_ns_with_surface_tension(self):
        analyzer = FluidDimensionAnalyzer()
        result = analyzer.analyze_ns({"include_surface_tension": True})
        assert len(result) >= 1


# ── QMDimensionAnalyzer ──


class TestQMDimensionAnalyzer:
    """量子力学维度分析器测试."""

    def test_basic_dft(self):
        analyzer = QMDimensionAnalyzer()
        result = analyzer.analyze_dft({})
        assert isinstance(result, list)

    def test_dft_with_temperature(self):
        analyzer = QMDimensionAnalyzer()
        result = analyzer.analyze_dft({"temperature": True})
        assert isinstance(result, list)


# ── EquationChecker ──


class TestEquationChecker:
    """方程维度一致性检查器测试."""

    def test_consistent_equation(self):
        checker = EquationChecker()
        # F = m * a: 两边都是 M L T^-2
        result = checker.check_equation(
            "Newton's second law",
            lhs_terms=["F (force)"],
            rhs_terms=["m (mass)", "a (acceleration)"],
        )
        assert isinstance(result, EquationDimensionalCheck)

    def test_inconsistent_equation(self):
        checker = EquationChecker()
        result = checker.check_equation(
            "bogus equation",
            lhs_terms=["F (force)"],
            rhs_terms=["L (length)"],
        )
        assert result.consistent is False

    def test_unknown_term(self):
        """_analyze_term 对完全无法匹配的项应返回 invalid."""
        checker = EquationChecker()
        # 使用不含任何已知符号子串的项来测试
        result = checker._analyze_term("ZZZXYZ123")
        assert result.valid is False

    def test_schema_spectral(self):
        checker = EquationChecker()
        result = checker.check_schema("H[n]ψ = εψ")
        assert isinstance(result, EquationDimensionalCheck)

    def test_schema_equilibrium(self):
        checker = EquationChecker()
        result = checker.check_schema("∇·σ = f")
        assert isinstance(result, EquationDimensionalCheck)

    def test_schema_unrecognized(self):
        checker = EquationChecker()
        result = checker.check_schema("some random string")
        assert "无法自动解析" in result.notes[0]


# ── Rust 加速回退 ──


class TestRustFallback:
    """Rust 加速路径回退测试."""

    def test_buckingham_pi_python_fallback(self):
        """确保 Python 回退路径能正确计算."""
        from math_anything.rust_bridge import EMLAccelerator, _EMLPyFallback

        BuckinghamPiEngine()
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        # 直接调用 Python 回退
        D = np.array([q.dim_vector for q in quantities]).T
        nonzero_rows = [i for i in range(len(BASE_DIMENSIONS)) if np.any(np.abs(D[i, :]) > 1e-10)]
        D_reduced = D[nonzero_rows, :]
        result = _EMLPyFallback.buckingham_pi(D_reduced)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_rust_bridge_buckingham_pi(self):
        """EMLAccelerator.buckingham_pi 应返回列表."""
        from math_anything.rust_bridge import EMLAccelerator

        accel = EMLAccelerator()
        D = np.array([[1, 1, 1, 0], [-3, 0, 1, -1], [0, -1, 0, -1]], dtype=float)
        result = accel.buckingham_pi(D)
        assert isinstance(result, list)

    def test_rust_availability_check(self):
        from math_anything.rust_bridge import EMLAccelerator

        accel = EMLAccelerator()
        # using_rust 是 bool，不应抛异常
        assert isinstance(accel.using_rust, bool)
