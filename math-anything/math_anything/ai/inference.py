"""智能推理引擎 — AI-Native 数学结构推理.

核心能力：
  1. 自动参数推断：从部分参数推导缺失参数
  2. 物理一致性检查：验证提取结果是否符合物理定律
  3. 跨引擎知识迁移：利用已知引擎的结果推断未知引擎
  4. 数学结构补全：从部分结构推断完整数学模型
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InferenceConfidence(Enum):
    """推理置信度."""

    HIGH = "high"  # 基于物理定律的确定性推理
    MEDIUM = "medium"  # 基于经验规律的统计推理
    LOW = "low"  # 基于启发式的猜测
    SPECULATIVE = "speculative"  # 跨域迁移的推测


@dataclass
class InferenceResult:
    """推理结果."""

    parameter: str
    inferred_value: Any
    confidence: InferenceConfidence
    reasoning: str  # 推理过程（可解释）
    source: str  # 推理来源（物理定律/经验/迁移/启发式）
    alternatives: list[tuple[Any, InferenceConfidence]] = field(default_factory=list)


class PhysicsKnowledgeBase:
    """物理知识库 — 用于推理的领域知识."""

    # 典型物理参数范围
    PARAMETER_RANGES = {
        # VASP
        "encut": {"min": 200, "max": 1000, "typical": 520, "unit": "eV"},
        "ediff": {"min": 1e-8, "max": 1e-4, "typical": 1e-6, "unit": "eV"},
        "kpoints_grid": {"min": 1, "max": 30, "typical": [4, 4, 4], "unit": "grid"},
        # LAMMPS
        "timestep": {"min": 0.1, "max": 10.0, "typical": 1.0, "unit": "fs"},
        "temperature": {"min": 0, "max": 10000, "typical": 300, "unit": "K"},
        "pressure": {"min": 0, "max": 1e6, "typical": 0, "unit": "bar"},
        # Abaqus/Ansys
        "youngs_modulus": {"min": 0.001, "max": 2000, "typical": 200, "unit": "GPa"},
        "poisson_ratio": {"min": 0.0, "max": 0.5, "typical": 0.3, "unit": ""},
        "density": {"min": 0.01, "max": 25, "typical": 7.8, "unit": "g/cm³"},
        # OpenFOAM
        "reynolds_number": {"min": 0.01, "max": 1e8, "typical": 1e4, "unit": ""},
        "viscosity": {"min": 1e-7, "max": 1e3, "typical": 1e-3, "unit": "Pa·s"},
        # Gaussian
        "basis_set_quality": {
            "values": [
                "sto-3g",
                "3-21g",
                "6-31g",
                "6-31g*",
                "6-311g**",
                "cc-pvdz",
                "cc-pvtz",
                "aug-cc-pvtz",
            ],
        },
        "method_quality": {
            "values": ["hf", "b3lyp", "mp2", "ccsd", "ccsd(t)"],
        },
    }

    # 物理定律约束
    PHYSICS_CONSTRAINTS = {
        "poisson_ratio_range": "0 ≤ ν ≤ 0.5 (热力学稳定性)",
        "bulk_modulus_positive": "K > 0 (材料稳定性)",
        "speed_of_sound": "c = sqrt(E(1-ν)/(ρ(1+ν)(1-2ν)))",
        "de_broglie": "λ = h/p (量子力学)",
        "cfl_condition": "Δt ≤ Δx / (c + |u|) (数值稳定性)",
        "energy_conservation": "E_total = E_kinetic + E_potential (守恒律)",
        "uncertainty_principle": "ΔxΔp ≥ ℏ/2",
    }

    # 跨引擎知识映射
    CROSS_ENGINE_MAP = {
        ("vasp", "quantum_espresso"): {
            "encut": "ecutwfc",  # VASP encut ≈ QE ecutwfc
            "ediff": "conv_thr",
            "kpoints": "k_points",
        },
        ("lammps", "openfoam"): {
            "timestep": "deltaT",  # 时间步长映射
            "temperature": "T",
        },
        ("abaqus", "ansys"): {
            "youngs_modulus": "EX",
            "poisson_ratio": "PRXY",
            "density": "DENS",
        },
    }

    @classmethod
    def infer_parameter(cls, param_name: str, known_params: dict[str, Any], engine: str) -> InferenceResult | None:
        """从已知参数推断缺失参数.

        Args:
            param_name: 需要推断的参数名
            known_params: 已知参数字典
            engine: 引擎名称

        Returns:
            推断结果，或 None（无法推断）
        """
        # 策略1: 查找参数典型值
        if param_name in cls.PARAMETER_RANGES:
            info = cls.PARAMETER_RANGES[param_name]
            if "typical" in info:
                return InferenceResult(
                    parameter=param_name,
                    inferred_value=info["typical"],
                    confidence=InferenceConfidence.MEDIUM,
                    reasoning=(f"使用 {param_name} 的典型值 {info['typical']} {info.get('unit', '')}"),
                    source="经验规律",
                )

        # 策略2: 物理定律推导
        if param_name == "timestep" and "reynolds_number" in known_params:
            Re = known_params["reynolds_number"]
            if Re > 0:
                # CFL 条件启发式
                dt = min(1.0, 10.0 / Re)
                return InferenceResult(
                    parameter=param_name,
                    inferred_value=round(dt, 4),
                    confidence=InferenceConfidence.MEDIUM,
                    reasoning=f"CFL 条件推导: Re={Re}, Δt ≈ 10/Re",
                    source="物理定律",
                )

        if param_name == "speed_of_sound" and "youngs_modulus" in known_params and "density" in known_params:
            E = known_params["youngs_modulus"]
            rho = known_params["density"]
            nu = known_params.get("poisson_ratio", 0.3)
            c = (E * 1e9 * (1 - nu) / (rho * 1000 * (1 + nu) * (1 - 2 * nu))) ** 0.5
            return InferenceResult(
                parameter=param_name,
                inferred_value=round(c, 2),
                confidence=InferenceConfidence.HIGH,
                reasoning=(f"c = sqrt(E(1-ν)/(ρ(1+ν)(1-2ν))), E={E}GPa, ρ={rho}g/cm³, ν={nu}"),
                source="物理定律",
            )

        # 策略3: 跨引擎迁移
        for (src_engine, tgt_engine), mapping in cls.CROSS_ENGINE_MAP.items():
            if tgt_engine == engine and param_name in mapping.values():
                # 找到源引擎的对应参数
                src_param = None
                for k, v in mapping.items():
                    if v == param_name:
                        src_param = k
                        break
                if src_param and src_param in known_params:
                    return InferenceResult(
                        parameter=param_name,
                        inferred_value=known_params[src_param],
                        confidence=InferenceConfidence.LOW,
                        reasoning=(f"从 {src_engine} 的 {src_param}={known_params[src_param]} 迁移"),
                        source="跨引擎迁移",
                    )

        return None


class IntelligentInferenceEngine:
    """智能推理引擎.

    能力：
    1. 自动参数推断
    2. 物理一致性验证
    3. 跨引擎知识迁移
    4. 数学结构补全
    """

    def __init__(self):
        self._kb = PhysicsKnowledgeBase()
        self._inference_history: list[InferenceResult] = []

    def infer_missing_parameters(
        self,
        engine: str,
        params: dict[str, Any],
        required_params: list[str],
    ) -> dict[str, InferenceResult]:
        """推断缺失的参数.

        Args:
            engine: 引擎名称
            params: 已提供的参数
            required_params: 需要的参数列表

        Returns:
            参数名 → 推断结果 的映射
        """
        results = {}
        missing = [p for p in required_params if p not in params]

        for param_name in missing:
            result = self._kb.infer_parameter(param_name, params, engine)
            if result is not None:
                results[param_name] = result
                self._inference_history.append(result)
                logger.info(
                    f"推断 {param_name}={result.inferred_value} "
                    f"(置信度={result.confidence.value}, 来源={result.source})"
                )

        return results

    def check_physics_consistency(self, params: dict[str, Any]) -> list[str]:
        """检查参数的物理一致性.

        Returns:
            不一致问题列表
        """
        issues = []

        # Poisson 比
        nu = params.get("poisson_ratio")
        if nu is not None and (nu < 0 or nu > 0.5):
            issues.append(f"Poisson 比 ν={nu} 超出 [0, 0.5] 范围，违反热力学稳定性")

        # 杨氏模量
        E = params.get("youngs_modulus")
        if E is not None and E <= 0:
            issues.append(f"杨氏模量 E={E} ≤ 0，违反材料稳定性")

        # 密度
        rho = params.get("density")
        if rho is not None and rho <= 0:
            issues.append(f"密度 ρ={rho} ≤ 0，物理上不可能")

        # 温度
        T = params.get("temperature")
        if T is not None and T < 0:
            issues.append(f"温度 T={T} < 0K，违反热力学第三定律")

        # CFL 条件
        dt = params.get("timestep")
        dx = params.get("mesh_size")
        c = params.get("speed_of_sound")
        if dt and dx and c:
            if dt > dx / c:
                issues.append(f"CFL 违反: Δt={dt} > Δx/c={dx / c:.4f}")

        return issues

    def suggest_improvements(self, engine: str, params: dict[str, Any]) -> list[str]:
        """建议参数改进.

        Returns:
            改进建议列表
        """
        suggestions = []

        # VASP: ENCUT 建议
        encut = params.get("encut")
        if encut and encut < 400:
            suggestions.append(f"ENCUT={encut}eV 较低，建议 ≥ 400eV 以确保收敛")

        # LAMMPS: 时间步长建议
        dt = params.get("timestep")
        if dt and dt > 2.0:
            suggestions.append(f"timestep={dt}fs 较大，建议 ≤ 2fs 以确保能量守恒")

        # OpenFOAM: Re 建议
        Re = params.get("reynolds_number")
        if Re and Re > 1e6:
            suggestions.append(f"Re={Re:.0e} 极高，建议使用 LES 或 DNS 而非 RANS")

        # Gaussian: 基组建议
        basis = params.get("basis")
        if basis and basis in ["sto-3g", "3-21g"]:
            suggestions.append(f"基组 {basis} 精度较低，建议至少 6-31G*")

        return suggestions

    def cross_engine_translate(
        self, src_engine: str, tgt_engine: str, src_params: dict[str, Any]
    ) -> dict[str, InferenceResult]:
        """跨引擎参数迁移."""
        results = {}
        key = (src_engine, tgt_engine)

        if key in self._kb.CROSS_ENGINE_MAP:
            mapping = self._kb.CROSS_ENGINE_MAP[key]
            for src_param, tgt_param in mapping.items():
                if src_param in src_params:
                    results[tgt_param] = InferenceResult(
                        parameter=tgt_param,
                        inferred_value=src_params[src_param],
                        confidence=InferenceConfidence.LOW,
                        reasoning=(f"从 {src_engine}.{src_param} 迁移到 {tgt_engine}.{tgt_param}"),
                        source="跨引擎迁移",
                    )

        return results

    @property
    def inference_history(self) -> list[InferenceResult]:
        return list(self._inference_history)
