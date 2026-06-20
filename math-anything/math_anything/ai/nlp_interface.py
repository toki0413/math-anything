"""自然语言接口 — AI-Native 用户交互.

允许用户用自然语言描述物理问题，自动映射到引擎和参数。
例如：
  "我想模拟一个钢梁在1000N载荷下的应力分布"
  → engine=abaqus, material=steel, load=1000N, analysis=static_structural

  "计算硅的能带结构"
  → engine=vasp, system=Si, calculation=band_structure
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── 关键词映射 ──

_ENGINE_KEYWORDS: dict[str, list[str]] = {
    "vasp": [
        "dft",
        "能带",
        "band",
        "电子结构",
        "第一性原理",
        "ab initio",
        "ab-initio",
        "密度泛函",
        "kohn-sham",
        "平面波",
        "pseudopotential",
        "赝势",
        "fermi",
        "dos",
        "态密度",
        "硅",
        "silicon",
        "半导体",
        "semiconductor",
    ],
    "lammps": [
        "分子动力学",
        "md",
        "molecular dynamics",
        "原子模拟",
        "势函数",
        "nvt",
        "npt",
        "扩散",
        "diffusion",
        "径向分布",
        "rdf",
        "轨迹",
        "trajectory",
    ],
    "abaqus": [
        "有限元",
        "fem",
        "应力",
        "stress",
        "应变",
        "strain",
        "梁",
        "beam",
        "壳",
        "shell",
        "接触",
        "contact",
        "断裂",
        "fracture",
        "疲劳",
        "fatigue",
        "载荷",
        "load",
        "位移",
        "displacement",
    ],
    "ansys": [
        "有限元",
        "fem",
        "热分析",
        "thermal",
        "模态",
        "modal",
        "谐响应",
        "harmonic",
        "流固耦合",
        "fsi",
        "电磁",
        "electromagnetic",
    ],
    "openfoam": [
        "流体",
        "cfd",
        "navier-stokes",
        "湍流",
        "turbulence",
        "雷诺",
        "reynolds",
        "边界层",
        "boundary layer",
        "绕流",
        "drag",
        "升力",
        "lift",
        "不可压",
        "incompressible",
        "可压",
        "compressible",
    ],
    "quantum_espresso": [
        "dft",
        "能带",
        "band",
        "声子",
        "phonon",
        "dfpt",
        "密度泛函",
        "平面波",
        "赝势",
    ],
    "gaussian": [
        "量子化学",
        "quantum chemistry",
        "分子轨道",
        "molecular orbital",
        "hf",
        "mp2",
        "ccsd",
        "耦合簇",
        "coupled cluster",
        "基组",
        "basis set",
        "优化",
        "optimization",
        "频率",
        "frequency",
    ],
}

_MATERIAL_KEYWORDS: dict[str, dict[str, Any]] = {
    "steel": {"youngs_modulus": 200, "poisson_ratio": 0.3, "density": 7.8},
    "aluminum": {"youngs_modulus": 70, "poisson_ratio": 0.33, "density": 2.7},
    "copper": {"youngs_modulus": 120, "poisson_ratio": 0.34, "density": 8.9},
    "titanium": {"youngs_modulus": 116, "poisson_ratio": 0.32, "density": 4.5},
    "concrete": {"youngs_modulus": 30, "poisson_ratio": 0.2, "density": 2.4},
    "silicon": {"youngs_modulus": 130, "poisson_ratio": 0.28, "density": 2.33},
    "water": {"density": 1.0, "viscosity": 1e-3},
    "air": {"density": 1.225, "viscosity": 1.81e-5},
}

_ANALYSIS_KEYWORDS: dict[str, str] = {
    "应力": "static_structural",
    "strain": "static_structural",
    "模态": "modal",
    "热": "thermal",
    "瞬态": "transient",
    "稳态": "steady_state",
    "能带": "band_structure",
    "态密度": "dos",
    "优化": "geometry_optimization",
    "分子动力学": "md",
    "扩散": "diffusion",
    "声子": "phonon",
}


@dataclass
class NLParseResult:
    """自然语言解析结果."""

    raw_query: str
    detected_engine: str | None = None
    detected_material: str | None = None
    detected_analysis: str | None = None
    extracted_parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)


class NaturalLanguageInterface:
    """自然语言接口.

    将自然语言描述映射到引擎选择和参数配置。
    """

    def __init__(self):
        self._engine_keywords = _ENGINE_KEYWORDS
        self._material_keywords = _MATERIAL_KEYWORDS
        self._analysis_keywords = _ANALYSIS_KEYWORDS

    def parse(self, query: str) -> NLParseResult:
        """解析自然语言查询.

        Args:
            query: 自然语言描述，如 "模拟钢梁在1000N载荷下的应力"

        Returns:
            解析结果，包含检测到的引擎、材料、分析类型和参数
        """
        result = NLParseResult(raw_query=query)
        query_lower = query.lower()

        # 1. 检测引擎
        engine_scores: dict[str, float] = {}
        for engine, keywords in self._engine_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                engine_scores[engine] = score

        if engine_scores:
            best_engine = max(engine_scores, key=engine_scores.get)  # type: ignore[arg-type]
            result.detected_engine = best_engine
            result.confidence = min(1.0, engine_scores[best_engine] / 3.0)
            result.alternatives = [
                (eng, score / 3.0)
                for eng, score in sorted(engine_scores.items(), key=lambda x: -x[1])
                if eng != best_engine
            ]

        # 2. 检测材料
        for material, props in self._material_keywords.items():
            if material in query_lower:
                result.detected_material = material
                result.extracted_parameters.update(props)
                break

        # 3. 检测分析类型
        for keyword, analysis_type in self._analysis_keywords.items():
            if keyword in query_lower:
                result.detected_analysis = analysis_type
                break

        # 4. 提取数值参数
        result.extracted_parameters.update(self._extract_numbers(query))

        # 5. 生成澄清问题
        if not result.detected_engine:
            result.clarification_questions.append("您想使用哪个仿真引擎？（VASP/LAMMPS/Abaqus/OpenFOAM/...）")
        if not result.detected_material and result.detected_engine in ["abaqus", "ansys"]:
            result.clarification_questions.append("请指定材料类型（钢/铝/铜/...）")
        if not result.detected_analysis:
            result.clarification_questions.append("您想进行什么类型的分析？")

        return result

    def _extract_numbers(self, query: str) -> dict[str, Any]:
        """从查询中提取数值参数."""
        params = {}

        # 载荷: "1000N", "1000 N"
        load_match = re.search(r"(\d+(?:\.\d+)?)\s*[Nn](?:ewton)?\b", query)
        if load_match:
            params["load"] = float(load_match.group(1))

        # 温度: "300K", "300 K"
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*[Kk]\b", query)
        if temp_match:
            params["temperature"] = float(temp_match.group(1))

        # 压力: "1.5MPa", "1.5 MPa"
        pressure_match = re.search(r"(\d+(?:\.\d+)?)\s*[Mm][Pp][Aa]\b", query)
        if pressure_match:
            params["pressure"] = float(pressure_match.group(1))

        # 能量截断: "520eV", "520 eV"
        encut_match = re.search(r"(\d+(?:\.\d+)?)\s*[Ee][Vv]\b", query)
        if encut_match:
            params["encut"] = float(encut_match.group(1))

        # 时间: "1.5ps", "1.5 ps"
        time_match = re.search(r"(\d+(?:\.\d+)?)\s*[Pp][Ss]\b", query)
        if time_match:
            params["timestep"] = float(time_match.group(1)) * 1000  # ps → fs

        # 雷诺数
        re_match = re.search(r"[Rr]e\s*[=≈~]\s*(\d+(?:\.\d+)?[eE]?[+-]?\d*)", query)
        if re_match:
            params["reynolds_number"] = float(re_match.group(1))

        return params

    def generate_prompt(self, parse_result: NLParseResult) -> str:
        """生成 LLM 友好的结构化 prompt."""
        parts = [
            f"Engine: {parse_result.detected_engine or 'unknown'}",
            f"Material: {parse_result.detected_material or 'unknown'}",
            f"Analysis: {parse_result.detected_analysis or 'unknown'}",
            f"Parameters: {parse_result.extracted_parameters}",
        ]
        if parse_result.clarification_questions:
            parts.append(f"Clarifications needed: {parse_result.clarification_questions}")
        return "\n".join(parts)
