"""Enhanced Abaqus extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体材料参数（弹性模量、泊松比、屈服强度等）
2. 生成用户友好的详细摘要
3. 保留原始文件的完整关键信息
4. 添加输入文件错误检查功能
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .extractor import AbaqusExtractor, FEMSettings
from .parser import BoundaryCondition as AbaqusBC
from .parser import Material, Step


@dataclass
class DetailedParameter:
    """详细参数定义"""

    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""
    line_number: int = 0


@dataclass
class FEMSimulationSummary:
    """FEM模拟摘要"""

    # 基本信息
    analysis_type: str = ""
    nlgeom: bool = False
    total_nodes: int = 0
    total_elements: int = 0
    element_types: List[str] = field(default_factory=list)

    # 材料信息
    material_name: str = ""
    material_type: str = ""  # 弹性/塑性/损伤等
    elastic_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    density: Optional[float] = None

    # 边界条件
    num_dirichlet_bc: int = 0  # 位移边界
    num_neumann_bc: int = 0  # 力边界
    bc_summary: List[str] = field(default_factory=list)

    # 载荷步
    num_steps: int = 0
    step_types: List[str] = field(default_factory=list)

    # 求解器设置
    solver_type: str = ""
    convergence_criteria: str = ""

    # 资源估计
    estimated_dof: int = 0  # 自由度估计
    estimated_memory: str = ""


class EnhancedAbaqusExtractor(AbaqusExtractor):
    """增强版Abaqus提取器"""

    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = FEMSimulationSummary()
        self.raw_keywords: List[Dict[str, Any]] = []
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        input_path = files.get("input")
        if not input_path:
            raise ValueError("Input file required")

        # 1. 调用父类的标准提取
        schema = self.extract(files)

        # 2. 解析原始关键字
        self.raw_keywords = self._parse_raw_keywords(input_path)

        # 3. 提取详细参数
        self.detailed_params = self._extract_detailed_parameters()

        # 4. 生成模拟摘要
        self.simulation_summary = self._generate_simulation_summary()

        # 5. 验证输入文件
        self._validate_input()

        return {
            "schema": schema,
            "detailed_params": self.detailed_params,
            "summary": self.simulation_summary,
            "raw_keywords": self.raw_keywords,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0,
            },
        }

    def _parse_raw_keywords(self, input_path: str) -> List[Dict[str, Any]]:
        """解析并保留原始关键字"""
        keywords = []
        with open(input_path, "r") as f:
            lines = f.readlines()

        current_keyword = None
        current_data = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("*"):
                # 保存前一个关键字
                if current_keyword:
                    keywords.append(
                        {
                            "keyword": current_keyword,
                            "data": current_data,
                            "line_number": current_keyword.get("line_number", 0),
                        }
                    )

                # 开始新关键字
                parts = stripped[1:].split(",")
                keyword_name = parts[0].strip().upper()
                params = {}
                for part in parts[1:]:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        params[key.strip().lower()] = value.strip()

                current_keyword = {
                    "name": keyword_name,
                    "params": params,
                    "line_number": i,
                }
                current_data = []
            elif current_keyword and stripped:
                current_data.append(stripped)

        # 保存最后一个关键字
        if current_keyword:
            keywords.append(
                {
                    "keyword": current_keyword,
                    "data": current_data,
                    "line_number": current_keyword.get("line_number", 0),
                }
            )

        return keywords

    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []

        # 从材料定义提取
        if self.settings and self.settings.material:
            mat = self.settings.material
            if hasattr(mat, "elastic") and mat.elastic:
                elastic = mat.elastic
                if isinstance(elastic, dict):
                    if "E" in elastic:
                        params.append(
                            DetailedParameter(
                                name="elastic_modulus",
                                value=elastic["E"],
                                unit="MPa" if elastic["E"] > 1000 else "GPa",
                                description="弹性模量",
                                source="*ELASTIC",
                            )
                        )
                    if "nu" in elastic:
                        params.append(
                            DetailedParameter(
                                name="poisson_ratio",
                                value=elastic["nu"],
                                unit="",
                                description="泊松比",
                                source="*ELASTIC",
                            )
                        )

        # 从原始关键字提取
        for kw in self.raw_keywords:
            keyword = kw["keyword"]
            name = keyword["name"]

            if name == "STEP":
                params.append(
                    DetailedParameter(
                        name="analysis_step",
                        value=keyword["params"].get("name", "unnamed"),
                        unit="",
                        description="分析步名称",
                        source=f"*STEP (line {keyword['line_number']})",
                    )
                )

            elif name == "STATIC":
                if kw["data"]:
                    line = kw["data"][0].split(",")
                    if len(line) >= 1:
                        try:
                            params.append(
                                DetailedParameter(
                                    name="initial_increment",
                                    value=float(line[0]),
                                    unit="",
                                    description="初始时间增量",
                                    source="*STATIC",
                                )
                            )
                        except (ValueError, IndexError):
                            pass

            elif name == "BOUNDARY":
                for data_line in kw["data"]:
                    parts = data_line.split(",")
                    if len(parts) >= 2:
                        params.append(
                            DetailedParameter(
                                name="boundary_constraint",
                                value=parts[0].strip(),
                                unit="",
                                description=f"边界条件: {parts[1].strip() if len(parts) > 1 else ''}",
                                source=f"*BOUNDARY (line {keyword['line_number']})",
                            )
                        )

        return params

    def _generate_simulation_summary(self) -> FEMSimulationSummary:
        """生成模拟摘要"""
        summary = FEMSimulationSummary()

        # 分析类型
        if self.settings:
            summary.analysis_type = self.settings.analysis_type
            summary.nlgeom = self.settings.nlgeom
            summary.total_nodes = self.settings.nodes
            summary.element_types = (
                list(set(self.settings.elements)) if self.settings.elements else []
            )
            summary.num_steps = len(self.settings.steps)

            # 材料
            if self.settings.material:
                summary.material_name = getattr(
                    self.settings.material, "name", "unnamed"
                )

        # 从原始关键字统计
        for kw in self.raw_keywords:
            name = kw["keyword"]["name"]
            if name == "BOUNDARY":
                summary.num_dirichlet_bc += 1
            elif name in ("CLOAD", "DLOAD"):
                summary.num_neumann_bc += 1
            elif name == "STEP":
                summary.step_types.append("static")

        # 自由度估计
        if summary.total_nodes > 0:
            summary.estimated_dof = summary.total_nodes * 3  # 3D问题
            if summary.estimated_dof < 10000:
                summary.estimated_memory = (
                    f"约 {summary.estimated_dof * 8 / 1024:.1f} MB"
                )
            else:
                summary.estimated_memory = (
                    f"约 {summary.estimated_dof * 8 / 1024 / 1024:.1f} GB"
                )

        return summary

    def _validate_input(self):
        """验证输入文件"""
        # 检查材料定义
        if not self.settings or not self.settings.material:
            self.validation_errors.append("未定义材料 (*MATERIAL)")

        # 检查分析步
        if not self.settings or not self.settings.steps:
            self.validation_errors.append("未定义分析步 (*STEP)")

        # 检查边界条件
        if self.settings and not self.settings.boundary_conditions:
            self.validation_warnings.append("未定义边界条件 (*BOUNDARY)")

        # 检查几何非线性
        if self.settings and self.settings.nlgeom:
            self.validation_warnings.append(
                "几何非线性已启用 (nlgeom=YES)，确保材料模型兼容"
            )

        # 检查单元类型
        if self.settings and self.settings.elements:
            element_types = set(self.settings.elements)
            if len(element_types) > 3:
                self.validation_warnings.append(
                    f"使用了 {len(element_types)} 种单元类型，检查是否必要"
                )


def generate_abaqus_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成Abaqus摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]

    report = []
    report.append("=" * 70)
    report.append("Abaqus FEM 模拟输入文件摘要")
    report.append("=" * 70)
    report.append("")

    report.append("【基本信息】")
    report.append(f"  分析类型: {summary.analysis_type}")
    report.append(f"  几何非线性: {'是' if summary.nlgeom else '否'}")
    report.append(f"  节点数: {summary.total_nodes:,}")
    report.append(
        f"  单元类型: {', '.join(summary.element_types) if summary.element_types else '未指定'}"
    )
    report.append("")

    report.append("【材料信息】")
    report.append(f"  材料名称: {summary.material_name or '未指定'}")
    if summary.elastic_modulus:
        report.append(f"  弹性模量: {summary.elastic_modulus} MPa")
    if summary.poisson_ratio:
        report.append(f"  泊松比: {summary.poisson_ratio}")
    report.append("")

    report.append("【边界条件】")
    report.append(f"  位移边界: {summary.num_dirichlet_bc} 个")
    report.append(f"  力边界: {summary.num_neumann_bc} 个")
    report.append("")

    report.append("【分析步】")
    report.append(f"  分析步数: {summary.num_steps}")
    report.append("")

    report.append("【资源估计】")
    report.append(f"  自由度: {summary.estimated_dof:,}")
    report.append(f"  内存需求: {summary.estimated_memory}")
    report.append("")

    if validation["errors"]:
        report.append("【错误】")
        for error in validation["errors"]:
            report.append(f"  ❌ {error}")
        report.append("")

    if validation["warnings"]:
        report.append("【警告】")
        for warning in validation["warnings"]:
            report.append(f"  ⚠️  {warning}")

    report.append("=" * 70)

    return "\n".join(report)
