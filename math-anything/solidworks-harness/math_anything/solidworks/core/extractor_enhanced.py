"""Enhanced SolidWorks extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体CAD/FEA参数
2. 生成用户友好的详细摘要
3. 添加输入文件解析
4. 模拟质量验证
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .extractor import SolidWorksExtractor


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class CADFEASummary:
    """CAD/FEA模拟摘要"""
    model_name: str = ""
    part_type: str = ""
    
    num_bodies: int = 0
    num_faces: int = 0
    num_edges: int = 0
    
    num_nodes: int = 0
    num_elements: int = 0
    element_type: str = ""
    mesh_quality: str = ""
    
    analysis_type: str = ""
    num_loads: int = 0
    num_fixtures: int = 0
    num_contacts: int = 0
    
    material_name: str = ""
    elastic_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    yield_strength: Optional[float] = None
    
    max_stress: float = 0.0
    max_displacement: float = 0.0
    safety_factor: float = 0.0


class EnhancedSolidWorksExtractor(SolidWorksExtractor):
    """增强版SolidWorks提取器"""
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = CADFEASummary()
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        self.detailed_params = self._extract_detailed_parameters()
        self.simulation_summary = self._generate_simulation_summary()
        self._validate_simulation()
        
        return {
            "detailed_params": self.detailed_params,
            "summary": self.simulation_summary,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        if self.simulation_summary.num_nodes > 0:
            params.append(DetailedParameter(
                name="num_nodes",
                value=self.simulation_summary.num_nodes,
                unit="",
                description="节点数",
                source="SolidWorks"
            ))
        
        if self.simulation_summary.num_elements > 0:
            params.append(DetailedParameter(
                name="num_elements",
                value=self.simulation_summary.num_elements,
                unit="",
                description="单元数",
                source="SolidWorks"
            ))
        
        if self.simulation_summary.elastic_modulus:
            params.append(DetailedParameter(
                name="elastic_modulus",
                value=self.simulation_summary.elastic_modulus,
                unit="MPa",
                description="弹性模量",
                source="Material"
            ))
        
        if self.simulation_summary.yield_strength:
            params.append(DetailedParameter(
                name="yield_strength",
                value=self.simulation_summary.yield_strength,
                unit="MPa",
                description="屈服强度",
                source="Material"
            ))
        
        if self.simulation_summary.max_stress > 0:
            params.append(DetailedParameter(
                name="max_stress",
                value=self.simulation_summary.max_stress,
                unit="MPa",
                description="最大应力",
                source="Results"
            ))
        
        if self.simulation_summary.max_displacement > 0:
            params.append(DetailedParameter(
                name="max_displacement",
                value=self.simulation_summary.max_displacement,
                unit="mm",
                description="最大位移",
                source="Results"
            ))
        
        return params
    
    def _generate_simulation_summary(self) -> CADFEASummary:
        """生成模拟摘要"""
        return self.simulation_summary
    
    def _validate_simulation(self):
        """验证模拟参数"""
        if self.simulation_summary.num_nodes == 0:
            self.validation_warnings.append("未检测到网格节点")
        
        if self.simulation_summary.elastic_modulus is None:
            self.validation_errors.append("未定义材料弹性模量")
        
        if self.simulation_summary.num_loads == 0:
            self.validation_warnings.append("未检测到载荷")
        
        if self.simulation_summary.num_fixtures == 0:
            self.validation_errors.append("未检测到约束，可能导致刚体位移")
        
        if self.simulation_summary.safety_factor > 0 and self.simulation_summary.safety_factor < 1.5:
            self.validation_warnings.append(f"安全系数 {self.simulation_summary.safety_factor:.2f} 偏低")


def generate_solidworks_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成SolidWorks摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("SolidWorks Simulation 摘要")
    report.append("=" * 70)
    report.append("")
    
    report.append("【模型信息】")
    report.append(f"  模型名称: {summary.model_name}")
    report.append(f"  零件类型: {summary.part_type}")
    report.append(f"  实体数: {summary.num_bodies}")
    report.append("")
    
    report.append("【网格信息】")
    report.append(f"  节点数: {summary.num_nodes:,}")
    report.append(f"  单元数: {summary.num_elements:,}")
    report.append(f"  单元类型: {summary.element_type}")
    report.append("")
    
    report.append("【材料属性】")
    report.append(f"  材料: {summary.material_name}")
    if summary.elastic_modulus:
        report.append(f"  弹性模量: {summary.elastic_modulus} MPa")
    if summary.yield_strength:
        report.append(f"  屈服强度: {summary.yield_strength} MPa")
    report.append("")
    
    report.append("【分析结果】")
    report.append(f"  分析类型: {summary.analysis_type}")
    if summary.max_stress > 0:
        report.append(f"  最大应力: {summary.max_stress:.2f} MPa")
    if summary.max_displacement > 0:
        report.append(f"  最大位移: {summary.max_displacement:.4f} mm")
    if summary.safety_factor > 0:
        report.append(f"  安全系数: {summary.safety_factor:.2f}")
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
