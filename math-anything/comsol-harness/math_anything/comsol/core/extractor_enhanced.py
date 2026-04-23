"""Enhanced COMSOL extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体多物理场参数
2. 生成用户友好的详细摘要
3. 添加MPH文件解析
4. 模拟质量验证
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .extractor import ComsolExtractor


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class MultiphysicsSummary:
    """多物理场模拟摘要"""
    model_name: str = ""
    physics_interfaces: List[str] = field(default_factory=list)
    study_types: List[str] = field(default_factory=list)
    
    num_domains: int = 0
    num_boundaries: int = 0
    num_edges: int = 0
    num_points: int = 0
    
    mesh_type: str = ""
    num_elements: int = 0
    num_nodes: int = 0
    
    materials: List[str] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    
    solver_type: str = ""
    is_nonlinear: bool = False


class EnhancedComsolExtractor(ComsolExtractor):
    """增强版COMSOL提取器"""
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = MultiphysicsSummary()
        self.model_params: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        if "mph" in files:
            self._parse_mph_file(files["mph"])
        
        self.detailed_params = self._extract_detailed_parameters()
        self.simulation_summary = self._generate_simulation_summary()
        self._validate_simulation()
        
        return {
            "detailed_params": self.detailed_params,
            "summary": self.simulation_summary,
            "model_params": self.model_params,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _parse_mph_file(self, mph_path: str):
        """解析MPH文件（简化版）"""
        try:
            import zipfile
            with zipfile.ZipFile(mph_path, 'r') as z:
                for name in z.namelist():
                    if 'model.xml' in name:
                        content = z.read(name).decode('utf-8', errors='ignore')
                        self._parse_model_xml(content)
        except Exception:
            self.validation_warnings.append("无法解析MPH文件结构")
    
    def _parse_model_xml(self, content: str):
        """解析模型XML内容"""
        if '<physics' in content:
            import re
            physics = re.findall(r'name="([^"]+)"', content.split('<physics')[1].split('</physics')[0] if '</physics' in content else content)
            self.simulation_summary.physics_interfaces = physics[:10]
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        for i, physics in enumerate(self.simulation_summary.physics_interfaces[:5]):
            params.append(DetailedParameter(
                name=f"physics_{i+1}",
                value=physics,
                unit="",
                description="物理场接口",
                source="MPH"
            ))
        
        if self.simulation_summary.num_elements > 0:
            params.append(DetailedParameter(
                name="num_elements",
                value=self.simulation_summary.num_elements,
                unit="",
                description="网格单元数",
                source="MPH"
            ))
        
        return params
    
    def _generate_simulation_summary(self) -> MultiphysicsSummary:
        """生成模拟摘要"""
        return self.simulation_summary
    
    def _validate_simulation(self):
        """验证模拟参数"""
        if not self.simulation_summary.physics_interfaces:
            self.validation_warnings.append("未检测到物理场接口")
        
        if self.simulation_summary.num_elements == 0:
            self.validation_warnings.append("未检测到网格信息")


def generate_comsol_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成COMSOL摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("COMSOL 多物理场模拟摘要")
    report.append("=" * 70)
    report.append("")
    
    report.append("【物理场】")
    for physics in summary.physics_interfaces[:5]:
        report.append(f"  • {physics}")
    report.append("")
    
    report.append("【研究类型】")
    for study in summary.study_types[:5]:
        report.append(f"  • {study}")
    report.append("")
    
    report.append("【几何信息】")
    report.append(f"  域: {summary.num_domains}")
    report.append(f"  边界: {summary.num_boundaries}")
    report.append(f"  单元数: {summary.num_elements:,}")
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
