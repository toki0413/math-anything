"""Enhanced ANSYS extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体FEA参数（网格、材料、边界条件等）
2. 生成用户友好的详细摘要
3. 添加APDL输入文件解析
4. 模拟质量验证
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from .extractor import AnsysExtractor
except ImportError:
    from extractor import AnsysExtractor


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class FEASimulationSummary:
    """FEA模拟摘要"""
    analysis_type: str = ""
    num_nodes: int = 0
    num_elements: int = 0
    element_types: List[str] = field(default_factory=list)
    
    num_materials: int = 0
    elastic_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    density: Optional[float] = None
    
    num_loads: int = 0
    num_constraints: int = 0
    
    num_steps: int = 1
    nlgeom: bool = False
    large_deflection: bool = False
    
    contact_types: List[str] = field(default_factory=list)
    num_contacts: int = 0


class EnhancedAnsysExtractor(AnsysExtractor):
    """增强版ANSYS提取器"""
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = FEASimulationSummary()
        self.apdl_commands: List[str] = []
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        if "apdl" in files:
            self.apdl_commands = self._parse_apdl_file(files["apdl"])
        
        self.detailed_params = self._extract_detailed_parameters()
        self.simulation_summary = self._generate_simulation_summary()
        self._validate_simulation()
        
        return {
            "detailed_params": self.detailed_params,
            "summary": self.simulation_summary,
            "apdl_commands": self.apdl_commands,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _parse_apdl_file(self, apdl_path: str) -> List[str]:
        """解析APDL输入文件"""
        commands = []
        with open(apdl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                commands.append(line)
        return commands
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        for cmd in self.apdl_commands:
            cmd_upper = cmd.upper()
            
            if cmd_upper.startswith('ET,'):
                parts = cmd.split(',')
                if len(parts) >= 2:
                    params.append(DetailedParameter(
                        name="element_type",
                        value=parts[1].strip(),
                        unit="",
                        description="单元类型",
                        source="APDL"
                    ))
            
            elif cmd_upper.startswith('MP,EX,'):
                parts = cmd.split(',')
                if len(parts) >= 3:
                    try:
                        params.append(DetailedParameter(
                            name="elastic_modulus",
                            value=float(parts[2]),
                            unit="MPa",
                            description="弹性模量",
                            source="APDL"
                        ))
                    except ValueError:
                        pass
            
            elif cmd_upper.startswith('MP,NUXY,'):
                parts = cmd.split(',')
                if len(parts) >= 3:
                    try:
                        params.append(DetailedParameter(
                            name="poisson_ratio",
                            value=float(parts[2]),
                            unit="",
                            description="泊松比",
                            source="APDL"
                        ))
                    except ValueError:
                        pass
            
            elif cmd_upper.startswith('MP,DENS,'):
                parts = cmd.split(',')
                if len(parts) >= 3:
                    try:
                        params.append(DetailedParameter(
                            name="density",
                            value=float(parts[2]),
                            unit="kg/m³",
                            description="密度",
                            source="APDL"
                        ))
                    except ValueError:
                        pass
            
            elif cmd_upper.startswith('ANTYPE,'):
                parts = cmd.split(',')
                if len(parts) >= 2:
                    analysis_map = {'0': '静力学', '1': '屈曲', '2': '模态', '3': '谐响应', '4': '瞬态'}
                    atype = parts[1].strip()
                    params.append(DetailedParameter(
                        name="analysis_type",
                        value=analysis_map.get(atype, f"类型{atype}"),
                        unit="",
                        description="分析类型",
                        source="APDL"
                    ))
        
        return params
    
    def _generate_simulation_summary(self) -> FEASimulationSummary:
        """生成模拟摘要"""
        summary = self.simulation_summary
        
        for cmd in self.apdl_commands:
            cmd_upper = cmd.upper()
            
            if cmd_upper.startswith('ANTYPE,'):
                parts = cmd.split(',')
                if len(parts) >= 2:
                    summary.analysis_type = parts[1].strip()
            
            elif cmd_upper.startswith('NLGEOM,'):
                if 'ON' in cmd_upper:
                    summary.nlgeom = True
            
            elif cmd_upper.startswith('MP,EX,'):
                parts = cmd.split(',')
                if len(parts) >= 3:
                    try:
                        summary.elastic_modulus = float(parts[2])
                    except ValueError:
                        pass
            
            elif cmd_upper.startswith('MP,NUXY,'):
                parts = cmd.split(',')
                if len(parts) >= 3:
                    try:
                        summary.poisson_ratio = float(parts[2])
                    except ValueError:
                        pass
        
        return summary
    
    def _validate_simulation(self):
        """验证模拟参数"""
        if self.simulation_summary.analysis_type == "":
            self.validation_warnings.append("未指定分析类型")
        
        if self.simulation_summary.elastic_modulus is None:
            self.validation_errors.append("未定义弹性模量")
        
        if self.simulation_summary.poisson_ratio is None:
            self.validation_warnings.append("未定义泊松比")
        
        if self.simulation_summary.elastic_modulus and self.simulation_summary.elastic_modulus <= 0:
            self.validation_errors.append("弹性模量必须为正")


def generate_ansys_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成ANSYS摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("ANSYS FEA 模拟摘要")
    report.append("=" * 70)
    report.append("")
    
    report.append("【分析类型】")
    report.append(f"  类型: {summary.analysis_type}")
    report.append(f"  几何非线性: {'是' if summary.nlgeom else '否'}")
    report.append("")
    
    report.append("【材料属性】")
    if summary.elastic_modulus:
        report.append(f"  弹性模量: {summary.elastic_modulus} MPa")
    if summary.poisson_ratio:
        report.append(f"  泊松比: {summary.poisson_ratio}")
    if summary.density:
        report.append(f"  密度: {summary.density} kg/m³")
    report.append("")
    
    report.append("【网格信息】")
    report.append(f"  节点数: {summary.num_nodes:,}")
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
