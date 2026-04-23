"""Enhanced Multiwfn extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体量子化学参数
2. 生成用户友好的详细摘要
3. 添加输入文件解析
4. 计算质量验证
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .extractor import MultiwfnExtractor


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class QCSimulationSummary:
    """量子化学计算摘要"""
    system_name: str = ""
    formula: str = ""
    num_atoms: int = 0
    num_electrons: int = 0
    charge: int = 0
    multiplicity: int = 1
    
    calculation_type: str = ""
    method: str = ""
    basis_set: str = ""
    
    homo_energy: float = 0.0
    lumo_energy: float = 0.0
    gap: float = 0.0
    
    total_energy: float = 0.0
    
    num_orbitals: int = 0
    num_basis_functions: int = 0


class EnhancedMultiwfnExtractor(MultiwfnExtractor):
    """增强版Multiwfn提取器"""
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = QCSimulationSummary()
        self.input_params: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        if "fch" in files:
            self._parse_fchk_file(files["fch"])
        elif "wfn" in files:
            self._parse_wfn_file(files["wfn"])
        
        self.detailed_params = self._extract_detailed_parameters()
        self.simulation_summary = self._generate_simulation_summary()
        self._validate_simulation()
        
        return {
            "detailed_params": self.detailed_params,
            "summary": self.simulation_summary,
            "input_params": self.input_params,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _parse_fchk_file(self, fchk_path: str):
        """解析FCHK文件"""
        with open(fchk_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "Number of atoms" in line:
                self.simulation_summary.num_atoms = int(lines[i+1].strip())
            elif "Charge" in line and "Multiplicity" not in line:
                try:
                    self.simulation_summary.charge = int(lines[i+1].strip())
                except ValueError:
                    pass
            elif "Multiplicity" in line:
                try:
                    self.simulation_summary.multiplicity = int(lines[i+1].strip())
                except ValueError:
                    pass
            elif "Number of electrons" in line:
                self.simulation_summary.num_electrons = int(lines[i+1].strip())
            elif "Number of basis functions" in line:
                self.simulation_summary.num_basis_functions = int(lines[i+1].strip())
            elif "Total Energy" in line:
                try:
                    self.simulation_summary.total_energy = float(lines[i+1].strip())
                except ValueError:
                    pass
    
    def _parse_wfn_file(self, wfn_path: str):
        """解析WFN文件"""
        with open(wfn_path, 'r') as f:
            content = f.read()
        
        import re
        atoms = re.findall(r'\s+(\d+)\s+\d+\.\d+', content)
        self.simulation_summary.num_atoms = len(atoms)
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        if self.simulation_summary.num_atoms > 0:
            params.append(DetailedParameter(
                name="num_atoms",
                value=self.simulation_summary.num_atoms,
                unit="",
                description="原子数",
                source="FCHK"
            ))
        
        if self.simulation_summary.num_electrons > 0:
            params.append(DetailedParameter(
                name="num_electrons",
                value=self.simulation_summary.num_electrons,
                unit="",
                description="电子数",
                source="FCHK"
            ))
        
        if self.simulation_summary.charge != 0:
            params.append(DetailedParameter(
                name="charge",
                value=self.simulation_summary.charge,
                unit="e",
                description="电荷",
                source="FCHK"
            ))
        
        if self.simulation_summary.multiplicity > 1:
            params.append(DetailedParameter(
                name="multiplicity",
                value=self.simulation_summary.multiplicity,
                unit="",
                description="自旋多重度",
                source="FCHK"
            ))
        
        if self.simulation_summary.total_energy != 0:
            params.append(DetailedParameter(
                name="total_energy",
                value=self.simulation_summary.total_energy,
                unit="Hartree",
                description="总能量",
                source="FCHK"
            ))
        
        if self.simulation_summary.num_basis_functions > 0:
            params.append(DetailedParameter(
                name="num_basis_functions",
                value=self.simulation_summary.num_basis_functions,
                unit="",
                description="基函数数",
                source="FCHK"
            ))
        
        return params
    
    def _generate_simulation_summary(self) -> QCSimulationSummary:
        """生成模拟摘要"""
        return self.simulation_summary
    
    def _validate_simulation(self):
        """验证计算参数"""
        if self.simulation_summary.num_atoms == 0:
            self.validation_errors.append("未检测到原子数")
        
        if self.simulation_summary.num_electrons == 0:
            self.validation_warnings.append("未检测到电子数")
        
        if self.simulation_summary.total_energy == 0:
            self.validation_warnings.append("未检测到总能量")


def generate_multiwfn_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成Multiwfn摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("Multiwfn 量子化学分析摘要")
    report.append("=" * 70)
    report.append("")
    
    report.append("【体系信息】")
    report.append(f"  原子数: {summary.num_atoms}")
    report.append(f"  电子数: {summary.num_electrons}")
    report.append(f"  电荷: {summary.charge}")
    report.append(f"  自旋多重度: {summary.multiplicity}")
    report.append("")
    
    report.append("【计算信息】")
    report.append(f"  基函数数: {summary.num_basis_functions}")
    if summary.total_energy != 0:
        report.append(f"  总能量: {summary.total_energy:.6f} Hartree")
    report.append("")
    
    if summary.homo_energy != 0 or summary.lumo_energy != 0:
        report.append("【轨道信息】")
        report.append(f"  HOMO能量: {summary.homo_energy:.4f} Hartree")
        report.append(f"  LUMO能量: {summary.lumo_energy:.4f} Hartree")
        report.append(f"  HOMO-LUMO Gap: {summary.gap:.4f} Hartree")
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
