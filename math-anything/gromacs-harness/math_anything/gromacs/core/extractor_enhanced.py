"""Enhanced GROMACS extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体MD参数（时间步长、温度、压强等）
2. 生成用户友好的详细摘要
3. 添加输入文件解析（MDP文件）
4. 模拟质量验证
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from .extractor import GromacsExtractor
except ImportError:
    from extractor import GromacsExtractor


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class MDSimulationSummary:
    """MD模拟摘要"""
    system_name: str = ""
    num_atoms: int = 0
    num_residues: int = 0
    num_molecules: int = 0
    box_type: str = ""
    box_dimensions: tuple = (0, 0, 0)
    
    integrator: str = ""
    timestep: float = 0.0
    nsteps: int = 0
    total_time: float = 0.0
    
    ensemble: str = ""
    temperature: float = 0.0
    pressure: float = 0.0
    
    force_field: str = ""
    water_model: str = ""
    
    constraints: str = ""
    cutoff_scheme: str = ""
    rcoulomb: float = 0.0
    rvdw: float = 0.0
    
    equilibrated: bool = False
    converged: bool = False


class EnhancedGromacsExtractor(GromacsExtractor):
    """增强版GROMACS提取器"""
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = MDSimulationSummary()
        self.mdp_params: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        if "mdp" in files:
            self.mdp_params = self._parse_mdp_file(files["mdp"])
        
        if "gro" in files:
            self._parse_gro_file(files["gro"])
        
        self.detailed_params = self._extract_detailed_parameters()
        self.simulation_summary = self._generate_simulation_summary()
        self._validate_simulation()
        
        return {
            "detailed_params": self.detailed_params,
            "summary": self.simulation_summary,
            "mdp_params": self.mdp_params,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _parse_mdp_file(self, mdp_path: str) -> Dict[str, Any]:
        """解析MDP参数文件"""
        params = {}
        with open(mdp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';') or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split(';')[0].strip()
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
        return params
    
    def _parse_gro_file(self, gro_path: str):
        """解析GRO结构文件"""
        with open(gro_path, 'r') as f:
            lines = f.readlines()
        
        self.simulation_summary.system_name = lines[0].strip()
        num_atoms = int(lines[1].strip())
        self.simulation_summary.num_atoms = num_atoms
        
        if len(lines) > num_atoms + 2:
            box_line = lines[num_atoms + 2].strip()
            box_parts = box_line.split()
            if len(box_parts) >= 3:
                self.simulation_summary.box_dimensions = tuple(float(x) for x in box_parts[:3])
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        param_info = {
            "dt": ("时间步长", "ps"),
            "nsteps": ("总步数", ""),
            "nstxout": ("坐标输出频率", "步"),
            "nstenergy": ("能量输出频率", "步"),
            "nstlog": ("日志输出频率", "步"),
            "init_temp": ("初始温度", "K"),
            "ref_t": ("参考温度", "K"),
            "ref_p": ("参考压强", "bar"),
            "rcoulomb": ("库仑截断", "nm"),
            "rvdw": ("范德华截断", "nm"),
            "rlist": ("近邻列表截断", "nm"),
        }
        
        for key, (desc, unit) in param_info.items():
            if key in self.mdp_params:
                params.append(DetailedParameter(
                    name=key,
                    value=self.mdp_params[key],
                    unit=unit,
                    description=desc,
                    source="MDP"
                ))
        
        if self.simulation_summary.num_atoms > 0:
            params.append(DetailedParameter(
                name="num_atoms",
                value=self.simulation_summary.num_atoms,
                unit="",
                description="原子数",
                source="GRO"
            ))
        
        return params
    
    def _generate_simulation_summary(self) -> MDSimulationSummary:
        """生成模拟摘要"""
        summary = self.simulation_summary
        
        if "integrator" in self.mdp_params:
            summary.integrator = str(self.mdp_params["integrator"])
        
        if "dt" in self.mdp_params:
            summary.timestep = float(self.mdp_params["dt"])
        
        if "nsteps" in self.mdp_params:
            summary.nsteps = int(self.mdp_params["nsteps"])
            if summary.timestep > 0:
                summary.total_time = summary.nsteps * summary.timestep
        
        if "tcoupl" in self.mdp_params:
            tcoupl = str(self.mdp_params["tcoupl"])
            if tcoupl != "no":
                summary.ensemble = "NVT" if "pcoupl" not in self.mdp_params or self.mdp_params.get("pcoupl") == "no" else "NPT"
            else:
                summary.ensemble = "NVE"
        
        if "ref_t" in self.mdp_params:
            summary.temperature = float(self.mdp_params["ref_t"])
        
        if "ref_p" in self.mdp_params:
            summary.pressure = float(self.mdp_params["ref_p"])
        
        return summary
    
    def _validate_simulation(self):
        """验证模拟参数"""
        if self.simulation_summary.timestep > 0.002:
            self.validation_warnings.append(f"时间步长 {self.simulation_summary.timestep} ps 可能过大，建议 <= 0.002 ps")
        
        if "rcoulomb" in self.mdp_params and "rvdw" in self.mdp_params:
            if self.mdp_params["rcoulomb"] != self.mdp_params["rvdw"]:
                self.validation_warnings.append("库仑截断和范德华截断不一致")
        
        if self.simulation_summary.num_atoms == 0:
            self.validation_errors.append("未找到原子数信息")
        
        if self.simulation_summary.nsteps == 0:
            self.validation_warnings.append("总步数为0")


def generate_gromacs_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成GROMACS摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("GROMACS MD 模拟摘要")
    report.append("=" * 70)
    report.append("")
    
    report.append("【体系信息】")
    report.append(f"  系统名称: {summary.system_name}")
    report.append(f"  原子数: {summary.num_atoms}")
    if summary.box_dimensions[0] > 0:
        report.append(f"  盒子尺寸: {summary.box_dimensions[0]:.2f} x {summary.box_dimensions[1]:.2f} x {summary.box_dimensions[2]:.2f} nm")
    report.append("")
    
    report.append("【模拟参数】")
    report.append(f"  积分器: {summary.integrator}")
    report.append(f"  时间步长: {summary.timestep} ps")
    report.append(f"  总步数: {summary.nsteps:,}")
    report.append(f"  总时间: {summary.total_time:.2f} ps")
    report.append("")
    
    report.append("【热力学条件】")
    report.append(f"  系综: {summary.ensemble}")
    report.append(f"  温度: {summary.temperature} K")
    if summary.pressure > 0:
        report.append(f"  压强: {summary.pressure} bar")
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
