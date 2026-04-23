"""Enhanced VASP extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体计算参数（ENCUT、KPOINTS、EDIFF等）
2. 生成用户友好的详细摘要
3. 保留原始文件的完整关键信息
4. 添加输入文件错误检查功能
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .extractor import VaspExtractor
from .parser import CrystalStructure, ElectronicParameters


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class DFTSimulationSummary:
    """DFT模拟摘要"""
    # 体系信息
    system_name: str = ""
    formula: str = ""
    num_atoms: int = 0
    num_species: int = 0
    crystal_system: str = ""
    space_group: str = ""
    
    # 计算参数
    encut: Optional[float] = None
    kpoints_mesh: Optional[tuple] = None
    ismear: Optional[int] = None
    sigma: Optional[float] = None
    ediff: Optional[float] = None
    ediffg: Optional[float] = None
    nsw: Optional[int] = None
    ibrion: Optional[int] = None
    
    # 电子结构
    nbands: Optional[int] = None
    ispin: Optional[int] = None
    magmom: Optional[float] = None
    
    # 交换关联
    xc_functional: str = ""
    pp_type: str = ""  # 赝势类型
    
    # 计算类型
    calculation_type: str = ""  # SCF/结构优化/MD等
    
    # 资源估计
    estimated_memory: str = ""
    estimated_time: str = ""


class EnhancedVaspExtractor(VaspExtractor):
    """增强版VASP提取器"""
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = DFTSimulationSummary()
        self.incar_params: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数"""
        # 1. 调用父类的标准提取
        schema = self.extract(files)
        
        # 2. 解析INCAR参数
        if "incar" in files:
            self.incar_params = self._parse_incar_detailed(files["incar"])
        
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
            "incar_params": self.incar_params,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _parse_incar_detailed(self, incar_path: str) -> Dict[str, Any]:
        """详细解析INCAR文件"""
        params = {}
        with open(incar_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('!'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    # 尝试转换数值
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
        return params
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        # 从INCAR提取
        param_info = {
            "ENCUT": ("截断能", "eV"),
            "EDIFF": ("电子收敛精度", "eV"),
            "EDIFFG": ("离子收敛精度", "eV/Å"),
            "SIGMA": ("展宽参数", "eV"),
            "POTIM": ("时间步长", "fs"),
            "NSW": ("离子步数", ""),
            "IBRION": ("离子优化算法", ""),
            "ISIF": ("应力张量计算", ""),
            "ISPIN": ("自旋极化", ""),
            "NBANDS": ("能带数", ""),
            "KPAR": ("K点并行", ""),
        }
        
        for key, (desc, unit) in param_info.items():
            if key in self.incar_params:
                params.append(DetailedParameter(
                    name=key.lower(),
                    value=self.incar_params[key],
                    unit=unit,
                    description=desc,
                    source="INCAR"
                ))
        
        # ISMEAR特殊处理
        if "ISMEAR" in self.incar_params:
            ismear = self.incar_params["ISMEAR"]
            ismear_desc = {
                -5: "四面体方法",
                0: "高斯展宽",
                1: "Methfessel-Paxton一阶",
                2: "Methfessel-Paxton二阶",
            }
            params.append(DetailedParameter(
                name="ismear",
                value=ismear,
                unit="",
                description=ismear_desc.get(ismear, f"ISMEAR={ismear}"),
                source="INCAR"
            ))
        
        # 从结构提取
        if self.structure:
            params.append(DetailedParameter(
                name="num_atoms",
                value=self.structure.num_atoms,
                unit="",
                description="原子数",
                source="POSCAR"
            ))
            if hasattr(self.structure, 'lattice'):
                params.append(DetailedParameter(
                    name="lattice_type",
                    value=self.structure.lattice_type,
                    unit="",
                    description="晶格类型",
                    source="POSCAR"
                ))
        
        return params
    
    def _generate_simulation_summary(self) -> DFTSimulationSummary:
        """生成模拟摘要"""
        summary = DFTSimulationSummary()
        
        # 从INCAR提取
        summary.encut = self.incar_params.get("ENCUT")
        summary.ediff = self.incar_params.get("EDIFF")
        summary.ediffg = self.incar_params.get("EDIFFG")
        summary.sigma = self.incar_params.get("SIGMA")
        summary.ismear = self.incar_params.get("ISMEAR")
        summary.nsw = self.incar_params.get("NSW")
        summary.ibrion = self.incar_params.get("IBRION")
        summary.nbands = self.incar_params.get("NBANDS")
        summary.ispin = self.incar_params.get("ISPIN")
        
        # 计算类型
        nsw = self.incar_params.get("NSW", 0)
        ibrion = self.incar_params.get("IBRION", 0)
        if nsw == 0:
            summary.calculation_type = "单点能计算 (SCF)"
        elif ibrion in (1, 2, 3):
            summary.calculation_type = "结构优化"
        elif ibrion == 0:
            summary.calculation_type = "分子动力学 (MD)"
        else:
            summary.calculation_type = f"IBRION={ibrion}"
        
        # 交换关联泛函
        gga = self.incar_params.get("GGA", "PE")
        xc_map = {"PE": "PBE", "91": "GGA91", "PS": "PBEsol"}
        summary.xc_functional = xc_map.get(gga, gga)
        
        # 从结构提取
        if self.structure:
            summary.num_atoms = self.structure.num_atoms
            summary.formula = getattr(self.structure, 'formula', '')
            summary.crystal_system = getattr(self.structure, 'crystal_system', '')
        
        # 资源估计
        if summary.num_atoms > 0 and summary.nbands:
            # 粗略估计
            mem_gb = summary.num_atoms * 0.1 + summary.nbands * 0.01
            summary.estimated_memory = f"约 {mem_gb:.1f} GB"
        
        return summary
    
    def _validate_input(self):
        """验证输入文件"""
        # 检查必要参数
        if "ENCUT" not in self.incar_params:
            self.validation_warnings.append("未设置ENCUT，将使用默认值（可能不够精确）")
        
        # 检查ENCUT是否足够
        encut = self.incar_params.get("ENCUT", 0)
        if encut > 0 and encut < 400:
            self.validation_warnings.append(f"ENCUT={encut} eV 可能偏低，建议 >= 400 eV")
        
        # 检查收敛参数
        ediff = self.incar_params.get("EDIFF", 1e-4)
        try:
            ediff_val = float(ediff) if isinstance(ediff, str) else ediff
            if ediff_val > 1e-4:
                self.validation_warnings.append(f"EDIFF={ediff_val} 可能不够精确，建议 <= 1e-5")
        except (ValueError, TypeError):
            pass
        
        # 检查KPOINTS
        if not self.incar_params.get("KPOINTS") and not hasattr(self, 'kpoints'):
            self.validation_warnings.append("未找到KPOINTS设置")
        
        # 检查结构优化参数
        nsw = self.incar_params.get("NSW", 0)
        ibrion = self.incar_params.get("IBRION")
        if nsw > 0 and ibrion is None:
            self.validation_errors.append("NSW>0 但未设置IBRION")
        
        # 检查自旋
        ispin = self.incar_params.get("ISPIN", 1)
        magmom = self.incar_params.get("MAGMOM")
        if ispin == 2 and not magmom:
            self.validation_warnings.append("ISPIN=2 但未设置MAGMOM，可能需要初始磁矩")


def generate_vasp_summary_report(enhanced_result: Dict[str, Any]) -> str:
    """生成VASP摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("VASP DFT 计算输入文件摘要")
    report.append("=" * 70)
    report.append("")
    
    report.append("【体系信息】")
    report.append(f"  化学式: {summary.formula or '未指定'}")
    report.append(f"  原子数: {summary.num_atoms}")
    report.append(f"  晶系: {summary.crystal_system or '未指定'}")
    report.append("")
    
    report.append("【计算类型】")
    report.append(f"  类型: {summary.calculation_type}")
    report.append(f"  交换关联: {summary.xc_functional}")
    report.append("")
    
    report.append("【计算参数】")
    if summary.encut:
        report.append(f"  截断能: {summary.encut} eV")
    if summary.ediff:
        report.append(f"  电子收敛: {summary.ediff} eV")
    if summary.nsw:
        report.append(f"  离子步数: {summary.nsw}")
    if summary.nbands:
        report.append(f"  能带数: {summary.nbands}")
    report.append("")
    
    report.append("【资源估计】")
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
