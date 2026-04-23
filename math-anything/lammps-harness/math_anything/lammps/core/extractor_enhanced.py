"""Enhanced LAMMPS extractor with detailed parameter extraction and summary generation.

改进点：
1. 提取具体参数值（温度、时间步长、截断距离等）
2. 生成用户友好的详细摘要
3. 保留原始文件的完整关键信息
4. 添加输入文件错误检查功能
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .extractor import LammpsExtractor
from .parser import LammpsInputParser, FixCommand


@dataclass
class DetailedParameter:
    """详细参数定义"""
    name: str
    value: Any
    unit: str = ""
    description: str = ""
    source: str = ""  # 从哪个命令提取的
    line_number: int = 0  # 在输入文件的行号


@dataclass
class SimulationSummary:
    """模拟摘要"""
    # 基本信息
    system_type: str = ""  # 分子动力学/能量最小化等
    total_atoms: int = 0
    atom_types: int = 0
    
    # 力场信息
    force_field: str = ""
    force_field_params: Dict[str, Any] = field(default_factory=dict)
    
    # 模拟参数
    timestep: float = 0.0
    total_steps: int = 0
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    
    # 边界条件
    boundary_conditions: List[str] = field(default_factory=list)
    box_dimensions: Optional[Tuple[float, float, float]] = None
    
    # 系综
    ensemble: str = ""  # NVE/NVT/NPT等
    thermostat: Optional[str] = None
    barostat: Optional[str] = None
    
    # 输出设置
    output_frequency: int = 0
    output_quantities: List[str] = field(default_factory=list)
    
    # 计算资源估计
    estimated_runtime: str = ""
    memory_requirement: str = ""


class EnhancedLammpsExtractor(LammpsExtractor):
    """增强版LAMMPS提取器
    
    在原有功能基础上，添加：
    1. 详细参数提取
    2. 用户友好摘要
    3. 输入文件验证
    """
    
    def __init__(self):
        super().__init__()
        self.detailed_params: List[DetailedParameter] = []
        self.simulation_summary = SimulationSummary()
        self.raw_commands: List[Dict[str, Any]] = []  # 保留原始命令
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def extract_enhanced(self, files: Dict[str, str]) -> Dict[str, Any]:
        """增强版提取函数
        
        Returns:
            包含以下内容的字典：
            - schema: 原始Math Schema
            - detailed_params: 详细参数列表
            - summary: 用户友好的模拟摘要
            - raw_commands: 原始命令保留
            - validation: 输入文件验证结果
        """
        input_path = files.get("input")
        if not input_path:
            raise ValueError("Input file required")
        
        # 1. 调用父类的标准提取
        schema = self.extract(files)
        
        # 2. 解析原始命令（保留完整信息）
        self.raw_commands = self._parse_raw_commands(input_path)
        
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
            "raw_commands": self.raw_commands,
            "validation": {
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "is_valid": len(self.validation_errors) == 0
            }
        }
    
    def _parse_raw_commands(self, input_path: str) -> List[Dict[str, Any]]:
        """解析并保留原始命令"""
        commands = []
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        line_num = 0
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            commands.append({
                "line_number": i,
                "command": line,
                "category": self._categorize_command(line)
            })
        
        return commands
    
    def _categorize_command(self, line: str) -> str:
        """分类命令类型"""
        keywords = {
            "units": "units",
            "dimension": "dimension",
            "boundary": "boundary",
            "atom_style": "atom_style",
            "read_data": "read_data",
            "pair_style": "pair_style",
            "pair_coeff": "pair_coeff",
            "bond_style": "bond_style",
            "angle_style": "angle_style",
            "timestep": "timestep",
            "fix": "fix",
            "compute": "compute",
            "thermo": "thermo",
            "thermo_style": "thermo_style",
            "dump": "dump",
            "run": "run",
            "minimize": "minimize",
            "velocity": "velocity",
            "group": "group",
            "region": "region",
        }
        
        first_word = line.split()[0].lower() if line.split() else ""
        return keywords.get(first_word, "other")
    
    def _extract_detailed_parameters(self) -> List[DetailedParameter]:
        """提取详细参数"""
        params = []
        
        # 1. 时间步长
        if self.settings.timestep:
            params.append(DetailedParameter(
                name="timestep",
                value=self.settings.timestep,
                unit="fs" if self.settings.units == "metal" else "time units",
                description="积分时间步长",
                source="timestep command"
            ))
        
        # 2. 力场参数
        if self.settings.pair_style:
            style = self.settings.pair_style.style
            args = self.settings.pair_style.args
            
            if style in ("lj/cut", "lj/cut/coul/long") and len(args) >= 1:
                params.append(DetailedParameter(
                    name="pair_cutoff",
                    value=float(args[0]),
                    unit="Å" if self.settings.units == "metal" else "distance units",
                    description="非键相互作用截断距离",
                    source="pair_style"
                ))
        
        # 3. 从pair_coeff提取具体参数
        for i, pc in enumerate(getattr(self.settings, 'pair_coeffs', [])):
            atom_type = pc.get('atom_type', i+1)
            if 'epsilon' in pc:
                params.append(DetailedParameter(
                    name=f"epsilon_type{atom_type}",
                    value=pc['epsilon'],
                    unit="eV" if self.settings.units == "metal" else "energy units",
                    description=f"原子类型 {atom_type} 的LJ能量参数",
                    source="pair_coeff"
                ))
            if 'sigma' in pc:
                params.append(DetailedParameter(
                    name=f"sigma_type{atom_type}",
                    value=pc['sigma'],
                    unit="Å" if self.settings.units == "metal" else "distance units",
                    description=f"原子类型 {atom_type} 的LJ长度参数",
                    source="pair_coeff"
                ))
        
        # 4. 系综参数（温度、压力）
        for fix in self.settings.fixes:
            if fix.fix_style == "nvt" and len(fix.args) >= 3:
                try:
                    t_start = float(fix.args[0])
                    t_stop = float(fix.args[1])
                    t_damp = float(fix.args[2])
                    
                    params.append(DetailedParameter(
                        name="temperature_start",
                        value=t_start,
                        unit="K" if self.settings.units == "metal" else "temperature units",
                        description="NVT系综初始温度",
                        source=f"fix nvt (ID: {fix.fix_id})"
                    ))
                    params.append(DetailedParameter(
                        name="temperature_stop",
                        value=t_stop,
                        unit="K" if self.settings.units == "metal" else "temperature units",
                        description="NVT系综目标温度",
                        source=f"fix nvt (ID: {fix.fix_id})"
                    ))
                    params.append(DetailedParameter(
                        name="temperature_damping",
                        value=t_damp,
                        unit="time units",
                        description="温度阻尼系数",
                        source=f"fix nvt (ID: {fix.fix_id})"
                    ))
                except (ValueError, IndexError):
                    pass
            
            elif fix.fix_style == "npt" and len(fix.args) >= 6:
                try:
                    t_start = float(fix.args[0])
                    t_stop = float(fix.args[1])
                    p_start = float(fix.args[3])
                    p_stop = float(fix.args[4])
                    
                    params.append(DetailedParameter(
                        name="temperature_npt",
                        value=t_stop,
                        unit="K",
                        description="NPT系综目标温度",
                        source=f"fix npt (ID: {fix.fix_id})"
                    ))
                    params.append(DetailedParameter(
                        name="pressure_npt",
                        value=p_stop,
                        unit="bar" if self.settings.units == "metal" else "pressure units",
                        description="NPT系综目标压力",
                        source=f"fix npt (ID: {fix.fix_id})"
                    ))
                except (ValueError, IndexError):
                    pass
        
        # 5. 输出频率 (从原始命令提取)
        for cmd in self.raw_commands:
            if cmd["category"] == "thermo":
                try:
                    args = cmd["command"].split()
                    if len(args) >= 2:
                        thermo_interval = int(args[1])
                        params.append(DetailedParameter(
                            name="thermo_output_frequency",
                            value=thermo_interval,
                            unit="timesteps",
                            description="热力学量输出频率",
                            source=f"thermo command (line {cmd['line_number']})"
                        ))
                except (ValueError, IndexError):
                    pass
        
        return params
    
    def _generate_simulation_summary(self) -> SimulationSummary:
        """生成用户友好的模拟摘要"""
        summary = SimulationSummary()
        
        # 1. 系统类型
        has_integrator = any(f.fix_style in ("nve", "nvt", "npt") for f in self.settings.fixes)
        has_minimize = any("minimize" in str(cmd["command"]) for cmd in self.raw_commands)
        
        if has_minimize:
            summary.system_type = "能量最小化"
        elif has_integrator:
            summary.system_type = "分子动力学模拟"
        else:
            summary.system_type = "静态计算"
        
        # 2. 原子信息（尝试从read_data推断）
        for cmd in self.raw_commands:
            if cmd["category"] == "read_data":
                # 这里可以解析data文件获取原子数
                summary.total_atoms = -1  # 需要实际解析
                break
        
        # 3. 力场信息
        if self.settings.pair_style:
            summary.force_field = self.settings.pair_style.style
            summary.force_field_params = {
                "cutoff": self.settings.pair_style.args[0] if self.settings.pair_style.args else "default"
            }
        
        # 4. 时间步长和总步数
        summary.timestep = self.settings.timestep or 0.0
        
        # 从run命令提取总步数
        for cmd in self.raw_commands:
            if cmd["category"] == "run":
                try:
                    args = cmd["command"].split()
                    if len(args) >= 2:
                        summary.total_steps = int(args[1])
                except (ValueError, IndexError):
                    pass
        
        # 5. 温度和压力
        for fix in self.settings.fixes:
            if fix.fix_style == "nvt" and len(fix.args) >= 2:
                try:
                    summary.temperature = float(fix.args[1])  # 目标温度
                    summary.ensemble = "NVT"
                    summary.thermostat = "Nosé-Hoover"
                except (ValueError, IndexError):
                    pass
            elif fix.fix_style == "npt" and len(fix.args) >= 2:
                try:
                    summary.temperature = float(fix.args[1])
                    summary.pressure = float(fix.args[4]) if len(fix.args) >= 5 else None
                    summary.ensemble = "NPT"
                    summary.thermostat = "Nosé-Hoover"
                    summary.barostat = "Nosé-Hoover"
                except (ValueError, IndexError):
                    pass
            elif fix.fix_style == "nve":
                summary.ensemble = "NVE"
        
        # 6. 边界条件
        if self.settings.boundary_style:
            bc_map = {"p": "周期性", "f": "固定", "s": "收缩包裹"}
            summary.boundary_conditions = [
                bc_map.get(bc, bc) for bc in self.settings.boundary_style
            ]
        
        # 7. 输出设置 (从原始命令提取)
        for cmd in self.raw_commands:
            if cmd["category"] == "thermo":
                try:
                    args = cmd["command"].split()
                    if len(args) >= 2:
                        summary.output_frequency = int(args[1])
                except (ValueError, IndexError):
                    pass
        
        # 8. 估计运行时间和内存需求
        if summary.total_steps > 0 and summary.timestep > 0:
            # 粗略估计：假设每1000步需要1分钟（CPU）
            est_minutes = summary.total_steps / 1000
            if est_minutes < 60:
                summary.estimated_runtime = f"约{int(est_minutes)}分钟"
            else:
                summary.estimated_runtime = f"约{est_minutes/60:.1f}小时"
        
        # 内存需求估计
        if summary.total_atoms > 0:
            # 粗略估计：每个原子约1KB
            mem_mb = summary.total_atoms * 1 / 1024
            if mem_mb < 1024:
                summary.memory_requirement = f"约{int(mem_mb)}MB"
            else:
                summary.memory_requirement = f"约{mem_mb/1024:.1f}GB"
        else:
            summary.memory_requirement = "取决于体系大小"
        
        return summary
    
    def _validate_input(self):
        """验证输入文件的合理性和潜在错误"""
        # 1. 检查时间步长
        if self.settings.timestep:
            units = self.settings.units
            dt = self.settings.timestep
            
            # 根据单位制检查时间步长是否合理
            if units == "metal":  # 金属单位：时间单位是ps
                if dt > 0.01:  # 10 fs以上可能太大
                    self.validation_warnings.append(
                        f"时间步长 {dt} ps (={dt*1000} fs) 可能过大，"
                        f"建议不超过 0.002 ps (2 fs) 以保证稳定性"
                    )
            elif units == "real":  # 真实单位：时间单位是fs
                if dt > 2.0:
                    self.validation_warnings.append(
                        f"时间步长 {dt} fs 可能过大，建议不超过 2 fs"
                    )
        else:
            self.validation_errors.append("未设置时间步长 (timestep)")
        
        # 2. 检查力场设置
        if not self.settings.pair_style:
            self.validation_errors.append("未设置力场 (pair_style)")
        
        # 3. 检查是否有积分器（对于MD）
        has_integrator = any(f.fix_style in ("nve", "nvt", "npt") for f in self.settings.fixes)
        has_run = any(cmd["category"] == "run" for cmd in self.raw_commands)
        
        if has_run and not has_integrator:
            self.validation_errors.append(
                "使用 'run' 命令但未设置时间积分器 (fix nve/nvt/npt)"
            )
        
        # 4. 检查温度设置（对于NVT/NPT）
        for fix in self.settings.fixes:
            if fix.fix_style in ("nvt", "npt"):
                if len(fix.args) < 3:
                    self.validation_errors.append(
                        f"fix {fix.fix_style} 参数不足，需要：Tstart Tstop Tdamp"
                    )
                else:
                    try:
                        t_start = float(fix.args[0])
                        t_stop = float(fix.args[1])
                        if t_start <= 0 or t_stop <= 0:
                            self.validation_errors.append(
                                f"温度必须为正数 (fix {fix.fix_id})"
                            )
                        if abs(t_stop - t_start) / max(t_start, 0.001) > 10:
                            self.validation_warnings.append(
                                f"温度从 {t_start} 变化到 {t_stop}，变化幅度很大"
                            )
                    except ValueError:
                        self.validation_errors.append(
                            f"fix {fix.fix_style} 温度参数格式错误"
                        )
        
        # 5. 检查边界条件与系综的一致性
        if self.settings.boundary_style:
            is_periodic = all(bc == "p" for bc in self.settings.boundary_style)
            for fix in self.settings.fixes:
                if fix.fix_style == "npt" and not is_periodic:
                    self.validation_warnings.append(
                        "NPT系综通常需要周期性边界条件"
                    )
        
        # 6. 检查输出设置
        has_thermo = any(cmd["category"] == "thermo" for cmd in self.raw_commands)
        if not has_thermo:
            self.validation_warnings.append("未设置热力学量输出频率 (thermo)")


def generate_human_readable_summary(enhanced_result: Dict[str, Any]) -> str:
    """生成人类可读的摘要报告"""
    summary = enhanced_result["summary"]
    params = enhanced_result["detailed_params"]
    validation = enhanced_result["validation"]
    
    report = []
    report.append("=" * 70)
    report.append("LAMMPS 模拟输入文件摘要")
    report.append("=" * 70)
    report.append("")
    
    # 基本信息
    report.append("【基本信息】")
    report.append(f"  模拟类型: {summary.system_type}")
    report.append(f"  系综: {summary.ensemble or '未指定'}")
    if summary.total_atoms > 0:
        report.append(f"  原子数: {summary.total_atoms}")
    report.append("")
    
    # 力场信息
    report.append("【力场设置】")
    report.append(f"  力场类型: {summary.force_field or '未指定'}")
    if summary.force_field_params:
        for k, v in summary.force_field_params.items():
            report.append(f"  - {k}: {v}")
    report.append("")
    
    # 模拟参数
    report.append("【模拟参数】")
    report.append(f"  时间步长: {summary.timestep} {'fs' if summary.timestep < 1 else 'ps'}")
    report.append(f"  总步数: {summary.total_steps:,}")
    if summary.total_steps > 0 and summary.timestep > 0:
        total_time = summary.total_steps * summary.timestep
        report.append(f"  总模拟时间: {total_time} {'ps' if summary.timestep < 1 else 'time units'}")
    report.append("")
    
    # 热力学条件
    report.append("【热力学条件】")
    if summary.temperature:
        report.append(f"  温度: {summary.temperature} K")
    if summary.pressure:
        report.append(f"  压力: {summary.pressure} bar")
    if summary.thermostat:
        report.append(f"  温控器: {summary.thermostat}")
    if summary.barostat:
        report.append(f"  压控器: {summary.barostat}")
    if summary.boundary_conditions:
        report.append(f"  边界条件: {', '.join(summary.boundary_conditions)}")
    report.append("")
    
    # 详细参数
    report.append("【详细参数】")
    for param in params[:10]:  # 只显示前10个
        unit_str = f" {param.unit}" if param.unit else ""
        report.append(f"  {param.name}: {param.value}{unit_str}")
        if param.description:
            report.append(f"    ({param.description})")
    if len(params) > 10:
        report.append(f"  ... 还有 {len(params) - 10} 个参数")
    report.append("")
    
    # 资源估计
    report.append("【资源需求估计】")
    report.append(f"  预计运行时间: {summary.estimated_runtime or '无法估计'}")
    report.append(f"  内存需求: {summary.memory_requirement}")
    report.append("")
    
    # 验证结果
    report.append("【输入验证】")
    if validation["is_valid"]:
        report.append("  ✅ 输入文件基本有效")
    else:
        report.append("  ❌ 发现错误:")
        for error in validation["errors"]:
            report.append(f"    - {error}")
    
    if validation["warnings"]:
        report.append("  ⚠️  警告:")
        for warning in validation["warnings"]:
            report.append(f"    - {warning}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


# 便捷函数
def extract_and_summarize(lmp_file: str) -> Tuple[Dict[str, Any], str]:
    """提取并生成摘要的便捷函数"""
    extractor = EnhancedLammpsExtractor()
    result = extractor.extract_enhanced({"input": lmp_file})
    summary_text = generate_human_readable_summary(result)
    return result, summary_text
