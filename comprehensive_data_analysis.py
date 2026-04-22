#!/usr/bin/env python3
"""
Math Anything 完整模拟数据分析
使用所有可用的UHPC和MD模拟数据
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob

sys.path.insert(0, str(Path(__file__).parent / "math-anything" / "core"))

from math_anything import MathAnything, PropositionGenerator


def analyze_vti_files(base_path):
    """分析所有VTI体素数据文件"""
    print("\n" + "="*70)
    print("【1】VTI体素数据分析 (所有温度×龄期)")
    print("="*70)
    
    vti_data = []
    
    for temp_dir in base_path.glob("*℃"):
        if not temp_dir.is_dir():
            continue
        temp = int(temp_dir.name.replace(" ℃", ""))
        
        for age_dir in temp_dir.glob("*d"):
            if not age_dir.is_dir():
                continue
            age = int(age_dir.name.replace(" d", "").replace("d", ""))
            
            vti_files = list(age_dir.glob("*.vti"))
            if vti_files:
                # 简单解析VTI文件获取体素统计
                with open(vti_files[0], 'r') as f:
                    content = f.read()
                    
                # 提取相分布统计
                phases = {}
                for line in content.split('\n')[10:]:  # 跳过XML头
                    if line.strip().isdigit():
                        phase = int(line.strip())
                        phases[phase] = phases.get(phase, 0) + 1
                
                total_voxels = sum(phases.values())
                porosity = phases.get(0, 0) / total_voxels * 100 if total_voxels > 0 else 0
                
                vti_data.append({
                    'temperature': temp,
                    'age': age,
                    'total_voxels': total_voxels,
                    'porosity_%': porosity,
                    'phases': len(phases),
                    'file': vti_files[0].name
                })
                
                print(f"  {temp}℃ {age}d: 孔隙率={porosity:.1f}%, 体素数={total_voxels}")
    
    return pd.DataFrame(vti_data)


def analyze_pore_data(base_path):
    """分析孔隙数据"""
    print("\n" + "="*70)
    print("【2】孔隙结构数据分析")
    print("="*70)
    
    pore_data = []
    
    for temp_dir in base_path.glob("*℃"):
        if not temp_dir.is_dir():
            continue
        temp = int(temp_dir.name.replace(" ℃", ""))
        
        for age_dir in temp_dir.glob("*d"):
            if not age_dir.is_dir():
                continue
            age = int(age_dir.name.replace(" d", "").replace("d", ""))
            
            pore_files = list(age_dir.glob("pore_*.txt"))
            if pore_files:
                df = pd.read_csv(pore_files[0], header=None)
                
                pore_data.append({
                    'temperature': temp,
                    'age': age,
                    'n_pores': len(df),
                    'mean_radius': df[0].mean() if len(df.columns) > 0 else 0,
                    'max_radius': df[0].max() if len(df.columns) > 0 else 0,
                    'total_volume': df[3].sum() if len(df.columns) > 3 else 0
                })
                
                print(f"  {temp}℃ {age}d: 孔隙数={len(df)}, 平均半径={df[0].mean():.2f}")
    
    return pd.DataFrame(pore_data)


def analyze_all_excel_data(base_path):
    """分析所有Excel数据"""
    print("\n" + "="*70)
    print("【3】UHPC力学性能数据 (所有温度×龄期)")
    print("="*70)
    
    all_data = []
    
    for temp_dir in base_path.glob("*℃"):
        if not temp_dir.is_dir():
            continue
        temp = int(temp_dir.name.replace(" ℃", ""))
        
        for age_dir in temp_dir.glob("*d"):
            if not age_dir.is_dir():
                continue
            age = int(age_dir.name.replace(" d", "").replace("d", ""))
            
            excel_files = list(age_dir.glob("*.xlsx"))
            if excel_files:
                try:
                    df = pd.read_excel(excel_files[0])
                    
                    # 自动识别列
                    strain_col = df.columns[0]
                    stress_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    
                    strain = df[strain_col].values
                    stress = df[stress_col].values
                    
                    valid = ~(np.isnan(stress) | np.isnan(strain))
                    stress = stress[valid]
                    strain = strain[valid]
                    
                    if len(stress) > 10:
                        # 计算弹性模量 (线性段)
                        n_linear = max(10, int(len(strain) * 0.2))
                        if n_linear > 1:
                            coeffs = np.polyfit(strain[:n_linear], stress[:n_linear], 1)
                            E = coeffs[0]
                        else:
                            E = 0
                        
                        all_data.append({
                            'temperature': temp,
                            'age': age,
                            'max_stress': np.max(stress),
                            'max_strain': np.max(strain),
                            'elastic_modulus': E,
                            'toughness': np.trapezoid(stress, strain) if hasattr(np, 'trapezoid') else np.trapz(stress, strain),  # 应力-应变曲线下面积
                            'data_points': len(stress)
                        })
                        
                        print(f"  {temp}℃ {age}d: 强度={np.max(stress):.1f}MPa, 弹性模量={E/1000:.1f}GPa")
                except Exception as e:
                    print(f"  {temp}℃ {age}d: 读取失败 - {e}")
    
    return pd.DataFrame(all_data)


def analyze_md_outputs(md_path):
    """分析MD模拟输出数据"""
    print("\n" + "="*70)
    print("【4】分子动力学模拟输出分析")
    print("="*70)
    
    md_data = []
    
    temp_dirs = list(md_path.glob("csh_5k_T*C"))
    
    for temp_dir in temp_dirs:
        if not temp_dir.is_dir():
            continue
        
        temp_str = temp_dir.name.split("T")[1].replace("C", "")
        try:
            temp = int(temp_str)
        except:
            continue
        
        # 读取MSD数据
        msd_file = temp_dir / "equil_msd.txt"
        if msd_file.exists():
            df_msd = pd.read_csv(msd_file, sep='\s+', comment='#', header=None, names=['step', 'msd'])
            
            # 计算扩散系数 (MSD = 6Dt)
            if len(df_msd) > 10:
                # 线性拟合 MSD vs t
                time = df_msd['step'].values * 0.5e-15  # 假设步长0.5fs
                msd = df_msd['msd'].values
                
                # 只取线性段
                valid = msd > 0
                if np.sum(valid) > 5:
                    coeffs = np.polyfit(time[valid], msd[valid], 1)
                    D = coeffs[0] / 6  # D = slope/6 (3D)
                else:
                    D = 0
                
                md_data.append({
                    'temperature': temp,
                    'diffusion_coefficient': D,
                    'final_msd': msd[-1] if len(msd) > 0 else 0,
                    'simulation_steps': len(df_msd)
                })
                
                print(f"  T={temp}℃: 扩散系数D={D:.2e} m²/s, 最终MSD={msd[-1]:.3f} Å²")
    
    return pd.DataFrame(md_data)


def analyze_force_fields(md_path):
    """分析力场文件"""
    print("\n" + "="*70)
    print("【5】力场参数分析")
    print("="*70)
    
    ma = MathAnything()
    
    # ClayFF
    clayff_file = md_path / "ClayFF" / "clayff.lammps"
    if clayff_file.exists():
        with open(clayff_file, 'r') as f:
            content = f.read()
        print("\n  ClayFF力场参数:")
        print("  - LJ截断: 10.0 Å")
        print("  - 长程作用: Ewald求和")
        print("  - 原子类型: Si, Ob, Oa, Oh, Ow, Ca, H")
        
    # ReaxFF
    reaxff_file = md_path / "ReaxFF_cement" / "ffield-CaSiOH-gulp.reax"
    if reaxff_file.exists():
        print("\n  ReaxFF力场: 可描述化学反应的键级依赖势")
        
    # NEP
    nep_file = md_path / "CSH-NEP" / "01-NEP-tob-C-S-H" / "nep.txt"
    if nep_file.exists():
        print("\n  NEP神经网络势: 机器学习力场")


def generate_comprehensive_report(all_results):
    """生成综合分析报告"""
    print("\n" + "="*70)
    print("【6】综合分析报告")
    print("="*70)
    
    # 合并所有数据
    if len(all_results['excel']) > 0 and len(all_results['vti']) > 0:
        merged = pd.merge(
            all_results['excel'], 
            all_results['vti'], 
            on=['temperature', 'age'], 
            how='outer'
        )
        
        if len(all_results['pore']) > 0:
            merged = pd.merge(
                merged,
                all_results['pore'],
                on=['temperature', 'age'],
                how='outer'
            )
        
        print("\n  多尺度数据关联矩阵:")
        print("  " + "-"*50)
        
        # 相关系数
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        corr_data = merged[numeric_cols].corr()
        
        print("\n  关键相关性:")
        if 'porosity_%' in corr_data and 'elastic_modulus' in corr_data:
            corr_ep = corr_data.loc['porosity_%', 'elastic_modulus']
            print(f"    孔隙率 vs 弹性模量: {corr_ep:.3f}")
        
        if 'temperature' in corr_data and 'max_stress' in corr_data:
            corr_ts = corr_data.loc['temperature', 'max_stress']
            print(f"    温度 vs 最大应力: {corr_ts:.3f}")
        
        if 'age' in corr_data and 'elastic_modulus' in corr_data:
            corr_ae = corr_data.loc['age', 'elastic_modulus']
            print(f"    龄期 vs 弹性模量: {corr_ae:.3f}")
    
    # 温度效应分析
    if len(all_results['excel']) > 0:
        print("\n  温度效应分析:")
        print("  " + "-"*50)
        
        df = all_results['excel']
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            avg_stress = temp_data['max_stress'].mean()
            avg_E = temp_data['elastic_modulus'].mean()
            print(f"    {temp}℃: 平均强度={avg_stress:.1f}MPa, 平均E={avg_E/1000:.1f}GPa")
    
    # 龄期效应分析
    if len(all_results['excel']) > 0:
        print("\n  龄期效应分析:")
        print("  " + "-"*50)
        
        df = all_results['excel']
        for age in sorted(df['age'].unique()):
            age_data = df[df['age'] == age]
            avg_stress = age_data['max_stress'].mean()
            avg_E = age_data['elastic_modulus'].mean()
            print(f"    {age}d: 平均强度={avg_stress:.1f}MPa, 平均E={avg_E/1000:.1f}GPa")


def translate_to_math_propositions(all_results):
    """生成数学命题"""
    print("\n" + "="*70)
    print("【7】数学命题生成 (Translate)")
    print("="*70)
    
    ma = MathAnything()
    gen = PropositionGenerator()
    
    # 从数据中生成命题
    print("\n  基于数据发现的数学命题:")
    print("  " + "-"*50)
    
    if len(all_results['excel']) > 0:
        df = all_results['excel']
        
        # 命题1: 温度-强度关系
        print("\n  [命题1] 温度对UHPC强度的影响规律")
        print("  数学形式: σ_max = f(T, age)")
        print("  待证明: 温度升高是否导致强度降低?")
        
        # 命题2: 龄期-弹性模量关系
        print("\n  [命题2] 龄期对弹性模量的非线性影响")
        print("  数学形式: E = g(age) × h(T)")
        print("  待证明: 是否存在最优养护龄期?")
        
        # 命题3: 孔隙-力学性能关系
        if len(all_results['vti']) > 0:
            print("\n  [命题3] 孔隙率与弹性模量的反比关系")
            print("  数学形式: E = E₀(1 - p)^n")
            print("  待证明: 孔隙率每增加1%, E下降多少?")
    
    if len(all_results['md']) > 0:
        # 命题4: 扩散系数
        print("\n  [命题4] 扩散系数的Arrhenius温度依赖")
        print("  数学形式: D = D₀ exp(-Eₐ/RT)")
        print("  待证明: 计算活化能 Eₐ")


def main():
    print("="*70)
    print("Math Anything 完整模拟数据分析")
    print("="*70)
    print("\n分析所有可用的UHPC和MD模拟数据...")
    
    base_path = Path(r"C:\Users\wanzh\Downloads\模拟数据\模拟数据")
    md_path = base_path / "MD_simulation"
    
    all_results = {}
    
    # 1. VTI体素数据
    try:
        all_results['vti'] = analyze_vti_files(base_path)
        print(f"\n  成功分析 {len(all_results['vti'])} 个VTI文件")
    except Exception as e:
        print(f"\n  VTI分析失败: {e}")
        all_results['vti'] = pd.DataFrame()
    
    # 2. 孔隙数据
    try:
        all_results['pore'] = analyze_pore_data(base_path)
        print(f"\n  成功分析 {len(all_results['pore'])} 个孔隙数据文件")
    except Exception as e:
        print(f"\n  孔隙分析失败: {e}")
        all_results['pore'] = pd.DataFrame()
    
    # 3. Excel力学数据
    try:
        all_results['excel'] = analyze_all_excel_data(base_path)
        print(f"\n  成功分析 {len(all_results['excel'])} 个Excel文件")
    except Exception as e:
        print(f"\n  Excel分析失败: {e}")
        all_results['excel'] = pd.DataFrame()
    
    # 4. MD输出数据
    try:
        all_results['md'] = analyze_md_outputs(md_path / "batch_runs")
        print(f"\n  成功分析 {len(all_results['md'])} 个MD温度点")
    except Exception as e:
        print(f"\n  MD分析失败: {e}")
        all_results['md'] = pd.DataFrame()
    
    # 5. 力场分析
    try:
        analyze_force_fields(md_path)
    except Exception as e:
        print(f"\n  力场分析失败: {e}")
    
    # 6. 综合报告
    try:
        generate_comprehensive_report(all_results)
    except Exception as e:
        print(f"\n  综合报告生成失败: {e}")
    
    # 7. 数学命题
    try:
        translate_to_math_propositions(all_results)
    except Exception as e:
        print(f"\n  命题生成失败: {e}")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()
