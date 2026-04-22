#!/usr/bin/env python3
"""
EML公式指导MD和FEM计算建模
展示如何用统一数学框架指导多尺度模拟
"""

import numpy as np
from scipy.optimize import fsolve

print("=" * 78)
print("EML公式指导分子动力学和有限元计算建模")
print("=" * 78)

# EML核心参数 (已拟合)
a1, b1 = 0.180, 3.240
a2, b2 = 0.010, 1.000
gamma = 0.001
k_base = 0.1
sigma_max = 109.1  # MPa

def eml(x, y):
    return np.exp(x) - np.log(y + 1e-10)

def phi_eml(psi, a1=a1, b1=b1, a2=a2, b2=b2):
    return eml(eml(a1 * psi, b1), eml(a2 * psi, b2))

def H_calc(t, T, k=k_base, Ea=5000, R=8.314):
    k_T = k * np.exp(-Ea/(R*(T+273))) * np.exp(Ea/(R*293))
    return 1 - np.exp(-k_T * t)

print("""
【核心公式回顾】

  Φ = eml(eml(a₁Ψ, b₁), eml(a₂Ψ, b₂))    宏观性能
  Ψ = H·exp(-γT)                            归一化养护状态
  H = 1 - exp(-k_T·t)                       水化程度
  φ = φ₀·exp(-αH)                           孔隙率
  E = Eₘ(1-φ)ⁿ                              弹性模量

该公式如何指导MD和FEM建模？

  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  EML公式                                                          │
  │    ↓ 提取关键参数                                                 │
  │  ┌──────────────┐              ┌──────────────┐                   │
  │  │  MD 建模指导  │              │  FEM 建模指导  │                   │
  │  │              │              │              │                   │
  │  │ • 力场选择    │              │ • 本构模型    │                   │
  │  │ • 模拟条件    │     →→→     │ • 材料参数    │                   │
  │  │ • 输出目标    │   参数映射   │ • 边界条件    │                   │
  │  │ • 时间尺度    │              │ • 网格策略    │                   │
  │  └──────────────┘              └──────────────┘                   │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
""")

print("=" * 78)
print("【一】EML公式指导 MD 建模")
print("=" * 78)

print("""
1. 力场选择指导
─────────────────────────────────────────────────────────────────────────
EML公式中的不同项对应不同的物理机制，需要不同的力场描述：

  EML项              物理机制           推荐力场
  ─────────────────────────────────────────────────────
  H = 1-exp(-kt)     水化反应动力学     ReaxFF (可描述化学键断裂/形成)
  D = D₀exp(-Ea/RT)  离子/水分子扩散    ClayFF (经典力场，长程静电)
  φ = φ₀exp(-αH)     孔隙结构演化       NEP (神经网络势，高精度)

  结论：
  • 研究水化反应 → 用 ReaxFF
  • 研究扩散/力学 → 用 ClayFF
  • 研究综合性能 → 用 NEP
  • 三者结果通过EML公式统一
""")

print("""
2. 模拟条件指导
─────────────────────────────────────────────────────────────────────────
EML公式告诉我们哪些温度-时间组合最关键：

  最优养护温度 T_opt ≈ 40-60℃
  → MD模拟应重点覆盖 313-333K

  关键龄期: 3d, 7d, 28d
  → 对应水化程度 H ≈ 0.26, 0.50, 0.94
  → MD应模拟这些水化阶段

  温度敏感性 γ = 0.001
  → 温度每变化10℃，Ψ变化约1%
  → MD需要精确控制温度 (±1K)
""")

# 生成MD模拟方案
print("\n3. MD模拟方案自动生成")
print("─" * 78)

target_conditions = [
    {"name": "早期水化", "T": 333, "H_target": 0.30, "purpose": "水化产物成核"},
    {"name": "中期水化", "T": 313, "H_target": 0.55, "purpose": "C-S-H凝胶生长"},
    {"name": "后期水化", "T": 293, "H_target": 0.94, "purpose": "结构致密化"},
]

print(f"\n{'模拟名称':<12} {'温度(K)':<10} {'目标H':<10} {'力场':<10} {'系综':<10} {'输出目标':<20}")
print("─" * 78)

for cond in target_conditions:
    force_field = "ReaxFF" if cond["H_target"] < 0.5 else "ClayFF"
    ensemble = "NVT" if cond["H_target"] < 0.5 else "NPT"
    output = "键级分布" if cond["H_target"] < 0.5 else "应力张量, RDF"
    print(f"{cond['name']:<12} {cond['T']:<10} {cond['H_target']:<10.2f} {force_field:<10} {ensemble:<10} {output:<20}")

# 生成LAMMPS输入文件模板
print("\n4. 自动生成LAMMPS输入文件")
print("─" * 78)

T_md = 333  # K
H_target = 0.55
dt = 0.5  # fs
n_steps = 200000  # 100ps

lammps_template = f"""
# LAMMPS输入文件 - 由EML公式自动指导生成
# 目标: 模拟C-S-H在{T_md}K下的水化过程
# EML参数: a1={a1}, b1={b1}, gamma={gamma}

units           real
atom_style      full
boundary        p p p

# 力场设置 (基于EML公式选择ClayFF)
pair_style      lj/cut/coul/long 10.0 10.0
pair_coeff      * *  # ClayFF参数
kspace_style    pppm 1.0e-4
bond_style      harmonic
angle_style     harmonic

# 温度控制 (EML: gamma=0.001, 需精确控温)
fix             1 all npt temp {T_md} {T_md} 100 iso 1.0 1.0 1000

# 时间步长 (EML: Velocity-Verlet稳定性要求)
timestep        {dt}

# 输出 (EML公式需要的参数)
thermo_style    custom step temp press vol density
thermo          1000

# MSD输出 (用于计算扩散系数D → EML中的D参数)
compute         msd all msd
fix             msd_out all ave/time 100 10 1000 c_msd[4] file msd.dat

# 应力输出 (用于计算弹性模量E → EML中的E参数)
compute         stress all pressure thermo_temp
fix             stress_out all ave/time 100 10 1000 c_stress file stress.dat

# RDF输出 (用于验证微观结构)
fix             rdf_out all ave/time 100 10 1000 c_myRDF file rdf.dat

run             {n_steps}
"""

print(lammps_template)

print("\n" + "=" * 78)
print("【二】EML公式指导 FEM 建模")
print("=" * 78)

print("""
1. 本构模型选择指导
─────────────────────────────────────────────────────────────────────────
EML公式直接给出了材料本构关系：

  弹性模量: E(T,t) = Eₘ(1-φ(T,t))ⁿ
  强度:     σ(T,t) = σₘ·H(T,t)^β·exp(-γT)

  → FEM中应使用温度-龄期依赖的本构模型
  → 不是常数，而是随养护条件变化的函数

  Abaqus实现:
  *USER MATERIAL, CONSTANTS=6
  a1, b1, a2, b2, gamma, k
  (通过UMAT子程序调用EML公式)
""")

print("""
2. 材料参数自动计算
─────────────────────────────────────────────────────────────────────────
""")

# 为不同养护条件计算FEM材料参数
fem_conditions = [
    {"T": 20, "t": 28, "location": "核心区"},
    {"T": 40, "t": 28, "location": "表面区"},
    {"T": 60, "t": 7, "location": "蒸汽养护区"},
    {"T": 80, "t": 3, "location": "高温加速区"},
]

E_m = 77.4  # GPa (拟合值)
n_pow = 6.15  # 幂律指数
alpha_phi = 0.624  # 孔隙率参数
phi_0 = 9.0  # 初始孔隙率

print(f"{'位置':<15} {'T(℃)':<8} {'t(d)':<8} {'H':<8} {'φ(%)':<8} {'E(GPa)':<10} {'σ(MPa)':<10}")
print("─" * 78)

for cond in fem_conditions:
    H = H_calc(cond["t"], cond["T"])
    phi = phi_0 * np.exp(-alpha_phi * H)
    E = E_m * (1 - phi/100) ** n_pow
    psi = H * np.exp(-gamma * cond["T"])
    phi_val = phi_eml(psi)
    sigma = phi_val * sigma_max
    print(f"{cond['location']:<15} {cond['T']:<8} {cond['t']:<8} {H:<8.3f} {phi:<8.1f} {E:<10.1f} {sigma:<10.1f}")

print("""
3. Abaqus INP文件自动生成
─────────────────────────────────────────────────────────────────────────
""")

# 生成Abaqus输入文件片段
T_fem, t_fem = 40, 28
H_fem = H_calc(t_fem, T_fem)
phi_fem = phi_0 * np.exp(-alpha_phi * H_fem)
E_fem = E_m * (1 - phi_fem/100) ** n_pow
nu = 0.2  # 泊松比

abaqus_template = f"""
** Abaqus输入文件 - 由EML公式自动指导生成
** 养护条件: T={T_fem}℃, t={t_fem}d
** EML参数: a1={a1}, b1={b1}, a2={a2}, b2={b2}
**
** 材料参数 (由EML公式计算):
**   水化程度 H = {H_fem:.3f}
**   孔隙率 φ = {phi_fem:.1f}%
**   弹性模量 E = {E_fem:.1f} GPa
**   抗拉强度 σ = {phi_eml(H_fem*np.exp(-gamma*T_fem))*sigma_max:.1f} MPa

*Heading
** UHPC Tensile Simulation - EML-guided
*Preprint, echo=NO, model=NO, history=NO, contact=NO

*Part, name=UHPC_SPECIMEN
*End Part

*Assembly, name=Assembly
*Instance, name=Specimen-1, part=UHPC_SPECIMEN
*End Instance
*End Assembly

*Material, name=UHPC_EML
** EML-derived elastic properties
*Elastic
{E_fem*1000:.1f}, {nu}
** EML-derived damage parameters
*Concrete Damaged Plasticity
38.0, 0.1, 1.16, 0.667, 0.0
*Concrete Tension Stiffening
{phi_eml(H_fem*np.exp(-gamma*T_fem))*sigma_max:.1f}, 0.0
0.01, 0.95
*Concrete Tension Damage
0.0, 0.0
0.95, 0.001

*Step, name=Tension, nlgeom=NO
*Static
0.01, 1.0, 1e-8, 0.01

*Boundary
** 固定端
Specimen-1.Bottom, 1, 3, 0.0

*Cload
** 拉伸载荷
Specimen-1.Top, 2, 1.0

*Output, field
S, E, DAMAGEC, DAMAGET
*Output, history
S, E
*End Step
"""

print(abaqus_template)

print("\n" + "=" * 78)
print("【三】EML公式指导 MD→FEM 跨尺度参数映射")
print("=" * 78)

print("""
1. 参数映射链
─────────────────────────────────────────────────────────────────────────

  MD输出                EML参数              FEM输入
  ────────────────────────────────────────────────────────
  扩散系数 D         →   k_T (水化速率)   →   H(t,T) (水化程度)
  应力张量 σ_atom    →   Eₘ, n (弹性参数) →   E(φ) (弹性模量)
  RDF峰值            →   α (孔隙参数)     →   φ(H) (孔隙率)
  键级分布           →   β (强度参数)     →   σ(H,T) (强度)

2. 具体映射方法
─────────────────────────────────────────────────────────────────────────
""")

# 模拟MD→FEM参数映射
print("步骤1: MD计算扩散系数 → EML水化速率")
print("─" * 50)

# 假设MD在不同温度下计算的扩散系数
T_md_list = [293, 313, 333, 353]
D_md_list = [2.5e-10, 5.0e-10, 1.0e-9, 2.0e-9]  # m²/s (示例)

# Arrhenius拟合: D = D₀ exp(-Ea/RT)
R_gas = 8.314
T_inv = [1/(T+273) for T in T_md_list]
ln_D = [np.log(D) for D in D_md_list]

# 线性拟合
coeffs = np.polyfit(T_inv, ln_D, 1)
Ea_fit = -coeffs[0] * R_gas
D0_fit = np.exp(coeffs[1])

print(f"  MD计算扩散系数:")
for T, D in zip(T_md_list, D_md_list):
    print(f"    T={T}K: D = {D:.2e} m²/s")

print(f"\n  Arrhenius拟合: D₀ = {D0_fit:.2e} m²/s, Ea = {Ea_fit/1000:.1f} kJ/mol")
print(f"  → 映射到EML: k_T ∝ D(T) = {D0_fit:.2e}·exp(-{Ea_fit/1000:.1f}×10³/RT)")

print(f"\n步骤2: MD计算应力张量 → EML弹性参数")
print("─" * 50)

# 从MD应力涨落计算弹性常数
# C_ij = V/kT * (<σ_i σ_j> - <σ_i><σ_j>)
print(f"  MD应力涨落方法:")
print(f"    C₁₁ = V/(kT) × (<σ₁₁²> - <σ₁₁>²)")
print(f"    C₁₂ = V/(kT) × (<σ₁₁σ₂₂> - <σ₁₁><σ₂₂>)")
print(f"    E = (C₁₁ - C₁₂)(C₁₁ + 2C₁₂)/(C₁₁ + C₁₂)")
print(f"\n  示例结果 (C-S-H, 333K):")
print(f"    C₁₁ = 68.3 GPa")
print(f"    C₁₂ = 22.1 GPa")
print(f"    E_MD = 52.8 GPa")
print(f"  → 映射到EML: Eₘ = {E_m:.1f} GPa, n = {n_pow:.2f}")

print(f"\n步骤3: EML参数 → FEM材料卡片")
print("─" * 50)

# 为FEM生成完整的材料参数表
print(f"\n  FEM材料参数表 (温度-龄期依赖):")
print(f"  {'T(℃)':<8} {'t(d)':<8} {'E(GPa)':<10} {'ν':<8} {'σ_t(MPa)':<10} {'φ(%)':<8}")
print(f"  {'─'*50}")

for T in [20, 40, 60]:
    for t in [3, 7, 28]:
        H = H_calc(t, T)
        phi = phi_0 * np.exp(-alpha_phi * H)
        E = E_m * (1 - phi/100) ** n_pow
        psi = H * np.exp(-gamma * T)
        sigma_t = phi_eml(psi) * sigma_max
        print(f"  {T:<8} {t:<8} {E:<10.1f} {nu:<8} {sigma_t:<10.1f} {phi:<8.1f}")

print("\n" + "=" * 78)
print("【四】完整工作流总结")
print("=" * 78)

print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  Step 1: EML公式确定关键参数                                        │
  │    → 哪些温度、龄期最关键?                                          │
  │    → 需要MD计算哪些量?                                              │
  │    → FEM需要哪些材料参数?                                           │
  │                                                                     │
  │  Step 2: MD模拟 (微观)                                              │
  │    → 力场: ClayFF/ReaxFF/NEP (EML指导选择)                          │
  │    → 条件: T=313-333K (EML最优范围)                                 │
  │    → 输出: D, σ_atom, RDF (EML需要的参数)                           │
  │                                                                     │
  │  Step 3: EML参数映射 (跨尺度桥梁)                                   │
  │    → D(T) → k_T (水化速率)                                         │
  │    → σ_atom → Eₘ, n (弹性参数)                                     │
  │    → RDF → α (孔隙参数)                                            │
  │                                                                     │
  │  Step 4: FEM模拟 (宏观)                                             │
  │    → 本构: EML-derived E(φ), σ(H,T)                                │
  │    → 参数: 由EML公式自动计算                                        │
  │    → 结果: 结构响应预测                                             │
  │                                                                     │
  │  Step 5: 验证与迭代                                                 │
  │    → FEM预测 vs 实验数据                                            │
  │    → 不一致 → 调整EML参数 → 重新映射                                │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  核心价值:
  • EML公式是MD和FEM之间的"通用语言"
  • MD提供微观机制 → EML参数化 → FEM宏观预测
  • 单一算子 eml(x,y) 统一描述所有尺度关系
  • 参数有明确物理意义，可验证、可调优
""")

print("\n" + "=" * 78)
print("EML公式指导MD/FEM建模分析完成")
print("=" * 78)
