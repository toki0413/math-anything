# Math-Anything 完整工作流：从模拟文件到可证明数学命题

## 工作流概述

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Math-Anything 工作流                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ 原始文件  │ →  │ 特征提取  │ →  │ 数学结构  │ →  │ 数学命题  │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│       ↓               ↓               ↓               ↓                   │
│  .lmp/.inp/      简单版/增强版      ODE/PDE/       存在性/唯一性/          │
│  .incar等        提取器            变分问题        稳定性/收敛性            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 第一阶段：原始文件输入

### 支持的文件类型

| 引擎 | 输入文件 | 物理问题 |
|------|----------|----------|
| LAMMPS | `.lmp`, `.in` | 分子动力学 |
| VASP | `INCAR`, `POSCAR`, `KPOINTS` | 第一性原理 |
| Abaqus | `.inp` | 有限元分析 |
| GROMACS | `.mdp`, `.gro`, `.top` | 分子动力学 |
| ANSYS | `.apdl`, `.cdb` | 有限元分析 |
| COMSOL | `.mph` | 多物理场 |
| Multiwfn | `.fchk`, `.wfn` | 波函数分析 |
| SolidWorks | `.cwr` | CAD/CAE |

---

## 第二阶段：特征提取

### 2.1 简单版提取器（`XxxExtractor`）

**用途**：提取数学结构，供LLM推理使用

**输出格式**：
```python
MathSchema(
    problem_type="initial_value_ode",      # 问题类型
    governing_equations=["m·r̈ = F(r)"],   # 控制方程
    symmetries=["E(3)"],                   # 对称性
    constraints=["能量守恒"],              # 约束条件
    parameters={"N": 5000, "T": 333}       # 基本参数
)
```

### 2.2 增强版提取器（`EnhancedXxxExtractor`）

**用途**：提取详细参数，供工程师验证使用

**输出格式**：
```python
{
    "detailed_params": [
        DetailedParameter(name="timestep", value=0.5, unit="fs", 
                         description="时间步长", source="input"),
        DetailedParameter(name="temperature", value=333.0, unit="K",
                         description="目标温度", source="input"),
        ...
    ],
    "summary": SimulationSummary(
        system_type="能量最小化",
        ensemble="NVT",
        force_field="lj/cut/coul/long",
        temperature=333.0,
        timestep=0.5,
        total_steps=80000
    ),
    "validation": {
        "errors": [],           # 致命错误
        "warnings": ["时间步长偏大"],  # 警告
        "is_valid": True
    }
}
```

---

## 第三阶段：数学结构识别

### 3.1 问题类型分类

```
┌─────────────────────────────────────────────────────────────────┐
│                    数学问题类型分类树                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  数学问题                                                       │
│  ├── 微分方程                                                   │
│  │   ├── 常微分方程 (ODE)                                       │
│  │   │   ├── 初值问题 (IVP) ── LAMMPS, GROMACS                 │
│  │   │   └── 边值问题 (BVP)                                     │
│  │   └── 偏微分方程 (PDE)                                       │
│  │       ├── 椭圆型 ── Abaqus静力学, ANSYS                      │
│  │       ├── 抛物型 ── 热传导                                   │
│  │       └── 双曲型 ── 波动方程                                 │
│  ├── 变分问题 ── VASP (DFT)                                     │
│  ├── 特征值问题 ── 量子化学                                     │
│  └── 随机方程 ── Langevin动力学                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 对称性识别

```
┌─────────────────────────────────────────────────────────────────┐
│                      物理对称性分类                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  连续对称群                                                      │
│  ├── SO(3) ── 各向同性系统（液体、气体）                         │
│  ├── O(3)  ── 各向同性 + 空间反演                               │
│  ├── E(3)  ── 欧几里得群（平移 + 旋转）                         │
│  ├── SE(3) ── 特殊欧几里得群                                    │
│  └── T(3)  ── 纯平移                                            │
│                                                                 │
│  离散对称群                                                      │
│  ├── 点群 (C₂v, D₆h, Oₕ) ── 分子、晶体                         │
│  └── 空间群 ── 周期性晶体                                       │
│                                                                 │
│  对称性 → ML架构推荐                                            │
│  ├── E(3)  → E(3)-等变网络 (SchNet, NequIP)                     │
│  ├── SE(3) → SE(3)-等变网络 (DimeNet++)                         │
│  └── 周期性 → 傅里叶神经算子 (FNO)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第四阶段：数学命题生成

### 4.1 LAMMPS 分子动力学

**输入文件** → **数学命题**

```
定理 (LAMMPS分子动力学适定性)

设系统包含 N 个原子，位置为 r = (r₁, ..., r_N) ∈ ℝ³ᴺ，
质量为 m = (m₁, ..., m_N) ∈ ℝ₊ᴺ。

控制方程:
  mᵢ(d²rᵢ/dt²) = Fᵢ(r) = -∂V/∂rᵢ

其中势能函数 V: ℝ³ᴺ → ℝ 由力场定义。

初值条件:
  r(0) = r⁰ ∈ ℝ³ᴺ
  ṙ(0) = v⁰ ∈ ℝ³ᴺ

假设:
  (H1) V ∈ C²(ℝ³ᴺ, ℝ) 且 ∇V 有界
  (H2) 初值 (r⁰, v⁰) 给定
  (H3) 时间步长 Δt 满足 CFL 条件

结论:
  1. 存在性: 存在 T > 0，使初值问题在 [0, T] 上有解
     证明: Picard-Lindelöf 定理
  
  2. 唯一性: 解在给定初值下唯一
     证明: Lipschitz 连续性 + Gronwall 不等式
  
  3. 数值稳定性: Velocity-Verlet 积分器稳定当
     Δt < 2/ω_max，其中 ω_max 为系统最高频率
  
  4. 收敛性: 能量误差 E(t) - E(0) = O(Δt²)

参数验证:
  ✓ 力场: lj/cut/coul/long (连续可微)
  ✓ 时间步长: Δt = 0.5 fs
  ✓ 总步数: 80,000
  ✓ 温度: T = 333 K (Nose-Hoover热浴)
```

### 4.2 VASP 第一性原理计算

**输入文件** → **数学命题**

```
定理 (Kohn-Sham DFT适定性)

设系统包含 N 个电子和 M 个原子核，原子核位置固定为 R = (R₁, ..., R_M)。

Kohn-Sham方程:
  [-½∇² + V_eff[ρ](r)]ψₙ(r) = εₙψₙ(r)

其中:
  V_eff[ρ] = V_ext + V_H[ρ] + V_xc[ρ]
  ρ(r) = Σₙ₌₁ᴺ |ψₙ(r)|²

变分原理:
  E[ρ] = min_{ρ ∈ A} {T_s[ρ] + E_ext[ρ] + E_H[ρ] + E_xc[ρ]}
  
  其中 A = {ρ ≥ 0 : ∫ρ = N, √ρ ∈ H¹(ℝ³)} 为允许集

假设:
  (H1) 原子核位置 R 固定
  (H2) 交换关联泛函 E_xc 满足特定条件
  (H3) 平面波截断 E_cut 充分大

结论:
  1. 存在性: 存在基态电子密度 ρ* 使 E[ρ] 最小
     证明: Hohenberg-Kohn 定理 + 直接变分法
  
  2. 唯一性: 基态能量 E₀ 唯一（基态密度可能不唯一）
     证明: 变分原理
  
  3. 收敛性: 自洽场迭代收敛当 E_cut 充分大
     证明: Banach 不动点定理
  
  4. 精度: 能量误差 |E - E_exact| = O(e^{-αE_cut})

参数验证:
  ✓ 交换关联: PBE (GGA泛函)
  ✓ 截断能: E_cut = 520 eV
  ✓ 收敛精度: ΔE < 1E-5
  ✓ 离子步数: 100 (共轭梯度法)
```

### 4.3 Abaqus 有限元分析

**输入文件** → **数学命题**

```
定理 (线弹性力学适定性)

设求解域 Ω ⊂ ℝ³ 为有界开集，边界 Γ = Γ_u ∪ Γ_t，
其中 Γ_u ∩ Γ_t = ∅，|Γ_u| > 0。

强形式:
  -∇·σ = f     在 Ω 内    (平衡方程)
  σ = C:ε      在 Ω 内    (本构方程)
  ε = ½(∇u + ∇uᵀ)  在 Ω 内 (几何方程)
  u = ū       在 Γ_u 上   (位移边界)
  σ·n = t̄     在 Γ_t 上   (力边界)

弱形式:
  找 u ∈ V = {v ∈ H¹(Ω)³ : v|_Γ_u = ū}
  使得 ∀v ∈ V₀ = {v ∈ H¹(Ω)³ : v|_Γ_u = 0}:
  
  a(u, v) = l(v)
  
  其中:
  a(u, v) = ∫_Ω C:ε(u):ε(v) dx
  l(v) = ∫_Ω f·v dx + ∫_Γ_t t̄·v ds

假设:
  (H1) Ω 为 Lipschitz 域
  (H2) 弹性张量 C 一致正定: ∃α > 0, C:ε:ε ≥ α|ε|²
  (H3) f ∈ L²(Ω), t̄ ∈ L²(Γ_t)

结论:
  1. 存在性: 存在弱解 u ∈ V
     证明: Lax-Milgram 定理
  
  2. 唯一性: 弱解唯一
     证明: 双线性形式 a(·,·) 的强制性
  
  3. 稳定性: ‖u‖_V ≤ C(‖f‖ + ‖t̄‖)
     证明: Lax-Milgram 估计
  
  4. 收敛性: 有限元解 u_h → u 当 h → 0
     证明: Céa 引理

参数验证:
  ✓ 分析类型: 静力学 (椭圆型PDE)
  ✓ 几何非线性: 否 (小变形假设)
  ✓ 材料定义: 需检查弹性模量正定
  ✓ 边界条件: 需检查约束充分
```

---

## 第五阶段：可解性验证

### 5.1 自动验证流程

```python
def verify_well_posedness(extracted_result):
    """验证问题的适定性"""
    
    validation = {
        "existence": verify_existence(extracted_result),
        "uniqueness": verify_uniqueness(extracted_result),
        "stability": verify_stability(extracted_result),
        "convergence": verify_convergence(extracted_result)
    }
    
    return validation

def verify_existence(result):
    """存在性验证"""
    checks = []
    
    # 检查方程类型
    if result.problem_type == "initial_value_ode":
        # Picard-Lindelöf 条件
        if result.force_field_continuous:
            checks.append("✓ 力场连续可微 → 局部解存在")
    
    elif result.problem_type == "variational":
        # 变分问题存在性
        if result.energy_bounded_below:
            checks.append("✓ 能量下有界 → 基态存在")
    
    elif result.problem_type == "boundary_value_pde":
        # Lax-Milgram 条件
        if result.material_positive_definite:
            checks.append("✓ 材料正定 → 弱解存在")
    
    return checks

def verify_uniqueness(result):
    """唯一性验证"""
    checks = []
    
    if result.problem_type == "initial_value_ode":
        if result.initial_conditions_defined:
            checks.append("✓ 初值条件给定 → 解唯一")
    
    elif result.problem_type == "variational":
        checks.append("✓ 基态能量唯一（密度可能不唯一）")
    
    elif result.problem_type == "boundary_value_pde":
        if result.constraints_sufficient:
            checks.append("✓ 约束充分 → 解唯一")
    
    return checks
```

### 5.2 验证报告生成

```
┌─────────────────────────────────────────────────────────────────┐
│                    适定性验证报告                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  文件: equil.lmp                                                │
│  问题类型: 初值ODE                                              │
│                                                                 │
│  【存在性验证】                                                  │
│  ✓ 力场连续可微 → 局部解存在                        │
│  ✓ 势能函数有界 → 全局解存在                                    │
│                                                                 │
│  【唯一性验证】                                                  │
│  ✓ 初值条件 (r⁰, v⁰) 给定                                       │
│  ✓ 力场 Lipschitz 连续                                          │
│  → 解唯一 (Gronwall 不等式)                                     │
│                                                                 │
│  【稳定性验证】                                                  │
│  ✓ 时间步长 Δt = 0.5 fs < 2/ω_max                               │
│  ✓ Velocity-Verlet 积分器稳定                                   │
│  → 数值解稳定                                                   │
│                                                                 │
│  【收敛性验证】                                                  │
│  ✓ Velocity-Verlet 是二阶辛积分器                               │
│  → 能量误差 O(Δt²)                                              │
│  → 轨迹误差 O(Δt²)                                              │
│                                                                 │
│  【结论】                                                        │
│  该模拟设置定义了一个适定的初值问题，                            │
│  存在唯一解，数值方法稳定收敛。                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第六阶段：ML架构推荐

### 6.1 基于数学结构的推荐

```python
def recommend_ml_architecture(math_schema):
    """基于数学结构推荐ML架构"""
    
    recommendations = []
    
    # 对称性匹配
    if "E(3)" in math_schema.symmetries:
        recommendations.append({
            "architecture": "SchNet / NequIP",
            "reason": "E(3)-等变，保持旋转不变性",
            "compatibility_proof": "消息传递在旋转下不变"
        })
    
    if "SE(3)" in math_schema.symmetries:
        recommendations.append({
            "architecture": "DimeNet++",
            "reason": "SE(3)-等变，捕获角度依赖",
            "compatibility_proof": "球谐基变换等变"
        })
    
    # 问题类型匹配
    if math_schema.problem_type == "boundary_value_pde":
        if "periodic" in math_schema.symmetries:
            recommendations.append({
                "architecture": "Fourier Neural Operator (FNO)",
                "reason": "周期边界条件天然适合傅里叶空间",
                "compatibility_proof": "傅里叶变换保持周期性"
            })
    
    return recommendations
```

### 6.2 推荐示例

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML架构推荐报告                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: LAMMPS分子动力学模拟                                      │
│  数学结构: {问题类型: IVP, 对称性: E(3), 拓扑: 点云}            │
│                                                                 │
│  【推荐架构】                                                    │
│                                                                 │
│  1. SchNet                                                      │
│     - 类型: E(3)-等变图神经网络                                  │
│     - 匹配原因: 系统具有 E(3) 对称性                             │
│     - 兼容性证明:                                                │
│       消息 m_ij = Σ_k φ(h_i, h_j, ||r_i - r_j||)                │
│       只依赖于距离 ||r_i - r_j||，旋转不变                       │
│     - 数学保证:                                                  │
│       ✓ E(3)-等变性                                             │
│       ✓ 置换不变性                                              │
│       ✓ 通用逼近                                                │
│                                                                 │
│  2. DimeNet++                                                   │
│     - 类型: SE(3)-等变方向消息传递                               │
│     - 匹配原因: 可捕获角度依赖                                   │
│     - 兼容性证明:                                                │
│       消息包含角度 θ_ijk，使用球谐基 Y_l^m                       │
│       Y_l^m(R·r̂) = Σ D_l^mm'(R) Y_l^m'(r̂)                      │
│     - 数学保证:                                                  │
│       ✓ SE(3)-等变性                                            │
│       ✓ 高阶角度相互作用                                        │
│                                                                 │
│  【不推荐架构】                                                  │
│                                                                 │
│  ✗ 标准CNN: 不满足旋转等变性                                     │
│  ✗ MLP: 忽略图结构                                              │
│  ✗ RNN: 不适合空间数据                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 完整工作流示例

### 输入：LAMMPS文件

```lammps
# LAMMPS input for C-S-H gel simulation
units           real
atom_style      full
dimension       3
boundary        p p p

# Force field
pair_style      lj/cut/coul/long 10.0 10.0
kspace_style    pppm 1.0e-4

# Simulation settings
timestep        0.5
run             80000

# Temperature control
fix             1 all nvt temp 333.0 333.0 100.0
```

### 输出：数学命题 + 验证报告

```
┌─────────────────────────────────────────────────────────────────┐
│              Math-Anything 分析报告                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│  第一部分：特征提取                                              │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  【简单版输出】                                                  │
│  - 问题类型: initial_value_ode                                  │
│  - 控制方程: m·r̈ = F(r)                                        │
│  - 对称性: E(3)                                                 │
│  - 约束: 能量守恒                                               │
│                                                                 │
│  【增强版输出】                                                  │
│  - 模拟类型: NVT分子动力学                                      │
│  - 力场: lj/cut/coul/long                                       │
│  - 时间步长: 0.5 fs                                             │
│  - 温度: 333.0 K                                                │
│  - 总步数: 80,000                                               │
│  - 截断距离: 10.0 Å                                             │
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│  第二部分：数学命题                                              │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  定理: 该模拟定义了一个适定的初值问题                            │
│                                                                 │
│  证明:                                                          │
│  1. 存在性: 力场 lj/cut/coul/long 连续可微                      │
│     → 由 Picard-Lindelöf 定理，局部解存在                       │
│                                                                 │
│  2. 唯一性: 初值条件 (r⁰, v⁰) 给定                              │
│     → 解唯一 (Gronwall 不等式)                                  │
│                                                                 │
│  3. 稳定性: Δt = 0.5 fs 满足数值稳定性条件                      │
│     → Velocity-Verlet 积分器稳定                                │
│                                                                 │
│  4. 收敛性: 能量误差 O(Δt²)                                     │
│     → 数值解收敛到真解                                          │
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│  第三部分：输入验证                                              │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  ✓ 力场定义完整                                                 │
│  ✓ 边界条件: 周期性边界                                         │
│  ✓ 温度控制: Nose-Hoover 热浴                                   │
│  ⚠ 时间步长偏大，建议 ≤ 1.0 fs                                  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│  第四部分：ML架构推荐                                            │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  基于数学结构 {E(3)对称性, 点云拓扑}:                            │
│                                                                 │
│  推荐: SchNet / NequIP                                          │
│  - E(3)-等变，保持物理对称性                                    │
│  - 适合势能面拟合                                               │
│  - 数学保证: 通用逼近 + 等变性                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## API使用示例

```python
from math_anything.lammps import LammpsExtractor, EnhancedLammpsExtractor
from math_anything.core.proposition_generator import generate_math_proposition
from math_anything.core.ml_recommender import recommend_architecture

# 第一步：简单版提取（数学结构）
simple_extractor = LammpsExtractor()
math_schema = simple_extractor.extract({"input": "equil.lmp"})

# 第二步：增强版提取（详细参数）
enhanced_extractor = EnhancedLammpsExtractor()
detailed_result = enhanced_extractor.extract_enhanced({"input": "equil.lmp"})

# 第三步：生成数学命题
proposition = generate_math_proposition(math_schema, detailed_result)
print(proposition)

# 第四步：适定性验证
validation = verify_well_posedness(detailed_result)
print(validation)

# 第五步：ML架构推荐
ml_recommendations = recommend_architecture(math_schema)
print(ml_recommendations)
```

---

## 总结

| 阶段 | 输入 | 输出 | 工具 |
|------|------|------|------|
| 1. 文件输入 | `.lmp`, `.inp`, `INCAR` | 原始文本 | 文件系统 |
| 2. 特征提取 | 原始文本 | 数学结构 + 详细参数 | `XxxExtractor` |
| 3. 结构识别 | 特征 | 问题类型 + 对称性 | 分类器 |
| 4. 命题生成 | 数学结构 | 可证明定理 | 命题生成器 |
| 5. 适定性验证 | 参数 | 存在/唯一/稳定/收敛 | 验证器 |
| 6. ML推荐 | 数学结构 | 架构 + 兼容性证明 | 推荐器 |
