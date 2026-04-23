# Math Anything 教程：储氢材料计算

> 跟随新能源材料博士生陈雪的 14 天学习历程，掌握储氢材料的吸附与扩散计算

---

## 人物背景

**陈雪**，新能源材料专业博二学生，研究方向为 MOFs 材料的储氢性能优化。

**背景**：
- 本科毕业于应用化学专业
- 熟练使用 VASP 和 LAMMPS 进行材料模拟
- 需要计算氢气在 MOFs 中的吸附等温线和扩散系数
- 希望通过机器学习预测新材料的储氢性能

**目标**：用 Math Anything 提取储氢材料计算的数学结构，建立从量子力学到分子动力学的完整分析流程。

---

## 第一章：储氢材料几何结构分析（第 1-3 天）

### 1.1 MOF 框架结构提取

陈雪分析一个典型的 MOF-5 结构用于储氢：

```python
from math_anything import MathAnything

ma = MathAnything()

# 从 VASP 结构文件提取
result = ma.extract_file("vasp", "MOF5/POSCAR")

print("晶体结构信息：")
print(f"  空间群: {result.schema['symmetry']['space_group']}")
print(f"  晶胞参数: a={result.schema['lattice']['a']:.3f} Å")
print(f"  孔隙率: {result.schema['porosity']:.2%}")
print(f"  比表面积: {result.schema['surface_area']} m²/g")

print("\n孔结构分析：")
print(f"  最大孔径: {result.schema['pore']['diameter_max']:.2f} Å")
print(f"  孔体积: {result.schema['pore']['volume']:.3f} cm³/g")
print(f"  孔道连通性: {result.schema['pore']['connectivity']}")
```

**输出**：
```
晶体结构信息：
  空间群: Fm-3m (No. 225)
  晶胞参数: a=25.832 Å
  孔隙率: 61.20%
  比表面积: 2900 m²/g

孔结构分析：
  最大孔径: 12.8 Å
  孔体积: 1.18 cm³/g
  孔道连通性: 3D 互联网络
```

**陈雪的发现**："MOF-5 有超高的比表面积和孔隙率，非常适合储氢！"

### 1.2 分级分析选择合适方法

```python
from math_anything import TieredAnalyzer, AnalysisTier

analyzer = TieredAnalyzer()

rec = analyzer.get_recommendation("MOF5/POSCAR")
print(f"推荐分析级别: {rec.recommended_tier}")
print(f"复杂度评分: {rec.complexity_score.total}/100")
```

**输出**：
```
推荐分析级别: PROFESSIONAL
复杂度评分: 65/100
原因: [
    "大晶胞系统 (>1000 原子)",
    "周期性边界条件复杂",
    "多孔拓扑结构需要分析",
    "吸附位点识别"
]
```

---

## 第二章：氢气吸附计算（第 4-7 天）

### 2.1 吸附等温线计算（GCMC）

陈雪使用巨正则蒙特卡洛模拟计算储氢性能：

```python
from math_anything import extract

# GCMC 模拟参数
result = extract("lammps", {
    "style": "gcmc",
    "temperature": 77,      # K 液氮温度
    "pressure": [1, 10, 50, 100],  # bar
    "gas_type": "H2",
    "adsorbent": "MOF5",
    "framework": "rigid",
})

print("GCMC 数学模型：")
print("  巨正则系综: μ, V, T 恒定")
print("  化学势: μ = μ° + kT ln(f/f°)")
print("  吸附量: N_ads = <N> - N_gas")

print("\nLangmuir 拟合参数:")
print(f"  单层吸附量 q_m: {result.schema['isotherm']['q_m']:.2f} wt%")
print(f"  Langmuir 常数 K_L: {result.schema['isotherm']['K_L']:.4f} bar⁻¹")
print(f"  R² 拟合优度: {result.schema['isotherm']['r_squared']:.4f}")
```

**输出**：
```
GCMC 数学模型：
  巨正则系综: μ, V, T 恒定
  化学势: μ = μ° + kT ln(f/f°)
  吸附量: N_ads = <N> - N_gas

Langmuir 拟合参数:
  单层吸附量 q_m: 4.52 wt%
  Langmuir 常数 K_L: 0.0234 bar⁻¹
  R² 拟合优度: 0.9987
```

### 2.2 吸附位点分析

```python
result = ma.extract_file("multiwfn", "MOF5_H2.wfn")

print("氢气吸附位点：")
for site in result.schema["adsorption_sites"]:
    print(f"  {site['name']}: {site['energy']:.2f} kJ/mol")
    print(f"    位置: {site['location']}")
    print(f"    配位数: {site['coordination']}")
    print(f"    吸附类型: {site['type']}")

print("\n吸附能分析：")
print(f"  最强吸附位: {result.schema['strongest_site']['energy']:.2f} kJ/mol")
print(f"  平均吸附能: {result.schema['average_energy']:.2f} kJ/mol")
print(f"  最优吸附温区: {result.schema['optimal_temperature']} K")
```

**输出**：
```
氢气吸附位点：
  Zn4O 簇: -7.25 kJ/mol
    位置: 金属簇中心
    配位数: 4
    吸附类型: 物理吸附 + 弱化学吸附
  苯环上方: -4.12 kJ/mol
    位置: 配体中心
    配位数: 6
    吸附类型: 物理吸附 (π-π 相互作用)
  
吸附能分析：
  最强吸附位: -7.25 kJ/mol
  平均吸附能: -5.18 kJ/mol
  最优吸附温区: 77-200 K
```

### 2.3 等温线模型对比

```python
from math_anything import MathDiffer

# Langmuir vs BET 模型对比
langmuir = extract("lammps", {"isotherm_model": "langmuir"})
bet = extract("lammps", {"isotherm_model": "bet"})

differ = MathDiffer()
report = differ.diff(langmuir.schema, bet.schema)

print("吸附模型对比：")
print("\nLangmuir 模型假设：")
print("  - 单层吸附")
print("  - 表面均匀")
print("  - 吸附热恒定")

print("\nBET 模型假设：")
print("  - 多层吸附")
print("  - 第一层吸附热不同")
print("  - 适用于相对压力 0.05-0.35")

print(f"\n{report.to_text()}")
```

---

## 第三章：氢气扩散动力学（第 8-10 天）

### 3.1 分子动力学模拟

陈雪计算氢气在 MOF 孔道中的扩散系数：

```python
from math_anything import extract

result = extract("lammps", {
    "ensemble": "nvt",
    "temperature": 300,
    "timestep": 1.0,        # fs
    "run": 10000000,        # 10 ns
    "dump": "trajectory.xyz",
    "diffusion": "msd",
})

print("扩散计算参数：")
print(f"  温度: {result.schema['temperature']} K")
print(f"  时间步长: {result.schema['timestep']} fs")
print(f"  总模拟时间: {result.schema['total_time']:.2f} ns")

print("\n均方位移 (MSD) 分析：")
print("  MSD(t) = <|r(t) - r(0)|²> = 6Dt")
print(f"  线性拟合 R²: {result.schema['msd']['r_squared']:.4f}")
```

**输出**：
```
扩散计算参数：
  温度: 300 K
  时间步长: 1.0 fs
  总模拟时间: 10.00 ns

均方位移 (MSD) 分析：
  MSD(t) = <|r(t) - r(0)|²> = 6Dt
  线性拟合 R²: 0.9992
```

### 3.2 扩散系数计算

```python
from math_anything import TieredAnalyzer

analyzer = TieredAnalyzer()
result = analyzer.analyze("diffusion_trajectory.xyz", tier=AnalysisTier.ENHANCED)

print("扩散动力学结果：")
print(f"  自扩散系数 D: {result.schema['diffusion']['D']:.2e} m²/s")
print(f"  跳跃频率: {result.schema['diffusion']['hopping_rate']:.2e} s⁻¹")
print(f"  停留时间: {result.schema['diffusion']['residence_time']:.2f} ps")

print("\n活化能计算：")
print("  D = D₀ exp(-E_a/RT)")
print(f"  活化能 E_a: {result.schema['diffusion']['activation_energy']:.2f} kJ/mol")
print(f"  指前因子 D₀: {result.schema['diffusion']['D0']:.2e} m²/s")
```

**输出**：
```
扩散动力学结果：
  自扩散系数 D: 2.35×10⁻⁹ m²/s
  跳跃频率: 1.25×10¹¹ s⁻¹
  停留时间: 8.00 ps

活化能计算：
  D = D₀ exp(-E_a/RT)
  活化能 E_a: 12.56 kJ/mol
  指前因子 D₀: 4.18×10⁻⁷ m²/s
```

### 3.3 扩散机理分析

```python
from math_anything import PropositionGenerator, TaskType

generator = PropositionGenerator()

theorem = generator.generate(
    engine="lammps",
    parameters={
        "system": "H2_in_MOF",
        "temperature": 300,
        "loading": 2.5,  # wt%
        "mechanism": "knudsen_diffusion",
    },
    task_type=TaskType.WELL_POSEDNESS
)

print(theorem)
```

**输出**：
```
定理（MOF 中氢气扩散的适定性）：
  考虑氢分子在 MOF 孔道中的扩散过程：
    dr/dt = -D∇c + √(2D)ξ(t)
  
  其中：
    - D = 2.35×10⁻⁹ m²/s 为自扩散系数
    - ξ(t) 为高斯白噪声（随机力）
    - 孔径 12.8 Å >> H₂ 动力学直径 2.89 Å
  
  扩散机理：
    - Knudsen 扩散主导（孔径 < 分子平均自由程）
    - 孔道连通性允许 3D 扩散路径
    - 吸附-脱附平衡时间 << 扩散时间尺度
  
  对于 300K、2.5 wt% 载量：
    - 氢分子主要以气相形式存在
    - 吸附相分数: θ = K_L·P/(1+K_L·P) = 0.23
    - 有效扩散系数: D_eff = D_gas·(1-θ) + D_ads·θ
```

---

## 第四章：多尺度建模（第 11-12 天）

### 4.1 DFT 计算吸附能

陈雪用第一性原理精确计算吸附能：

```python
from math_anything import extract

result = extract("vasp", {
    "calculation": "relax",
    "encut": 520,
    "kpoints": "3 3 3",
    "functional": "pbe",
    "vdw_correction": "dft-d3",
    "adsorption": {
        "adsorbate": "H2",
        "site": "Zn4O",
        "coverage": 0.25,
    }
})

print("吸附能计算：")
print("  E_ads = E(MOF+H2) - E(MOF) - E(H2)")
print(f"  吸附能: {result.schema['adsorption']['energy']:.3f} eV")
print(f"  对应: {result.schema['adsorption']['energy_kjmol']:.2f} kJ/mol")

print("\n电子结构分析：")
print(f"  电荷转移: {result.schema['charge_transfer']:.3f} e")
print(f"  重叠布居: {result.schema['overlap_population']:.3f}")
```

**输出**：
```
吸附能计算：
  E_ads = E(MOF+H2) - E(MOF) - E(H2)
  吸附能: -0.185 eV
  对应: -17.85 kJ/mol

电子结构分析：
  电荷转移: 0.023 e
  重叠布居: 0.089
```

### 4.2 跨尺度参数传递

```python
from math_anything import CrossEngineSession

session = CrossEngineSession()

# DFT 模型（量子尺度）
session.add_model("dft_adsorption", {
    "engine": "vasp",
    "scale": "quantum",
    "adsorption_energy": -17.85,  # kJ/mol
    "charge_transfer": 0.023,
})

# GCMC 模型（统计力学尺度）
session.add_model("gcmc_adsorption", {
    "engine": "lammps",
    "scale": "statistical",
    "temperature": 77,
    "pressures": [1, 10, 50, 100],
})

# MD 模型（动力学尺度）
session.add_model("md_diffusion", {
    "engine": "lammps",
    "scale": "atomistic",
    "temperature": 300,
    "diffusion_coefficient": 2.35e-9,
})

# 建立耦合
coupling = session.add_interface(
    "dft_adsorption", "gcmc_adsorption",
    coupling_type="adsorption_isotherm",
    shared_variables=["adsorption_energy", "temperature"]
)
```

---

## 第五章：机器学习预测（第 13-14 天）

### 5.1 结构-性能关系

陈雪建立 ML 模型预测新 MOFs 的储氢性能：

```python
from math_anything import MLArchitectureRecommender

recommender = MLArchitectureRecommender()

recommendation = recommender.recommend(
    schema=result.schema,
    task="hydrogen_storage_prediction",
    available_data=500,  # 500 个 MOFs
    features=["surface_area", "pore_volume", "pore_diameter", "metal_type"]
)

print(f"推荐 ML 架构: {recommendation['architecture']}")
print(f"原因: {recommendation['reasoning']}")
```

**输出**：
```
推荐 ML 架构: Graph Neural Network (GNN) - Crystal Graph Convolution
原因:
  - MOF 结构天然是图（原子为节点，键为边）
  - CGCNN 保持晶体周期性对称性
  - 可以学习局域化学环境到宏观性质的映射
  - 500 个样本足够训练中等规模 GNN
  - 预测目标: 77K, 100 bar 下储氢量 (wt%)
```

### 5.2 高通量筛选

```python
def predict_storage_capacity(mof_features):
    """预测 MOF 的储氢性能"""
    model = load_pretrained_model("cgcnn_hydrogen")
    
    prediction = model.predict(mof_features)
    
    return {
        "77K_100bar": prediction[0],  # wt%
        "298K_100bar": prediction[1],  # wt%
        "delivery_capacity": prediction[0] - prediction[2],  # 5-100 bar
    }

# 筛选高性能材料
targets = load_mof_database("hMOF")
high_performance = []

for mof in targets:
    pred = predict_storage_capacity(mof.features)
    if pred["77K_100bar"] > 5.0 and pred["delivery_capacity"] > 4.0:
        high_performance.append((mof, pred))

print(f"筛选出 {len(high_performance)} 个高性能 MOFs")
```

### 5.3 生成项目报告

```python
from math_anything import generate_report

report = generate_report({
    "structure_analysis": result.schema,
    "adsorption_isotherm": gcmc_result.schema,
    "diffusion_kinetics": md_result.schema,
    "ml_prediction": recommendation,
}, format="pdf")

with open("MOF5_hydrogen_storage_report.pdf", "wb") as f:
    f.write(report)
```

---

## 储氢材料专用技巧

### 技巧 1：美国能源部目标检查

```python
def check_doe_targets(result):
    """检查是否满足 DOE 储氢目标"""
    targets = {
        "2025_system": 5.5,      # wt%
        "2030_system": 8.0,      # wt%
        "ultimate_system": 10.0,  # wt%
    }
    
    capacity = result.schema["storage_capacity"]["77K_100bar"]
    
    for target, value in targets.items():
        if capacity >= value:
            print(f"✓ 达到 DOE {target} 目标 ({value} wt%)")
        else:
            gap = value - capacity
            print(f"✗ 距离 DOE {target} 目标差 {gap:.2f} wt%")
```

### 技巧 2：储氢容量单位转换

```python
def convert_storage_units(value, from_unit, to_unit):
    """储氢容量单位转换"""
    conversions = {
        "wt_to_vol": 0.0899,  # 1 wt% = 0.0899 kg H2/m³ tank
        "wt_to_g_l": 8.99,    # 1 wt% = 8.99 g H2/L tank
    }
    
    if from_unit == "wt" and to_unit == "g_l":
        return value * 8.99
    elif from_unit == "wt" and to_unit == "kg_m3":
        return value * 0.899
```

---

## 总结：陈雪的研究成果

14 天后，陈雪完成了：

1. ✅ MOF 结构拓扑分析（孔隙率、比表面积）
2. ✅ GCMC 吸附等温线计算（77K, 298K）
3. ✅ MD 扩散动力学模拟（扩散系数、活化能）
4. ✅ DFT 吸附能计算（精确到 kJ/mol）
5. ✅ GNN 模型预测新材料储氢性能

**研究成果**：
- MOF-5 在 77K、100 bar 下储氢量: 4.52 wt%
- 氢气扩散系数: 2.35×10⁻⁹ m²/s (300K)
- 筛选出 15 个潜在高性能 MOFs

**发表论文**：基于 Math Anything 提取的数据发表在《ACS Applied Materials & Interfaces》
