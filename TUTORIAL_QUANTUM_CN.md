# Math Anything 教程：量子化学计算

> 跟随量子化学博士生王芳的 14 天学习历程，掌握多尺度波函数分析

---

## 人物背景

**王芳**，物理化学专业博三学生，研究方向为激发态电子结构和光催化机理。

**背景**：
- 精通 Gaussian、ORCA 进行量子化学计算
- 使用 Multiwfn 进行波函数分析
- 需要理解复杂体系的电子相关效应
- 希望用 AI 辅助分析激发态和电子跃迁

**目标**：用 Math Anything 提取量子化学计算的数学结构，让 AI 理解电子结构计算。

---

## 第一章：基础 DFT 计算分析（第 1-3 天）

### 1.1 首次提取分子计算

王芳分析一个光催化剂的基态电子结构：

```python
from math_anything import MathAnything

ma = MathAnything()

# 从 Gaussian 输出提取
result = ma.extract_file("gaussian", "catalyst.log")

print("数学结构：")
print(result.schema["mathematical_structure"]["canonical_form"])
# 输出: F[ρ] = T[ρ] + ∫v(r)ρ(r)dr + E_H[ρ] + E_xc[ρ]

print("\nKohn-Sham 方程：")
print(result.schema["equations"]["ks_equations"])
# 输出: [-½∇² + v_eff[n](r)]φᵢ(r) = εᵢφᵢ(r)
```

### 1.2 泛函和基组分析

```python
from math_anything import extract

# 分析计算设置
result = extract("gaussian", {
    "method": "wb97xd",
    "basis": "def2tzvp",
    "scf_convergence": "tight",
    "integral_accuracy": "ultrafine",
})

print("交换-相关泛函：")
print(f"  类型: {result.schema['xc_functional']['type']}")
print(f"  长程修正: {result.schema['xc_functional']['range_separated']}")
print(f"  色散校正: {result.schema['xc_functional']['dispersion']}")

print("\n基组质量检查：")
basis_checks = result.schema["basis_set"]["quality_metrics"]
for check in basis_checks:
    status = "✓" if check["passed"] else "✗"
    print(f"  {status} {check['name']}: {check['value']}")
```

**输出**：
```
交换-相关泛函：
  类型: hybrid_GGA
  长程修正: True (ωB97X-D)
  色散校正: D3(BJ)

基组质量检查：
  ✓ 完备性: def2-TZVP 是三重zeta极化基组
  ✓ 扩散函数: 没有 (适合基态，激发态需要)
  ! 警告: 对于阴离子建议加 diffuse 函数
```

---

## 第二章：波函数分析（第 4-7 天）

### 2.1 用 Multiwfn 提取拓扑分析

王芳用 Multiwfn 分析电子密度拓扑：

```python
from math_anything import MathAnything

ma = MathAnything()

# 从 Multiwfn 输入提取
result = ma.extract_file("multiwfn", "catalyst.wfn")

print("电子密度拓扑：")
print(f"  临界点数量: {result.schema['topology']['critical_points']}")
print(f"  BCP (键临界点): {result.schema['topology']['bond_critical_points']}")
print(f"  RCP (环临界点): {result.schema['topology']['ring_critical_points']}")

print("\nQTAIM 分析：")
for atom in result.schema["atoms"][:5]:  # 前5个原子
    print(f"  {atom['symbol']}: 电荷 = {atom['charge']:.3f}, "
          f"|∇ρ| = {atom['rho_gradient']:.4f}")
```

### 2.2 激发态分析（TD-DFT）

```python
# 分析含时 DFT 计算
result = extract("gaussian", {
    "method": "td",
    "functional": "cam-b3lyp",
    "roots": 10,
    "target_state": "singlet",
})

print("激发态信息：")
for i, state in enumerate(result.schema["excited_states"][:5]):
    print(f"  S{i+1}: {state['energy']:.3f} eV, "
          f"f = {state['oscillator_strength']:.4f}")
    print(f"       主导跃迁: {state['dominant_transition']}")

print("\n自然跃迁轨道 (NTO) 分析：")
print(f"  空穴轨道: {result.schema['nto']['hole']}")
print(f"  电子轨道: {result.schema['nto']['particle']}")
print(f"  重叠: {result.schema['nto']['overlap']:.3f}")
```

### 2.3 非共价相互作用（NCI）分析

```python
from math_anything import TieredAnalyzer, AnalysisTier

analyzer = TieredAnalyzer()

# 完整分析包含拓扑信息
result = analyzer.analyze("complex_dimer.wfn", tier=AnalysisTier.PROFESSIONAL)

print("非共价相互作用：")
print(f"  氢键数量: {result.topology_info.hydrogen_bonds}")
print(f"  π-π 堆积: {result.topology_info.pi_stacking}")
print(f"  范德华接触: {result.topology_info.vdw_contacts}")

print("\nMorse 理论分析能量景观：")
print(f"  临界点: {result.morse_info.critical_points}")
print(f"  鞍点: {result.morse_info.saddle_points}")
```

---

## 第三章：多尺度计算（第 8-10 天）

### 3.1 ONIOM 分层计算

王芳做 QM/MM 计算分析蛋白质活性位点：

```python
from math_anything import extract

# ONIOM 分层设置
result = extract("gaussian", {
    "oniom": True,
    "high_level": "B3LYP/6-311G(d,p)",
    "medium_level": "PM6",
    "low_level": "UFF",
    "embedding": "mechanical",
})

print("ONIOM 分层结构：")
print(f"  高层 (QM): {result.schema['oniom']['high_layer_atoms']} 原子")
print(f"  中层 (SE): {result.schema['oniom']['medium_layer_atoms']} 原子")
print(f"  低层 (MM): {result.schema['oniom']['low_layer_atoms']} 原子")

print("\n能量表达式：")
print("  E(Real) = E(QM,High) + E(SE,Medium) + E(MM,Low)")
print("  E(Model) = E(QM,High) + E(SE,Medium) + E(MM,Low)")
print("  E(ONIOM) = E(Real) - E(Model) + E(High-Level,Model)")
```

### 3.2 跨尺度分析

```python
from math_anything import CrossEngineSession

session = CrossEngineSession()

# 量子力学模型（活性位点）
session.add_model("active_site_qm", {
    "engine": "gaussian",
    "theory": "dft",
    "functional": "b3lyp",
    "basis": "6-311g_d_p",
    "scale": "electronic",
})

# 半经验模型（蛋白质环境）
session.add_model("protein_se", {
    "engine": "gaussian",
    "theory": "semiempirical",
    "method": "pm6",
    "scale": "atomistic",
})

# 分子力学模型（溶剂环境）
session.add_model("solvent_mm", {
    "engine": "amber",
    "forcefield": "tip3p",
    "scale": "continuum",
})

# 建立耦合
session.add_interface(
    "active_site_qm", "protein_se",
    coupling_type="electrostatic_embedding",
)
```

---

## 第四章：数学命题生成（第 11-12 天）

### 4.1 变分原理定理

王芳生成量子化学基础定理：

```python
from math_anything import PropositionGenerator, TaskType

generator = PropositionGenerator()

theorem = generator.generate(
    engine="gaussian",
    parameters={
        "theory": "dft",
        "functional": "b3lyp",
        "basis": "cc-pvtz",
    },
    task_type=TaskType.WELL_POSEDNESS
)

print(theorem)
```

**输出**：
```
定理（Kohn-Sham DFT 的变分原理）：
  考虑 N 电子体系的基态能量泛函：
    E[ρ] = T_s[ρ] + ∫v_ext(r)ρ(r)dr + E_H[ρ] + E_xc[ρ]
  
  其中：
    - T_s[ρ]: 无相互作用动能
    - v_ext(r): 外势（核吸引 + 外场）
    - E_H[ρ]: Hartree 库仑能
    - E_xc[ρ]: 交换-相关能（采用 B3LYP 近似）
  
  根据 Hohenberg-Kohn 第一定理，基态能量满足变分原理：
    E₀ = min_ρ {E[ρ] | ∫ρ(r)dr = N, ρ(r) ≥ 0}
  
  Kohn-Sham 方程通过自洽迭代求解：
    [-½∇² + v_eff[ρ](r)]φᵢ = εᵢφᵢ
    ρ(r) = Σᵢ |φᵢ(r)|²
  
  收敛条件：
    |E^(n+1) - E^(n)| < 10⁻⁸ Hartree
    max|ρ^(n+1) - ρ^(n)| < 10⁻⁶
```

### 4.2 激发态收敛定理

```python
# TD-DFT 激发态定理
theorem = generator.generate(
    engine="gaussian",
    parameters={
        "method": "td",
        "roots": 5,
        "triplet": False,
    },
    task_type=TaskType.CONVERGENCE
)
```

---

## 第五章：完整项目（第 13-14 天）

### 5.1 光催化机理研究

王芳完成一个完整的光催化项目：

```python
from math_anything import MathAnything, generate_report

ma = MathAnything()

# 基态优化
ground_state = ma.extract_file("gaussian", "ground_state.log")

# 激发态计算
excited_state = ma.extract_file("gaussian", "excited_state.log")

# 瞬态吸收
transient = ma.extract_file("gaussian", "transient.log")

# 生成论文支撑信息
report = generate_report({
    "ground_state": ground_state.schema,
    "excited_states": excited_state.schema,
    "transient_absorption": transient.schema,
}, format="latex")

with open("catalysis_si.tex", "w") as f:
    f.write(report)
```

---

## 量子化学专用技巧

### 技巧 1：基组收敛检查

```python
def check_basis_convergence(results):
    """检查基组收敛性"""
    energies = []
    for basis in ["sto-3g", "6-31g", "6-311g", "cc-pvdz", "cc-pvtz"]:
        result = extract("gaussian", {"basis": basis})
        energies.append((basis, result.energy))
    
    for i in range(1, len(energies)):
        delta = abs(energies[i][1] - energies[i-1][1])
        print(f"{energies[i-1][0]} → {energies[i][0]}: ΔE = {delta:.6f} Hartree")
```

### 技巧 2：激发态表征

```python
def characterize_excitation(state):
    """表征激发态类型"""
    if state["oscillator_strength"] < 0.01:
        return "暗态 (f < 0.01)"
    elif state["ct_character"] > 0.5:
        return "电荷转移态"
    elif state["pi_pi_star"]:
        return "π→π* 跃迁"
    elif state["n_pi_star"]:
        return "n→π* 跃迁"
    else:
        return "混合跃迁"
```

---

## 总结：王芳的学习成果

14 天后，王芳掌握了：

1. ✅ DFT 计算的数学结构提取（Kohn-Sham 方程）
2. ✅ 波函数拓扑分析（QTAIM、NCI）
3. ✅ 激发态计算（TD-DFT、NTO 分析）
4. ✅ ONIOM QM/MM 多尺度计算
5. ✅ 自动生成论文支撑信息和定理

**影响**：支撑信息撰写时间减少 60%，理论分析更严谨。
