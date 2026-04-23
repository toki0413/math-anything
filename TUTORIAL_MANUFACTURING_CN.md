# Math Anything 教程：材料加工与制造仿真

> 跟随制造工程师张伟的 14 天学习历程，掌握焊接和增材制造工艺仿真

---

## 人物背景

**张伟**，某重工企业高级仿真工程师，专注于焊接残余应力和金属 3D 打印工艺优化。

**背景**：
- 硕士毕业于材料加工工程专业
- 熟练使用 Abaqus 和 ANSYS 进行热-力耦合分析
- 负责大型结构件的焊接工艺评定
- 需要验证打印件的质量和性能

**目标**：用 Math Anything 提取制造仿真的数学结构，建立标准化的工艺仿真流程。

---

## 第一章：焊接热传导分析（第 1-3 天）

### 1.1 首次焊接仿真提取

张伟分析厚板对接焊的温度场：

```python
from math_anything import MathAnything

ma = MathAnything()

# 从 Abaqus 输入文件提取
result = ma.extract_file("abaqus", "welding_thermal.inp")

print("控制方程：")
print(result.schema["mathematical_structure"]["canonical_form"])
# 输出: ρc_p ∂T/∂t = ∇·(k∇T) + Q(x,t)

print("\n热源模型：")
print(result.schema["heat_source"]["model"])
# 输出: 双椭球 (Goldak) 热源模型
```

### 1.2 热源参数分析

```python
from math_anything import extract

# 分析焊接工艺参数
result = extract("abaqus", {
    "weld_type": "double_ellipsoid",
    "heat_input": 1500,  # W
    "voltage": 28,       # V
    "current": 250,      # A
    "velocity": 5,       # mm/s
    "efficiency": 0.85,
})

print("热源几何参数：")
a = result.schema["heat_source"]["a"]  # 前半轴
b = result.schema["heat_source"]["b"]  # 宽度
c1 = result.schema["heat_source"]["c1"]  # 后半轴前
c2 = result.schema["heat_source"]["c2"]  # 后半轴后

print(f"  前半轴 a = {a:.2f} mm")
print(f"  宽度 b = {b:.2f} mm")
print(f"  后半轴 c1 = {c1:.2f} mm, c2 = {c2:.2f} mm")

print("\n热输入验证：")
q = result.schema["process"]["heat_input_calculated"]
print(f"  计算热输入: {q:.2f} kJ/mm")
print(f"  ISO 标准: 必须 < 1.5 kJ/mm 对于高强钢")
```

**输出**：
```
热源几何参数：
  前半轴 a = 3.50 mm
  宽度 b = 4.20 mm
  后半轴 c1 = 4.00 mm, c2 = 8.00 mm

热输入验证：
  计算热输入: 1.19 kJ/mm
  ✓ 满足 ISO 标准 (< 1.5 kJ/mm)
```

### 1.3 材料相变分析

```python
# 提取材料模型
result = ma.extract_file("abaqus", "material_phase.inp")

print("相变动力学：")
print("  Johnson-Mehl-Avrami-Kolmogorov (JMAK) 模型:")
print(f"    X(t) = 1 - exp(-kt^n)")
print(f"    k = {result.schema['phase']['k']:.2e}")
print(f"    n = {result.schema['phase']['n']:.2f}")

print("\n连续冷却转变 (CCT) 曲线:")
for phase in result.schema["phases"]:
    print(f"  {phase['name']}: T_start = {phase['t_start']}°C, "
          f"T_finish = {phase['t_finish']}°C")
```

---

## 第二章：焊接残余应力（第 4-6 天）

### 2.1 热-力耦合分析

张伟进行热-弹塑性分析：

```python
from math_anything import extract

result = extract("abaqus", {
    "analysis_type": "thermomechanical",
    "coupling": "sequential",
    "thermal_step": "transient",
    "mechanical_step": "static",
    "nlgeom": True,  # 几何非线性
})

print("热-力耦合框架：")
print("  热分析: ρc_p ∂T/∂t = ∇·(k∇T) + Q")
print("  结构分析: ∇·σ + f = 0")
print("  本构关系: σ = C(ε - ε^th - ε^pl)")

print("\n耦合策略：")
print(f"  类型: {result.schema['coupling']['type']}")
print(f"  时间步长: Δt = {result.schema['time']['dt']:.2f} s")
print(f"  最大温度变化: ΔT_max = {result.schema['thermal']['max_delta_t']}°C")
```

### 2.2 残余应力提取

```python
result = ma.extract_file("abaqus", "residual_stress.odb")

print("残余应力分布：")
print(f"  纵向应力 σ_x 峰值: {result.schema['stress']['longitudinal_max']:.2f} MPa")
print(f"  横向应力 σ_y 峰值: {result.schema['stress']['transverse_max']:.2f} MPa")
print(f"  von Mises 应力峰值: {result.schema['stress']['von_mises_max']:.2f} MPa")

print("\n应力分类：")
if result.schema["stress"]["von_mises_max"] > 0.8 * result.schema["material"]["yield"]:
    print("  ⚠ 高残余应力区域 - 需要后热处理 (PWHT)")
else:
    print("  ✓ 残余应力在可接受范围内")
```

### 2.3 变形预测

```python
print("焊接变形：")
print(f"  角变形: {result.schema['deformation']['angular']:.3f}°")
print(f"  纵向收缩: {result.schema['deformation']['longitudinal']:.2f} mm")
print(f"  横向收缩: {result.schema['deformation']['transverse']:.2f} mm")

print("\n变形控制建议：")
if abs(result.schema['deformation']['angular']) > 3:
    print("  - 使用夹具约束")
    print("  - 采用分段退焊")
    print("  - 增加反变形量")
```

---

## 第三章：增材制造仿真（第 7-10 天）

### 3.1 金属 3D 打印分析

张伟分析选区激光熔化（SLM）过程：

```python
from math_anything import TieredAnalyzer, AnalysisTier

analyzer = TieredAnalyzer()

rec = analyzer.get_recommendation("slm_build.inp")
print(f"推荐分析级别: {rec.recommended_tier}")
print(f"复杂度评分: {rec.complexity_score.total}/100")
```

**输出**：
```
推荐分析级别: COMPLETE
复杂度评分: 90/100
原因: [
    "瞬态多道次激光扫描",
    "粉末-实体相变",
    "复杂热循环历史",
    "微尺度熔池流体流动"
]
```

### 3.2 熔池尺度分析

```python
result = analyzer.analyze("slm_build.inp", tier=AnalysisTier.COMPLETE)

print("熔池物理：")
print("  能量守恒: ρc_p(∂T/∂t + u·∇T) = ∇·(k∇T) + Q_laser")
print("  Navier-Stokes: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f_buoyancy")

print("\n熔池几何：")
print(f"  深度: {result.manifold_info.melt_pool_depth:.2f} μm")
print(f"  宽度: {result.manifold_info.melt_pool_width:.2f} μm")
print(f"  长宽比: {result.manifold_info.aspect_ratio:.2f}")

print("\nMarangoni 对流：")
print(f"  表面张力梯度: ∂σ/∂T = {result.manifold_info.dsigma_dt:.2e} N/(m·K)")
print(f"  最大流速: {result.manifold_info.max_velocity:.2f} m/s")
```

### 3.3 微观组织预测

```python
from math_anything import PropositionGenerator, TaskType

generator = PropositionGenerator()

theorem = generator.generate(
    engine="abaqus",
    parameters={
        "process": "slm",
        "material": "inconel718",
        "power": 200,  # W
        "speed": 800,  # mm/s
        "hatch": 120,  # μm
    },
    task_type=TaskType.WELL_POSEDNESS
)

print(theorem)
```

**输出**：
```
定理（SLM 凝固微观组织）：
  考虑熔池凝固过程中的晶粒生长：
    ∂T/∂t + V_solidification · ∇T = α∇²T
  
  温度梯度 G 和凝固速率 R 决定微观组织：
    G × R = 冷却速率 (K/s)
    G/√R = 控制柱状晶→等轴晶转变
  
  柱状晶生长条件：
    G/√R > (G/√R)_critical ≈ 1.2 × 10^6 K^(1/2)/m^(1/2)/s^(1/2)
  
  对于 Inconel 718：
    熔池边界处 G = 1.5 × 10^6 K/m, R = 0.5 m/s
    G/√R = 2.1 × 10^6 > 临界值
    → 形成柱状晶组织，外延生长
  
  胞晶间距预测：
    λ = A(G × V)^(-n), n ≈ 0.3-0.5
    估算值: λ ≈ 0.5-1.0 μm
```

---

## 第四章：工艺优化（第 11-12 天）

### 4.1 多道焊优化

张伟优化焊接顺序：

```python
from math_anything import CrossEngineSession

session = CrossEngineSession()

# 不同焊接顺序
session.add_model("sequence_a", {
    "weld_order": [1, 2, 3, 4],  # 顺序焊接
    "interpass_temp": 150,
})

session.add_model("sequence_b", {
    "weld_order": [1, 3, 2, 4],  # 跳焊
    "interpass_temp": 150,
})

# 对比残余应力
checks = session.check_consistency()
for check in checks:
    print(f"{'✓' if check.passed else '✗'} {check.name}: {check.message}")
```

### 4.2 打印参数优化

```python
# 参数敏感性分析
parameters = {
    "laser_power": [150, 200, 250],  # W
    "scan_speed": [600, 800, 1000],  # mm/s
    "layer_thickness": [30, 40, 50],  # μm
}

for power in parameters["laser_power"]:
    for speed in parameters["scan_speed"]:
        energy_density = power / (speed * 0.12)  # J/mm^3
        print(f"功率 {power}W, 速度 {speed}mm/s: "
              f"能量密度 = {energy_density:.2f} J/mm³")
        
        if 50 < energy_density < 100:
            print("  ✓ 适宜范围")
        elif energy_density < 50:
            print("  ✗ 未熔合风险")
        else:
            print("  ✗ 球化/气孔风险")
```

---

## 第五章：完整工艺评定（第 13-14 天）

### 5.1 WPS/PQR 文档生成

```python
from math_anything import generate_report

wps_data = {
    "welding_process": result.schema["process"],
    "heat_input": result.schema["thermal"],
    "residual_stress": result.schema["stress"],
    "distortion": result.schema["deformation"],
    "pwht_recommendation": result.schema["heat_treatment"],
}

report = generate_report(wps_data, format="pdf")

with open("WPS_PQR_Document.pdf", "wb") as f:
    f.write(report)
```

### 5.2 质量检验标准

```python
def check_compliance(result, standard="ASME_IX"):
    """检查是否符合焊接标准"""
    checks = []
    
    # 热输入检查
    hi = result.schema["process"]["heat_input_calculated"]
    if hi < 1.5:
        checks.append(("✓", "热输入", f"{hi:.2f} kJ/mm < 1.5"))
    else:
        checks.append(("✗", "热输入", f"{hi:.2f} kJ/mm 超标"))
    
    # 层间温度
    ipt = result.schema["process"].get("interpass_temp", 0)
    if ipt < 250:
        checks.append(("✓", "层间温度", f"{ipt}°C < 250°C"))
    else:
        checks.append(("✗", "层间温度", f"{ipt}°C 过高"))
    
    # 残余应力
    rs = result.schema["stress"]["von_mises_max"]
    ys = result.schema["material"]["yield"]
    if rs < 0.8 * ys:
        checks.append(("✓", "残余应力", f"{rs/ys:.1%} < 80% 屈服强度"))
    else:
        checks.append(("⚠", "残余应力", "建议 PWHT"))
    
    return checks
```

---

## 制造仿真专用技巧

### 技巧 1：热输入计算器

```python
def calculate_heat_input(voltage, current, velocity, efficiency=0.85):
    """计算焊接热输入 (kJ/mm)"""
    power = voltage * current  # W
    travel_speed = velocity / 1000  # m/s
    
    q = (power * efficiency) / (travel_speed * 1000)  # J/mm
    return q / 1000  # kJ/mm

# 示例
hi = calculate_heat_input(voltage=28, current=250, velocity=5)
print(f"热输入: {hi:.3f} kJ/mm")
```

### 技巧 2：冷却时间预测

```python
def calculate_cooling_time(t_800, t_500, thickness):
    """计算 t8/5 冷却时间"""
    # 根据厚度和热输入估算
    return (t_800 - t_500) * (thickness / 20) ** 0.5

# 预测硬度
if t_85 < 5:  # 秒
    print("快速冷却 → 高硬度，马氏体风险")
elif t_85 > 20:
    print("慢速冷却 → 软化和晶粒长大")
```

---

## 总结：张伟的学习成果

14 天后，张伟掌握了：

1. ✅ 焊接热传导和残余应力分析提取
2. ✅ 增材制造熔池物理建模
3. ✅ 工艺参数优化和敏感性分析
4. ✅ 自动生成 WPS/PQR 文档
5. ✅ 质量检验标准符合性检查

**影响**：工艺评定文档编写时间减少 70%，产品质量一致性显著提高。
