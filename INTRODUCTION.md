# Math Anything 项目介绍

## 一句话概括

**Math Anything** 是计算材料科学的数学语义层——让 AI 真正理解你在算什么，而不只是读取参数值。

---

## 完整功能介绍

### 1. 核心能力：从输入文件到数学结构

传统方式：
```
你看到：ENCUT = 520
AI 看到：ENCUT = 520（只是一个数字）
```

Math Anything 方式：
```
AI 看到：ENCUT = 520
       → 约束：ENCUT > 0 ✓
       → 约束：ENCUT > max(ENMAX) ✓
       → 关系：ENCUT = ENMAX × factor(PREC)
       → 语义："平面波基组展开的截断能"
       → 数学：H[n]ψ = εψ（非线性特征值问题）
```

### 2. 支持的计算引擎

| 引擎 | 数学问题 | AI 理解 |
|------|---------|---------|
| **VASP** | H[n]ψ = εψ（非线性特征值） | "需要 SCF 迭代，V_eff 依赖密度" |
| **LAMMPS** | m d²r/dt² = F(r)（初值 ODE） | "时间积分，非迭代求解" |
| **Abaqus** | ∇·σ + f = 0（边值问题） | "FEM 求解平衡方程" |
| **ANSYS** | Kφ = λMφ（特征值） | "求解固有频率" |
| **COMSOL** | 耦合 PDE 系统 | "多种物理场耦合" |
| **GROMACS** | 随机 ODE | "生物分子动力学含约束" |
| **Multiwfn** | ∇ρ(r) = 0（拓扑） | "寻找密度中的临界点" |

### 3. 五级分析体系（Tiered Analysis）

根据系统复杂度自动选择分析深度：

```
Level 1 (Basic)      → 快速筛选，简单特征提取
Level 2 (Enhanced)   → 详细参数和验证
Level 3 (Professional) → + 拓扑分析（Betti 数）
Level 4 (Advanced)   → + 几何方法（辛积分器）
Level 5 (Complete)   → 五层统一框架 + 潜空间加速
```

### 4. 五层统一数学框架

完整分析结合高级数学方法：

1. **辛积分器**：保持能量和相空间结构
2. **约束流形**：有约束系统降维
3. **拓扑分析**：Betti 数识别连通分量、环路、空洞
4. **Morse 理论**：能量景观临界点分析
5. **潜空间**：E(3)-等变神经表示加速

### 5. 数学命题生成

自动形式化定理：

```python
from math_anything import PropositionGenerator

# 生成 MD 适定性定理
theorem = generator.generate(
    engine="lammps",
    parameters={"timestep": 0.5, "run": 80000},
    task_type=TaskType.WELL_POSEDNESS
)
```

**输出**：
```
定理（MD 模拟的适定性）：
  给定 m·r̈ = F(r)，F 是 Lipschitz 连续的，
  且初始条件 r(0) = r₀, ṙ(0) = v₀，
  存在唯一解对于 t ∈ [0, T]。
```

### 6. 机器学习架构推荐

基于提取的数学结构推荐 ML 模型：

```
输入：E(3) 对称性 + 图结构 + 连续滤波
推荐：SchNet（E(3)-等变图神经网络）
理由：消息传递保持旋转不变性
兼容性证明：✓
```

### 7. 安全与隐私

- **零数据上传**：计算完全在本地进行
- **路径安全验证**：防止路径遍历攻击
- **文件大小限制**：防止资源耗尽
- **LLM 只读取抽象数学结构**：原始数据不会离开本地

### 8. 异常处理与类型安全

- 完整的异常层次结构
- Pydantic 模型验证
- 跨平台路径安全
- 详细的错误信息

---

## 使用方式

### 方式一：Python API

```python
from math_anything import MathAnything, TieredAnalyzer, AnalysisTier

# 基础提取
ma = MathAnything()
result = ma.extract_file("vasp", "INCAR")

# 分级分析
analyzer = TieredAnalyzer()
result = analyzer.analyze("simulation.lmp", tier=AnalysisTier.PROFESSIONAL)
```

### 方式二：CLI 工具

```bash
# 交互式 REPL
math-anything repl

# 一次性提取
math-anything extract vasp INCAR --output schema.json

# 对比两个计算
math-anything diff calc1.json calc2.json
```

### 方式三：AI 工具集成

将 whl 包传给 Claude Code / Cursor / Trae：

```
"请分析这个 LAMMPS 输入文件的数学结构"
→ AI 调用 Math Anything 提取
→ 返回形式化数学描述
→ 生成适定性定理
→ 推荐 ML 架构
```

---

## 安装

```bash
pip install math_anything-1.0.0-py3-none-any.whl
```

依赖：Python ≥3.10，click，pydantic，numpy

---

## 文档与教程

- **README.md**：项目概述
- **QUICK_START.md**：5 分钟快速开始
- **API.md**：完整 API 参考
- **TUTORIAL.md**：材料科学入门（中文）
- **TUTORIAL_EN.md**：计算生物物理（英文）
- **TUTORIAL_AERO_EN.md**：航空航天（英文）
- **TUTORIAL_QUANTUM_CN.md**：量子化学（中文）
- **TUTORIAL_FLUID_EN.md**：计算流体力学（英文）
- **TUTORIAL_MANUFACTURING_CN.md**：材料加工（中文）
- **TUTORIAL_HYDROGEN_CN.md**：储氢材料（中文）

---

## 隐私说明

**数据安全**：
- 所有计算在本地完成
- LLM 只读取抽象的数学结构（如 "H[n]ψ = εψ"）
- 原始输入文件内容不会传输给 LLM
- 符合实验室数据安全要求

---

## 反馈

欢迎大家试用并反馈：
- 使用体验
- 功能需求
- 问题报告
- 新引擎支持请求
