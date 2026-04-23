# Math Anything 快速介绍

## 组内分享版

大家好，我这周做了一个名为 **Math Anything** 的计算材料科学小工具。

### 它能解决什么问题？

做计算模拟时，我们经常遇到这些问题：
- **参数设置困惑**："ENCUT 应该设多少？这个值合理吗？"
- **AI 不理解计算**：给 LLM 看 INCAR 文件，它只能读数字，不懂物理意义
- **ML 选型盲目**：不知道哪种神经网络适合自己的体系
- **论文写作繁琐**：理论部分需要手动整理数学描述

### 核心功能

**1. 数学语义提取**
```
传统方式：
  你看到：ENCUT = 520
  AI 看到：ENCUT = 520（只是一个数字）

Math Anything：
  AI 看到：ENCUT = 520
         → 约束：ENCUT > 0 ✓
         → 关系：ENCUT = ENMAX × factor(PREC)
         → 语义："平面波基组展开的截断能"
         → 数学：H[n]ψ = εψ（非线性特征值问题）
```

**2. 自动验证**
- 检查参数是否满足数学约束（如 SIGMA > 0）
- 发现参数不匹配（如 ISMEAR=-5 与 SIGMA=0.2 冲突）

**3. 分级分析**
根据系统复杂度自动选择分析深度：
- Level 1：快速筛选
- Level 2：详细参数
- Level 3：拓扑分析（Betti 数）
- Level 4：几何方法（辛积分器）
- Level 5：完整数学框架

**4. ML 架构推荐**
基于提取的数学结构推荐模型：
```
输入：E(3) 对称性 + 图结构
推荐：SchNet（E(3)-等变图神经网络）
理由：消息传递保持旋转不变性
兼容性证明：✓
```

**5. 适定性定理生成**
自动生成形式化定理，可直接用于论文：
```
定理（MD 模拟的适定性）：
  给定 m·r̈ = F(r)，F 是 Lipschitz 连续的，
  且初始条件 r(0) = r₀, ṙ(0) = v₀，
  存在唯一解对于 t ∈ [0, T]。
```

### 支持的引擎

| 引擎 | 数学问题 | 示例 |
|------|---------|------|
| VASP | H[n]ψ = εψ | DFT 计算 |
| LAMMPS | m d²r/dt² = F(r) | 分子动力学 |
| Abaqus | ∇·σ + f = 0 | 有限元分析 |
| GROMACS | 随机 ODE | 生物分子模拟 |
| ANSYS/COMSOL/Multiwfn | ... | ... |

### 使用方式

**方式一：Python API**
```python
from math_anything import MathAnything, TieredAnalyzer

ma = MathAnything()
result = ma.extract_file("vasp", "INCAR")

analyzer = TieredAnalyzer()
result = analyzer.analyze("simulation.lmp")
```

**方式二：AI 工具集成**
把 wheel 包传给 Claude Code / Cursor / Trae：
```
"请分析这个 LAMMPS 输入文件的数学结构"
→ AI 自动调用 Math Anything
→ 返回形式化数学描述
→ 生成适定性定理
→ 推荐 ML 架构
```

### 隐私安全

- **计算完全本地**：所有分析在本地完成
- **LLM 只读数学结构**：如 "H[n]ψ = εψ"，不接触原始数据
- **原始数据不离开本地**：符合实验室数据安全要求

### 安装

```bash
pip install math_anything-1.0.0-py3-none-any.whl
```

### 文档

- **README.md**：项目概述
- **QUICK_START.md**：5 分钟快速开始
- **7 个教程**：覆盖材料、生物、航空、量子化学、流体、制造、储氢

### 反馈

欢迎大家试用，有任何问题或需求随时反馈给我！
