# Math Anything

计算材料科学的数学语义层。不只是从模拟输入文件中提取参数值，Math Anything 提取的是背后的*方程、约束和关系*，让 LLM 能够推理物理，而不只是读取数字。

## 为什么需要这个

如果你用过 VASP、LAMMPS 或其他模拟软件，你一定深有体会：输入文件里有 `ENCUT = 520`，但这*意味着*什么？够了吗？太多了？它是否满足平面波 DFT 计算的数学约束？

传统的解析器给你数值。Math Anything 给你数学：

```
传统方式:  ENCUT = 520
Math Anything: ENCUT = 520
               约束: ENCUT > 0 ✓
               约束: ENCUT > max(ENMAX) ✓
               关系: ENCUT = ENMAX × factor(PREC)
               语义: "平面波基组展开的截断能"
```

这很重要，因为 LLM（和对代码不熟悉的人）无法验证他们不理解的东西。像 `SIGMA > 0` 这样的符号约束是智能体可以检查的。裸的 `0.05` 则不行。

## 它能做什么

- **提取数学结构**：控制方程、边界条件、本构关系
- **揭示数学本质**：不只是"ENCUT=520"，而是"这是一个需要 SCF 迭代的非线性特征值问题"
- **验证符号约束**：参数在物理/数学上是否一致？
- **跨引擎映射参数**：VASP 的 `ENCUT` ↔ Quantum ESPRESSO 的 `ecutwfc`
- **语义对比计算**：数学上什么改变了，而不只是哪行不同
- **生成数学命题**：形式化存在性、唯一性、稳定性定理
- **提供分级分析**：从快速筛选到完整数学框架的 5 个级别

## 支持的引擎

| 引擎 | 类型 | 数学问题 | LLM 理解 |
|------|------|---------|---------|
| VASP | DFT | H[n]ψ = εψ (非线性特征值) | "需要 SCF 迭代，V_eff 依赖密度" |
| LAMMPS | MD | m d²r/dt² = F(r) (初值 ODE) | "时间积分，非迭代求解" |
| Abaqus | FEM | ∇·σ + f = 0 (边值问题) | "FEM 求解平衡方程" |
| Ansys | FEM | Kφ = λMφ (特征值) | "求解固有频率" |
| COMSOL | 多物理场 | 耦合 PDE 系统 | "多种物理场耦合" |
| GROMACS | 生物分子 MD | 随机 ODE | "生物分子动力学含约束" |
| Multiwfn | 波函数 | ∇ρ(r) = 0 (拓扑) | "寻找密度中的临界点" |

运行提取时，你会得到数学结构、变量依赖关系，以及从物理到数值的完整近似层次结构。

## 快速开始

### 安装

```bash
# 从 Gitee（中国用户，更快）
git clone https://gitee.com/crested-ibis-0413/math-anything.git
cd math-anything
pip install -e .
```

### 命令行使用

```bash
# 交互式 REPL
math-anything repl

# 一次性提取
math-anything extract vasp INCAR POSCAR KPOINTS --output schema.json

# 对比两个计算
math-anything diff calc1.json calc2.json

# 跨引擎映射
math-anything cross vasp_schema.json quantum_espresso
```

### Python API

#### 基础提取

```python
from math_anything import extract, MathAnything

# 简单提取
result = extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
print(result.schema["mathematical_structure"]["canonical_form"])
# 输出: H[n]ψ = εψ

# 文件解析
ma = MathAnything()
result = ma.extract_file("lammps", "in.file")
print(result.to_mermaid())  # 可视化为图表
```

#### 分级分析（新功能）

```python
from math_anything import TieredAnalyzer, AnalysisTier, tiered_analyze

# 根据复杂度自动检测分析级别
result = tiered_analyze("large_simulation.lmp")
print(f"推荐层级: {result.tier.name}")
# 输出: ADVANCED（大型系统、长时间模拟）

# 或指定确切层级
analyzer = TieredAnalyzer()
result = analyzer.analyze("simulation.lmp", tier=AnalysisTier.PROFESSIONAL)

# 只获取推荐不运行
rec = analyzer.get_recommendation("simulation.lmp")
print(f"复杂度评分: {rec.complexity_score.total}")
print(f"预估时间: {rec.estimated_time}")
print(f"适合层级: {[t.name for t in rec.suitable_tiers]}")
```

#### 数学命题生成

```python
from math_anything import PropositionGenerator, MathematicalTask, TaskType

# 从模拟生成数学命题
extractor = PropositionGenerator()
propositions = extractor.generate(
    engine="lammps",
    parameters={"timestep": 0.5, "run": 80000, "fix": "nvt"},
    task_type=TaskType.WELL_POSEDNESS
)

print(propositions)
# 输出: 定理（MD 模拟的适定性）：
#   给定 m·r̈ = F(r)，F 是 Lipschitz 连续的，
#   且初始条件 r(0) = r₀, ṙ(0) = v₀，
#   存在唯一解对于 t ∈ [0, T]。
```

## 分级分析系统

Math Anything 提供 5 级分析深度，根据系统复杂度自动选择：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         分级分析层级                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  第 1 级：基础层      → 快速筛选，简单特征提取                           │
│  第 2 级：增强层      → 详细参数和验证                                   │
│  第 3 级：专业层      → + 拓扑分析（Betti 数）                           │
│  第 4 级：高级层      → + 几何方法（辛积分器）                           │
│  第 5 级：完整层      → 五层统一框架 + 潜空间                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 五层统一框架（第 5 级）

完整分析结合了高级数学方法：

1. **辛积分器**：保持能量和相空间结构（O(Δt²) 误差）
2. **约束流形**：为有约束的系统降维
3. **拓扑分析**：Betti 数识别连通分量、环路、空洞
4. **Morse 理论**：能量景观的临界点分析
5. **潜空间**：E(3)-等变神经表示用于加速

```python
# 第 5 级：完整分析，包含所有方法
result = analyzer.analyze("complex_system.lmp", tier=AnalysisTier.COMPLETE)

print(result.topology_info)    # Betti 数: [1, 0, 0]
print(result.manifold_info)    # 维度、度量、辛结构
print(result.morse_info)       # 能量景观的临界点
print(result.latent_info)      # 推荐编码器、加速估计
```

## 实际用例

### 及早发现错误输入

```
✓ ENCUT > 0
✓ EDIFF > 0
✗ SIGMA > 0  ← SIGMA = -0.2 是无效的！
```

### 理解你在计算什么

```
VASP 不只是"运行 DFT"：
  问题类型: nonlinear_eigenvalue
  标准形式: H[n]ψ = εψ
  变量依赖: V_eff → n → ψ → V_eff（循环）
  → 需要 SCF 迭代
```

### 解密黑盒计算

```
LAMMPS 输入脚本：
  核心问题: initial_value_ode
  近似方法: 经典力学 → 力场 → 截断
  层次结构: 量子 → Born-Oppenheimer → 经典 → 经验
```

### 用物理上下文指导 ML

```
预测形成能的 ML 模型：
  近似: DFT 总能量计算
  缺失: 显式物理约束（对称性、守恒律）

基于 E(3) 对称性的推荐：
  → 使用 SchNet 或 NequIP（E(3)-等变网络）
  → 兼容性证明：消息传递旋转不变
```

### 跨物理尺度对比

```
VASP (DFT):     H[n]ψ = εψ              (量子)
LAMMPS (MD):    m d²r/dt² = F(r)        (经典)
Abaqus (FEM):   ∇·σ + f = 0             (连续介质)
→ 不同数学框架，需要谨慎上尺度
```

## 设计理念

**零侵入**：从不修改你的输入文件。只读取和报告。

**零评判**：不会告诉你"ENCUT=200 是错的"。而是报告"ENCUT=200 超出典型范围 200-800 eV"。决定权在你。

**数学精确性**：用标准形式表达结构。`H[n]ψ = εψ` 对任何物理学家都意味着相同的东西。

**分层复杂度**：从快速检查到深度数学分析，让工作量与问题匹配。

## 文档

- [QUICK_START.md](QUICK_START.md) - 5 分钟快速开始
- [WORKFLOW.md](WORKFLOW.md) - 从文件到数学命题的完整工作流
- [TIERED_SYSTEM.md](TIERED_SYSTEM.md) - 分级分析系统设计
- [UNIFIED_MATH_FRAMEWORK.md](UNIFIED_MATH_FRAMEWORK.md) - 五层统一框架
- [TOPOLOGY_MANIFOLD_ANALYSIS.md](TOPOLOGY_MANIFOLD_ANALYSIS.md) - 拓扑和流形效率分析
- [LATENT_SPACE_ANALYSIS.md](LATENT_SPACE_ANALYSIS.md) - 潜空间加速分析
- [API.md](API.md) - 完整 API 参考

## 致谢

受 [CLI-Anything](https://github.com/fzdwx/cli-anything) 启发，它展示了通过结构化提取让 CLI 工具对 AI 智能体可理解。我们将其从 CLI 语义扩展到数学语义。

EML（Exp-Minus-Log）符号回归实现基于 **Andrzej Odrzywołek** 及其论文 *"All elementary functions from a single binary operator"* (arXiv:2603.21852) 的工作，该论文证明了所有初等函数都可以从单一二元运算符 `eml(x,y) = exp(x) - ln(y)` 构造。

## 许可证

MIT
