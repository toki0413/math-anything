# PSRN 集成指南

本指南说明如何将 PSRN (Parallel Symbolic Regression Network) 集成到 math-anything 项目中。

## 架构概览

```
math_anything/psrn/
├── __init__.py          # 模块导出
├── symbol_layer.py      # SymbolLayer - 子树复用核心
├── gpu_evaluator.py     # GPUEvaluator - GPU/CPU 并行评估
├── psrn_network.py      # PSRN - 并行符号回归网络
├── token_generator.py   # TokenGenerator - GP/MCTS 子表达式生成
├── pse_engine.py        # PSEEngine - 顶层协调引擎
├── bridge.py            # Bridge - 与现有 EML/GP API 兼容
├── test_psrn.py         # 测试套件
├── demo.py              # 演示脚本
└── INTEGRATION_GUIDE.md # 本文件
```

## 核心设计决策

### 1. 与现有代码的关系

| 现有组件 | PSRN 对应组件 | 关系 |
|---------|-------------|------|
| `ImprovedSymbolicRegression` | `PSRNSymbolicRegression` | 替代/升级 |
| `Node` / `ExprBuilder` | `SymbolLayer` + 表达式字符串 | 保留兼容 |
| `ExpressionSimplifier` | `SymbolLayer._normalize_expr` + DR Mask | 增强 |
| `discover_equation()` | `PSEEngine.discover()` | 替代 |

### 2. 关键集成点

#### API 兼容层 (`bridge.py`)

`PSRNSymbolicRegression` 继承自 `ImprovedSymbolicRegression` 的接口，但完全重写内部实现：

```python
# 原有代码（无需修改）
from math_anything import ImprovedSymbolicRegression
sr = ImprovedSymbolicRegression(population_size=200, generations=100)
best_tree = sr.fit(X, y, variable_names=['x'])

# 替换为 PSRN（只需改类名）
from math_anything.psrn import PSRNSymbolicRegression
sr = PSRNSymbolicRegression(n_layers=2, max_iterations=5)
best_tree = sr.fit(X, y, variable_names=['x'])
```

#### MathAnything API 扩展

可以在 `api.py` 的 `MathAnything.discover()` 方法中添加 `use_psrn` 参数：

```python
def discover(self, X, y, variable_names=None, use_psrn=False, **kwargs):
    if use_psrn:
        from .psrn import PSRNSymbolicRegression
        sr = PSRNSymbolicRegression(**kwargs)
    else:
        sr = ImprovedSymbolicRegression(**kwargs)
    # ... 其余逻辑不变
```

### 3. 性能对比预期

| 指标 | 传统 GP | PSRN | 提升 |
|-----|--------|------|------|
| 评估方式 | 串行独立 | 并行共享子树 | 10-10000x |
| 搜索空间 | 百万级 | 亿级 | 100x |
| 高维问题 | 困难 | 可扩展 | - |
| GPU 支持 | 无 | 有 | 硬件加速 |

## 使用示例

### 基础使用

```python
from math_anything.psrn import PSEEngine, PSEConfig, PSRNConfig

# 配置
config = PSEConfig(
    psrn_config=PSRNConfig(n_layers=2, n_input_slots=3),
    max_iterations=5,
)

# 创建引擎
engine = PSEEngine(config)

# 发现方程
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = X.flatten() ** 2

best_expr, pareto_front = engine.discover(X, y, variable_names=['x'])
print(f"发现: {best_expr}")
```

### 与现有工作流集成

```python
from math_anything import MathAnything
from math_anything.psrn import PSEConfig, PSEEngine, PSRNConfig

ma = MathAnything()

# 从 VASP 提取数学结构
result = ma.extract_file("vasp", "INCAR")

# 如果有仿真输出数据，可以用 PSRN 发现经验关系
# 例如：发现能量与晶格常数的关系
lattice_params = np.array([...])
energies = np.array([...])

config = PSEConfig(psrn_config=PSRNConfig(n_layers=2))
engine = PSEEngine(config)
best_expr, _ = engine.discover(
    lattice_params.reshape(-1, 1),
    energies,
    variable_names=['a'],
)
print(f"E(a) = {best_expr}")
```

## 扩展方向

### 1. 添加新的算子

在 `SymbolConfig` 中扩展算子列表：

```python
config = SymbolConfig(
    unary_ops=["identity", "neg", "sin", "cos", "exp", "log", "tanh"],
    binary_triangled_ops=["add", "mul", "pow"],
)
```

### 2. 自定义 Token Generator

继承 `TokenGenerator` 基类：

```python
class MyTokenGenerator(TokenGenerator):
    def generate(self, X, y, variable_names, current_tokens=None, reward_history=None):
        # 自定义生成逻辑
        return token_exprs, token_values
```

### 3. 物理约束集成

在 `PSEEngine._compute_reward()` 中加入量纲分析：

```python
def _compute_reward(self, mse, complexity, expr):
    base_reward = super()._compute_reward(mse, complexity)
    # 量纲一致性奖励
    if self._check_dimensional_consistency(expr):
        base_reward *= 1.2
    return base_reward
```

## 测试

运行测试套件：

```bash
cd math-anything/math-anything/core
python -m pytest math_anything/psrn/test_psrn.py -v
```

运行演示：

```bash
python -m math_anything.psrn.demo
```

## 注意事项

1. **GPU 依赖**: GPU 加速需要 CuPy 或 Numba，未安装时自动回退到 CPU
2. **内存限制**: 深层 PSRN 可能消耗大量 GPU 内存，使用 `use_dr_mask=True` 缓解
3. **表达式解析**: 当前 `bridge.py` 中的字符串到 Node 转换是简化实现，生产环境建议添加完整解析器
