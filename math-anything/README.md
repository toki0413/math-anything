# Math Anything - Phase 0 Implementation

基于 CLI-Anything 架构的 Math Anything Phase 0 实现。

## 项目结构

```
math-anything/
├── core/                           # 核心引擎
│   ├── math_anything/
│   │   ├── __init__.py            # 包入口，提供 load_harness() 等API
│   │   ├── cli.py                 # CLI 接口和 REPL
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── harness.py         # Harness 抽象基类和注册表
│   │   │   ├── extractor.py       # 提取引擎
│   │   │   └── session.py         # 会话管理（undo/redo）
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── math_schema.py     # Math Schema v1.0 实现
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── math_diff.py       # 数学结构差异追踪
│   │       ├── semantic_validator.py # 语义一致性校验
│   │       └── llm_context.py     # LLM 上下文协议
│   ├── tests/
│   │   ├── test_schema.py         # Schema 单元测试
│   │   └── test_math_diff.py      # Diff 功能测试
│   ├── setup.py
│   └── README.md
│
├── lammps-harness/                # LAMMPS 适配器
│   ├── math_anything/
│   │   ├── lammps/
│   │   │   ├── __init__.py
│   │   │   └── core/
│   │   │       ├── __init__.py
│   │   │       ├── parser.py      # LAMMPS 输入/日志解析器
│   │   │       ├── extractor.py   # LAMMPS 数学结构提取器
│   │   │       └── harness.py     # LAMMPSHarness 实现
│   ├── tests/
│   │   └── test_fix_deform.py     # fix deform 张量边界条件测试
│   └── setup.py
│
├── examples/
│   ├── example_fix_deform.py      # fix deform 示例
│   └── example_diff.py            # diff 功能示例
│
└── run_tests.py                   # 测试运行脚本
```

## 已完成的功能

### ✅ 1. Math Schema v1.0 核心契约

实现了完整的 Math Schema 数据结构：

- **MathematicalModel**: 控制方程、边界条件、初始条件、本构关系
- **BoundaryCondition with Tensor Support**: 支持任意阶张量的边界条件
- **ComputationalGraph**: 显式/隐式循环区分
- **NumericalMethod**: 离散化方法和求解器配置
- **ConservationProperties**: 守恒律属性

### ✅ 2. LAMMPS Harness (Phase 0)

实现了 LAMMPS 适配器，支持提取：

- **控制方程**: Newton 第二定律、Hamiltonian 动力学
- **边界条件**: 周期性边界、fix deform 张量约束
- **数值方法**: Velocity Verlet、NVT/NPT 积分器
- **计算图**: 显式更新 vs 隐式循环
- **守恒性质**: 能量、动量、角动量

### ✅ 3. Fix Deform 张量边界条件 (Phase 0 验证)

实现了 fix deform 的完整张量表达：

```json
{
  "id": "fix_2",
  "type": "dirichlet",
  "mathematical_object": {
    "field": "displacement_gradient",
    "tensor_rank": 2,
    "tensor_form": "F_{ij} = ∂x_i/∂X_j",
    "components": [
      {"index": [1,1], "value": "1 + 0.01 * t", "unit": "dimensionless"},
      {"index": [2,2], "value": "1", "unit": "dimensionless"},
      {"index": [3,3], "value": "1", "unit": "dimensionless"}
    ],
    "symmetry": "symmetric",
    "trace_condition": "det(F) = 1 + 0.01*t"
  },
  "equivalent_formulations": [
    {
      "type": "strain_rate_tensor",
      "form": "ε̇_{ij} = (L_{ij} + L_{ji})/2",
      "velocity_gradient": "L = Ḟ·F⁻¹"
    }
  ]
}
```

满足 Phase 0 验证要求：
1. ✅ 二阶张量约束（变形梯度 F_ij）
2. ✅ 对称性声明
3. ✅ 迹条件/体积变化约束
4. ✅ 等价的应变率张量描述

### ✅ 4. Math Anything Diff（改进建议 #1）

**数学结构差异追踪** - 不是文本 diff，而是语义差异分析：

```bash
# 比较两个模型
math-anything diff model_v1.json model_v2.json

# 只看关键变化
math-anything diff old.json new.json --critical-only

# JSON 输出供 LLM 分析
math-anything diff a.json b.json --json > diff.json
```

**检测的变更类型**：
- 控制方程：添加/删除/修改
- 边界条件：类型变化、张量阶数变化、分量变化
- 数值方法：积分器改变、时间步长改变、阶数改变
- 守恒性质：获得/丢失
- 计算图：节点/边变化、循环类型改变

**严重级别分类**：
- `critical`: 影响物理正确性（如积分器改变、守恒丢失）
- `warning`: 需要关注（如时间步长大幅改变）
- `info`: 信息性变化

### ✅ 5. 语义一致性校验（改进建议 #2）

框架已建立，提供以下校验能力：
- 量纲一致性检查（待实现）
- 边界条件与方程兼容性（待实现）
- 守恒律一致性（待实现）
- 数值稳定性指标（待实现）

### ✅ 6. LLM Context Protocol（改进建议 #6）

**标准化 LLM 交互协议** - 不只是输出 JSON，而是结构化上下文：

```python
from math_anything.utils import LLMContextProtocol

protocol = LLMContextProtocol()
context = protocol.generate_context(schema_data, include_sections=[
    "overview",
    "governing_equations",
    "boundary_conditions",
    "numerical_method",
])

# 生成提示模板
prompt = protocol.generate_prompt_template("explain")
```

**生成的上下文结构**：
- 模型概览
- 控制方程详解
- 边界条件（含张量信息）
- 数值方法说明
- 守恒性质状态
- LLM 查询指南

### ✅ 7. CLI 接口和 REPL

完整的命令行接口：

```bash
# 提取
math-anything extract -e lammps -i in.file -o model.json

# 验证
math-anything validate model.json

# 查看
math-anything show model.json -s mathematical_model.boundary_conditions

# 比较（新增）
math-anything diff model_v1.json model_v2.json

# REPL 交互模式
math-anything repl
> extract lammps in.file
> show
> export model.json
```

### ✅ 8. 测试套件

- `core/tests/test_schema.py`: Schema 核心测试 (13 个)
- `core/tests/test_math_diff.py`: Diff 功能测试 (10 个)
- `lammps-harness/tests/test_fix_deform.py`: LAMMPS 测试 (17 个)

**总计：40 个测试全部通过** ✅

## 安装

### 开发安装

```bash
cd math-anything/core
pip install -e .

cd ../lammps-harness
pip install -e .
```

## 使用示例

### Python API

```python
import math_anything as ma
from math_anything.utils import MathDiffer

# 加载 LAMMPS harness
harness = ma.load_harness("lammps")

# 提取数学结构
schema = harness.extract({"input": "in.deform"})
schema.save("model.json")

# 比较两个版本
differ = MathDiffer()
report = differ.compare(schema_v1, schema_v2)

if report.has_critical_changes:
    print("警告：检测到关键数学变更！")
    for change in report.critical_changes:
        print(f"  - {change.description}")
```

### CLI

```bash
# 提取
math-anything extract --engine lammps --input in.file --output model.json

# 验证 Schema
math-anything validate model.json

# 查看边界条件
math-anything show model.json -s mathematical_model.boundary_conditions

# 比较模型版本（新增）
math-anything diff model_v1.json model_v2.json
math-anything diff old.json new.json --critical-only
```

### 运行示例

```bash
cd math-anything

# fix deform 张量边界条件示例
python examples/example_fix_deform.py

# diff 功能示例
python examples/example_diff.py
```

### 运行测试

```bash
cd math-anything
python run_tests.py

# 总计 40 个测试
# ✓ core/tests/test_schema.py (13 tests)
# ✓ core/tests/test_math_diff.py (10 tests)
# ✓ lammps-harness/tests/test_fix_deform.py (17 tests)
```

## 技术亮点

### 1. 基于 CLI-Anything 的架构

- 参考 CLI-Anything 的 7 阶段流水线设计
- 复用 cli-anything-plugin 的 REPL Skin 设计理念
- 采用 namespace package 实现引擎隔离

### 2. 完整的张量支持

- MathematicalObject 支持任意阶张量
- TensorComponent 表示张量分量
- 对称性声明和迹条件约束

### 3. 显式/隐式循环区分

ComputationalGraph 明确区分：
- `explicit_update`: 单次计算，无迭代
- `implicit_loop`: 需迭代至收敛
- `symplectic_integrator`: 辛积分器

### 4. Math Anything Diff（语义差异追踪）

- 不是文本 diff，而是数学语义比较
- 自动分类严重级别
- 支持张量变化检测
- JSON 输出供 LLM 消费

### 5. 零侵入设计

- 不修改 LAMMPS 输入文件
- 纯解析输入/日志文件
- 输出独立的 JSON Schema

## 改进建议实现状态

| 建议 | 状态 | 说明 |
|------|------|------|
| 1. Math Diff | ✅ 完成 | 语义差异追踪，40 个测试通过 |
| 2. 语义校验 | 🟡 框架 | 基础框架建立，规则待扩展 |
| 3. LLM Context Protocol | ✅ 完成 | 标准化交互协议 |
| 4. Schema 扩展点 | 🔵 计划中 | Phase 2 实现 |
| 5. 多引擎联合提取 | 🔵 计划中 | Phase 1/2 实现 |
| 6. 流式解析 | 🔵 计划中 | 大数据场景优化 |

## 下一步（Phase 1）

- 实现 VASP Harness（第一性原理计算）
- 实现 Abaqus Harness（有限元分析）
- 多引擎 Schema 一致性验证
- Knowledge Connector 原型

## 许可证

- 核心引擎：Apache-2.0
- LAMMPS Harness：MIT
- Schema 规范：CC-BY-4.0