# Math Anything Core

数学结构提取层 - 从计算软件中提取与具体材料无关的普适数学结构。

## 概述

Math Anything 是材料科学计算模拟领域的**数学结构提取层**。它从主流计算软件（VASP、LAMMPS、Abaqus、Quantum ESPRESSO、GROMACS、COMSOL、自研代码等）的输入文件、日志和输出中，提取**与具体材料体系无关的普适数学结构**（控制方程、离散化格式、边界条件、迭代算法、守恒律实现、计算图拓扑等），并按照标准 Math Schema 组织成 LLM 原生可消费的结构化数据（JSON/YAML）。

## 核心原则

- **零侵入**：不替代现有软件，不修改用户工作流，旁路运行
- **零判断**：不做"对错"判定，只负责"提取与结构化"
- **LLM 原生**：输出为结构化数据，供用户自己的 LLM、本地知识库或外部学术库消费
- **完全开源**：MIT/Apache-2.0 双许可，无商业闭源组件

## Math Schema v1.0

Schema 是产品的核心契约。所有 Harness 必须将提取结果映射到此 Schema。

### 顶层结构

```json
{
  "schema_version": "1.0.0",
  "meta": {
    "extracted_by": "math-anything-lammps",
    "extractor_version": "0.1.0",
    "extracted_at": "2026-04-20T13:54:00Z",
    "source_files": {...},
    "material_context_declaration": {...}
  },
  "mathematical_model": {
    "governing_equations": [],
    "boundary_conditions": [],
    "initial_conditions": [],
    "constitutive_relations": [],
    "coupling_conditions": []
  },
  "numerical_method": {
    "discretization": {},
    "solver": {},
    "parallelization": {}
  },
  "conservation_properties": {},
  "computational_graph": {},
  "raw_symbols": {}
}
```

## 安装

```bash
pip install math-anything-core
```

## 使用

### Python API

```python
import math_anything as ma

# 加载 Harness
harness = ma.load_harness("lammps")

# 提取
model = harness.extract({
    "input": "in.file",
    "log": "log.lammps"
})

# 保存为 JSON
model.save("model.json")
```

### CLI

```bash
# 提取
math-anything extract --engine lammps --input in.file --output model.json

# REPL 模式
math-anything repl --engine lammps
> load in.file
> extract
> show --json
> export model.json
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│  消费层（Consumer Layer）                                      │
├─────────────────────────────────────────────────────────────┤
│  核心引擎层（Core Engine）                                     │
│  ├── 解析器（Parser）：AST / 正则 / 语法分析                   │
│  ├── 提取器（Extractor）：数学对象识别与标准化                 │
│  ├── Schema 校验器（Validator）                               │
│  └── 序列化器（Serializer）：JSON / YAML                      │
├─────────────────────────────────────────────────────────────┤
│  适配层（Harness Layer）                                       │
│  ├── lammps-harness                                           │
│  ├── vasp-harness                                             │
│  ├── abaqus-harness                                           │
│  └── ...                                                      │
└─────────────────────────────────────────────────────────────┘
```

## 许可证

- 核心引擎：Apache-2.0
- 官方 Harness：MIT
- Schema 规范：CC-BY-4.0