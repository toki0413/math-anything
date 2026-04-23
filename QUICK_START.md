# Math-Anything 快速开始指南

## 一键安装

### Windows
双击运行 `install.bat`

### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

### 手动安装
```bash
# 从PyPI安装
pip install math-anything

# 或从本地wheel安装
pip install dist/math_anything-1.0.0-py3-none-any.whl

# 或从本地源码安装
pip install dist/math_anything-1.0.0.tar.gz
```

## 验证安装

```bash
math-anything --help
math-anything --version
```

## 基本用法

### 1. 提取LAMMPS文件的数学结构
```bash
math-anything extract lammps simulation.lmp
```

### 2. 提取Abaqus文件的数学结构
```bash
math-anything extract abaqus model.inp
```

### 3. 提取VASP文件的数学结构
```bash
math-anything extract vasp INCAR
```

### 4. 启动交互式REPL
```bash
math-anything interactive
```

## Python API 使用

```python
from math_anything import MathAnything

# 创建实例
ma = MathAnything()

# 提取LAMMPS文件的数学结构
result = ma.extract_file("lammps", "simulation.lmp")
print(result.schema)

# 提取Abaqus文件
result = ma.extract_file("abaqus", "model.inp")
print(result.equations)

# 算法推荐
from math_anything.algorithm_knowledge_graph import recommend_ml_architecture
recommendation = recommend_ml_architecture(result.schema, "abaqus")
print(f"推荐架构: {recommendation.primary.name}")
```

## 分发包说明

- `math_anything-1.0.0-py3-none-any.whl` (165KB) - 预编译wheel包，安装最快
- `math_anything-1.0.0.tar.gz` (254KB) - 源码分发包，需要编译

## 系统要求

- Python >= 3.10
- 支持平台: Windows, Linux, macOS

## 获取更多帮助

- GitHub: https://github.com/toki0413/math-anything
- Gitee: https://gitee.com/crested-ibis-0413/math-anything
