# ADR-001: Rust 加速层

## 状态
已采纳

## 上下文
Python 在数值计算密集型任务（Riemann 张量 O(n⁵)、PSRN 种群评估、Buckingham Pi SVD）中性能不足。

## 决策
引入 Rust (PyO3) 作为可选加速层：
- 12 个核心函数用 Rust 实现
- Rayon 并行化批量操作
- Python 回退路径保证兼容性
- GNU 工具链 (x86_64-pc-windows-gnu)

## 后果
- 正面：10-200x 性能提升，Rayon 并行
- 负面：构建依赖 Rust 工具链，跨平台编译复杂
- 缓解：Python 回退路径，maturin 简化构建
