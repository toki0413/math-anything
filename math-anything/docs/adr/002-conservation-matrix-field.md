# ADR-002: 守恒矩阵场

## 状态
已采纳

## 上下文
守恒律在代码中以扁平字典表示，丢失了守恒量之间的代数关系。Noether 对应只是字符串描述，不是可计算关系。

## 决策
引入 ConservationMatrixField 类：
- 将多个守恒律编码为统一矩阵算子 dU/dt + div(F(U)) = S(U)
- Noether 对应表：SymmetryGroup → ConservedQuantity 双射
- 辛矩阵表示（Hamilton 系统）
- 18 种方程构建器覆盖 8 大物理领域

## 后果
- 正面：守恒律可计算、可验证、可传播
- 负面：FieldConservedQuantity 与 evolution.ConservedQuantity 命名需区分
- 缓解：前缀 Field 区分，re-export 保持兼容
