"""五层分层分析与符号回归的整合模块.

整合点：
- Level 3+: PSRN 用于发现变量间的数学关系
- Level 4+: 符号回归辅助几何结构分析
- Level 5: 统一的符号-几何分析框架

业务逻辑连贯性：
1. Level 1-2: 基础数据提取
2. Level 3: PSRN 发现隐藏的数学规律
3. Level 4: 结合符号表达式分析几何结构
4. Level 5: 符号-几何统一的完整分析
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..eml_v2 import ImprovedSymbolicRegression
from ..psrn import ConservativePSRNConfig, EnhancedPSRNSymbolicRegression


class TieredSymbolicRegressionAnalyzer:
    """分层符号回归分析器 - 在五层框架中集成 PSRN.

    这是五层分层算法的符号回归扩展，根据层级选择合适的分析方法。

    Example:
        >>> analyzer = TieredSymbolicRegressionAnalyzer()
        >>>
        >>> # Level 3: 发现变量关系
        >>> result = analyzer.analyze_relationships(
        ...     X, y, tier=3, variable_names=['x', 'y', 'z']
        ... )
        >>>
        >>> # Level 5: 完整符号-几何分析
        >>> full_result = analyzer.analyze_complete(X, y, context={})
    """

    def __init__(self):
        self.psrn_cache = {}
        self.gp_cache = {}

    def analyze_relationships(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        tier: int = 3,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """在指定层级分析变量关系.

        Args:
            X: 输入特征 (n_samples, n_features)
            y: 目标值 (n_samples,)
            variable_names: 变量名列表
            tier: 分析层级 (1-5)
            context: 上下文信息（来自前几层的结果）

        Returns:
            包含发现关系和符号表达式的字典
        """
        result = {
            "tier": tier,
            "variable_names": variable_names,
            "discovered_relationships": [],
        }

        if tier >= 3:
            # Level 3+: 使用保守增强型 PSRN
            result["discovered_relationships"] = self._analyze_with_psrn(
                X, y, variable_names, tier
            )

        if tier >= 4:
            # Level 4+: 深度符号分析 + 几何关联
            result["geometric_insights"] = self._analyze_geometric_relationships(
                result["discovered_relationships"], context
            )

        if tier >= 5:
            # Level 5: 完整的符号-几何统一分析
            result["unified_representation"] = self._build_unified_representation(
                result, context
            )

        return result

    def _analyze_with_psrn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        tier: int,
    ) -> List[Dict[str, Any]]:
        """使用 PSRN 分析关系.

        根据层级选择不同复杂度的 PSRN 配置。
        """
        # 根据层级选择配置
        if tier == 3:
            # Level 3: 快速探索，中等复杂度
            config = ConservativePSRNConfig(
                n_layers=2,
                max_layer_size=200,
            )
        elif tier == 4:
            # Level 4: 深度探索，允许更复杂表达式
            config = ConservativePSRNConfig(
                n_layers=2,
                max_layer_size=500,
            )
        else:  # tier >= 5
            # Level 5: 完整探索
            config = ConservativePSRNConfig(
                n_layers=3,
                max_layer_size=800,
            )

        from ..psrn.enhanced_psrn_v2 import ConservativeEnhancedPSRN

        psrn = ConservativeEnhancedPSRN(config)
        best_expr, best_mse, top_k = psrn.fit(X, y, variable_names)

        relationships = []

        # 主关系
        relationships.append(
            {
                "expression": best_expr,
                "mse": best_mse,
                "complexity": self._estimate_complexity(best_expr),
                "confidence": self._compute_confidence(best_mse, y),
                "type": "primary",
            }
        )

        # Top-K 备选关系
        for expr, mse in top_k[:3]:
            relationships.append(
                {
                    "expression": expr,
                    "mse": mse,
                    "complexity": self._estimate_complexity(expr),
                    "confidence": self._compute_confidence(mse, y),
                    "type": "alternative",
                }
            )

        return relationships

    def _analyze_geometric_relationships(
        self,
        relationships: List[Dict],
        context: Optional[Dict],
    ) -> Dict[str, Any]:
        """分析符号表达式与几何结构的关系.

        Level 4+: 将发现的数学关系与几何特性关联。
        """
        insights = {
            "symmetry_analysis": {},
            "conservation_laws": [],
            "constraint_structure": {},
        }

        for rel in relationships:
            expr = rel["expression"]

            # 检测对称性
            if "sin" in expr and "cos" in expr:
                insights["symmetry_analysis"]["rotational"] = True

            if "exp" in expr:
                insights["symmetry_analysis"]["scale_invariance"] = True

            # 检测守恒量
            if "+" in expr and "-" in expr:
                insights["conservation_laws"].append("possible_energy_conservation")

        return insights

    def _build_unified_representation(
        self,
        analysis_result: Dict,
        context: Optional[Dict],
    ) -> Dict[str, Any]:
        """构建符号-几何统一表示.

        Level 5: 将符号回归结果与几何分析整合为统一框架。
        """
        unified = {
            "symbolic_core": {},
            "geometric_embedding": {},
            "latent_structure": {},
        }

        # 提取核心符号关系
        primary = None
        for rel in analysis_result.get("discovered_relationships", []):
            if rel["type"] == "primary":
                primary = rel
                break

        if primary:
            unified["symbolic_core"] = {
                "canonical_form": primary["expression"],
                "confidence": primary["confidence"],
                "complexity": primary["complexity"],
            }

            # 构建几何嵌入
            unified["geometric_embedding"] = {
                "manifold_dimension": primary["complexity"],
                "metric_tensor": "derived_from_expression",
                "symmetries": analysis_result.get("geometric_insights", {}).get(
                    "symmetry_analysis", {}
                ),
            }

            # 潜在空间结构
            unified["latent_structure"] = {
                "dimension": max(16, min(128, primary["complexity"] * 8)),
                "encoder_type": "SymbolicEGNN",
                "equivariance_groups": (
                    ["E(3)"]
                    if "sin" in primary["expression"] or "cos" in primary["expression"]
                    else []
                ),
            }

        return unified

    def _estimate_complexity(self, expr: str) -> int:
        """估计表达式复杂度."""
        complexity = 0
        ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "eml"]
        for op in ops:
            complexity += expr.count(op)
        return max(complexity, 1)

    def _compute_confidence(self, mse: float, y: np.ndarray) -> float:
        """计算置信度."""
        y_var = np.var(y)
        if y_var == 0:
            return 1.0 if mse < 1e-10 else 0.0

        r_squared = 1 - mse / y_var
        return max(0.0, min(1.0, r_squared))


class IntegratedTieredAnalyzer:
    """集成式五层分析器 - 将符号回归整合到五层框架中.

    这是 TieredAnalyzer 的扩展版本，在 Level 3+ 自动调用 PSRN。
    """

    def __init__(self):
        from .tiered_analyzer import TieredAnalyzer

        self.base_analyzer = TieredAnalyzer()
        self.sr_analyzer = TieredSymbolicRegressionAnalyzer()

    def analyze_with_symbolic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        file_path: Optional[str] = None,
        tier: int = 3,
    ) -> Dict[str, Any]:
        """完整分析：五层框架 + 符号回归.

        如果提供了 file_path，先运行基础五层分析，
        然后在 Level 3+ 加入符号回归结果。

        Args:
            X: 输入数据
            y: 目标值
            variable_names: 变量名
            file_path: 可选的仿真文件路径
            tier: 目标层级

        Returns:
            包含五层分析和符号回归的完整结果
        """
        result = {
            "tier": tier,
            "has_file_analysis": file_path is not None,
            "has_symbolic_regression": tier >= 3,
        }

        # Step 1: 基础五层分析（如果提供了文件）
        if file_path and tier >= 1:
            base_result = self.base_analyzer.analyze(file_path, tier=min(tier, 2))
            result["base_analysis"] = base_result
            context = self._extract_context(base_result)
        else:
            context = {}

        # Step 2: 符号回归分析（Level 3+）
        if tier >= 3:
            sr_result = self.sr_analyzer.analyze_relationships(
                X, y, variable_names, tier, context
            )
            result["symbolic_regression"] = sr_result

        # Step 3: 整合分析（Level 5）
        if tier >= 5 and file_path:
            result["integrated_insights"] = self._integrate_analyses(
                result.get("base_analysis"),
                result.get("symbolic_regression"),
            )

        return result

    def _extract_context(self, base_result) -> Dict:
        """从基础分析结果中提取上下文."""
        context = {}

        if hasattr(base_result, "file_analysis"):
            fa = base_result.file_analysis
            context["engine"] = fa.engine
            context["num_atoms"] = fa.num_atoms
            context["simulation_time"] = fa.simulation_time

        if hasattr(base_result, "topology_info"):
            context["topology"] = base_result.topology_info

        return context

    def _integrate_analyses(
        self,
        base_analysis,
        sr_analysis: Dict,
    ) -> Dict[str, Any]:
        """整合基础分析和符号回归结果."""
        insights = {
            "physical_interpretation": {},
            "mathematical_structure": {},
            "recommendations": [],
        }

        # 物理意义解释
        if sr_analysis and "discovered_relationships" in sr_analysis:
            primary = None
            for rel in sr_analysis["discovered_relationships"]:
                if rel["type"] == "primary":
                    primary = rel
                    break

            if primary:
                expr = primary["expression"]

                # 根据表达式形式推断物理意义
                if "exp" in expr:
                    insights["physical_interpretation"]["growth_decay"] = True
                if "sin" in expr or "cos" in expr:
                    insights["physical_interpretation"]["oscillatory"] = True
                if "x*x" in expr or "x**2" in expr:
                    insights["physical_interpretation"]["quadratic"] = True

        return insights


def tiered_symbolic_regression_analysis(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: List[str],
    tier: int = 3,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """便捷函数：分层符号回归分析.

    Args:
        X: 输入数据
        y: 目标值
        variable_names: 变量名
        tier: 分析层级 (1-5)
        file_path: 可选的仿真文件路径

    Returns:
        完整分析结果

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> result = tiered_symbolic_regression_analysis(
        ...     x.reshape(-1, 1), y, ['x'], tier=4
        ... )
        >>> print(result["symbolic_regression"]["discovered_relationships"][0])
    """
    analyzer = IntegratedTieredAnalyzer()
    return analyzer.analyze_with_symbolic_regression(
        X, y, variable_names, file_path, tier
    )
