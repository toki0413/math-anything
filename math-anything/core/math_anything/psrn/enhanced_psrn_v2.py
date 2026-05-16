"""Enhanced PSRN v2 - 更保守的实现，避免内存爆炸.

核心改进：
1. 严格限制每层输出维度 (max 1000)
2. 渐进式剪枝，在早期层更激进
3. 使用稀疏评估避免内存问题
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import warnings

from .psrn_network import PSRN, PSRNConfig
from .symbol_layer import SymbolLayer, SymbolConfig


@dataclass
class ConservativePSRNConfig(PSRNConfig):
    """保守型 PSRN 配置."""
    
    max_layer_size: int = 500  # 每层最大候选数
    progressive_pruning: bool = True
    
    def __post_init__(self):
        # 强制限制层数
        self.n_layers = min(self.n_layers, 3)


class ConservativeEnhancedPSRN(PSRN):
    """保守增强型 PSRN - 严格控制内存使用."""
    
    def __init__(self, config: Optional[ConservativePSRNConfig] = None):
        self.conservative_config = config or ConservativePSRNConfig()
        super().__init__(self.conservative_config)
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
        token_exprs: Optional[List[str]] = None,
        token_values: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """运行保守型 PSRN."""
        
        # 构建基础表达式
        base_exprs, base_values = self._build_base_expressions(
            X, variable_names, token_exprs, token_values
        )
        
        # 限制输入大小
        if len(base_exprs) > 20:
            base_exprs = base_exprs[:20]
            base_values = base_values[:, :20]
        
        self._base_exprs = base_exprs
        self._layer_outputs = []
        self._layer_values = []
        
        current_exprs = base_exprs
        current_values = base_values
        all_candidates = []  # 收集所有层的候选
        
        for layer_idx, layer in enumerate(self.layers):
            # 前向传播
            new_exprs, new_values, _ = layer.forward(
                current_exprs, current_values, layer_idx
            )
            
            # 评估并排序
            mses = []
            for j in range(new_values.shape[1]):
                mse = np.mean((new_values[:, j] - y) ** 2)
                mses.append(mse)
            
            mses = np.array(mses)
            
            # 严格剪枝
            max_size = self.conservative_config.max_layer_size
            keep_size = min(max_size, int(len(new_exprs) * 0.2))  # 只保留20%
            keep_size = max(10, keep_size)  # 至少保留10个
            
            # 选择最优
            sorted_indices = np.argsort(mses)[:keep_size]
            
            pruned_exprs = [new_exprs[i] for i in sorted_indices]
            pruned_values = new_values[:, sorted_indices]
            
            print(f"  Layer {layer_idx}: {len(new_exprs)} → {len(pruned_exprs)}")
            
            # 保存该层候选
            for i, idx in enumerate(sorted_indices[:10]):  # 只保存前10
                all_candidates.append((new_exprs[idx], mses[idx], layer_idx))
            
            self._layer_outputs.append(pruned_exprs)
            self._layer_values.append(pruned_values)
            
            current_exprs = pruned_exprs
            current_values = pruned_values
        
        # 从所有候选中选择最优
        all_candidates.sort(key=lambda x: x[1])
        best_expr, best_mse, _ = all_candidates[0] if all_candidates else ("x", float('inf'))
        
        top_k = [(expr, mse) for expr, mse, _ in all_candidates[:10]]
        
        return best_expr, best_mse, top_k
