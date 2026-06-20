"""PSRN (Parallel Symbolic Regression Network) for Math Anything.

基于 "Discovering physical laws with parallel symbolic enumeration" (Nature Computational Science, 2025)
的核心思想，将并行符号枚举集成到 math-anything 的符号回归框架中。

核心组件:
- SymbolLayer: 符号层，实现表达式树的层次化构建和子树复用
- PSRN: 并行符号回归网络，GPU 加速的表达式评估引擎
- TokenGenerator: 令牌生成器，产生有希望的子表达式作为 PSRN 输入
- PSEEngine: 顶层引擎，协调 PSRN 和 TokenGenerator 的迭代循环

与现有代码的集成点:
- 替代 ImprovedSymbolicRegression 的评估核心
- 保留 ExprBuilder/Node 的表达式树表示
- 兼容现有的 simplify 和 visualization 流程
"""

from .adaptive_attention import (
    AdaptiveAttentionGenerator,
    HybridAttentionGenerator,
)
from .bridge import (
    EnhancedPSRNSymbolicRegression,
    PSRNSymbolicRegression,
    upgrade_to_enhanced_psrn,
    upgrade_to_psrn,
)
from .csa_hca_attention import CSAHCAAttentionGenerator
from .enhanced_psrn_v2 import ConservativeEnhancedPSRN, ConservativePSRNConfig
from .gpu_evaluator import GPUEvaluator, has_gpu_support
from .pse_engine import PSEConfig, PSEEngine
from .psrn_network import PSRN, PSRNConfig
from .symbol_layer import OperatorType, SymbolConfig, SymbolLayer
from .token_generator import (
    FastTokenGenerator,
    GPTokenGenerator,
    MCTSTokenGenerator,
    RandomTokenGenerator,
    TokenGenerator,
)

__all__ = [
    # Symbol Layer
    "SymbolLayer",
    "OperatorType",
    "SymbolConfig",
    # PSRN Network
    "PSRN",
    "PSRNConfig",
    # Enhanced PSRN
    "ConservativeEnhancedPSRN",
    "ConservativePSRNConfig",
    "EnhancedPSRNSymbolicRegression",
    "upgrade_to_enhanced_psrn",
    # Token Generator
    "TokenGenerator",
    "RandomTokenGenerator",
    "FastTokenGenerator",
    "GPTokenGenerator",
    "MCTSTokenGenerator",
    # PSE Engine
    "PSEEngine",
    "PSEConfig",
    # GPU Evaluator
    "GPUEvaluator",
    "has_gpu_support",
    # Bridge (API compatible)
    "PSRNSymbolicRegression",
    "upgrade_to_psrn",
]
