"""GPU Evaluator - GPU 加速的表达式评估引擎.

提供 CUDA 加速的数值计算，自动回退到 CPU（当 GPU 不可用时）。
使用 Numba 或 CuPy 实现 GPU 并行化。
"""

import os
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np


def has_gpu_support() -> bool:
    """检查是否有 GPU 支持."""
    try:
        import cupy as cp

        _ = cp.cuda.Device(0)
        return True
    except Exception:
        pass

    try:
        import numba.cuda

        return numba.cuda.is_available()
    except Exception:
        pass

    return False


class GPUEvaluator:
    """GPU 表达式评估器.

    通过批量并行评估大量表达式，避免重复计算公共子树。
    当 GPU 不可用时，使用优化的 NumPy CPU 实现。

    Example:
        >>> evaluator = GPUEvaluator()
        >>>
        >>> # 准备数据: 1000 样本, 5 个输入变量
        >>> X = np.random.randn(1000, 5)
        >>>
        >>> # 定义一批表达式（共享子树会被自动复用）
        >>> expressions = ["x0+x1", "sin(x0+x1)", "exp(x0+x1)"]
        >>>
        >>> # 并行评估
        >>> results = evaluator.evaluate_batch(expressions, X)
        >>> print(results.shape)  # (1000, 3)
    """

    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Args:
            use_gpu: 是否使用 GPU。None 则自动检测。
        """
        self._has_gpu = has_gpu_support()
        self.use_gpu = use_gpu if use_gpu is not None else self._has_gpu

        if self.use_gpu and not self._has_gpu:
            warnings.warn("GPU requested but not available, falling back to CPU")
            self.use_gpu = False

        self._cupy = None
        self._numba = None
        self._cache: Dict[str, np.ndarray] = {}  # 子树值缓存

        if self.use_gpu:
            try:
                import cupy as cp

                self._cupy = cp
            except ImportError:
                try:
                    import numba.cuda

                    self._numba = numba.cuda
                except ImportError:
                    self.use_gpu = False

    def evaluate_batch(
        self,
        expressions: List[str],
        X: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """批量评估多个表达式.

        Args:
            expressions: 表达式字符串列表
            X: 输入数据 (n_samples, n_features)
            variable_names: 变量名列表（默认 x0, x1, ...）

        Returns:
            评估结果 (n_samples, n_expressions)
        """
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        if self.use_gpu and self._cupy is not None:
            return self._evaluate_batch_cupy(expressions, X, variable_names)
        elif self.use_gpu and self._numba is not None:
            return self._evaluate_batch_numba(expressions, X, variable_names)
        else:
            return self._evaluate_batch_cpu(expressions, X, variable_names)

    def _evaluate_batch_cpu(
        self,
        expressions: List[str],
        X: np.ndarray,
        variable_names: List[str],
    ) -> np.ndarray:
        """CPU 批量评估 - 使用编译优化."""
        from .compiled_evaluator import CompiledEvaluator

        evaluator = CompiledEvaluator()
        return evaluator.evaluate_batch_vec(expressions, X, variable_names)

    def _evaluate_batch_cupy(
        self,
        expressions: List[str],
        X: np.ndarray,
        variable_names: List[str],
    ) -> np.ndarray:
        """CuPy GPU 批量评估."""
        cp = self._cupy

        # 将数据转移到 GPU
        X_gpu = cp.asarray(X)
        var_dict = {name: X_gpu[:, i] for i, name in enumerate(variable_names)}

        n_samples = X.shape[0]
        n_exprs = len(expressions)
        results = cp.zeros((n_samples, n_exprs))

        for expr_idx, expr in enumerate(expressions):
            try:
                result = self._safe_eval_gpu(expr, var_dict, cp)
                results[:, expr_idx] = result
            except Exception:
                results[:, expr_idx] = cp.nan

        # 将结果转移回 CPU
        return cp.asnumpy(results)

    def _safe_eval_gpu(self, expr: str, var_dict: Dict, cp):
        """GPU 安全评估."""
        safe_dict = {
            "sin": cp.sin,
            "cos": cp.cos,
            "exp": lambda x: cp.exp(cp.clip(x, -700, 700)),
            "log": lambda x: cp.log(cp.abs(x) + 1e-10),
            "sqrt": lambda x: cp.sqrt(cp.abs(x)),
            "abs": cp.abs,
            "pi": cp.pi,
            "e": cp.e,
        }
        safe_dict.update(var_dict)

        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            if isinstance(result, (int, float)):
                return cp.full(var_dict[list(var_dict.keys())[0]].shape, float(result))
            return result
        except Exception as e:
            raise ValueError(f"GPU eval failed for '{expr}': {e}")

    def _evaluate_batch_numba(
        self,
        expressions: List[str],
        X: np.ndarray,
        variable_names: List[str],
    ) -> np.ndarray:
        """Numba GPU 批量评估（简化实现）."""
        # Numba 实现更复杂，这里回退到优化的 CPU 版本
        return self._evaluate_batch_cpu(expressions, X, variable_names)

    def clear_cache(self):
        """清除子树缓存."""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        """返回缓存统计信息."""
        return {
            "cached_expressions": len(self._cache),
            "cache_size_mb": sum(v.nbytes for v in self._cache.values())
            / (1024 * 1024),
        }
