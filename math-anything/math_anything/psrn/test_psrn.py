"""Tests for PSRN (Parallel Symbolic Regression Network).

验证 PSRN 各组件的正确性和与现有代码的兼容性。
"""

import numpy as np
import pytest

from .bridge import PSRNSymbolicRegression, upgrade_to_psrn
from .gpu_evaluator import GPUEvaluator, has_gpu_support
from .pse_engine import PSEConfig, PSEEngine
from .psrn_network import PSRN, PSRNConfig
from .symbol_layer import SymbolConfig, SymbolLayer
from .token_generator import GPTokenGenerator, MCTSTokenGenerator


class TestSymbolLayer:
    """测试 SymbolLayer 的核心功能."""

    def test_forward_generates_expressions(self):
        layer = SymbolLayer()
        inputs = ["x0", "x1"]

        exprs, values, offsets = layer.forward(inputs, layer_idx=0)

        assert len(exprs) > 0
        assert len(exprs) == len(offsets)
        # 2 变量 + 一元算子 + 二元算子
        # 至少应该有变量本身
        assert "x0" in exprs or "sin(x0)" in exprs

    def test_unary_operators(self):
        layer = SymbolLayer(SymbolConfig(unary_ops=["sin", "cos"], binary_squared_ops=[], binary_triangled_ops=[]))
        inputs = ["x"]

        exprs, _, _ = layer.forward(inputs, layer_idx=0)

        assert "sin(x)" in exprs
        assert "cos(x)" in exprs

    def test_binary_operators(self):
        layer = SymbolLayer(SymbolConfig(unary_ops=[], binary_squared_ops=["sub"], binary_triangled_ops=["add"]))
        inputs = ["x", "y"]

        exprs, _, _ = layer.forward(inputs, layer_idx=0)

        assert "(x+y)" in exprs or "(x+x)" in exprs
        assert "(x-y)" in exprs or "(y-x)" in exprs

    def test_deduce_expression(self):
        layer = SymbolLayer(SymbolConfig(unary_ops=["sin"], binary_squared_ops=[], binary_triangled_ops=["add"]))

        base_exprs = ["x", "y"]
        offsets = np.array([("sin", 0, -1), ("add", 0, 1), ("add", 1, 1)], dtype=object)

        # 测试 sin(x)
        result = layer.deduce_expression(0, [offsets], base_exprs)
        assert "sin" in result and "x" in result

    def test_memory_estimate(self):
        layer = SymbolLayer()
        dims = layer.memory_estimate(n_inputs=2, n_layers=2)

        assert 0 in dims
        assert 1 in dims
        assert 2 in dims
        assert dims[0] == 2
        assert dims[1] > dims[0]

    def test_duplicate_mask(self):
        layer = SymbolLayer()
        exprs = ["x", "y", "x", "(x+y)", "(y+x)"]

        unique_indices, unique_exprs = layer.build_duplicate_mask(exprs)

        assert len(unique_exprs) <= len(exprs)
        assert len(unique_indices) == len(unique_exprs)


class TestGPUEvaluator:
    """测试 GPU Evaluator."""

    def test_cpu_fallback(self):
        evaluator = GPUEvaluator(use_gpu=False)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        expressions = ["x0+x1", "sin(x0)"]

        results = evaluator.evaluate_batch(expressions, X, ["x0", "x1"])

        assert results.shape == (2, 2)
        # x0+x1: [3.0, 7.0]
        assert np.isclose(results[0, 0], 3.0)
        assert np.isclose(results[1, 0], 7.0)

    def test_cache_works(self):
        evaluator = GPUEvaluator(use_gpu=False)
        X = np.array([[1.0]])

        # 第一次评估
        evaluator.evaluate_batch(["x0"], X, ["x0"])
        stats1 = evaluator.cache_stats()

        # 第二次评估相同表达式
        evaluator.evaluate_batch(["x0"], X, ["x0"])
        stats2 = evaluator.cache_stats()

        assert stats2["cached_expressions"] >= stats1["cached_expressions"]

    def test_has_gpu_support(self):
        # 这个测试只是确保函数不会崩溃
        result = has_gpu_support()
        assert isinstance(result, bool)


class TestPSRN:
    """测试 PSRN 网络."""

    def test_fit_simple_function(self):
        config = PSRNConfig(n_layers=1, n_input_slots=2)
        psrn = PSRN(config)

        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = X.flatten() ** 2

        best_expr, best_mse, top_k = psrn.fit(X, y, variable_names=["x"])

        assert isinstance(best_expr, str)
        assert best_mse >= 0
        assert len(top_k) > 0

    def test_search_space_size(self):
        config = PSRNConfig(n_layers=2, n_input_slots=2)
        psrn = PSRN(config)

        sizes = psrn.get_search_space_size()

        assert 0 in sizes
        assert 1 in sizes
        assert 2 in sizes

    def test_build_base_expressions(self):
        config = PSRNConfig(n_input_slots=4, max_constants=2)
        psrn = PSRN(config)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        base_exprs, base_values = psrn._build_base_expressions(X, variable_names=["x", "y"])

        assert "x" in base_exprs
        assert "y" in base_exprs
        assert len(base_exprs) >= 2  # 至少变量
        assert base_values.shape[0] == 2  # 2 样本


class TestTokenGenerators:
    """测试 Token Generator."""

    def test_gp_token_generator(self):
        gen = GPTokenGenerator(n_tokens=3)
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = X.flatten() ** 2

        token_exprs, token_values = gen.generate(X, y, ["x"])

        assert len(token_exprs) <= 3
        assert token_values.shape[1] == len(token_exprs)
        assert token_values.shape[0] == 20

    def test_mcts_token_generator(self):
        gen = MCTSTokenGenerator(n_tokens=3)
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = X.flatten() ** 2

        token_exprs, token_values = gen.generate(X, y, ["x"])

        assert len(token_exprs) > 0
        assert token_values.shape[1] == len(token_exprs)


class TestPSEEngine:
    """测试 PSE 引擎."""

    def test_discover_quadratic(self):
        config = PSEConfig(
            psrn_config=PSRNConfig(n_layers=1, n_input_slots=2),
            max_iterations=2,
        )
        engine = PSEEngine(config)

        X = np.linspace(0, 1, 30).reshape(-1, 1)
        y = X.flatten() ** 2

        best_expr, pareto_front = engine.discover(X, y, ["x"], verbose=False)

        assert isinstance(best_expr, str)
        assert len(pareto_front) > 0

    def test_pareto_front_updated(self):
        config = PSEConfig(
            psrn_config=PSRNConfig(n_layers=1, n_input_slots=2),
            max_iterations=2,
        )
        engine = PSEEngine(config)

        # 手动添加一些解
        engine._update_pareto_front("x", 0.5, 1, 0.8)
        engine._update_pareto_front("x^2", 0.1, 2, 0.6)

        assert len(engine.pareto_front) >= 1

    def test_reward_computation(self):
        config = PSEConfig(reward_discount=0.99)
        engine = PSEEngine(config)

        reward = engine._compute_reward(mse=0.0, complexity=1)
        assert reward > 0

        # MSE 越大，reward 越小
        reward_high_mse = engine._compute_reward(mse=100.0, complexity=1)
        reward_low_mse = engine._compute_reward(mse=0.01, complexity=1)
        assert reward_low_mse > reward_high_mse


class TestBridge:
    """测试与现有代码的桥接."""

    def test_psrn_regression_api(self):
        """测试 PSRNSymbolicRegression 与 ImprovedSymbolicRegression API 兼容."""
        sr = PSRNSymbolicRegression(n_layers=1, max_iterations=2)

        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = X.flatten() ** 2

        best_tree = sr.fit(X, y, variable_names=["x"])

        assert best_tree is not None
        assert sr.best_fitness_ < float("inf")

    def test_predict_api(self):
        sr = PSRNSymbolicRegression(n_layers=1, max_iterations=2)

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 4.0, 9.0])

        sr.fit(X, y, variable_names=["x"])
        predictions = sr.predict(X)

        assert predictions.shape == (3,)
        assert not np.any(np.isnan(predictions))

    def test_upgrade_function(self):
        from ..eml_v2 import ImprovedSymbolicRegression

        old_sr = ImprovedSymbolicRegression()
        new_sr = upgrade_to_psrn(old_sr)

        assert isinstance(new_sr, PSRNSymbolicRegression)


class TestIntegration:
    """集成测试."""

    def test_end_to_end_discovery(self):
        """端到端测试：从数据到发现表达式."""
        config = PSEConfig(
            psrn_config=PSRNConfig(n_layers=2, n_input_slots=3),
            max_iterations=3,
        )
        engine = PSEEngine(config)

        # 生成测试数据: y = x^2 + 1
        X = np.linspace(-2, 2, 50).reshape(-1, 1)
        y = X.flatten() ** 2 + 1.0

        best_expr, pareto = engine.discover(X, y, ["x"], verbose=False)

        assert best_expr != ""
        assert len(pareto) > 0

        # 检查 Pareto 前沿的格式
        for expr, mse, comp, reward in pareto:
            assert isinstance(expr, str)
            assert isinstance(mse, float)
            assert isinstance(comp, int)
            assert isinstance(reward, float)

    def test_multivariable_discovery(self):
        """测试多变量发现."""
        config = PSEConfig(
            psrn_config=PSRNConfig(n_layers=1, n_input_slots=4),
            max_iterations=2,
        )
        engine = PSEEngine(config)

        x1 = np.linspace(0, 1, 20)
        x2 = np.linspace(0, 1, 20)
        X = np.column_stack([x1, x2])
        y = x1 * x2  # y = x1 * x2

        best_expr, pareto = engine.discover(X, y, ["x1", "x2"], verbose=False)

        assert best_expr != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
