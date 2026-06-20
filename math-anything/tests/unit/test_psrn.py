"""Unit tests for PSRN module: CompiledEvaluator, PSEEngine, PSRNBridge."""

import math

import numpy as np
import pytest

from math_anything.psrn.compiled_evaluator import (
    CompiledEvaluator,
    FastSymbolLayer,
    NumbaEvaluator,
    _eml,
    _VECTORIZED_OPS,
)
from math_anything.psrn.pse_engine import PSEConfig, PSEEngine
from math_anything.psrn.bridge import (
    PSRNSymbolicRegression,
    EnhancedPSRNSymbolicRegression,
    upgrade_to_psrn,
    upgrade_to_enhanced_psrn,
)


# ── CompiledEvaluator fixtures ──

@pytest.fixture
def evaluator():
    return CompiledEvaluator()


@pytest.fixture
def sample_data():
    """2-variable sample data: x=1..5, y=10..14."""
    X = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0], [5.0, 14.0]])
    return X


# ── CompiledEvaluator: creation ──

class TestCompiledEvaluatorCreation:
    def test_creates_with_empty_cache(self, evaluator):
        assert evaluator._compiled_cache == {}

    def test_has_vectorized_ops(self, evaluator):
        assert "sin" in evaluator._vectorized_ops
        assert "add" in evaluator._vectorized_ops
        assert "eml" in evaluator._vectorized_ops

    def test_cache_stats_initial(self, evaluator):
        stats = evaluator.cache_stats()
        assert stats["cached_expressions"] == 0


# ── CompiledEvaluator: compile + evaluate_single ──

class TestCompiledEvaluatorCompile:
    def test_compile_simple_add(self, evaluator):
        func = evaluator.compile("x + y", ["x", "y"])
        result = func(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]))
        np.testing.assert_array_almost_equal(result, [4.0, 6.0])

    def test_compile_simple_mul(self, evaluator):
        func = evaluator.compile("x * y", ["x", "y"])
        result = func(x=np.array([2.0, 3.0]), y=np.array([5.0, 7.0]))
        np.testing.assert_array_almost_equal(result, [10.0, 21.0])

    def test_compile_sin(self, evaluator):
        func = evaluator.compile("sin(x)", ["x"])
        result = func(x=np.array([0.0, math.pi / 2]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])

    def test_compile_cos(self, evaluator):
        func = evaluator.compile("cos(x)", ["x"])
        result = func(x=np.array([0.0, math.pi]))
        np.testing.assert_array_almost_equal(result, [1.0, -1.0])

    def test_compile_exp(self, evaluator):
        func = evaluator.compile("exp(x)", ["x"])
        result = func(x=np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(result, [1.0, math.e])

    def test_compile_log(self, evaluator):
        func = evaluator.compile("log(x)", ["x"])
        result = func(x=np.array([1.0, math.e]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])

    def test_compile_sqrt(self, evaluator):
        func = evaluator.compile("sqrt(x)", ["x"])
        result = func(x=np.array([4.0, 9.0]))
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_compile_abs(self, evaluator):
        func = evaluator.compile("abs(x)", ["x"])
        result = func(x=np.array([-3.0, 5.0]))
        np.testing.assert_array_almost_equal(result, [3.0, 5.0])

    def test_compile_constant(self, evaluator):
        func = evaluator.compile("3.14", ["x"])
        result = func(x=np.array([1.0, 2.0]))
        np.testing.assert_array_almost_equal(result, [3.14, 3.14])

    def test_compile_pi(self, evaluator):
        func = evaluator.compile("pi", ["x"])
        result = func(x=np.array([0.0]))
        assert abs(result - math.pi) < 1e-10

    def test_compile_e_constant(self, evaluator):
        func = evaluator.compile("e", ["x"])
        result = func(x=np.array([0.0]))
        assert abs(result - math.e) < 1e-10

    def test_compile_sub(self, evaluator):
        func = evaluator.compile("x - y", ["x", "y"])
        result = func(x=np.array([5.0]), y=np.array([3.0]))
        np.testing.assert_array_almost_equal(result, [2.0])

    def test_compile_div(self, evaluator):
        func = evaluator.compile("x / y", ["x", "y"])
        result = func(x=np.array([10.0]), y=np.array([2.0]))
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_compile_pow(self, evaluator):
        func = evaluator.compile("x ** 2", ["x"])
        result = func(x=np.array([3.0, 4.0]))
        np.testing.assert_array_almost_equal(result, [9.0, 16.0])

    def test_compile_unary_neg(self, evaluator):
        func = evaluator.compile("-x", ["x"])
        result = func(x=np.array([3.0, -2.0]))
        np.testing.assert_array_almost_equal(result, [-3.0, 2.0])

    def test_compile_unary_plus(self, evaluator):
        func = evaluator.compile("+x", ["x"])
        result = func(x=np.array([3.0]))
        np.testing.assert_array_almost_equal(result, [3.0])

    def test_compile_eml(self, evaluator):
        func = evaluator.compile("eml(x, y)", ["x", "y"])
        x_val = np.array([0.0])
        y_val = np.array([1.0])
        result = func(x=x_val, y=y_val)
        expected = np.exp(0.0) - np.log(1.0 + 1e-10)
        assert abs(result[0] - expected) < 1e-6

    def test_compile_caches_result(self, evaluator):
        f1 = evaluator.compile("sin(x)", ["x"])
        f2 = evaluator.compile("sin(x)", ["x"])
        assert f1 is f2
        assert evaluator.cache_stats()["cached_expressions"] == 1

    def test_compile_nested_expr(self, evaluator):
        func = evaluator.compile("sin(x) + x ** 2", ["x"])
        result = func(x=np.array([0.0, 1.0]))
        expected = np.sin(np.array([0.0, 1.0])) + np.array([0.0, 1.0]) ** 2
        np.testing.assert_array_almost_equal(result, expected)


# ── CompiledEvaluator: error handling ──

class TestCompiledEvaluatorErrors:
    def test_unknown_variable_raises(self, evaluator):
        with pytest.raises(ValueError, match="Unknown variable"):
            evaluator.compile("z + 1", ["x"])

    def test_invalid_syntax_raises(self, evaluator):
        with pytest.raises(SyntaxError):
            evaluator.compile("sin(", ["x"])

    def test_eml_wrong_arg_count_raises(self, evaluator):
        with pytest.raises(ValueError, match="2 arguments"):
            evaluator.compile("eml(x)", ["x"])

    def test_multi_arg_func_raises(self, evaluator):
        with pytest.raises(ValueError, match="single-argument"):
            evaluator.compile("sin(x, y)", ["x", "y"])


# ── CompiledEvaluator: evaluate ──

class TestCompiledEvaluatorEvaluate:
    def test_evaluate_single_expr(self, evaluator, sample_data):
        result = evaluator.evaluate("x + y", sample_data, ["x", "y"])
        expected = sample_data[:, 0] + sample_data[:, 1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_uses_cache(self, evaluator, sample_data):
        evaluator.evaluate("sin(x)", sample_data, ["x", "y"])
        assert evaluator.cache_stats()["cached_expressions"] == 1


# ── CompiledEvaluator: evaluate_batch_vec ──

class TestCompiledEvaluatorBatch:
    def test_batch_basic(self, evaluator, sample_data):
        exprs = ["x + y", "x * y"]
        result = evaluator.evaluate_batch_vec(exprs, sample_data, ["x", "y"])
        assert result.shape == (5, 2)
        np.testing.assert_array_almost_equal(result[:, 0], sample_data[:, 0] + sample_data[:, 1])
        np.testing.assert_array_almost_equal(result[:, 1], sample_data[:, 0] * sample_data[:, 1])

    def test_batch_with_invalid_expr(self, evaluator, sample_data):
        exprs = ["x + y", "invalid_func(x)"]
        result = evaluator.evaluate_batch_vec(exprs, sample_data, ["x", "y"])
        assert result.shape == (5, 2)
        # First expression should work
        np.testing.assert_array_almost_equal(result[:, 0], sample_data[:, 0] + sample_data[:, 1])
        # Second should be NaN
        assert np.all(np.isnan(result[:, 1]))

    def test_batch_empty_exprs(self, evaluator, sample_data):
        result = evaluator.evaluate_batch_vec([], sample_data, ["x", "y"])
        assert result.shape == (5, 0)


# ── CompiledEvaluator: clear_cache ──

class TestCompiledEvaluatorCache:
    def test_clear_cache(self, evaluator):
        evaluator.compile("sin(x)", ["x"])
        assert evaluator.cache_stats()["cached_expressions"] == 1
        evaluator.clear_cache()
        assert evaluator.cache_stats()["cached_expressions"] == 0


# ── FastSymbolLayer ──

class TestFastSymbolLayer:
    def test_unary_op(self):
        layer = FastSymbolLayer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        offsets = [("sin", 0, -1)]
        result = layer.compute_layer(offsets, X)
        np.testing.assert_array_almost_equal(result[:, 0], np.sin(X[:, 0]))

    def test_binary_op(self):
        layer = FastSymbolLayer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        offsets = [("add", 0, 1)]
        result = layer.compute_layer(offsets, X)
        np.testing.assert_array_almost_equal(result[:, 0], X[:, 0] + X[:, 1])

    def test_unknown_op_returns_zero(self):
        layer = FastSymbolLayer()
        X = np.array([[1.0, 2.0]])
        offsets = [("unknown_op", 0, -1)]
        result = layer.compute_layer(offsets, X)
        assert result[0, 0] == 0.0

    def test_multiple_offsets(self):
        layer = FastSymbolLayer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        offsets = [("add", 0, 1), ("mul", 0, 1), ("sin", 0, -1)]
        result = layer.compute_layer(offsets, X)
        assert result.shape == (2, 3)


# ── _eml function ──

class TestEMLFunction:
    def test_eml_basic(self):
        result = _eml(0.0, 1.0)
        expected = math.exp(0.0) - math.log(1.0 + 1e-10)
        assert abs(result - expected) < 1e-6

    def test_eml_with_negative_y(self):
        result = _eml(1.0, -2.0)
        # abs(-2) + 1e-10 = 2.0000000001, log of that
        expected = math.exp(1.0) - math.log(2.0 + 1e-10)
        assert abs(result - expected) < 1e-4


# ── PSEConfig ──

class TestPSEConfig:
    def test_default_config(self):
        config = PSEConfig()
        assert config.max_iterations == 10
        assert config.n_top_expressions == 5
        assert config.reward_discount == 0.99
        assert config.early_stop_mse == 1e-12
        assert config.no_improvement_limit == 3
        assert config.token_generator_type == "fast"

    def test_custom_config(self):
        config = PSEConfig(max_iterations=5, token_generator_type="random")
        assert config.max_iterations == 5
        assert config.token_generator_type == "random"


# ── PSEEngine: creation ──

class TestPSEEngineCreation:
    def test_creates_with_default_config(self):
        engine = PSEEngine()
        assert engine.config.token_generator_type == "fast"
        assert engine.pareto_front == []
        assert engine.reward_history == {}

    def test_creates_with_random_generator(self):
        config = PSEConfig(token_generator_type="random")
        engine = PSEEngine(config)
        assert engine.config.token_generator_type == "random"

    def test_unknown_generator_raises(self):
        config = PSEConfig(token_generator_type="nonexistent")
        with pytest.raises(ValueError, match="Unknown token generator"):
            PSEEngine(config)


# ── PSEEngine: internal methods ──

class TestPSEEngineInternals:
    def test_compute_complexity_simple(self):
        engine = PSEEngine()
        c = engine._compute_complexity("x + y")
        assert c >= 1

    def test_compute_complexity_nested(self):
        engine = PSEEngine()
        c1 = engine._compute_complexity("x + y")
        c2 = engine._compute_complexity("sin(x) + cos(y) + exp(z)")
        assert c2 > c1

    def test_compute_complexity_empty(self):
        engine = PSEEngine()
        c = engine._compute_complexity("")
        assert c == 1  # max(complexity, 1)

    def test_compute_reward(self):
        engine = PSEEngine()
        r = engine._compute_reward(0.0, 1)
        # discount^1 / (1 + sqrt(0)) = 0.99 / 1.0 = 0.99
        assert abs(r - 0.99) < 1e-6

    def test_compute_reward_high_mse(self):
        engine = PSEEngine()
        r = engine._compute_reward(100.0, 1)
        # 0.99 / (1 + 10) = 0.09
        assert r < 0.1

    def test_update_pareto_front(self):
        engine = PSEEngine()
        engine._update_pareto_front("x+y", 0.1, 3, 0.5)
        assert len(engine.pareto_front) == 1
        assert engine.pareto_front[0][0] == "x+y"

    def test_pareto_front_dominated_not_added(self):
        engine = PSEEngine()
        engine._update_pareto_front("x+y", 0.1, 3, 0.5)
        # Dominated: same complexity but worse MSE
        engine._update_pareto_front("x-y", 0.2, 3, 0.4)
        # Should not add dominated entry
        assert len(engine.pareto_front) == 1

    def test_pareto_front_non_dominated_added(self):
        engine = PSEEngine()
        engine._update_pareto_front("x+y", 0.1, 5, 0.5)
        # Non-dominated: lower complexity, higher MSE
        engine._update_pareto_front("x", 0.5, 1, 0.3)
        assert len(engine.pareto_front) == 2

    def test_safe_eval_basic(self):
        engine = PSEEngine()
        var_dict = {"x": np.array([1.0, 2.0, 3.0])}
        result = engine._safe_eval("sin(x)", var_dict)
        np.testing.assert_array_almost_equal(result, np.sin([1.0, 2.0, 3.0]))

    def test_safe_eval_constant(self):
        engine = PSEEngine()
        var_dict = {"x": np.array([1.0, 2.0])}
        result = engine._safe_eval("3.14", var_dict)
        np.testing.assert_array_almost_equal(result, [3.14, 3.14])

    def test_safe_eval_error_returns_zeros(self):
        engine = PSEEngine()
        var_dict = {"x": np.array([1.0, 2.0])}
        result = engine._safe_eval("1/0", var_dict)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_get_pareto_summary(self):
        engine = PSEEngine()
        engine._update_pareto_front("x+y", 0.01, 3, 0.9)
        summary = engine.get_pareto_summary()
        assert "Pareto Front" in summary
        assert "x+y" in summary


# ── PSRNSymbolicRegression (Bridge) ──

class TestPSRNSymbolicRegressionCreation:
    def test_creates_with_defaults(self):
        sr = PSRNSymbolicRegression()
        assert sr.n_layers == 2
        assert sr.n_input_slots == 5
        assert sr.max_iterations == 5
        assert sr._best_expr == ""
        assert sr.best_fitness_ == float("inf")

    def test_creates_with_custom_params(self):
        sr = PSRNSymbolicRegression(n_layers=3, max_iterations=10, token_generator="random")
        assert sr.n_layers == 3
        assert sr.max_iterations == 10
        assert sr.token_generator_type == "random"


class TestPSRNSymbolicRegressionPredict:
    def test_predict_before_fit_raises(self):
        sr = PSRNSymbolicRegression()
        with pytest.raises(ValueError, match="not fitted"):
            sr.predict(np.array([[1.0]]))

    def test_get_pareto_front_before_fit(self):
        sr = PSRNSymbolicRegression()
        assert sr.get_pareto_front() == []


class TestPSRNSymbolicRegressionEval:
    def test_eval_expr_basic(self):
        sr = PSRNSymbolicRegression()
        result = sr._eval_expr("sin(0)", {})
        assert abs(result) < 1e-10

    def test_eval_expr_with_vars(self):
        sr = PSRNSymbolicRegression()
        result = sr._eval_expr("x + y", {"x": 3.0, "y": 4.0})
        assert abs(result - 7.0) < 1e-10

    def test_eval_expr_error_returns_zero(self):
        sr = PSRNSymbolicRegression()
        result = sr._eval_expr("1/0", {})
        assert result == 0.0

    def test_eval_expr_pi(self):
        sr = PSRNSymbolicRegression()
        result = sr._eval_expr("pi", {})
        assert abs(result - math.pi) < 1e-10

    def test_expr_to_node_empty(self):
        sr = PSRNSymbolicRegression()
        node = sr._expr_to_node("")
        assert node is not None

    def test_expr_to_node_with_vars(self):
        sr = PSRNSymbolicRegression()
        sr.variables = ["x"]
        node = sr._expr_to_node("sin(x)")
        assert node is not None


# ── upgrade functions ──

class TestUpgradeFunctions:
    def test_upgrade_to_psrn(self):
        from math_anything.eml_v2 import ImprovedSymbolicRegression
        sr = ImprovedSymbolicRegression()
        psrn_sr = upgrade_to_psrn(sr, n_layers=3, max_iterations=7)
        assert isinstance(psrn_sr, PSRNSymbolicRegression)
        assert psrn_sr.n_layers == 3
        assert psrn_sr.max_iterations == 7

    def test_upgrade_to_enhanced_psrn(self):
        from math_anything.eml_v2 import ImprovedSymbolicRegression
        sr = ImprovedSymbolicRegression()
        enhanced = upgrade_to_enhanced_psrn(sr, n_layers=2, max_layer_size=200)
        assert isinstance(enhanced, EnhancedPSRNSymbolicRegression)
        assert enhanced.max_layer_size == 200


# ── NumbaEvaluator ──

class TestNumbaEvaluator:
    def test_creates(self):
        ev = NumbaEvaluator()
        assert hasattr(ev, "_has_numba")

    def test_evaluate_batch_delegates(self):
        ev = NumbaEvaluator()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ev.evaluate_batch(["x + y"], X, ["x", "y"])
        assert result.shape == (2, 1)
        np.testing.assert_array_almost_equal(result[:, 0], X[:, 0] + X[:, 1])


# ── _VECTORIZED_OPS coverage ──

class TestVectorizedOps:
    def test_identity(self):
        result = _VECTORIZED_OPS["identity"](5.0)
        assert result == 5.0

    def test_neg(self):
        result = _VECTORIZED_OPS["neg"](3.0)
        assert result == -3.0

    def test_inv(self):
        result = _VECTORIZED_OPS["inv"](2.0)
        assert abs(result - 0.5) < 1e-6

    def test_sub(self):
        result = _VECTORIZED_OPS["sub"](5.0, 3.0)
        assert result == 2.0

    def test_div(self):
        result = _VECTORIZED_OPS["div"](10.0, 2.0)
        assert abs(result - 5.0) < 1e-6

    def test_pow(self):
        result = _VECTORIZED_OPS["pow"](2.0, 3.0)
        assert abs(result - 8.0) < 1e-6
