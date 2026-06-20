"""Tests for SymbolicDimensionalAnalyzer."""

import numpy as np
import pytest

from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer


class TestSymbolicDimensionalAnalyzer:
    @pytest.fixture()
    def analyzer(self):
        return SymbolicDimensionalAnalyzer()

    def test_parse_simple_variable(self, analyzer):
        dim = analyzer.parse_expression("F")
        assert dim is not None
        np.testing.assert_allclose(dim, [1, 1, -2, 0, 0, 0, 0], atol=1e-10)

    def test_parse_multiplication(self, analyzer):
        dim = analyzer.parse_expression("m * a")
        assert dim is not None
        np.testing.assert_allclose(dim, [1, 1, -2, 0, 0, 0, 0], atol=1e-10)

    def test_parse_division(self, analyzer):
        dim = analyzer.parse_expression("v / t")
        assert dim is not None
        np.testing.assert_allclose(dim, [1, 0, -2, 0, 0, 0, 0], atol=1e-10)

    def test_parse_power(self, analyzer):
        dim = analyzer.parse_expression("v ^ 2")
        assert dim is not None
        np.testing.assert_allclose(dim, [2, 0, -2, 0, 0, 0, 0], atol=1e-10)

    def test_check_equation_consistent(self, analyzer):
        result = analyzer.check_equation("F", "m * a")
        assert result["consistent"]

    def test_check_equation_inconsistent(self, analyzer):
        result = analyzer.check_equation("F", "m * v")
        assert not result["consistent"]

    def test_rho_v_squared(self, analyzer):
        """rho*v^2 has dimensions of pressure."""
        dim = analyzer.parse_expression("rho * v * v")
        assert dim is not None
        np.testing.assert_allclose(dim, [-1, 1, -2, 0, 0, 0, 0], atol=1e-10)

    def test_rho_v_squared_equals_p(self, analyzer):
        result = analyzer.check_equation("rho * v * v", "p")
        assert result["consistent"]

    def test_register_custom_variable(self, analyzer):
        analyzer.register_variable("custom_var", [1, 2, -3, 0, 0, 0, 0])
        dim = analyzer.parse_expression("custom_var")
        assert dim is not None
        np.testing.assert_allclose(dim, [1, 2, -3, 0, 0, 0, 0], atol=1e-10)
