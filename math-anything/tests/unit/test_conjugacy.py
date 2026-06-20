"""Tests for RigorousConjugacyChecker."""

import numpy as np
import pytest

from math_anything.conjugacy import RigorousConjugacyChecker


class TestRigorousConjugacyChecker:
    @pytest.fixture()
    def checker(self):
        return RigorousConjugacyChecker(tol=1e-6, n_test_points=50)

    def test_identity_conjugacy(self, checker):
        """h(x)=x conjugates f to f: h(f(x)) = f(h(x))."""
        f = lambda x: x**2
        g = lambda x: x**2
        h = lambda x: x
        result = checker.check_conjugacy(f, g, h)
        assert result["conjugate"]
        assert result["max_error"] == pytest.approx(0.0, abs=1e-5)

    def test_non_conjugacy(self, checker):
        """f(x)=x^2 and g(x)=x^3 are not conjugate via h(x)=x."""
        f = lambda x: x**2
        g = lambda x: x**3
        h = lambda x: x
        result = checker.check_conjugacy(f, g, h)
        assert not result["conjugate"]

    def test_log_conjugacy(self, checker):
        """f(x)=2x and g(x)=x^2 are conjugate via h(x)=exp(x):
        h(f(x)) = exp(2x) = (exp(x))^2 = g(h(x))."""
        f = lambda x: 2 * x
        g = lambda x: x**2
        h = lambda x: np.exp(x)
        result = checker.check_conjugacy(f, g, h, domain=(0.1, 2.0))
        assert result["conjugate"]

    def test_find_conjugacy(self, checker):
        """Find the correct conjugacy from a family."""
        f = lambda x: x**2
        g = lambda x: x**2
        h_family = [
            lambda x: x,       # identity — correct
            lambda x: 2 * x,   # wrong
            lambda x: x**3,    # wrong
        ]
        result = checker.find_conjugacy(f, g, h_family)
        assert result["best_match_index"] == 0
