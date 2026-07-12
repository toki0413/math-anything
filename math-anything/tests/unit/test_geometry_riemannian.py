"""Tests for Riemannian geometry — MetricFunction, Christoffel symbols, curvature, Lie derivatives."""

import numpy as np
import pytest

from math_anything.structures.geometry_riemannian import (
    MetricFunction,
    flat_metric,
    lie_derivative_metric,
    lie_derivative_scalar,
    lie_derivative_vector_field,
    schwarzschild_metric,
    spherical_metric,
)


class TestMetricFunction:
    def test_flat_metric_at_origin(self):
        """Flat metric should be identity at any point."""
        mf = flat_metric(dim=3)
        g = mf.at({"x0": 0.0, "x1": 0.0, "x2": 0.0})
        np.testing.assert_allclose(g, np.eye(3), atol=1e-10)

    def test_flat_metric_christoffel_zero(self):
        """Flat metric should have zero Christoffel symbols."""
        mf = flat_metric(dim=2)
        coords = {"x0": 0.0, "x1": 0.0}
        gamma = mf.christoffel_at(coords)
        np.testing.assert_allclose(gamma, 0, atol=1e-5)

    def test_flat_metric_riemann_zero(self):
        """Flat metric should have zero Riemann curvature."""
        mf = flat_metric(dim=2)
        coords = {"x0": 0.0, "x1": 0.0}
        R = mf.riemann_at(coords)
        np.testing.assert_allclose(R, 0, atol=1e-3)

    def test_inverse_at(self):
        """Inverse metric should satisfy g * g_inv = I."""
        mf = flat_metric(dim=3)
        coords = {"x0": 1.0, "x1": 2.0, "x2": 3.0}
        g = mf.at(coords)
        g_inv = mf.inverse_at(coords)
        np.testing.assert_allclose(g @ g_inv, np.eye(3), atol=1e-10)

    def test_schwarzschild_metric_shape(self):
        """Schwarzschild metric should be 4x4."""
        mf = schwarzschild_metric(M=1.0)
        coords = {"t": 0.0, "r": 5.0, "theta": 1.0, "phi": 0.0}
        g = mf.at(coords)
        assert g.shape == (4, 4)
        # g_{00} should be negative (time component)
        assert g[0, 0] < 0
        # g_{11} should be positive (radial component)
        assert g[1, 1] > 0

    def test_schwarzschild_christoffel_nonzero(self):
        """Schwarzschild metric should have nonzero Christoffel symbols."""
        mf = schwarzschild_metric(M=1.0)
        coords = {"t": 0.0, "r": 5.0, "theta": 1.0, "phi": 0.0}
        gamma = mf.christoffel_at(coords)
        assert gamma.shape == (4, 4, 4)
        # At least some components should be nonzero
        assert np.max(np.abs(gamma)) > 1e-6

    def test_schwarzschild_ricci_curvature(self):
        """Schwarzschild is a vacuum solution: Ricci should be near zero."""
        mf = schwarzschild_metric(M=1.0)
        coords = {"t": 0.0, "r": 5.0, "theta": 1.0, "phi": 0.0}
        ricci = mf.ricci_at(coords, epsilon=1e-4)
        # Vacuum solution: R_{ij} = 0 (up to numerical error)
        np.testing.assert_allclose(ricci, 0, atol=0.1)

    def test_spherical_metric_diagonal(self):
        """Spherical metric should be diagonal."""
        mf = spherical_metric()
        coords = {"t": 0.0, "r": 2.0, "theta": 1.0, "phi": 0.0}
        g = mf.at(coords)
        # Off-diagonal should be zero
        off_diag = g - np.diag(np.diag(g))
        np.testing.assert_allclose(off_diag, 0, atol=1e-10)

    def test_scalar_curvature_flat(self):
        """Flat metric should have zero scalar curvature."""
        mf = flat_metric(dim=3)
        coords = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        R = mf.scalar_curvature_at(coords)
        assert abs(R) < 1e-3


class TestLieDerivatives:
    def test_lie_derivative_scalar_constant_field(self):
        """Lie derivative of a constant scalar along any field should be zero."""

        def X(c):
            return np.array([1.0, 0.0])

        def f(c):
            return 5.0  # constant

        coords = {"x0": 0.0, "x1": 0.0}
        result = lie_derivative_scalar(X, f, coords)
        assert abs(result) < 1e-5

    def test_lie_derivative_scalar_linear(self):
        """L_X f for X = d/dx and f = x should give 1."""

        def X(c):
            return np.array([1.0, 0.0])

        def f(c):
            return c["x0"]

        coords = {"x0": 1.0, "x1": 0.0}
        result = lie_derivative_scalar(X, f, coords)
        assert abs(result - 1.0) < 1e-4

    def test_lie_derivative_vector_commuting(self):
        """Lie derivative of d/dy along d/dx should be zero (commuting fields)."""

        def X(c):
            return np.array([1.0, 0.0])

        def Y(c):
            return np.array([0.0, 1.0])

        coords = {"x0": 0.0, "x1": 0.0}
        result = lie_derivative_vector_field(X, Y, coords)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-4)

    def test_lie_derivative_metric_killing(self):
        """For a Killing vector field, L_X g = 0."""

        # Translation in flat space is a Killing vector
        def X(c):
            return np.array([1.0, 0.0])

        mf = flat_metric(dim=2)
        coords = {"x0": 0.0, "x1": 0.0}
        result = lie_derivative_metric(X, mf, coords)
        np.testing.assert_allclose(result, 0, atol=1e-4)

    def test_lie_derivative_metric_non_killing(self):
        """For a non-Killing vector, L_X g != 0."""

        # Scaling field X = x d/dx is NOT a Killing vector for flat metric
        def X(c):
            return np.array([c["x0"], 0.0])

        mf = flat_metric(dim=2)
        coords = {"x0": 1.0, "x1": 0.0}
        result = lie_derivative_metric(X, mf, coords)
        # L_X g_{00} = 2 * dX^0/dx^0 = 2, so result should be nonzero
        assert abs(result[0, 0]) > 0.1
