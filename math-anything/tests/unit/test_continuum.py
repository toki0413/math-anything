"""Tests for continuum mechanics — DeformationGradient."""

import numpy as np
import pytest

from math_anything.structures.geometry_continuum import DeformationGradient


class TestDeformationGradient:
    def test_identity_deformation(self):
        F = np.eye(3)
        dg = DeformationGradient(F)
        E = dg.green_lagrange_strain()
        np.testing.assert_allclose(E, np.zeros((3, 3)), atol=1e-10)

    def test_jacobian_identity(self):
        F = np.eye(3)
        dg = DeformationGradient(F)
        assert dg.jacobian() == pytest.approx(1.0, abs=1e-10)

    def test_incompressible(self):
        # Simple shear: volume-preserving
        F = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dg = DeformationGradient(F)
        assert dg.is_incompressible()

    def test_right_cauchy_green(self):
        F = np.array([[2.0, 0.0], [0.0, 1.0]])
        dg = DeformationGradient(F)
        C = dg.right_cauchy_green()
        expected = F.T @ F
        np.testing.assert_allclose(C, expected, atol=1e-10)

    def test_principal_stretches(self):
        F = np.array([[2.0, 0.0], [0.0, 1.0]])
        dg = DeformationGradient(F)
        stretches = dg.principal_stretches()
        np.testing.assert_allclose(sorted(stretches), [1.0, 2.0], atol=1e-10)

    def test_polar_decomposition(self):
        F = np.array([[2.0, 0.0], [0.0, 1.0]])
        dg = DeformationGradient(F)
        R, U = dg.polar_decomposition()
        # R should be orthogonal
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)
        # F = R @ U
        np.testing.assert_allclose(R @ U, F, atol=1e-10)

    def test_cauchy_stress(self):
        F = np.eye(2)
        dg = DeformationGradient(F)
        sigma = dg.cauchy_stress(lame_lambda=100.0, lame_mu=50.0)
        # Zero strain → zero stress
        np.testing.assert_allclose(sigma, np.zeros((2, 2)), atol=1e-10)

    def test_von_mises_zero(self):
        F = np.eye(2)
        dg = DeformationGradient(F)
        vm = dg.von_mises_stress(lame_lambda=100.0, lame_mu=50.0)
        assert vm == pytest.approx(0.0, abs=1e-10)

    def test_von_mises_nonzero(self):
        F = np.array([[1.1, 0.0], [0.0, 1.0]])
        dg = DeformationGradient(F)
        vm = dg.von_mises_stress(lame_lambda=100.0, lame_mu=50.0)
        assert vm > 0.0
