"""Tests for equilibrium numerical solver — VariationalSolver."""

import numpy as np
import pytest

from math_anything.structures.equilibrium import VariationalSolver


class TestVariationalSolver:
    @pytest.fixture
    def solver(self):
        return VariationalSolver()

    def test_1d_poisson_basic(self, solver):
        result = solver.solve_1d_poisson(n_elements=10)
        assert "solution" in result
        assert "nodes" in result
        assert len(result["solution"]) == 11  # 10 elements + 1
        assert len(result["nodes"]) == 11

    def test_1d_poisson_positive_displacement(self, solver):
        """For f=1 source, the solution should be positive (parabolic)."""
        result = solver.solve_1d_poisson(n_elements=20)
        # Interior values should be positive for f=1
        interior = result["solution"][1:-1]
        assert max(interior) > 0

    def test_1d_poisson_boundary_conditions(self, solver):
        """Dirichlet BC: u(0) = u(L) = 0."""
        result = solver.solve_1d_poisson(n_elements=10)
        assert result["solution"][0] == 0.0
        assert result["solution"][-1] == 0.0

    def test_1d_poisson_convergence(self, solver):
        """Finer mesh should give smaller element size."""
        r10 = solver.solve_1d_poisson(n_elements=10)
        r20 = solver.solve_1d_poisson(n_elements=20)
        assert r20["h"] < r10["h"]
        assert abs(r10["h"] - 0.1) < 1e-10
        assert abs(r20["h"] - 0.05) < 1e-10

    def test_1d_poisson_custom_source(self, solver):
        """Test with a custom source function."""
        result = solver.solve_1d_poisson(
            n_elements=10,
            source=lambda x: 2.0,
        )
        assert len(result["solution"]) == 11

    def test_residual_norm(self, solver):
        K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        u = np.array([1.0, 1.0])
        f = np.array([1.0, 1.0])
        norm = solver.residual_norm(K, u, f)
        # Ku = [1, 1], f = [1, 1], Ku - f = [0, 0]
        assert abs(norm) < 1e-10

    def test_residual_norm_nonzero(self, solver):
        K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        u = np.array([0.0, 0.0])
        f = np.array([1.0, 1.0])
        norm = solver.residual_norm(K, u, f)
        assert norm > 0

    def test_condition_number_identity(self, solver):
        K = np.eye(3)
        cond = solver.condition_number(K)
        assert abs(cond - 1.0) < 1e-5

    def test_condition_number_ill_conditioned(self, solver):
        K = np.array([[1e10, 0], [0, 1e-10]])
        cond = solver.condition_number(K)
        assert cond > 1e10

    def test_energy_norm_error(self, solver):
        K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        u_h = np.array([1.0, 0.0])
        u_exact = np.array([1.0, 0.0])
        err = solver.energy_norm_error(K, u_h, u_exact)
        assert abs(err) < 1e-10

    def test_energy_norm_error_nonzero(self, solver):
        K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        u_h = np.array([0.0, 0.0])
        u_exact = np.array([1.0, 1.0])
        err = solver.energy_norm_error(K, u_h, u_exact)
        assert err > 0
