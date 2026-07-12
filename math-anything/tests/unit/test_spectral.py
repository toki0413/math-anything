"""Tests for spectral problems — EigenvalueSolver and SelfConsistentSolver."""

import numpy as np
import pytest

from math_anything.structures.spectral import EigenvalueSolver, SelfConsistentSolver


class TestEigenvalueSolver:
    def test_2x2_symmetric(self):
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        solver = EigenvalueSolver(M)
        evals = np.sort(solver.eigenvalues().real)
        assert evals[0] == pytest.approx(1.0, abs=1e-10)
        assert evals[1] == pytest.approx(3.0, abs=1e-10)

    def test_eigenvectors(self):
        M = np.array([[2.0, 0.0], [0.0, 3.0]])
        solver = EigenvalueSolver(M)
        evals, evecs = solver.eigenvectors()
        sorted_evals = np.sort(evals.real)
        assert sorted_evals[0] == pytest.approx(2.0, abs=1e-10)
        assert sorted_evals[1] == pytest.approx(3.0, abs=1e-10)

    def test_spectral_gap(self):
        M = np.array([[1.0, 0.0], [0.0, 3.0]])
        solver = EigenvalueSolver(M)
        gap = solver.spectral_gap()
        assert gap == pytest.approx(2.0, abs=1e-10)

    def test_condition_number(self):
        M = np.array([[1.0, 0.0], [0.0, 10.0]])
        solver = EigenvalueSolver(M)
        cond = solver.condition_number()
        assert cond == pytest.approx(10.0, abs=1e-5)

    def test_positive_definite(self):
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        solver = EigenvalueSolver(M)
        assert solver.is_positive_definite()

    def test_not_positive_definite(self):
        M = np.array([[-1.0, 0.0], [0.0, 2.0]])
        solver = EigenvalueSolver(M)
        assert not solver.is_positive_definite()

    def test_self_adjoint(self):
        M = np.array([[2.0, 1.0], [1.0, 3.0]])
        solver = EigenvalueSolver(M)
        assert solver.is_self_adjoint()

    def test_not_self_adjoint(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        solver = EigenvalueSolver(M)
        assert not solver.is_self_adjoint()


class TestSelfConsistentSolver:
    def test_convergence(self):
        """SCF convergence on a simple model Hamiltonian."""

        def hamiltonian_builder(density):
            H0 = np.array([[-1.0, 0.5], [0.5, 1.0]])
            V = 0.5 * density
            return H0 + V

        solver = SelfConsistentSolver(
            hamiltonian_builder=hamiltonian_builder,
            n_states=1,
            mixing=0.3,
            max_iter=100,
            tol=1e-6,
        )
        result = solver.solve(np.eye(2) * 0.5)
        assert result["converged"]
        assert result["iterations"] < 100
        assert len(result["eigenvalues"]) == 2

    def test_max_iter_reached(self):
        """Solver returns converged=False when max_iter is reached."""

        def stubborn_hamiltonian(density):
            H0 = np.array([[0.0, 1.0], [1.0, 0.0]])
            return H0 + 10.0 * (density - 0.5 * np.eye(2))

        solver = SelfConsistentSolver(
            hamiltonian_builder=stubborn_hamiltonian,
            n_states=1,
            mixing=0.1,
            max_iter=5,
            tol=1e-10,
        )
        result = solver.solve(np.eye(2) * 0.5)
        assert not result["converged"]
        assert result["iterations"] == 5
