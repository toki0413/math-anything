"""Tests for evolution numerical solvers — SymplecticIntegrator and ConservationLawSolver."""

import pytest
import numpy as np

from math_anything.structures.evolution import SymplecticIntegrator, ConservationLawSolver


class TestSymplecticIntegrator:

    @pytest.fixture
    def harmonic_oscillator(self):
        """1D harmonic oscillator: H = p^2/2 + q^2/2."""
        return SymplecticIntegrator(lambda q, p: 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2), dim=1)

    def test_harmonic_oscillator_trajectory(self, harmonic_oscillator):
        result = harmonic_oscillator.integrate(
            q0=np.array([1.0]),
            p0=np.array([0.0]),
            dt=0.01,
            n_steps=1000,
        )
        assert result["q"].shape == (1001, 1)
        assert result["p"].shape == (1001, 1)
        assert len(result["energy"]) == 1001

    def test_energy_conservation(self, harmonic_oscillator):
        result = harmonic_oscillator.integrate(
            q0=np.array([1.0]),
            p0=np.array([0.0]),
            dt=0.01,
            n_steps=1000,
        )
        # Velocity Verlet should conserve energy well
        assert result["energy_drift"] < 0.01

    def test_trajectory_oscillation(self, harmonic_oscillator):
        """Harmonic oscillator should oscillate, not diverge."""
        result = harmonic_oscillator.integrate(
            q0=np.array([1.0]),
            p0=np.array([0.0]),
            dt=0.01,
            n_steps=2000,
        )
        # Position should stay bounded
        assert np.max(np.abs(result["q"])) < 2.0

    def test_verify_symplecticity(self, harmonic_oscillator):
        result = harmonic_oscillator.verify_symplecticity(
            q0=np.array([1.0]),
            p0=np.array([0.0]),
            dt=0.01,
        )
        assert result["is_symplectic"] == True
        assert abs(result["det_J"] - 1.0) < 1e-3

    def test_2d_harmonic_oscillator(self):
        H = lambda q, p: 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2)
        integrator = SymplecticIntegrator(H, dim=2)
        result = integrator.integrate(
            q0=np.array([1.0, 0.0]),
            p0=np.array([0.0, 1.0]),
            dt=0.01,
            n_steps=500,
        )
        assert result["q"].shape == (501, 2)
        assert result["energy_drift"] < 0.01

    def test_zero_initial_conditions(self, harmonic_oscillator):
        result = harmonic_oscillator.integrate(
            q0=np.array([0.0]),
            p0=np.array([0.0]),
            dt=0.01,
            n_steps=100,
        )
        # At equilibrium, should stay at zero
        np.testing.assert_allclose(result["q"][-1], [0.0], atol=1e-10)
        np.testing.assert_allclose(result["p"][-1], [0.0], atol=1e-10)


class TestConservationLawSolver:

    @pytest.fixture
    def linear_advection(self):
        """Linear advection: F(U) = U, characteristic speed = 1."""
        return ConservationLawSolver(lambda U: U, n_vars=1)

    @pytest.fixture
    def double_advection(self):
        """F(U) = 2U, characteristic speed = 2."""
        return ConservationLawSolver(lambda U: 2 * U, n_vars=1)

    def test_flux_jacobian_linear(self, linear_advection):
        J = linear_advection.flux_jacobian(np.array([1.0]))
        np.testing.assert_allclose(J, [[1.0]], atol=1e-5)

    def test_flux_jacobian_nonlinear(self):
        """Burgers flux: F(U) = U^2/2, dF/dU = U."""
        solver = ConservationLawSolver(lambda U: 0.5 * U**2, n_vars=1)
        J = solver.flux_jacobian(np.array([3.0]))
        np.testing.assert_allclose(J, [[3.0]], atol=1e-5)

    def test_characteristic_speeds(self, linear_advection):
        speeds = linear_advection.characteristic_speeds(np.array([1.0]))
        np.testing.assert_allclose(speeds, [1.0], atol=1e-5)

    def test_max_wave_speed(self, double_advection):
        speed = double_advection.max_wave_speed(np.array([1.0]))
        assert abs(speed - 2.0) < 1e-5

    def test_cfl_condition(self, linear_advection):
        dt = linear_advection.cfl_condition(np.array([1.0]), dx=0.1, cfl_number=0.5)
        assert abs(dt - 0.05) < 1e-5

    def test_cfl_condition_zero_speed(self):
        """Zero wave speed should return inf."""
        solver = ConservationLawSolver(lambda U: np.zeros_like(U), n_vars=1)
        dt = solver.cfl_condition(np.array([1.0]), dx=0.1, cfl_number=0.5)
        assert dt == float('inf')

    def test_lax_friedrichs_step(self, linear_advection):
        """Lax-Friedrichs step should not blow up for smooth data."""
        n = 10
        U = np.ones(n)
        U_new = linear_advection.lax_friedrichs_step(U, dx=0.1, dt=0.01)
        assert U_new.shape == (n,)
        # Boundaries should be unchanged
        assert U_new[0] == U[0]
        assert U_new[-1] == U[-1]

    def test_lax_friedrichs_preserves_constant(self, linear_advection):
        """Constant state should remain constant (up to boundary)."""
        n = 10
        U = np.ones(n) * 5.0
        U_new = linear_advection.lax_friedrichs_step(U, dx=0.1, dt=0.01)
        # Interior should stay constant for advection of constant
        np.testing.assert_allclose(U_new[1:-1], 5.0, atol=1e-10)
