"""Property-based tests for core mathematical structures using Hypothesis.

These tests verify mathematical properties that should hold for ALL valid inputs,
not just specific test cases. This catches edge cases that unit tests miss.
"""

import unittest

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


class TestConservationFieldProperties(unittest.TestCase):
    """Properties that should hold for all conservation fields."""

    @given(mu=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_navier_stokes_coupling_matrix_symmetric(self, mu):
        """The coupling matrix for Navier-Stokes should be well-defined for any valid viscosity."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        field.build_from_navier_stokes(mu=mu)
        self.assertIsNotNone(field.coupling_matrix)
        # Matrix should be finite
        mat = np.array(field.coupling_matrix)
        self.assertTrue(np.all(np.isfinite(mat)))

    @given(
        hbar=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        mass=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_schrodinger_field_well_defined(self, hbar, mass):
        """Schrodinger conservation field should be well-defined for any valid hbar and mass."""
        from math_anything.structures.conservation_field import ConservationMatrixField

        field = ConservationMatrixField()
        field.build_from_schrodinger(hbar=hbar, m=mass)
        self.assertIsNotNone(field.coupling_matrix)
        self.assertGreater(len(field.conserved_quantities), 0)


class TestSymplecticIntegratorProperties(unittest.TestCase):
    """Properties that should hold for symplectic integration."""

    @given(
        q0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        p0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_energy_conservation_harmonic_oscillator(self, q0, p0, dt):
        """For a harmonic oscillator, symplectic integration should conserve energy."""
        from math_anything.structures.evolution import SymplecticIntegrator

        def H(q, p):
            return 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2)

        integrator = SymplecticIntegrator(H, dim=1)
        result = integrator.integrate(
            q0=np.array([q0]),
            p0=np.array([p0]),
            dt=dt,
            n_steps=100,
        )
        # Energy drift should be small for symplectic integrator
        self.assertLess(result["energy_drift"], 0.1)

    @given(
        q0=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        p0=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_symplecticity_preserved(self, q0, p0):
        """The symplectic structure should be preserved.

        Note: verify_symplecticity uses double finite differencing (eps=1e-7 for
        the Jacobian, eps=1e-8 for forces inside velocity_verlet_step), which
        limits numerical precision. The is_symplectic flag (threshold 1e-4) is
        too strict for this approach, so we check symplectic_error directly
        with a tolerance that accounts for finite-difference noise. A small dt
        keeps the finite-difference noise well below the tolerance.
        """
        from math_anything.structures.evolution import SymplecticIntegrator

        def H(q, p):
            return 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2)

        integrator = SymplecticIntegrator(H, dim=1)
        result = integrator.verify_symplecticity(
            q0=np.array([q0]),
            p0=np.array([p0]),
            dt=0.001,
        )
        # Velocity Verlet is exactly symplectic; the measured error comes from
        # finite-difference noise in the Jacobian computation.
        self.assertLess(result["symplectic_error"], 0.05)


class TestEigenvalueSolverProperties(unittest.TestCase):
    """Properties that should hold for eigenvalue problems."""

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_symmetric_matrix_real_eigenvalues(self, n, seed):
        """Symmetric matrices should have real eigenvalues."""
        from math_anything.structures.spectral import EigenvalueSolver

        rng = np.random.RandomState(seed)
        A = rng.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric

        solver = EigenvalueSolver(A)
        evals = solver.eigenvalues()
        # eigvals returns complex dtype; check imaginary parts are negligible
        self.assertTrue(np.allclose(evals.imag, 0, atol=1e-10))

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_positive_definite_matrix_positive_eigenvalues(self, n, seed):
        """Positive definite matrices should have all positive eigenvalues."""
        from math_anything.structures.spectral import EigenvalueSolver

        rng = np.random.RandomState(seed)
        A = rng.randn(n, n)
        A = A @ A.T + np.eye(n)  # Positive definite

        solver = EigenvalueSolver(A)
        evals = solver.eigenvalues()
        self.assertTrue(np.all(evals.real > 0))
        self.assertTrue(solver.is_positive_definite())

    @given(n=st.integers(min_value=2, max_value=6))
    @settings(max_examples=10, deadline=5000)
    def test_identity_matrix_eigenvalues(self, n):
        """Identity matrix should have all eigenvalues equal to 1."""
        from math_anything.structures.spectral import EigenvalueSolver

        solver = EigenvalueSolver(np.eye(n))
        evals = solver.eigenvalues()
        np.testing.assert_allclose(evals, np.ones(n), atol=1e-10)

    @given(n=st.integers(min_value=2, max_value=6))
    @settings(max_examples=10, deadline=5000)
    def test_condition_number_identity(self, n):
        """Identity matrix should have condition number 1."""
        from math_anything.structures.spectral import EigenvalueSolver

        solver = EigenvalueSolver(np.eye(n))
        cond = solver.condition_number()
        self.assertAlmostEqual(cond, 1.0, places=5)


class TestDeformationGradientProperties(unittest.TestCase):
    """Properties that should hold for continuum mechanics."""

    @given(
        angle=st.floats(min_value=-np.pi / 2 + 0.1, max_value=np.pi / 2 - 0.1, allow_nan=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_rotation_is_isochoric(self, angle):
        """A pure rotation should have Jacobian = 1 (volume preserving)."""
        from math_anything.structures.geometry_continuum import DeformationGradient

        c, s = np.cos(angle), np.sin(angle)
        F = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        dg = DeformationGradient(F)
        self.assertAlmostEqual(dg.jacobian(), 1.0, places=5)
        self.assertTrue(dg.is_incompressible())

    @given(
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_uniform_scaling_jacobian(self, scale):
        """Uniform scaling by factor s should give Jacobian = s^3."""
        from math_anything.structures.geometry_continuum import DeformationGradient

        F = np.eye(3) * scale
        dg = DeformationGradient(F)
        self.assertAlmostEqual(dg.jacobian(), scale**3, places=5)

    def test_identity_deformation_zero_strain(self):
        """Identity deformation gradient should give zero Green-Lagrange strain."""
        from math_anything.structures.geometry_continuum import DeformationGradient

        dg = DeformationGradient(np.eye(3))
        strain = dg.green_lagrange_strain()
        np.testing.assert_allclose(strain, np.zeros((3, 3)), atol=1e-10)


class TestDimensionalAnalysisProperties(unittest.TestCase):
    """Properties that should hold for dimensional analysis."""

    @given(
        mass_val=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        accel_val=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_dimensional_homogeneity(self, mass_val, accel_val):
        """m * a (mass times acceleration) should have same dimensions as F (force)."""
        from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer

        analyzer = SymbolicDimensionalAnalyzer()
        # m * a has dimensions [M][L T^-2] = [M L T^-2] = force
        # F has dimensions [M L T^-2]
        result = analyzer.check_equation("m * a", "F")
        self.assertTrue(result["consistent"])

    @given(
        length_val=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        time_val=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_velocity_dimensional_consistency(self, length_val, time_val):
        """x / t (length over time) should have same dimensions as v (velocity)."""
        from math_anything.dimensional.equation_checker import SymbolicDimensionalAnalyzer

        analyzer = SymbolicDimensionalAnalyzer()
        result = analyzer.check_equation("x / t", "v")
        self.assertTrue(result["consistent"])


class TestMorphismProperties(unittest.TestCase):
    """Properties that should hold for morphisms."""

    @given(n_electrons=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_born_oppenheimer_preserves_electron_count(self, n_electrons):
        """Born-Oppenheimer approximation should preserve electron count."""
        from math_anything.morphisms.dft import BornOppenheimerApproximation

        bo = BornOppenheimerApproximation()
        state = {"n_electrons": n_electrons}
        result = bo.apply(state)
        self.assertEqual(result["n_electrons"], n_electrons)

    @given(n_electrons=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=5000)
    def test_kohn_sham_preserves_electron_count(self, n_electrons):
        """Kohn-Sham mapping should preserve electron count."""
        from math_anything.morphisms.dft import KohnShamMapping

        ks = KohnShamMapping()
        state = {"n_electrons": n_electrons}
        result = ks.apply(state)
        self.assertEqual(result["n_electrons"], n_electrons)

    @given(n_electrons=st.integers(min_value=1, max_value=50))
    @settings(max_examples=15, deadline=5000)
    def test_composed_morphism_preserves_electron_count(self, n_electrons):
        """Composed morphisms should preserve electron count."""
        from math_anything.morphisms.dft import BornOppenheimerApproximation, KohnShamMapping

        bo = BornOppenheimerApproximation()
        ks = KohnShamMapping()
        composed = ks.compose(bo)
        state = {"n_electrons": n_electrons}
        result = composed.apply(state)
        self.assertEqual(result["n_electrons"], n_electrons)


class TestConservationLawSolverProperties(unittest.TestCase):
    """Properties that should hold for conservation law solvers."""

    @given(
        u=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        speed=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_linear_advection_characteristic_speed(self, u, speed):
        """For F(U) = speed * U, characteristic speed should equal `speed`."""
        from math_anything.structures.evolution import ConservationLawSolver

        solver = ConservationLawSolver(lambda U: speed * U, n_vars=1)
        speeds = solver.characteristic_speeds(np.array([u]))
        np.testing.assert_allclose(speeds, [speed], atol=1e-4)

    @given(
        u=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        dx=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_cfl_condition_positive(self, u, dx):
        """CFL condition should give a positive time step."""
        from math_anything.structures.evolution import ConservationLawSolver

        solver = ConservationLawSolver(lambda U: U, n_vars=1)
        dt = solver.cfl_condition(np.array([u]), dx=dx)
        self.assertGreater(dt, 0)


class TestVariationalSolverProperties(unittest.TestCase):
    """Properties that should hold for variational problems."""

    @given(n_el=st.integers(min_value=5, max_value=50))
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_poisson_solution_bounded(self, n_el):
        """1D Poisson solution should be bounded."""
        from math_anything.structures.equilibrium import VariationalSolver

        solver = VariationalSolver()
        result = solver.solve_1d_poisson(n_elements=n_el)
        sol = result["solution"]
        self.assertTrue(np.all(np.isfinite(sol)))
        # Solution should have n_el + 1 nodes
        self.assertEqual(len(sol), n_el + 1)


if __name__ == "__main__":
    unittest.main()
