"""Unit tests for structures — aligned with actual APIs."""

import pytest


class TestSpectralStructures:
    def test_linear_eigenvalue_creates(self):
        from math_anything.structures import LinearEigenvalueProblem

        s = LinearEigenvalueProblem()
        assert s.family.value == "spectral"

    def test_nonlinear_eigenvalue_creates(self):
        from math_anything.structures import NonlinearEigenvalueProblem

        s = NonlinearEigenvalueProblem(nonlinearity_source="density_dependent")
        assert s.self_consistency_required
        assert len(s.structural_invariants) > 0

    def test_self_consistent_invariants(self, self_consistent_structure):
        invs = self_consistent_structure.structural_invariants
        names = [i.name for i in invs]
        assert "charge_conservation" in names

    def test_spectral_to_dict(self, self_consistent_structure):
        d = self_consistent_structure.to_dict()
        assert d["family"] == "spectral"


class TestEvolutionStructures:
    def test_hamiltonian_invariants(self, hamiltonian_structure):
        invs = hamiltonian_structure.structural_invariants
        names = [i.name for i in invs]
        assert any("symplectic" in n for n in names)

    def test_navier_stokes_creates(self, navier_stokes_structure):
        assert navier_stokes_structure.regime == "incompressible"
        assert navier_stokes_structure.buckingham_pi_count >= 2

    def test_ns_named_pi_groups(self, navier_stokes_structure):
        groups = navier_stokes_structure.named_pi_groups
        assert "Re" in groups

    def test_conservation_law_invariants(self):
        from math_anything.structures import ConservationLawSystem

        s = ConservationLawSystem(hyperbolic=True, has_diffusion=True)
        invs = s.structural_invariants
        [i.name for i in invs]


class TestEquilibriumStructures:
    def test_variational_convex(self, variational_structure):
        names = [i.name for i in variational_structure.structural_invariants]
        assert "unique_minimizer" in names

    def test_variational_nonconvex(self):
        from math_anything.structures import VariationalMinimizationProblem

        s = VariationalMinimizationProblem(convex=False)
        names = [i.name for i in s.structural_invariants]
        assert "multiple_minimizers_possible" in names

    def test_fixed_point_contraction(self):
        from math_anything.structures import FixedPointProblem

        s = FixedPointProblem(contraction=True, lipschitz_constant=0.5)
        names = [i.name for i in s.structural_invariants]
        assert "unique_fixed_point" in names


class TestGeometryStructures:
    def test_manifold_creates(self):
        from math_anything.structures import SmoothManifold

        s = SmoothManifold()
        assert len(s.structural_invariants) > 0

    def test_deformation_mapping(self):
        from math_anything.structures import DeformationMapping

        s = DeformationMapping()
        assert len(s.structural_invariants) > 0


class TestAlgebraStructures:
    def test_cstar_algebra(self):
        from math_anything.structures import CStarAlgebra

        s = CStarAlgebra()
        assert len(s.structural_invariants) > 0

    def test_gns_construction(self):
        from math_anything.structures import GNSConstruction

        s = GNSConstruction()
        assert len(s.structural_invariants) > 0


class TestNumericalAnalysis:
    def test_lax_theorem(self):
        from math_anything.structures import LaxEquivalenceTheorem

        s = LaxEquivalenceTheorem()
        assert len(s.structural_invariants) > 0

    def test_hyperbolic_cfl(self):
        from math_anything.structures import HyperbolicCFL

        s = HyperbolicCFL(cfl_number=0.5)
        assert s.cfl_number == 0.5

    def test_algebraic_convergence(self):
        from math_anything.structures import AlgebraicConvergence

        s = AlgebraicConvergence(order=2)
        assert s.order == 2


class TestGroupRepresentation:
    def test_finite_group_rep(self):
        from math_anything.structures import FiniteGroupRepresentation

        s = FiniteGroupRepresentation(group_order=24, conjugacy_classes=5)
        assert s.group_order == 24

    def test_wigner_eckart(self):
        from math_anything.structures import WignerEckartTheorem

        s = WignerEckartTheorem()
        assert len(s.structural_invariants) > 0


class TestCategoryTheory:
    def test_category_creates(self):
        from math_anything.structures import Category

        s = Category()
        assert len(s.structural_invariants) > 0

    def test_functor_invariants(self):
        from math_anything.structures import Functor

        s = Functor()
        names = [i.name for i in s.structural_invariants]
        assert any("identity" in n for n in names)

    def test_adjunction(self):
        from math_anything.structures import Adjunction

        s = Adjunction()
        names = [i.name for i in s.structural_invariants]
        assert any("triangle" in n for n in names)

    def test_monad(self):
        from math_anything.structures import Monad

        s = Monad()
        assert len(s.structural_invariants) > 0

    def test_kan_extension(self):
        from math_anything.structures import KanExtension

        s = KanExtension()
        assert len(s.structural_invariants) > 0


class TestCharacterTables:
    def test_32_point_groups(self):
        from math_anything.groups.character_tables import CHARACTER_TABLES

        assert len(CHARACTER_TABLES) == 32

    def test_oh_group(self):
        from math_anything.groups.character_tables import CHARACTER_TABLES

        oh = CHARACTER_TABLES["Oh"]
        assert oh["order"] == 48
        assert "T2g" in oh["irreps"]
