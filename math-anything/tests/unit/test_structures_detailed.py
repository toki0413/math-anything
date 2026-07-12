"""Detailed unit tests for structures — enums, invariant_registry, evolution, conservation_field.

Covers:
  - All enum values (SymmetryGroup, OperatorType, SpectrumType, etc.)
  - InvariantRegistry: query_invariants, get_invariants, predefined groups
  - Evolution: ConservationLawSystem, HamiltonianSystem, dimensional_rank
  - ConservationMatrixField: edge cases, verify_conservation, to_dict
"""

import numpy as np
import pytest

from math_anything.structures._core import StructuralInvariant
from math_anything.structures.conservation_field import (
    NOETHER_CORRESPONDENCE,
    TIME_TRANSLATION_CONSERVATION,
    ConservationMatrixField,
    FieldConservedQuantity,
    NoetherCurrent,
)
from math_anything.structures.enums import (
    OperatorType,
    SpectrumType,
    StructureDomain,
    SymmetryGroup,
    VariationalPrinciple,
)
from math_anything.structures.evolution import (
    ConservationLawSystem,
    ConservedQuantity,
    DissipativeSystem,
    EvolutionProblem,
    FluxType,
    HamiltonianSystem,
    NavierStokesProblem,
    NSRegime,
    NSTurbulenceModel,
    StochasticSystem,
)
from math_anything.structures.invariant_registry import (
    CONSERVATION_LAW_INVARIANTS,
    HAMILTONIAN_INVARIANTS,
    INVARIANT_REGISTRY,
    SPECTRAL_SELF_ADJOINT_INVARIANTS,
    VARIATIONAL_INVARIANTS,
    get_invariants,
    query_invariants,
)

# ── Fixtures ──


@pytest.fixture
def hamiltonian():
    return HamiltonianSystem(phase_space_dim=6, symplectic=True, autonomous=True)


@pytest.fixture
def conservation_law():
    return ConservationLawSystem(
        hyperbolic=True,
        has_diffusion=True,
        has_source=False,
        spatial_dim=3,
    )


@pytest.fixture
def conservation_law_with_thermal():
    return ConservationLawSystem(
        conserved_variables=[
            ConservedQuantity(name="energy", symbol="E", dimensions={"temperature": 1.0}),
        ],
        hyperbolic=False,
        has_diffusion=True,
    )


@pytest.fixture
def ns_incompressible():
    return NavierStokesProblem(
        regime=NSRegime.INCOMPRESSIBLE,
        has_diffusion=True,
    )


@pytest.fixture
def ns_compressible():
    return NavierStokesProblem(
        regime=NSRegime.COMPRESSIBLE_SUPERSONIC,
        has_diffusion=True,
    )


@pytest.fixture
def empty_field():
    return ConservationMatrixField()


@pytest.fixture
def euler_field():
    f = ConservationMatrixField()
    f.build_from_euler_equations()
    return f


@pytest.fixture
def schrodinger_field():
    f = ConservationMatrixField()
    f.build_from_schrodinger()
    return f


# ── Enum tests ──


class TestOperatorType:
    def test_self_adjoint(self):
        assert OperatorType.SELF_ADJOINT == "self_adjoint"

    def test_normal(self):
        assert OperatorType.NORMAL == "normal"

    def test_non_self_adjoint(self):
        assert OperatorType.NON_SELF_ADJOINT == "non_self_adjoint"

    def test_unitary(self):
        assert OperatorType.UNITARY == "unitary"

    def test_positive_definite(self):
        assert OperatorType.POSITIVE_DEFINITE == "positive_definite"

    def test_bounded_below(self):
        assert OperatorType.BOUNDED_BELOW == "bounded_below"

    def test_compact(self):
        assert OperatorType.COMPACT == "compact"

    def test_total_count(self):
        assert len(OperatorType) == 7


class TestSpectrumType:
    def test_pure_point(self):
        assert SpectrumType.PURE_POINT == "pure_point"

    def test_continuous(self):
        assert SpectrumType.CONTINUOUS == "continuous"

    def test_mixed(self):
        assert SpectrumType.MIXED == "mixed"

    def test_band(self):
        assert SpectrumType.BAND == "band"

    def test_total_count(self):
        assert len(SpectrumType) == 4


class TestSymmetryGroup:
    def test_translation(self):
        assert SymmetryGroup.TRANSLATION == "translation"

    def test_rotation_so3(self):
        assert SymmetryGroup.ROTATION_SO3 == "rotation_SO3"

    def test_reflection(self):
        assert SymmetryGroup.REFLECTION == "reflection"

    def test_point_group(self):
        assert SymmetryGroup.POINT_GROUP == "point_group"

    def test_space_group(self):
        assert SymmetryGroup.SPACE_GROUP == "space_group"

    def test_gauge_u1(self):
        assert SymmetryGroup.GAUGE_U1 == "gauge_U1"

    def test_gauge_su2(self):
        assert SymmetryGroup.GAUGE_SU2 == "gauge_SU2"

    def test_lorentz(self):
        assert SymmetryGroup.LORENTZ == "lorentz"

    def test_galilean(self):
        assert SymmetryGroup.GALILEAN == "galilean"

    def test_scaling(self):
        assert SymmetryGroup.SCALING == "scaling"

    def test_permutation(self):
        assert SymmetryGroup.PERMUTATION == "permutation"

    def test_total_count(self):
        assert len(SymmetryGroup) == 11


class TestVariationalPrinciple:
    def test_stationary(self):
        assert VariationalPrinciple.STATIONARY == "stationary"

    def test_minimum(self):
        assert VariationalPrinciple.MINIMUM == "minimum"

    def test_maximum(self):
        assert VariationalPrinciple.MAXIMUM == "maximum"

    def test_hamiltonian(self):
        assert VariationalPrinciple.HAMILTONIAN == "hamiltonian"

    def test_self_consistent(self):
        assert VariationalPrinciple.SELF_CONSISTENT == "self_consistent"

    def test_constrained(self):
        assert VariationalPrinciple.CONSTRAINED == "constrained"

    def test_total_count(self):
        assert len(VariationalPrinciple) == 6


class TestStructureDomain:
    def test_continuum(self):
        assert StructureDomain.CONTINUUM == "continuum"

    def test_lattice(self):
        assert StructureDomain.LATTICE == "lattice"

    def test_particles(self):
        assert StructureDomain.PARTICLES == "particles"

    def test_mesh(self):
        assert StructureDomain.MESH == "mesh"

    def test_graph(self):
        assert StructureDomain.GRAPH == "graph"

    def test_total_count(self):
        assert len(StructureDomain) == 5


# ── InvariantRegistry tests ──


class TestInvariantRegistryPredefined:
    def test_spectral_invariants_count(self):
        assert len(SPECTRAL_SELF_ADJOINT_INVARIANTS) >= 3

    def test_variational_invariants_count(self):
        assert len(VARIATIONAL_INVARIANTS) >= 2

    def test_hamiltonian_invariants_count(self):
        assert len(HAMILTONIAN_INVARIANTS) >= 3

    def test_conservation_law_invariants_count(self):
        assert len(CONSERVATION_LAW_INVARIANTS) >= 3

    def test_registry_keys(self):
        assert "spectral_self_adjoint" in INVARIANT_REGISTRY
        assert "variational" in INVARIANT_REGISTRY
        assert "hamiltonian" in INVARIANT_REGISTRY
        assert "conservation_law" in INVARIANT_REGISTRY

    def test_registry_total_count(self):
        assert len(INVARIANT_REGISTRY) == 4


class TestGetInvariants:
    def test_get_spectral(self):
        invs = get_invariants("spectral_self_adjoint")
        assert len(invs) > 0
        assert all(isinstance(i, StructuralInvariant) for i in invs)

    def test_get_variational(self):
        invs = get_invariants("variational")
        assert len(invs) > 0

    def test_get_hamiltonian(self):
        invs = get_invariants("hamiltonian")
        assert len(invs) > 0

    def test_get_conservation_law(self):
        invs = get_invariants("conservation_law")
        assert len(invs) > 0

    def test_get_unknown_returns_empty(self):
        assert get_invariants("nonexistent_category") == []


class TestQueryInvariants:
    def test_query_by_keyword_eigenvalue(self):
        results = query_invariants(keyword="eigenvalue")
        assert len(results) > 0
        names = [r.name for r in results]
        assert any("eigenvalue" in n for n in names)

    def test_query_by_keyword_energy(self):
        results = query_invariants(keyword="energy")
        assert len(results) > 0

    def test_query_by_theorem_noether(self):
        results = query_invariants(theorem="Noether")
        assert len(results) > 0

    def test_query_by_affected_quantity_eigenvalues(self):
        results = query_invariants(affected_quantity="eigenvalues")
        assert len(results) > 0

    def test_query_no_filters_returns_all(self):
        results = query_invariants()
        # Should return all unique invariants across all groups
        assert len(results) >= 8

    def test_query_combined_filters(self):
        results = query_invariants(keyword="conservation", theorem="Noether")
        assert len(results) > 0

    def test_query_no_match(self):
        results = query_invariants(keyword="xyz_nonexistent_12345")
        assert len(results) == 0

    def test_query_deduplication(self):
        # Same invariant shouldn't appear twice
        results = query_invariants(keyword="conservation")
        names = [r.name for r in results]
        assert len(names) == len(set(names))


# ── Evolution tests ──


class TestEvolutionProblem:
    def test_default_creation(self):
        e = EvolutionProblem()
        assert e.phase_space_dim == 0
        assert e.time_dependent is True
        assert e.autonomous is True

    def test_function_space_zero_dim(self):
        e = EvolutionProblem()
        assert "Γ" in e.function_space

    def test_function_space_with_dim(self):
        e = EvolutionProblem(phase_space_dim=6)
        assert "ℝ^6" in e.function_space

    def test_structural_invariants_empty(self):
        e = EvolutionProblem()
        assert e.structural_invariants == []


class TestHamiltonianSystem:
    def test_default_creation(self, hamiltonian):
        assert hamiltonian.symplectic is True
        assert hamiltonian.reversible is True
        assert hamiltonian.integrable is False

    def test_structural_invariants_nonempty(self, hamiltonian):
        invs = hamiltonian.structural_invariants
        assert len(invs) > 0

    def test_symplectic_invariant(self, hamiltonian):
        names = [i.name for i in hamiltonian.structural_invariants]
        assert "symplectic_form" in names

    def test_integrable_adds_liouville(self):
        h = HamiltonianSystem(symplectic=True, integrable=True)
        names = [i.name for i in h.structural_invariants]
        assert "liouville_integrability" in names

    def test_non_symplectic_no_symplectic_form(self):
        h = HamiltonianSystem(symplectic=False)
        names = [i.name for i in h.structural_invariants]
        assert "symplectic_form" not in names

    def test_energy_conservation_active_for_autonomous(self, hamiltonian):
        names = [i.name for i in hamiltonian.structural_invariants]
        assert "energy_conservation" in names


class TestConservationLawSystem:
    def test_default_creation(self):
        c = ConservationLawSystem()
        assert c.hyperbolic is True
        assert c.has_diffusion is False
        assert c.spatial_dim == 3

    def test_structural_invariants_hyperbolic(self, conservation_law):
        names = [i.name for i in conservation_law.structural_invariants]
        assert "finite_signal_speed" in names

    def test_structural_invariants_diffusion(self, conservation_law):
        names = [i.name for i in conservation_law.structural_invariants]
        assert "energy_dissipation" in names

    def test_dimensional_rank_base(self):
        c = ConservationLawSystem()
        assert c.dimensional_rank == 3

    def test_dimensional_rank_with_temperature(self, conservation_law_with_thermal):
        assert conservation_law_with_thermal.dimensional_rank == 4

    def test_function_space(self):
        c = ConservationLawSystem(
            conserved_variables=[
                ConservedQuantity(name="mass", symbol="ρ"),
                ConservedQuantity(name="momentum", symbol="ρu"),
            ],
            spatial_dim=3,
        )
        assert "L²" in c.function_space
        assert "ℝ^3" in c.function_space


class TestDissipativeSystem:
    def test_default_creation(self):
        d = DissipativeSystem()
        assert d.conserved is False

    def test_structural_invariants_lyapunov(self):
        d = DissipativeSystem()
        names = [i.name for i in d.structural_invariants]
        assert "lyapunov_functional" in names
        assert "energy_monotonic" in names

    def test_conserved_adds_mass_conservation(self):
        d = DissipativeSystem(conserved=True)
        names = [i.name for i in d.structural_invariants]
        assert "mass_conservation" in names


class TestStochasticSystem:
    def test_default_creation(self):
        s = StochasticSystem()
        assert s.noise_type == "additive"
        assert s.noise_strength == 0.0
        assert s.fluctuation_dissipation is False

    def test_structural_invariants_ito(self):
        s = StochasticSystem()
        names = [i.name for i in s.structural_invariants]
        assert "ito_isometry" in names

    def test_fdt_adds_invariant(self):
        s = StochasticSystem(fluctuation_dissipation=True)
        names = [i.name for i in s.structural_invariants]
        assert "fluc_diss_theorem" in names


class TestNavierStokesProblem:
    def test_incompressible_regime(self, ns_incompressible):
        assert ns_incompressible.regime == NSRegime.INCOMPRESSIBLE

    def test_incompressible_divergence_free(self, ns_incompressible):
        names = [i.name for i in ns_incompressible.structural_invariants]
        assert "divergence_free" in names

    def test_compressible_has_eos(self, ns_compressible):
        names = [i.name for i in ns_compressible.structural_invariants]
        assert "equation_of_state" in names

    def test_buckingham_pi_base(self, ns_incompressible):
        assert ns_incompressible.buckingham_pi_count >= 2

    def test_named_pi_groups_re(self, ns_incompressible):
        assert "Re" in ns_incompressible.named_pi_groups

    def test_named_pi_groups_compressible_ma(self, ns_compressible):
        assert "Ma" in ns_compressible.named_pi_groups

    def test_rans_turbulence_model(self):
        ns = NavierStokesProblem(
            regime=NSRegime.INCOMPRESSIBLE,
            turbulence_model=NSTurbulenceModel.RANS_KEPSILON,
        )
        names = [i.name for i in ns.structural_invariants]
        assert "reynolds_stress_closure" in names


# ── ConservationMatrixField tests ──


class TestConservationMatrixFieldEmpty:
    def test_empty_field_n_conserved(self, empty_field):
        assert empty_field.n_conserved == 0

    def test_empty_field_not_hyperbolic(self, empty_field):
        assert empty_field.is_hyperbolic is False

    def test_empty_field_not_hamiltonian(self, empty_field):
        assert empty_field.is_hamiltonian is False

    def test_empty_field_no_characteristic_speeds(self, empty_field):
        assert empty_field.characteristic_speeds is None

    def test_empty_field_noether_map(self, empty_field):
        assert empty_field.noether_map == {}

    def test_verify_conservation_no_coupling_matrix(self, empty_field):
        # No coupling_matrix → empty results
        U = np.array([1.0])
        results = empty_field.verify_conservation(U)
        assert results == {}

    def test_structural_invariants_empty(self, empty_field):
        invs = empty_field.structural_invariants()
        assert invs == []

    def test_to_dict_empty(self, empty_field):
        d = empty_field.to_dict()
        assert d["n_conserved"] == 0
        assert d["is_hyperbolic"] is False
        assert d["is_hamiltonian"] is False
        assert d["characteristic_speeds"] is None


class TestConservationMatrixFieldEuler:
    def test_n_conserved(self, euler_field):
        assert euler_field.n_conserved == 3

    def test_conserved_quantity_names(self, euler_field):
        names = [q.name for q in euler_field.conserved_quantities]
        assert "mass" in names
        assert "momentum" in names
        assert "energy" in names

    def test_noether_map(self, euler_field):
        nm = euler_field.noether_map
        assert SymmetryGroup.GAUGE_U1 in nm
        assert SymmetryGroup.TRANSLATION in nm

    def test_noether_currents(self, euler_field):
        assert len(euler_field.noether_currents) >= 2

    def test_structural_invariants(self, euler_field):
        invs = euler_field.structural_invariants()
        assert len(invs) >= 3

    def test_to_dict(self, euler_field):
        d = euler_field.to_dict()
        assert d["n_conserved"] == 3
        assert "conserved_quantities" in d
        assert "noether_currents" in d


class TestConservationMatrixFieldSchrodinger:
    def test_is_hamiltonian(self, schrodinger_field):
        assert schrodinger_field.is_hamiltonian is True

    def test_symplectic_matrix(self, schrodinger_field):
        assert schrodinger_field.symplectic_matrix is not None
        assert schrodinger_field.symplectic_matrix.shape == (2, 2)

    def test_hamiltonian_expression(self, schrodinger_field):
        assert schrodinger_field.hamiltonian != ""

    def test_structural_invariants_hamiltonian(self, schrodinger_field):
        invs = schrodinger_field.structural_invariants()
        names = [i.name for i in invs]
        assert "symplectic_structure" in names


class TestConservationMatrixFieldVerify:
    def test_verify_with_zero_coupling(self):
        # Zero coupling matrix → everything conserved
        f = ConservationMatrixField(
            conserved_quantities=[
                FieldConservedQuantity("mass", "ρ"),
                FieldConservedQuantity("momentum", "ρu"),
            ],
            coupling_matrix=np.zeros((2, 2)),
        )
        U = np.array([1.0, 2.0])
        results = f.verify_conservation(U)
        assert all(results.values())

    def test_verify_with_nonzero_coupling(self):
        # Non-zero coupling → not conserved
        f = ConservationMatrixField(
            conserved_quantities=[
                FieldConservedQuantity("mass", "ρ"),
            ],
            coupling_matrix=np.array([[1.0]]),
        )
        U = np.array([1.0])
        results = f.verify_conservation(U)
        assert not results["mass"]

    def test_verify_with_source_vector(self):
        # coupling produces dU/dt = source → conserved
        f = ConservationMatrixField(
            conserved_quantities=[
                FieldConservedQuantity("mass", "ρ"),
            ],
            coupling_matrix=np.array([[1.0]]),
            source_vector=np.array([1.0]),
        )
        U = np.array([1.0])
        results = f.verify_conservation(U)
        assert results["mass"]


class TestConservationMatrixFieldBuildMethods:
    def test_build_navier_stokes(self):
        f = ConservationMatrixField()
        f.build_from_navier_stokes()
        assert f.n_conserved == 3

    def test_build_maxwell(self):
        f = ConservationMatrixField()
        f.build_from_maxwell()
        assert f.n_conserved == 2

    def test_build_elasticity(self):
        f = ConservationMatrixField()
        f.build_from_elasticity()
        assert f.n_conserved == 3

    def test_build_heat_equation(self):
        f = ConservationMatrixField()
        f.build_from_heat_equation()
        assert f.n_conserved == 1

    def test_build_advection_diffusion(self):
        f = ConservationMatrixField()
        f.build_from_advection_diffusion()
        assert f.n_conserved == 1

    def test_build_mhd(self):
        f = ConservationMatrixField()
        f.build_from_mhd()
        assert f.n_conserved == 4

    def test_build_kohn_sham(self):
        f = ConservationMatrixField()
        f.build_from_kohn_sham()
        assert f.n_conserved == 3

    def test_build_boltzmann(self):
        f = ConservationMatrixField()
        f.build_from_boltzmann()
        assert f.n_conserved == 3

    def test_build_shallow_water(self):
        f = ConservationMatrixField()
        f.build_from_shallow_water()
        assert f.n_conserved == 2

    def test_build_wave_equation(self):
        f = ConservationMatrixField()
        f.build_from_wave_equation()
        assert f.is_hamiltonian is True

    def test_build_dirac(self):
        f = ConservationMatrixField()
        f.build_from_dirac()
        assert f.n_conserved == 3

    def test_build_klein_gordon(self):
        f = ConservationMatrixField()
        f.build_from_klein_gordon()
        assert f.n_conserved == 2

    def test_build_einstein_field(self):
        f = ConservationMatrixField()
        f.build_from_einstein_field()
        assert f.n_conserved == 2

    def test_build_schrodinger_nonlinear(self):
        f = ConservationMatrixField()
        f.build_from_schrodinger_nonlinear()
        assert f.is_hamiltonian is True

    def test_build_vlasov(self):
        f = ConservationMatrixField()
        f.build_from_vlasov()
        assert f.n_conserved == 3

    def test_build_hartree_fock(self):
        f = ConservationMatrixField()
        f.build_from_hartree_fock()
        assert f.n_conserved == 3


class TestNoetherCorrespondence:
    def test_translation_to_momentum(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.TRANSLATION] == "momentum_conservation"

    def test_rotation_to_angular_momentum(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.ROTATION_SO3] == "angular_momentum_conservation"

    def test_gauge_u1_to_charge(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.GAUGE_U1] == "charge_conservation"

    def test_lorentz_to_four_momentum(self):
        assert NOETHER_CORRESPONDENCE[SymmetryGroup.LORENTZ] == "four_momentum_conservation"

    def test_time_translation(self):
        assert TIME_TRANSLATION_CONSERVATION == "energy_conservation"


class TestFieldConservedQuantity:
    def test_creation(self):
        q = FieldConservedQuantity("mass", "ρ", "dρ/dt = 0", SymmetryGroup.GAUGE_U1, 3)
        assert q.name == "mass"
        assert q.symbol == "ρ"
        assert q.symmetry == SymmetryGroup.GAUGE_U1
        assert q.spatial_dim == 3

    def test_default_symmetry_none(self):
        q = FieldConservedQuantity("energy", "E")
        assert q.symmetry is None

    def test_default_spatial_dim(self):
        q = FieldConservedQuantity("energy", "E")
        assert q.spatial_dim == 3


class TestNoetherCurrent:
    def test_creation(self):
        nc = NoetherCurrent(
            name="mass_current",
            symmetry=SymmetryGroup.GAUGE_U1,
            current_components=["ρ", "ρu"],
            continuity_equation="dρ/dt + div(ρu) = 0",
        )
        assert nc.name == "mass_current"
        assert nc.symmetry == SymmetryGroup.GAUGE_U1
        assert len(nc.current_components) == 2

    def test_default_components(self):
        nc = NoetherCurrent(name="test", symmetry=SymmetryGroup.TRANSLATION)
        assert nc.current_components == []
        assert nc.continuity_equation == ""


class TestConservedQuantity:
    def test_creation(self):
        q = ConservedQuantity(name="mass", symbol="ρ", dimensions={"M": 1, "L": -3})
        assert q.name == "mass"
        assert q.dimensions == {"M": 1, "L": -3}

    def test_default_dimensions(self):
        q = ConservedQuantity(name="mass", symbol="ρ")
        assert q.dimensions == {}


class TestHyperbolicity:
    def test_real_eigenvalues_hyperbolic(self):
        # Diagonal matrix with real eigenvalues
        J = np.diag([1.0, 2.0, 3.0])
        f = ConservationMatrixField(jacobian=J)
        assert f.is_hyperbolic is True

    def test_complex_eigenvalues_not_hyperbolic(self):
        # Matrix with complex eigenvalues
        J = np.array([[0, -1], [1, 0]], dtype=float)
        f = ConservationMatrixField(jacobian=J)
        assert f.is_hyperbolic is False

    def test_characteristic_speeds(self):
        J = np.diag([3.0, 1.0, 2.0])
        f = ConservationMatrixField(jacobian=J)
        speeds = f.characteristic_speeds
        assert speeds is not None
        np.testing.assert_array_equal(speeds, [1.0, 2.0, 3.0])


class TestFluxType:
    def test_convective(self):
        assert FluxType.CONVECTIVE == "convective"

    def test_diffusive(self):
        assert FluxType.DIFFUSIVE == "diffusive"

    def test_rotational(self):
        assert FluxType.ROTATIONAL == "rotational"
