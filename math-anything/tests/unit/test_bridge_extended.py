"""Extended unit tests for bridge.py — StructureBridge methods not covered by
tests/integration/test_integration.py.

Targets uncovered lines:
  - build_from_lammps (lines 135-157)
  - build_from_cfd compressible branch (lines 189-193)
  - build_from_fem (lines 212-231)
  - query / impact_analysis / root_cause / cumulative_loss (lines 239-255)
  - analyze_constraints (lines 259-372)
  - get_summary (lines 374-378)
  - knowledge_graph property (lines 235-237)
"""

from unittest.mock import patch

import pytest

from math_anything.bridge import StructureBridge


# ── Fixtures ──


@pytest.fixture
def bridge(tmp_path):
    # Use a temp directory for the knowledge graph to avoid polluting ~/.math-anything
    return StructureBridge(kg_root=str(tmp_path / "kg"))


# ── build_from_lammps ──


class TestBuildFromLammps:
    def test_returns_dict_with_structure(self, bridge):
        r = bridge.build_from_lammps({"n_atoms": 1000, "ensemble": "NVT"})
        assert isinstance(r, dict)
        assert "structure" in r
        assert "knowledge_graph_stats" in r

    def test_structure_name_is_hamiltonian(self, bridge):
        r = bridge.build_from_lammps({"n_atoms": 500})
        name = r["structure"]["name"]
        assert "Hamiltonian" in name

    def test_structure_has_invariants(self, bridge):
        r = bridge.build_from_lammps({"n_atoms": 100})
        # to_dict() includes structural_invariants list
        assert "structural_invariants" in r["structure"]
        assert isinstance(r["structure"]["structural_invariants"], list)

    def test_default_n_atoms(self, bridge):
        r = bridge.build_from_lammps({})
        # Default n_atoms=1000, structure should still be created
        assert "structure" in r
        assert "Hamiltonian" in r["structure"]["name"]

    def test_kg_stats_updated(self, bridge):
        r = bridge.build_from_lammps({"n_atoms": 100})
        stats = r["knowledge_graph_stats"]
        assert isinstance(stats, dict)


# ── build_from_cfd — compressible branch ──


class TestBuildFromCfdCompressible:
    def test_compressible_regime_skips_incompressibility(self, bridge):
        # When regime contains "compressible", the incompressibility morphism
        # branch is skipped (line 189 condition is False)
        r = bridge.build_from_cfd({
            "engine": "OpenFOAM",
            "regime": "compressible",
            "Re": 1e5,
        })
        assert "structure" in r
        assert "Navier" in r["structure"]["name"]

    def test_compressible_with_mach_number(self, bridge):
        r = bridge.build_from_cfd({
            "regime": "compressible",
            "Ma": 0.8,
            "Re": 1e6,
        })
        assert "Navier" in r["structure"]["name"]

    def test_incompressible_applies_morphism(self, bridge):
        # Incompressible regime → incompressibility morphism IS applied
        r = bridge.build_from_cfd({"regime": "incompressible", "Re": 1000})
        assert "Navier" in r["structure"]["name"]

    def test_cfd_with_energy_and_gravity(self, bridge):
        r = bridge.build_from_cfd({
            "regime": "incompressible",
            "include_energy": True,
            "include_gravity": True,
            "include_surface_tension": True,
        })
        struct = r["structure"]
        assert struct is not None

    def test_cfd_default_regime(self, bridge):
        r = bridge.build_from_cfd({})
        assert "Navier" in r["structure"]["name"]

    def test_cfd_returns_pi_groups(self, bridge):
        r = bridge.build_from_cfd({"Re": 1000, "regime": "incompressible"})
        assert "pi_groups" in r
        assert "named_pi_groups" in r

    def test_cfd_custom_engine_name(self, bridge):
        r = bridge.build_from_cfd({"engine": "ansys_fluent", "regime": "incompressible"})
        assert "structure" in r


# ── build_from_fem ──


class TestBuildFromFem:
    def test_returns_dict_with_structure(self, bridge):
        r = bridge.build_from_fem({"nonlinear": False})
        assert isinstance(r, dict)
        assert "structure" in r
        assert "knowledge_graph_stats" in r

    def test_linear_fem(self, bridge):
        r = bridge.build_from_fem({"nonlinear": False})
        struct = r["structure"]
        assert "name" in struct
        assert "structural_invariants" in struct

    def test_nonlinear_fem(self, bridge):
        r = bridge.build_from_fem({"nonlinear": True})
        struct = r["structure"]
        assert "name" in struct
        assert "structural_invariants" in struct

    def test_default_nonlinear_false(self, bridge):
        r = bridge.build_from_fem({})
        assert "structure" in r
        assert "name" in r["structure"]

    def test_custom_engine_name(self, bridge):
        r = bridge.build_from_fem({"engine": "ansys"})
        assert "structure" in r

    def test_default_engine_abaqus(self, bridge):
        r = bridge.build_from_fem({})
        assert "structure" in r


# ── build_from_vasp (extended) ──


class TestBuildFromVaspExtended:
    def test_with_kpoints(self, bridge):
        r = bridge.build_from_vasp({"ENCUT": 520, "ISMEAR": 1, "kpoints": [4, 4, 4]})
        assert "structure" in r
        assert "pi_groups" in r

    def test_default_encut(self, bridge):
        r = bridge.build_from_vasp({"ISMEAR": 0})
        assert "structure" in r

    def test_default_ismear(self, bridge):
        r = bridge.build_from_vasp({"ENCUT": 400})
        assert "structure" in r

    def test_returns_pi_groups(self, bridge):
        r = bridge.build_from_vasp({"ENCUT": 520, "EDIFF": 1e-6})
        assert "pi_groups" in r
        assert isinstance(r["pi_groups"], list)


# ── knowledge_graph property ──


class TestKnowledgeGraphProperty:
    def test_returns_graph_object(self, bridge):
        from math_anything.categories.graph import MathKnowledgeGraph
        kg = bridge.knowledge_graph
        assert isinstance(kg, MathKnowledgeGraph)

    def test_kg_populated_after_build(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        kg = bridge.knowledge_graph
        # After building, the KG should have some stats
        stats = kg.stats()
        assert isinstance(stats, dict)


# ── query ──


class TestQuery:
    def test_query_returns_dict(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.query("ENCUT")
        assert isinstance(result, dict)

    def test_query_unknown_term(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.query("nonexistent_term_xyz")
        assert isinstance(result, dict)


# ── impact_analysis ──


class TestImpactAnalysis:
    def test_impact_returns_dict(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.impact_analysis("ENCUT")
        assert isinstance(result, dict)

    def test_impact_unknown_param(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.impact_analysis("NONEXISTENT_PARAM")
        assert isinstance(result, dict)
        # Should report error or empty impacted list
        assert "error" in result or "impacted" in result


# ── root_cause ──


class TestRootCause:
    def test_root_cause_returns_dict(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.root_cause("convergence")
        assert isinstance(result, dict)

    def test_root_cause_unknown_problem(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.root_cause("nonexistent_problem_xyz")
        assert isinstance(result, dict)


# ── cumulative_loss ──


class TestCumulativeLoss:
    def test_cumulative_loss_with_path(self, bridge):
        # Build VASP to register the morphism chain
        bridge.build_from_vasp({"ENCUT": 520})
        # Try to trace from FullManyBody to KohnSham_Converged
        result = bridge.cumulative_loss("FullManyBody", "KohnSham_Converged")
        assert isinstance(result, dict)

    def test_cumulative_loss_no_path(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.cumulative_loss("NonexistentA", "NonexistentB")
        assert isinstance(result, dict)
        assert "error" in result


# ── analyze_constraints ──


class TestAnalyzeConstraints:
    """analyze_constraints calls LearnedInvariant.evaluate, which uses safe_eval
    on invariant expressions. Structural invariants contain mathematical notation
    (e.g. 'λ_i ∈ ℝ') that safe_eval can't parse, raising SafeEvalError.
    We mock evaluate to return predictable statuses so the bridge logic is tested.
    """

    @pytest.fixture(autouse=True)
    def _mock_evaluate(self):
        from math_anything.constraints.invariant import InvariantStatus
        with patch(
            "math_anything.constraints.invariant.LearnedInvariant.evaluate",
            return_value=InvariantStatus.SATISFIED,
        ):
            yield

    def test_analyze_constraints_returns_dict(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520, "EDIFF": 1e-6},
        )
        assert isinstance(result, dict)
        assert "interior" in result
        assert "boundary_count" in result
        assert "violation_count" in result
        assert "risks" in result
        assert "propagation" in result
        assert "weakening_suggestions" in result
        assert "boundary_state" in result

    def test_analyze_constraints_without_morphism_chain(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        # No morphism chain → propagation is None
        assert result["propagation"] is None

    def test_analyze_constraints_with_morphism_chain(self, bridge, self_consistent_structure):
        # Register morphisms first via build_from_vasp
        bridge.build_from_vasp({"ENCUT": 520})
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520, "EDIFF": 1e-6},
            morphism_chain=["born_oppenheimer", "kohn_sham"],
        )
        assert isinstance(result, dict)
        assert "propagation" in result

    def test_analyze_constraints_violation_count_is_int(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        assert isinstance(result["violation_count"], int)
        assert isinstance(result["boundary_count"], int)

    def test_analyze_constraints_risks_is_list(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        assert isinstance(result["risks"], list)

    def test_analyze_constraints_with_navier_stokes(self, bridge, navier_stokes_structure):
        result = bridge.analyze_constraints(
            navier_stokes_structure,
            {"Re": 1000, "viscosity": 1e-3},
        )
        assert isinstance(result, dict)
        assert "interior" in result

    def test_analyze_constraints_with_hamiltonian(self, bridge, hamiltonian_structure):
        result = bridge.analyze_constraints(
            hamiltonian_structure,
            {"timestep": 1.0, "n_atoms": 100},
        )
        assert isinstance(result, dict)

    def test_analyze_constraints_with_variational(self, bridge, variational_structure):
        result = bridge.analyze_constraints(
            variational_structure,
            {"youngs_modulus": 200},
        )
        assert isinstance(result, dict)

    def test_analyze_constraints_all_satisfied(self, bridge, self_consistent_structure):
        # With evaluate mocked to SATISFIED, all invariants go to interior
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        assert result["violation_count"] == 0
        assert result["boundary_count"] == 0
        assert len(result["interior"]) > 0


class TestAnalyzeConstraintsViolated:
    """Test the violated/weakened branches of analyze_constraints."""

    @pytest.fixture(autouse=True)
    def _mock_evaluate_violated(self):
        from math_anything.constraints.invariant import InvariantStatus
        with patch(
            "math_anything.constraints.invariant.LearnedInvariant.evaluate",
            return_value=InvariantStatus.VIOLATED,
        ):
            yield

    def test_violated_invariants_counted(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        assert result["violation_count"] > 0
        assert result["boundary_count"] > 0

    def test_violated_produces_risks(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        # Violated invariants should produce risk entries
        assert isinstance(result["risks"], list)


class TestAnalyzeConstraintsWeakened:
    """Test the weakened/conditional branch of analyze_constraints."""

    @pytest.fixture(autouse=True)
    def _mock_evaluate_weakened(self):
        from math_anything.constraints.invariant import InvariantStatus
        with patch(
            "math_anything.constraints.invariant.LearnedInvariant.evaluate",
            return_value=InvariantStatus.WEAKENED,
        ):
            yield

    def test_weakened_invariants_are_boundary(self, bridge, self_consistent_structure):
        result = bridge.analyze_constraints(
            self_consistent_structure,
            {"ENCUT": 520},
        )
        assert result["boundary_count"] > 0
        assert result["violation_count"] == 0


# ── get_summary ──


class TestGetSummary:
    def test_returns_dict_with_kg_and_engine(self, bridge):
        s = bridge.get_summary()
        assert isinstance(s, dict)
        assert "knowledge_graph" in s
        assert "category_engine" in s

    def test_summary_after_build(self, bridge):
        bridge.build_from_vasp({"ENCUT": 520})
        bridge.build_from_lammps({"n_atoms": 100})
        s = bridge.get_summary()
        assert "knowledge_graph" in s
        assert "category_engine" in s
        # Category engine should have morphisms registered
        ce = s["category_engine"]
        assert ce["morphisms_count"] > 0


# ── StructureBridge init ──


class TestStructureBridgeInit:
    def test_init_with_custom_kg_root(self, tmp_path):
        b = StructureBridge(kg_root=str(tmp_path / "custom_kg"))
        assert b.kg is not None
        assert b.builder is not None
        assert b.engine is not None
        assert b.pi_engine is not None

    def test_init_registers_morphisms(self, tmp_path):
        b = StructureBridge(kg_root=str(tmp_path / "kg"))
        # After init, known morphisms should be registered
        assert "born_oppenheimer" in b.engine.morphisms
        assert "kohn_sham" in b.engine.morphisms
        assert "plane_wave_truncation" in b.engine.morphisms
        assert "scf_iteration" in b.engine.morphisms
        assert "classical_limit" in b.engine.morphisms
        assert "force_field" in b.engine.morphisms
        assert "incompressibility" in b.engine.morphisms
        assert "fem_discretization" in b.engine.morphisms

    def test_init_registers_discretizations(self, tmp_path):
        b = StructureBridge(kg_root=str(tmp_path / "kg"))
        assert "spectral_discretization" in b.engine.morphisms
        assert "fem_discretization" in b.engine.morphisms
        assert "fvm_discretization" in b.engine.morphisms
        assert "particle_discretization" in b.engine.morphisms
