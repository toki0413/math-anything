"""Unit tests for morphisms, categories, and dimensional analysis."""


class TestMorphisms:
    def test_born_oppenheimer(self, born_oppenheimer):
        m = born_oppenheimer
        assert m.name == "born_oppenheimer"
        assert "nuclear_quantum_effects" in m.invariants_lost
        assert len(m.invariants_kept) > 0

    def test_plane_wave_truncation(self, plane_wave_truncation):
        m = plane_wave_truncation
        assert m.encut == 520
        assert len(m.invariants_kept) > 0

    def test_kohn_sham(self, kohn_sham_mapping):
        m = kohn_sham_mapping
        assert len(m.invariants_kept) > 0

    def test_cfl_morphism(self):
        from math_anything.morphisms.convergence import CFLConditionMorphism
        m = CFLConditionMorphism()
        assert m.scheme_type == "explicit"
        assert len(m.invariants_kept) > 0

    def test_convergence_morphism(self):
        from math_anything.morphisms.convergence import ConvergenceMorphism
        m = ConvergenceMorphism()
        assert len(m.invariants_kept) > 0

    def test_symmetry_reduction(self):
        from math_anything.morphisms.symmetry import SymmetryReductionMorphism
        m = SymmetryReductionMorphism()
        assert len(m.invariants_kept) > 0

    def test_bloch_theorem(self):
        from math_anything.morphisms.symmetry import BlochTheoremMorphism
        m = BlochTheoremMorphism()
        assert len(m.invariants_kept) > 0


class TestCategoryEngine:
    def test_register_and_chain(self, category_engine):
        ce = category_engine
        chain = ce.get_morphism_chain("FullManyBody", "KohnSham_Truncated")
        assert len(chain) > 0

    def test_cumulative_loss(self, category_engine):
        result = category_engine.cumulative_loss("FullManyBody", "KohnSham_Truncated")
        assert "chain" in result
        assert len(result["total_invariants_lost"]) > 0

    def test_invariant_under(self, category_engine):
        assert category_engine.invariant_under(
            "self_adjointness (orthogonal projection preserves it)",
            "plane_wave_truncation"
        )

    def test_kernel_of(self, category_engine):
        kernel = category_engine.kernel_of("plane_wave_truncation")
        assert len(kernel) > 0


class TestDimensional:
    def test_buckingham_basic(self, buckingham_engine):
        from math_anything.dimensional.scaling_group import BUILTIN_QUANTITIES
        quantities = [
            BUILTIN_QUANTITIES["density"],
            BUILTIN_QUANTITIES["velocity"],
            BUILTIN_QUANTITIES["length"],
            BUILTIN_QUANTITIES["dynamic_viscosity"],
        ]
        pi_groups = buckingham_engine.compute(quantities)
        assert len(pi_groups) > 0

    def test_fluid_analyzer(self, fluid_analyzer):
        pi_groups = fluid_analyzer.analyze_ns({"regime": "incompressible"})
        assert len(pi_groups) > 0


class TestEquationChecker:
    def test_ode_check(self):
        from math_anything.dimensional.equation_checker import EquationChecker
        checker = EquationChecker()
        result = checker.check_schema("m_i d²r_i/dt² = F_i")
        assert result.consistent
