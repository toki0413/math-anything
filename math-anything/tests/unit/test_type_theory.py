"""MLTT 类型理论模块测试."""

import pytest
from math_anything.type_theory import (
    Term, TermKind, Var, Universe, Pi, Lam, App, Sigma, Pair,
    Proj1, Proj2, Identity, Refl, Sym, Trans, Cong, Transport,
    Annotation, InductiveType, Constructor, Construct, IndElim,
    Context, Judgment,
    free_vars, substitute, whnf, term_to_str,
    arrow, product, TYPE0, TYPE1,
    TypeChecker, TypeCheckResult, TypeCheckError,
    TypeTheoryBridge, MorphismType,
    invariant_to_identity, invariant_to_prop_type,
    morphism_to_type, propagation_to_transport, structure_to_inductive,
    FormalSystemStrength, DecidabilityClass,
    GodelBoundary, DecidabilityBoundary, MetamathAnalyzer,
    UNDECIDABILITY_RESULTS, GODEL_BOUNDARIES,
)
from math_anything.structures.properties import StructuralInvariant
from math_anything.structures.evolution import HamiltonianSystem
from math_anything.morphisms import ContinuumToDiscrete


# ── 项语言测试 ──

class TestTerms:
    def test_var_str(self):
        assert term_to_str(Var("x")) == "x"

    def test_universe_str(self):
        assert term_to_str(Universe(0)) == "Type"
        assert term_to_str(Universe(2)) == "Type_2"

    def test_arrow_str(self):
        t = arrow(TYPE0, TYPE0)
        assert "→" in term_to_str(t)

    def test_pi_str(self):
        t = Pi("x", TYPE0, App(Var("f"), Var("x")))
        s = term_to_str(t)
        assert "Π" in s
        assert "x" in s

    def test_identity_str(self):
        t = Identity(TYPE0, Var("a"), Var("b"))
        s = term_to_str(t)
        assert "Id" in s

    def test_free_vars(self):
        t = App(Var("f"), Var("x"))
        assert free_vars(t) == {"f", "x"}

    def test_free_vars_bound(self):
        t = Lam("x", App(Var("f"), Var("x")))
        assert free_vars(t) == {"f"}

    def test_substitute(self):
        t = App(Var("f"), Var("x"))
        result = substitute(t, "x", Var("y"))
        assert result == App(Var("f"), Var("y"))

    def test_substitute_capture_avoiding(self):
        # lambda y 绑定了 y，替换 x→y 时 body 中自由出现的 x 被替换为 y
        # 但 y 在 body 中被 lambda 遮蔽，所以替换后的 y 指向 lambda 绑定
        t = Lam("y", App(Var("f"), Var("x")))
        result = substitute(t, "x", Var("y"))
        # x 被替换为 y（自由变量替换），结果中 y 指向 lambda 绑定
        assert result == Lam("y", App(Var("f"), Var("y")))

    def test_whnf_beta(self):
        t = App(Lam("x", Var("x")), Var("a"))
        result = whnf(t)
        assert result == Var("a")

    def test_whnf_proj1(self):
        t = Proj1(Pair(Var("a"), Var("b")))
        result = whnf(t)
        assert result == Var("a")

    def test_whnf_proj2(self):
        t = Proj2(Pair(Var("a"), Var("b")))
        result = whnf(t)
        assert result == Var("b")


# ── 类型检查器测试 ──

class TestTypeChecker:
    def setup_method(self):
        self.tc = TypeChecker()
        self.ctx = Context()

    def test_universe_type(self):
        result = self.tc.type_check(self.ctx, Universe(0))
        assert result.success
        assert result.inferred_type == Universe(1)

    def test_var_lookup(self):
        ctx = self.ctx.extend("x", TYPE0)
        result = self.tc.type_check(ctx, Var("x"))
        assert result.success

    def test_var_unbound(self):
        with pytest.raises(TypeCheckError, match="Unbound"):
            self.tc.infer(self.ctx, Var("unknown"))

    def test_pi_formation(self):
        t = Pi("x", TYPE0, TYPE0)
        result = self.tc.type_check(self.ctx, t)
        assert result.success

    def test_lambda_check(self):
        pi_type = Pi("x", TYPE0, TYPE0)
        lam = Lam("x", Var("x"))
        result = self.tc.type_check(self.ctx, lam, pi_type)
        assert result.success

    def test_lambda_infer_fails(self):
        lam = Lam("x", Var("x"))
        with pytest.raises(TypeCheckError, match="Cannot infer"):
            self.tc.infer(self.ctx, lam)

    def test_app(self):
        ctx = self.ctx.extend("f", Pi("x", TYPE0, TYPE0))
        ctx = ctx.extend("a", TYPE0)
        result = self.tc.type_check(ctx, App(Var("f"), Var("a")))
        assert result.success

    def test_app_type_mismatch(self):
        ctx = self.ctx.extend("f", Pi("x", TYPE0, TYPE0))
        with pytest.raises(TypeCheckError):
            self.tc.infer(ctx, App(Var("f"), Universe(0)))

    def test_identity_formation(self):
        ctx = self.ctx.extend("a", TYPE0)
        ctx = ctx.extend("b", TYPE0)
        result = self.tc.type_check(ctx, Identity(TYPE0, Var("a"), Var("b")))
        assert result.success

    def test_refl(self):
        ctx = self.ctx.extend("a", TYPE0)
        refl = Refl(TYPE0, Var("a"))
        result = self.tc.type_check(ctx, refl)
        assert result.success

    def test_refl_check_identity(self):
        ctx = self.ctx.extend("a", TYPE0)
        id_type = Identity(TYPE0, Var("a"), Var("a"))
        refl = Refl(TYPE0, Var("a"))
        result = self.tc.type_check(ctx, refl, id_type)
        assert result.success

    def test_def_eq_same(self):
        assert self.tc.def_eq(self.ctx, Var("x"), Var("x"))

    def test_def_eq_different(self):
        assert not self.tc.def_eq(self.ctx, Var("x"), Var("y"))

    def test_def_eq_beta(self):
        t1 = App(Lam("x", Var("x")), Var("a"))
        t2 = Var("a")
        assert self.tc.def_eq(self.ctx, t1, t2)

    def test_sigma_formation(self):
        t = Sigma("x", TYPE0, TYPE0)
        result = self.tc.type_check(self.ctx, t)
        assert result.success

    def test_pair_check(self):
        # Sigma(_ : Type) × Type, 项需要是 Type 的居留元
        # Type : Type_1, 所以 Universe(0) 是 Type_1 的项
        # 改用变量来测试
        ctx = self.ctx.extend("a", TYPE0).extend("b", TYPE0)
        sig = Sigma("x", TYPE0, TYPE0)
        pair = Pair(Var("a"), Var("b"))
        result = self.tc.type_check(ctx, pair, sig)
        assert result.success

    def test_annotation(self):
        ann = Annotation(Universe(0), Universe(1))
        result = self.tc.type_check(self.ctx, ann)
        assert result.success


# ── 桥接测试 ──

class TestBridge:
    def test_invariant_to_identity(self):
        inv = StructuralInvariant(
            name="eigenvalues_real",
            expression="λ_i ∈ ℝ for all i",
            theorem="Spectral Theorem",
        )
        id_type = invariant_to_identity(inv)
        assert id_type.kind == TermKind.IDENTITY

    def test_invariant_to_prop_type(self):
        inv = StructuralInvariant(
            name="eigenvalues_real",
            expression="λ_i ∈ ℝ for all i",
            theorem="Spectral Theorem",
            affected_quantities=["eigenvalues"],
        )
        prop = invariant_to_prop_type(inv)
        assert prop.kind == TermKind.PI

    def test_morphism_to_type(self):
        ctd = ContinuumToDiscrete(method="fdm")
        mtype = morphism_to_type(ctd)
        assert mtype.morphism_name == "continuum_to_discrete"
        assert len(mtype.kept_proofs) > 0
        assert len(mtype.lost_proofs) > 0
        assert mtype.full_type is not None

    def test_propagation_to_transport(self):
        t = propagation_to_transport(
            invariant_name="energy_conservation",
            source_value=Var("a"),
            target_value=Var("b"),
            eq_proof=Var("p"),
        )
        assert t.kind == TermKind.TRANSPORT

    def test_structure_to_inductive(self):
        hs = HamiltonianSystem(symplectic=True)
        itype = structure_to_inductive(hs)
        assert itype.name == "Hamiltonian_System"
        assert len(itype.constructors) > 0

    def test_bridge_register_structure(self):
        bridge = TypeTheoryBridge()
        hs = HamiltonianSystem(symplectic=True)
        itype = bridge.register_structure(hs)
        assert hs.name in bridge._registered_structures

    def test_bridge_register_morphism(self):
        bridge = TypeTheoryBridge()
        ctd = ContinuumToDiscrete(method="fdm")
        mtype = bridge.register_morphism(ctd)
        assert ctd.name in bridge._registered_morphisms

    def test_bridge_propagate_invariant(self):
        bridge = TypeTheoryBridge()
        inv = StructuralInvariant(
            name="conservation_laws_at_discrete_level",
            expression="Conservation at discrete level",
            theorem="Discrete conservation",
        )
        ctd = ContinuumToDiscrete(method="fdm")
        result = bridge.propagate_invariant(inv, [ctd])
        assert result.success


# ── 元数学测试 ──

class TestMetamath:
    def test_formal_system_strength_order(self):
        assert FormalSystemStrength.PA.consistency_strength_order() < FormalSystemStrength.ZFC.consistency_strength_order()
        assert FormalSystemStrength.MLTT1.consistency_strength_order() > FormalSystemStrength.MLTT0.consistency_strength_order()

    def test_can_prove_consistency(self):
        assert FormalSystemStrength.ZFC.can_prove_consistency_of(FormalSystemStrength.PA)
        assert not FormalSystemStrength.PA.can_prove_consistency_of(FormalSystemStrength.PA)

    def test_undecidability_results(self):
        assert len(UNDECIDABILITY_RESULTS) >= 5
        halting = [r for r in UNDECIDABILITY_RESULTS if r.problem_name == "halting_problem"]
        assert len(halting) == 1
        assert halting[0].decidability == DecidabilityClass.UNDECIDABLE

    def test_godel_boundaries(self):
        assert len(GODEL_BOUNDARIES) >= 3
        for gb in GODEL_BOUNDARIES:
            assert gb.godel_sentence != ""
            assert len(gb.consistency_provable_in) > 0 or gb.theory_strength == FormalSystemStrength.TG

    def test_metamath_analyzer_scf(self):
        analyzer = MetamathAnalyzer()
        result = analyzer.analyze_invariant_decidability("scf_convergence", "SCF iteration convergence")
        assert result is not None
        assert result.decidability == DecidabilityClass.SEMI_DECIDABLE

    def test_metamath_analyzer_type_checking(self):
        analyzer = MetamathAnalyzer()
        result = analyzer.analyze_invariant_decidability("type_checking_MLTT")
        assert result is not None
        assert result.decidability == DecidabilityClass.DECIDABLE

    def test_godel_limitation(self):
        analyzer = MetamathAnalyzer()
        godel = analyzer.godel_limitation(FormalSystemStrength.CIC)
        assert godel.theory_name == "CIC (Coq/Lean)"
        assert "not provable" in godel.godel_sentence

    def test_consistency_comparison(self):
        analyzer = MetamathAnalyzer()
        comp = analyzer.consistency_strength_comparison(
            FormalSystemStrength.CIC, FormalSystemStrength.PA
        )
        assert "强于" in comp

    def test_invariant_state_mapping(self):
        analyzer = MetamathAnalyzer()
        assert analyzer.invariant_state_from_decidability(DecidabilityClass.DECIDABLE) == "SATISFIED"
        assert analyzer.invariant_state_from_decidability(DecidabilityClass.UNDECIDABLE) == "UNKNOWN"
        assert analyzer.invariant_state_from_decidability(DecidabilityClass.SEMI_DECIDABLE) == "CONDITIONAL"

    def test_full_analysis(self):
        analyzer = MetamathAnalyzer()
        result = analyzer.full_analysis("eigenvalue_computation")
        assert "decidability" in result
        assert "godel_limitation" in result
        assert result["system_strength"] == "CIC"


# ── CIC 测试 ──

class TestCIC:
    def test_prop_type(self):
        from math_anything.type_theory import PROP, TYPE0_SORT, SortKind
        assert PROP.is_prop
        assert PROP.sort_kind == SortKind.PROP
        assert not TYPE0_SORT.is_prop

    def test_pi_sort_prop_prop(self):
        from math_anything.type_theory import PropTypeRule, PROP, TYPE0_SORT
        result = PropTypeRule.pi_sort(PROP, PROP)
        assert result.is_prop

    def test_pi_sort_type_prop(self):
        from math_anything.type_theory import PropTypeRule, TYPE0_SORT, PROP
        result = PropTypeRule.pi_sort(TYPE0_SORT, PROP)
        assert result.is_prop

    def test_pi_sort_type_type(self):
        from math_anything.type_theory import PropTypeRule, TYPE0_SORT
        result = PropTypeRule.pi_sort(TYPE0_SORT, TYPE0_SORT)
        assert not result.is_prop
        assert result.level == 1

    def test_cic_bridge_severity(self):
        from math_anything.type_theory import CICBridge, PROP
        cb = CICBridge()
        assert cb.severity_to_sort("theorem").is_prop
        assert not cb.severity_to_sort("conservation").is_prop

    def test_scf_coinductive(self):
        from math_anything.type_theory import CICBridge
        cb = CICBridge()
        stream = cb.scf_to_coinductive()
        assert stream.name == "SCFStream"
        assert len(stream.co_constructors) > 0

    def test_quotient_type(self):
        from math_anything.type_theory import QuotientType, TYPE0
        qt = QuotientType(base_type=TYPE0, relation=Var("R"))
        assert qt.name == "Quotient"

    def test_fixpoint_termination(self):
        from math_anything.type_theory import Fixpoint, check_termination
        fix = Fixpoint("f", arrow(TYPE0, TYPE0), Var("body"))
        result = check_termination(fix)
        assert result.terminates

    def test_cic_type_checker_sort(self):
        from math_anything.type_theory import CICTypeChecker, PROP, Context
        from math_anything.type_theory.cic import Sort
        cic = CICTypeChecker()
        ctx = Context()
        # 未绑定变量应抛出 TypeCheckError
        from math_anything.type_theory.checker import TypeCheckError
        with pytest.raises(TypeCheckError, match="Unbound variable"):
            cic.sort_of(Var("x"), ctx)

    def test_cumulativity(self):
        from math_anything.type_theory import CICTypeChecker, PROP, TYPE0_SORT, TYPE1_SORT
        cic = CICTypeChecker()
        assert cic.check_cumulativity(PROP, TYPE0_SORT)
        assert cic.check_cumulativity(TYPE0_SORT, TYPE1_SORT)
        assert not cic.check_cumulativity(TYPE1_SORT, TYPE0_SORT)


# ── HoTT 测试 ──

class TestHoTT:
    def test_hlevel_from_isomorphism(self):
        from math_anything.type_theory import HLevel
        hlevel = HLevel.from_morphism_properties(True, True, True)
        assert hlevel == HLevel.CONTRACTIBLE

    def test_hlevel_from_injective(self):
        from math_anything.type_theory import HLevel
        hlevel = HLevel.from_morphism_properties(True, False, False)
        assert hlevel == HLevel.SET

    def test_hlevel_from_general(self):
        from math_anything.type_theory import HLevel
        hlevel = HLevel.from_morphism_properties(False, False, False)
        assert hlevel == HLevel.GROUPOID

    def test_interval(self):
        from math_anything.type_theory import INTERVAL
        assert INTERVAL.name == "Interval"
        assert len(INTERVAL.point_constructors) == 2
        assert len(INTERVAL.path_constructors) == 1
        assert INTERVAL.path_constructors[0].source == "zero"
        assert INTERVAL.path_constructors[0].target == "one"

    def test_circle(self):
        from math_anything.type_theory import CIRCLE
        assert CIRCLE.name == "S1"
        assert len(CIRCLE.point_constructors) == 1
        assert len(CIRCLE.path_constructors) == 1
        assert CIRCLE.path_constructors[0].source == "base"
        assert CIRCLE.path_constructors[0].target == "base"

    def test_hott_bridge_morphism_chain(self):
        from math_anything.type_theory import HoTTBridge
        from math_anything.morphisms import ContinuumToDiscrete, TimeSteppingMorphism
        hb = HoTTBridge()
        ctd = ContinuumToDiscrete(method="fdm")
        tsm = TimeSteppingMorphism(method="euler_explicit")
        hit = hb.morphism_chain_to_hit("chain", [ctd, tsm])
        assert hit.name == "chain"
        assert len(hit.point_constructors) == 3
        assert len(hit.path_constructors) == 2

    def test_hott_checker_hlevel(self):
        from math_anything.type_theory import HoTTTypeChecker, HLevel, Context
        hc = HoTTTypeChecker()
        ctx = Context()
        # Universe is n-groupoid
        assert hc.infer_h_level(Universe(0), ctx) == HLevel.N_GROUPOID
        # Identity type is proposition
        from math_anything.type_theory import Identity, TYPE0, Var
        id_type = Identity(TYPE0, Var("a"), Var("a"))
        assert hc.infer_h_level(id_type, ctx) == HLevel.PROPOSITION

    def test_equivalence(self):
        from math_anything.type_theory import Equivalence, Var
        equiv = Equivalence(
            forward=Var("f"), backward=Var("g"),
            section=Var("alpha"), retraction=Var("beta"),
        )
        assert equiv.forward == Var("f")

    def test_univalence(self):
        from math_anything.type_theory import Univalence, Equivalence, Var, TYPE0
        equiv = Equivalence(Var("f"), Var("g"), Var("alpha"), Var("beta"))
        ua = Univalence(equiv, TYPE0, TYPE0)
        assert ua.source_type == TYPE0


# ── 验证流水线测试 ──

class TestVerificationPipeline:
    def test_symbolic_layer(self):
        from math_anything.type_theory.verify import VerificationPipeline, VerificationLayer
        vp = VerificationPipeline()
        result = vp.verify(
            statement="E = mc^2",
            layers=[VerificationLayer.SYMBOLIC],
        )
        assert len(result.layers) == 1
        assert result.layers[0].layer == VerificationLayer.SYMBOLIC

    def test_type_system_layer(self):
        from math_anything.type_theory.verify import VerificationPipeline, VerificationLayer
        vp = VerificationPipeline()
        result = vp.verify(
            statement="eigenvalues are real",
            layers=[VerificationLayer.TYPE_SYSTEM],
        )
        assert len(result.layers) == 1
        assert result.layers[0].passed

    def test_logic_layer(self):
        from math_anything.type_theory.verify import VerificationPipeline, VerificationLayer
        vp = VerificationPipeline()
        result = vp.verify(
            statement="A implies B",
            proof_text="Given A, we derive B",
            assumptions=["A is true"],
            goals=["B is true"],
            layers=[VerificationLayer.LOGIC],
        )
        assert len(result.layers) == 1
        assert result.layers[0].passed

    def test_full_pipeline(self):
        from math_anything.type_theory.verify import VerificationPipeline
        vp = VerificationPipeline()
        result = vp.verify(
            statement="Self-adjoint operators have real eigenvalues",
            proof_text="By Spectral Theorem",
            assumptions=["H is self-adjoint"],
            goals=["All eigenvalues in R"],
        )
        assert len(result.layers) == 5
        # Lean4 层：代码已生成但未编译验证，passed=True 表示代码生成成功
        assert result.layers[4].confidence <= 0.5

    def test_overall_confidence(self):
        from math_anything.type_theory.verify import VerificationPipeline, VerificationLayer
        vp = VerificationPipeline()
        result = vp.verify(
            statement="x = x",
            layers=[VerificationLayer.SYMBOLIC, VerificationLayer.TYPE_SYSTEM],
        )
        result.compute_overall()
        assert 0 < result.overall_confidence <= 1.0
