"""Unit tests for the constraints module — invariant, propagation, boundary, domain."""

import pytest

from math_anything.constraints.invariant import (
    DomainCondition,
    InvariantStatus,
    LearnedInvariant,
    PropagationOutcome,
    WeakeningRule,
    from_structural_invariant,
)
from math_anything.constraints.propagation import (
    ConstraintPropagation,
    PropagationChain,
    PropagationResult,
)
from math_anything.constraints.boundary import (
    BoundaryEvolution,
    BoundaryState,
    ExecutionRecord,
    RiskItem,
)
from math_anything.constraints.domain import (
    DomainHypothesis,
    DomainLearner,
    TrainingExample,
)
from math_anything.structures._core import StructuralInvariant


# ── LearnedInvariant ──

class TestLearnedInvariant:

    def test_creation(self):
        inv = LearnedInvariant(
            name="energy_negative",
            expression="E_total < 0",
            theorem="variational principle",
        )
        assert inv.name == "energy_negative"
        assert inv.expression == "E_total < 0"
        assert inv.effective_expression == "E_total < 0"

    def test_default_values(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        assert inv.domain_confidence == 1.0
        assert inv.is_probabilistic is False
        assert inv.probability_threshold == 0.95
        assert inv.active_weakening == 0
        assert inv.violation_count == 0
        assert inv.satisfaction_count == 0
        assert inv.severity == "theorem"

    def test_is_active_no_conditions(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        assert inv.is_active({"x": 1}) is True

    def test_is_active_with_satisfied_condition(self):
        inv = LearnedInvariant(
            name="test", expression="x > 0",
            domain_conditions=[DomainCondition("is_bound", "==", True)],
        )
        assert inv.is_active({"is_bound": True}) is True

    def test_is_active_with_unsatisfied_condition(self):
        inv = LearnedInvariant(
            name="test", expression="x > 0",
            domain_conditions=[DomainCondition("is_bound", "==", True)],
        )
        assert inv.is_active({"is_bound": False}) is False

    def test_evaluate_satisfied(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        status = inv.evaluate({"x": 5})
        assert status == InvariantStatus.SATISFIED
        assert inv.satisfaction_count == 1

    def test_evaluate_violated(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        status = inv.evaluate({"x": -1})
        assert status == InvariantStatus.VIOLATED
        assert inv.violation_count == 1

    def test_evaluate_inactive(self):
        inv = LearnedInvariant(
            name="test", expression="x > 0",
            domain_conditions=[DomainCondition("is_bound", "==", True)],
        )
        status = inv.evaluate({"is_bound": False})
        assert status == InvariantStatus.INACTIVE

    def test_evaluate_unknown_when_expression_non_boolean(self):
        # safe_eval returns a non-bool result (like a number), which
        # falls through the bool check and returns UNKNOWN
        inv = LearnedInvariant(name="test", expression="x + 1")
        status = inv.evaluate({"x": 5})
        assert status == InvariantStatus.UNKNOWN

    def test_weaken_applies_rule(self):
        inv = LearnedInvariant(
            name="convergence",
            expression="abs(E_new - E_old) < 1e-6",
            weakening_rules=[
                WeakeningRule(
                    name="relax_1e4",
                    weakened_expression="abs(E_new - E_old) < 1e-4",
                    trigger_condition="iterations > 50",
                    consequence="accuracy reduced",
                ),
            ],
        )
        result = inv.weaken(0)
        assert result is inv
        assert inv.active_weakening == 1
        assert inv.effective_expression == "abs(E_new - E_old) < 1e-4"

    def test_weaken_without_rules_is_noop(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        result = inv.weaken()
        assert result is inv
        assert inv.active_weakening == 0

    def test_restore(self):
        inv = LearnedInvariant(
            name="test", expression="x > 0",
            weakening_rules=[
                WeakeningRule("w1", "x > -1", "always", "relaxed"),
            ],
        )
        inv.weaken(0)
        assert inv.active_weakening > 0
        inv.restore()
        assert inv.active_weakening == 0
        assert inv.effective_expression == "x > 0"

    def test_record_propagation(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        inv.record_propagation("born_oppenheimer", "PRESERVED")
        assert len(inv.propagation_history) == 1
        assert inv.propagation_history[0] == ("born_oppenheimer", "PRESERVED")

    def test_to_dict(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        d = inv.to_dict()
        assert d["name"] == "test"
        assert d["expression"] == "x > 0"
        assert d["original_expression"] == "x > 0"
        assert d["status"] == "original"

    def test_to_dict_weakened(self):
        inv = LearnedInvariant(
            name="test", expression="x > 0",
            weakening_rules=[WeakeningRule("w1", "x > -1", "always", "relaxed")],
        )
        inv.weaken(0)
        d = inv.to_dict()
        assert d["status"] == "weakened"

    def test_evaluate_weakened_returns_weakened_status(self):
        inv = LearnedInvariant(
            name="test", expression="x > 10",
            weakening_rules=[WeakeningRule("w1", "x > 0", "always", "relaxed")],
        )
        inv.weaken(0)
        # x=5 satisfies "x > 0" but is weakened
        status = inv.evaluate({"x": 5})
        assert status == InvariantStatus.WEAKENED


class TestFromStructuralInvariant:

    def test_creates_learned_invariant(self):
        si = StructuralInvariant(
            name="real_eigenvalues",
            expression="λ_i ∈ ℝ for all i",
            theorem="Spectral Theorem",
            severity="theorem",
        )
        li = from_structural_invariant(si)
        assert isinstance(li, LearnedInvariant)
        assert li.name == "real_eigenvalues"
        assert li.expression == "λ_i ∈ ℝ for all i"
        assert li.theorem == "Spectral Theorem"
        assert li.domain_confidence == 1.0
        assert li.is_probabilistic is False


class TestDomainCondition:

    def test_equality_operator(self):
        dc = DomainCondition(feature="is_bound", operator="==", threshold=True)
        assert dc.feature == "is_bound"
        assert dc.operator == "=="

    def test_comparison_operator(self):
        dc = DomainCondition(feature="encut", operator=">", threshold=400.0)
        assert dc.threshold == 400.0

    def test_default_confidence(self):
        dc = DomainCondition(feature="x", operator=">", threshold=0)
        assert dc.confidence == 0.5


class TestWeakeningRule:

    def test_creation(self):
        wr = WeakeningRule(
            name="relax",
            weakened_expression="x > -1",
            trigger_condition="always",
            consequence="less strict",
        )
        assert wr.name == "relax"
        assert wr.recovery_path == ""


# ── ConstraintPropagation ──

class TestConstraintPropagation:

    @pytest.fixture
    def simple_morphism(self):
        """Create a minimal morphism-like object for testing."""
        class FakeMorphism:
            name = "test_morph"
            invariants_kept = ["energy_negative"]
            invariants_lost = ["exact_solution"]
            invariants_introduced = ["discrete_approximation"]
            kernel_description = ""
        return FakeMorphism()

    @pytest.fixture
    def invariant(self):
        return LearnedInvariant(name="energy_negative", expression="E_total < 0")

    def test_propagate_single_preserved(self, simple_morphism, invariant):
        cp = ConstraintPropagation()
        result = cp.propagate_single(invariant, simple_morphism, "A", "B")
        assert result.outcome == PropagationOutcome.PRESERVED
        assert result.morphism_name == "test_morph"

    def test_propagate_single_lost(self, simple_morphism):
        inv = LearnedInvariant(name="exact_solution", expression="u = u_exact")
        cp = ConstraintPropagation()
        result = cp.propagate_single(inv, simple_morphism, "A", "B")
        assert result.outcome == PropagationOutcome.LOST
        assert result.loss_reason != ""

    def test_propagate_single_emerged(self, simple_morphism):
        inv = LearnedInvariant(name="discrete_approximation", expression="u_h ≈ u")
        cp = ConstraintPropagation()
        result = cp.propagate_single(inv, simple_morphism, "A", "B")
        assert result.outcome == PropagationOutcome.EMERGED

    def test_propagate_single_conditional(self):
        class MorphWithKernel:
            name = "kernel_morph"
            invariants_kept = []
            invariants_lost = []
            invariants_introduced = []
            kernel_description = "is_bound"
        inv = LearnedInvariant(
            name="energy_negative", expression="E < 0",
            domain_conditions=[DomainCondition("is_bound", "==", True)],
        )
        cp = ConstraintPropagation()
        result = cp.propagate_single(inv, MorphWithKernel(), "A", "B")
        assert result.outcome == PropagationOutcome.CONDITIONAL

    def test_propagate_chain(self, simple_morphism, invariant):
        cp = ConstraintPropagation()
        chain = cp.propagate_chain(
            [invariant],
            [simple_morphism],
            ["A"],
            ["B"],
        )
        assert isinstance(chain, PropagationChain)
        assert len(chain.results) == 1

    def test_propagation_result_to_dict(self, simple_morphism, invariant):
        cp = ConstraintPropagation()
        result = cp.propagate_single(invariant, simple_morphism, "A", "B")
        d = result.to_dict()
        assert "invariant" in d
        assert "outcome" in d
        assert "morphism" in d

    def test_compose_propagation_lost_dominates(self):
        cp = ConstraintPropagation()
        r1 = PropagationResult(
            invariant=LearnedInvariant(name="t", expression="x"),
            outcome=PropagationOutcome.PRESERVED,
            morphism_name="f", source_structure="A", target_structure="B",
        )
        r2 = PropagationResult(
            invariant=LearnedInvariant(name="t", expression="x"),
            outcome=PropagationOutcome.LOST,
            morphism_name="g", source_structure="B", target_structure="C",
        )
        assert cp.compose_propagation(r1, r2) == PropagationOutcome.LOST

    def test_compose_propagation_preserved_preserved(self):
        cp = ConstraintPropagation()
        r1 = PropagationResult(
            invariant=LearnedInvariant(name="t", expression="x"),
            outcome=PropagationOutcome.PRESERVED,
            morphism_name="f", source_structure="A", target_structure="B",
        )
        r2 = PropagationResult(
            invariant=LearnedInvariant(name="t", expression="x"),
            outcome=PropagationOutcome.PRESERVED,
            morphism_name="g", source_structure="B", target_structure="C",
        )
        assert cp.compose_propagation(r1, r2) == PropagationOutcome.PRESERVED


class TestPropagationChain:

    def test_final_state_empty(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        chain = PropagationChain(invariants=[inv], chain=["f"])
        fs = chain.final_state
        assert "test" in fs
        assert fs["test"] == PropagationOutcome.PRESERVED

    def test_preserved_invariants(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        chain = PropagationChain(invariants=[inv], chain=["f"])
        assert inv in chain.preserved_invariants

    def test_lost_invariants_empty(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        chain = PropagationChain(invariants=[inv], chain=["f"])
        assert len(chain.lost_invariants) == 0


# ── BoundaryState & BoundaryEvolution ──

class TestBoundaryState:

    def test_default_state(self):
        bs = BoundaryState()
        assert len(bs.interior_invariants) == 0
        assert len(bs.boundary_invariants) == 0
        assert len(bs.exterior_invariants) == 0
        assert bs.expansion_count == 0
        assert bs.contraction_count == 0

    def test_is_interior_empty(self):
        bs = BoundaryState()
        assert bs.is_interior({"x": 1}) is True

    def test_is_interior_satisfied(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        bs = BoundaryState(interior_invariants=[inv])
        assert bs.is_interior({"x": 5}) is True

    def test_is_interior_violated(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        bs = BoundaryState(interior_invariants=[inv])
        assert bs.is_interior({"x": -1}) is False

    def test_risk_assessment_empty(self):
        bs = BoundaryState()
        risks = bs.risk_assessment({"x": 1})
        assert risks == []

    def test_to_dict(self):
        bs = BoundaryState()
        d = bs.to_dict()
        assert "interior_count" in d
        assert "boundary_count" in d
        assert "exterior_count" in d
        assert d["interior_count"] == 0


class TestBoundaryEvolution:

    def test_evolve_success(self):
        be = BoundaryEvolution()
        record = ExecutionRecord(success=True, params={"x": 1})
        state = be.evolve(record)
        assert state.expansion_count == 1
        assert state.total_experiences == 1

    def test_evolve_failure(self):
        be = BoundaryEvolution()
        record = ExecutionRecord(success=False, params={"x": -1})
        state = be.evolve(record)
        assert state.contraction_count == 1
        assert state.total_experiences == 1

    def test_consecutive_successes_promote(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        inv.violation_count = 0
        bs = BoundaryState(
            boundary_invariants=[(inv, 0.2)],
            expansion_threshold=3,
        )
        be = BoundaryEvolution(state=bs)
        for _ in range(3):
            be.evolve(ExecutionRecord(success=True, params={"x": 5}))
        # low-risk boundary invariant should be promoted
        assert len(be.state.interior_invariants) > 0

    def test_consecutive_failures_demote(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        inv.violation_count = 5
        bs = BoundaryState(
            interior_invariants=[inv],
            contraction_threshold=2,
        )
        be = BoundaryEvolution(state=bs)
        for _ in range(2):
            be.evolve(ExecutionRecord(
                success=False,
                params={"x": -1},
                invariant_results=[("test", InvariantStatus.VIOLATED)],
            ))
        assert len(be.state.boundary_invariants) > 0


class TestExecutionRecord:

    def test_creation(self):
        rec = ExecutionRecord(success=True, params={"x": 1})
        assert rec.success is True
        assert rec.wall_time == 0.0
        assert rec.error_message == ""


class TestRiskItem:

    def test_creation(self):
        inv = LearnedInvariant(name="test", expression="x > 0")
        ri = RiskItem(invariant=inv, risk_score=0.8, status=InvariantStatus.VIOLATED)
        assert ri.risk_score == 0.8
        assert ri.status == InvariantStatus.VIOLATED


# ── DomainLearner ──

class TestDomainLearner:

    def test_creation(self):
        dl = DomainLearner()
        assert len(dl.examples) == 0

    def test_add_example(self):
        dl = DomainLearner()
        ex = TrainingExample(
            params={"encut": 520},
            invariant_name="energy_negative",
            status=InvariantStatus.SATISFIED,
        )
        dl.add_example(ex)
        assert len(dl.examples) == 1

    def test_learn_insufficient_data(self):
        dl = DomainLearner()
        for i in range(3):
            dl.add_example(TrainingExample(
                params={"encut": 520},
                invariant_name="energy_negative",
                status=InvariantStatus.SATISFIED,
            ))
        result = dl.learn("energy_negative")
        assert result is None

    def test_learn_with_enough_data(self):
        dl = DomainLearner()
        # positive examples
        for _ in range(10):
            dl.add_example(TrainingExample(
                params={"encut": 520, "is_bound": True},
                invariant_name="energy_negative",
                status=InvariantStatus.SATISFIED,
            ))
        # negative examples
        for _ in range(10):
            dl.add_example(TrainingExample(
                params={"encut": 100, "is_bound": False},
                invariant_name="energy_negative",
                status=InvariantStatus.VIOLATED,
            ))
        hyp = dl.learn("energy_negative")
        assert hyp is not None
        assert isinstance(hyp, DomainHypothesis)
        assert hyp.invariant_name == "energy_negative"
        assert len(hyp.conditions) > 0

    def test_apply_to_invariant(self):
        dl = DomainLearner()
        for _ in range(10):
            dl.add_example(TrainingExample(
                params={"encut": 520, "is_bound": True},
                invariant_name="energy_negative",
                status=InvariantStatus.SATISFIED,
            ))
        for _ in range(10):
            dl.add_example(TrainingExample(
                params={"encut": 100, "is_bound": False},
                invariant_name="energy_negative",
                status=InvariantStatus.VIOLATED,
            ))
        inv = LearnedInvariant(name="energy_negative", expression="E < 0")
        result = dl.apply_to_invariant(inv)
        assert result is inv
        # domain conditions may or may not be set depending on accuracy threshold
        # but the method should not raise

    def test_stats(self):
        dl = DomainLearner()
        dl.add_example(TrainingExample(
            params={"x": 1}, invariant_name="t", status=InvariantStatus.SATISFIED,
        ))
        stats = dl.stats()
        assert stats["total_examples"] == 1
        assert stats["unique_invariants"] == 1

    def test_example_cap(self):
        dl = DomainLearner()
        for i in range(11000):
            dl.add_example(TrainingExample(
                params={"x": i}, invariant_name="t", status=InvariantStatus.SATISFIED,
            ))
        assert len(dl.examples) <= 10000


class TestDomainHypothesis:

    def test_creation(self):
        dh = DomainHypothesis(
            invariant_name="test",
            conditions=[DomainCondition("x", ">", 0)],
            accuracy=0.9,
            coverage=0.8,
            support=50,
        )
        assert dh.invariant_name == "test"
        assert dh.accuracy == 0.9
        assert dh.support == 50


class TestTrainingExample:

    def test_creation(self):
        te = TrainingExample(
            params={"encut": 520},
            invariant_name="energy_negative",
            status=InvariantStatus.SATISFIED,
            source="vasp",
        )
        assert te.source == "vasp"
        assert te.status == InvariantStatus.SATISFIED

    def test_default_source(self):
        te = TrainingExample(
            params={}, invariant_name="t", status=InvariantStatus.UNKNOWN,
        )
        assert te.source == ""
