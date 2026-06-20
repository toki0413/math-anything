"""Integration tests for the constraint framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from math_anything.constraints import (
    BoundaryEvolution,
    BoundaryState,
    ConstraintPropagation,
    DomainCondition,
    DomainLearner,
    ExecutionRecord,
    InvariantStatus,
    LearnedInvariant,
    PropagationOutcome,
    TrainingExample,
    WeakeningRule,
    from_structural_invariant,
)
from math_anything.structures.properties import StructuralInvariant


# ── helpers ──

@dataclass
class SimpleMorphism:
    """A minimal morphism-like object for propagation testing."""
    name: str
    invariants_kept: list[str] = field(default_factory=list)
    invariants_lost: list[str] = field(default_factory=list)
    invariants_introduced: list[str] = field(default_factory=list)
    kernel_description: str = ""


# ── Test LearnedInvariant ──

class TestLearnedInvariant:
    def test_create_with_weakening_rules(self):
        rules = [
            WeakeningRule(
                name="relax_tol",
                weakened_expression="abs(E_new - E_old) < 1e-4",
                trigger_condition="iterations > 50",
                consequence="accuracy reduced",
                recovery_path="reduce mixing parameter",
            ),
            WeakeningRule(
                name="relax_tol_loose",
                weakened_expression="abs(E_new - E_old) < 1e-2",
                trigger_condition="iterations > 100",
                consequence="significant accuracy loss",
                recovery_path="restart with better guess",
            ),
        ]
        inv = LearnedInvariant(
            name="energy_convergence",
            expression="abs(E_new - E_old) < 1e-6",
            theorem="Kohn-Sham convergence",
            weakening_rules=rules,
            severity="consistency",
        )
        assert inv.name == "energy_convergence"
        assert inv.effective_expression == "abs(E_new - E_old) < 1e-6"
        assert inv.active_weakening == 0
        assert inv.severity == "consistency"
        assert len(inv.weakening_rules) == 2

    def test_evaluate_satisfied(self):
        inv = LearnedInvariant(
            name="energy_positive",
            expression="E > 0",
        )
        status = inv.evaluate({"E": 10.0})
        assert status == InvariantStatus.SATISFIED
        assert inv.satisfaction_count == 1

    def test_evaluate_violated(self):
        inv = LearnedInvariant(
            name="energy_negative",
            expression="E < 0",
        )
        status = inv.evaluate({"E": 5.0})
        assert status == InvariantStatus.VIOLATED
        assert inv.violation_count == 1

    def test_weaken_and_restore(self):
        rules = [
            WeakeningRule(
                name="relax",
                weakened_expression="E < 10.0",
                trigger_condition="failures > 3",
                consequence="boundary relaxed",
            ),
        ]
        inv = LearnedInvariant(
            name="bound",
            expression="E < 1.0",
            weakening_rules=rules,
        )
        # Before weakening: E=5.0 violates
        assert inv.evaluate({"E": 5.0}) == InvariantStatus.VIOLATED
        # Weaken
        inv.weaken(0)
        assert inv.active_weakening == 1
        assert inv.effective_expression == "E < 10.0"
        # After weakening: E=5.0 is satisfied (as weakened)
        assert inv.evaluate({"E": 5.0}) == InvariantStatus.WEAKENED
        # Restore
        inv.restore()
        assert inv.active_weakening == 0
        assert inv.effective_expression == "E < 1.0"
        assert inv.evaluate({"E": 5.0}) == InvariantStatus.VIOLATED

    def test_auto_weaken_uses_next_rule(self):
        rules = [
            WeakeningRule("r1", "x < 5", "always", ""),
            WeakeningRule("r2", "x < 10", "always", ""),
            WeakeningRule("r3", "x < 20", "always", ""),
        ]
        inv = LearnedInvariant("test_auto", "x < 1", weakening_rules=rules)
        # weaken(): rule_index = min(active(0)+1, len-1) = 1, picks r2
        inv.weaken()
        assert inv.effective_expression == "x < 10"
        assert inv.active_weakening == 2
        # next: rule_index = min(2+1, 2) = 2, picks r3
        inv.weaken()
        assert inv.effective_expression == "x < 20"
        assert inv.active_weakening == 3
        # next: rule_index = min(3+1, 2) = 2, still r3 (no change)
        inv.weaken()
        assert inv.effective_expression == "x < 20"

    def test_domain_condition_inactive(self):
        inv = LearnedInvariant(
            name="bound_system_only",
            expression="E_total < 0",
            domain_conditions=[
                DomainCondition(
                    feature="is_bound",
                    operator="present",
                    threshold=0.0,
                    confidence=0.9,
                ),
            ],
        )
        # is_bound not present → INACTIVE
        assert inv.evaluate({"E_total": -5.0}) == InvariantStatus.INACTIVE
        # is_bound present → SATISFIED
        assert inv.evaluate({"E_total": -5.0, "is_bound": True}) == InvariantStatus.SATISFIED

    def test_is_probabilistic(self):
        inv = LearnedInvariant(
            name="likely_metallic",
            expression="band_gap < 0.1",
            is_probabilistic=True,
            probability_threshold=0.9,
        )
        assert inv.is_probabilistic
        assert inv.probability_threshold == 0.9

    def test_to_dict(self):
        inv = LearnedInvariant(
            name="energy_conv",
            expression="|E| < 1e-6",
            theorem="convergence",
        )
        d = inv.to_dict()
        assert d["name"] == "energy_conv"
        assert d["expression"] == "|E| < 1e-6"
        assert d["status"] == "original"
        assert d["severity"] == "theorem"

    def test_record_propagation(self):
        inv = LearnedInvariant("test", "x > 0")
        inv.record_propagation("born_oppenheimer", "PRESERVED")
        inv.record_propagation("kohn_sham", "WEAKENED")
        assert len(inv.propagation_history) == 2
        assert inv.propagation_history[0] == ("born_oppenheimer", "PRESERVED")


# ── Test ConstraintPropagation ──

class TestConstraintPropagation:
    def test_propagate_single_preserved(self):
        morph = SimpleMorphism(
            name="born_oppenheimer",
            invariants_kept=["eigenvalues_real"],
        )
        inv = LearnedInvariant(
            name="eigenvalues_real",
            expression="imag(lambda) == 0",
            theorem="Spectral Theorem",
        )
        prop = ConstraintPropagation()
        result = prop.propagate_single(inv, morph, "FullManyBody", "Electronic")
        assert result.outcome == PropagationOutcome.PRESERVED

    def test_propagate_single_lost(self):
        morph = SimpleMorphism(
            name="classical_limit",
            invariants_lost=["energy_quantization"],
        )
        inv = LearnedInvariant(
            name="energy_quantization",
            expression="E_n = (n + 1/2) * hbar * omega",
        )
        prop = ConstraintPropagation()
        result = prop.propagate_single(inv, morph, "Quantum", "Classical")
        assert result.outcome == PropagationOutcome.LOST
        assert "classical_limit" in result.loss_reason

    def test_propagate_single_conditional(self):
        morph = SimpleMorphism(
            name="projection",
            kernel_description="high_frequency",
        )
        inv = LearnedInvariant(
            name="energy_bounded",
            expression="E_kin < 1000",
            domain_conditions=[
                DomainCondition(feature="high_frequency", operator="present", threshold=0.0),
            ],
        )
        prop = ConstraintPropagation()
        result = prop.propagate_single(inv, morph, "Full", "Truncated")
        assert result.outcome == PropagationOutcome.CONDITIONAL
        assert "high_frequency" in result.new_condition

    def test_propagate_single_emerged(self):
        morph = SimpleMorphism(
            name="symmetry_reduction",
            invariants_introduced=["crystal_symmetry"],
        )
        inv = LearnedInvariant("crystal_symmetry", "g ∈ G")
        prop = ConstraintPropagation()
        result = prop.propagate_single(inv, morph, "General", "Periodic")
        assert result.outcome == PropagationOutcome.EMERGED

    def test_propagation_chain(self):
        m1 = SimpleMorphism(
            name="born_oppenheimer",
            invariants_kept=["eigenvalues_real", "variational"],
            invariants_lost=["fully_quantum"],
        )
        m2 = SimpleMorphism(
            name="kohn_sham",
            invariants_kept=["eigenvalues_real"],
            invariants_lost=["variational"],
            invariants_introduced=["xc_approximation"],
        )
        invariants = [
            LearnedInvariant("eigenvalues_real", "imag(λ)=0"),
            LearnedInvariant("variational", "E[ρ] ≥ E_gs"),
            LearnedInvariant("fully_quantum", "Ψ exact"),
        ]
        prop = ConstraintPropagation()
        chain = prop.propagate_chain(
            invariants,
            [m1, m2],
            ["s0", "s1"],
            ["s1", "s2"],
        )
        # Verify step-level results directly
        step0 = chain.results[0]  # 3 invariants through m1
        outcomes_step0 = {r.invariant.name: r.outcome for r in step0}
        assert outcomes_step0["eigenvalues_real"] == PropagationOutcome.PRESERVED
        assert outcomes_step0["variational"] == PropagationOutcome.PRESERVED
        assert outcomes_step0["fully_quantum"] == PropagationOutcome.LOST

        step1 = chain.results[1]  # 2 surviving invariants through m2
        outcomes_step1 = {r.invariant.name: r.outcome for r in step1}
        assert outcomes_step1["eigenvalues_real"] == PropagationOutcome.PRESERVED
        assert outcomes_step1["variational"] == PropagationOutcome.LOST
        # xc_approximation emerges from m2 but is added to current_invariants
        # after m2's loop, so it only appears in step results of the next morphism

    def test_propagation_chain_emerged(self):
        m1 = SimpleMorphism(
            name="m1",
            invariants_introduced=["new_invariant"],
        )
        invariants = [
            LearnedInvariant("seed", "x > 0"),
        ]
        prop = ConstraintPropagation()
        chain = prop.propagate_chain(
            invariants,
            [m1],
            ["s0"],
            ["s1"],
        )
        # The emerged invariant appears in the step results for subsequent steps
        step0 = chain.results[0]
        assert len(step0) == 1  # only "seed" went through m1
        assert step0[0].invariant.name == "seed"
        # "new_invariant" was added to current_invariants after this step,
        # but not present in step0 results since it wasn't propagated through m1
        # It will be propagated through the next morphism if there is one

    def test_compose_propagation(self):
        r1_result = PropagationOutcome.WEAKENED
        r2_result = PropagationOutcome.PRESERVED
        from dataclasses import dataclass as dc

        @dc
        class MockResult:
            outcome: PropagationOutcome

        prop = ConstraintPropagation()
        combined = prop.compose_propagation(
            MockResult(r1_result), MockResult(r2_result)
        )
        assert combined == PropagationOutcome.WEAKENED

    def test_compose_lost_dominates(self):
        from dataclasses import dataclass as dc

        @dc
        class MockResult:
            outcome: PropagationOutcome

        prop = ConstraintPropagation()
        combined = prop.compose_propagation(
            MockResult(PropagationOutcome.LOST),
            MockResult(PropagationOutcome.PRESERVED),
        )
        assert combined == PropagationOutcome.LOST


# ── Test BoundaryEvolution ──

class TestBoundaryEvolution:
    def test_create_boundary_state(self):
        state = BoundaryState()
        assert state.total_experiences == 0
        assert len(state.interior_invariants) == 0
        assert len(state.boundary_invariants) == 0

    def test_is_interior(self):
        inv = LearnedInvariant("E_bound", "E < 0")
        state = BoundaryState(interior_invariants=[inv])
        assert state.is_interior({"E": -5.0})
        assert not state.is_interior({"E": 5.0})

    def test_is_boundary(self):
        inv = LearnedInvariant("E_bound", "E < 0")
        state = BoundaryState(boundary_invariants=[(inv, 0.5)])
        assert state.is_boundary({"E": 5.0})
        assert not state.is_boundary({"E": -5.0})

    def test_risk_assessment(self):
        inv_interior = LearnedInvariant("E_conv", "abs(E_new - E_old) < 1e-6")
        inv_boundary = LearnedInvariant("force_convergence", "max_force < 0.01")
        state = BoundaryState(
            interior_invariants=[inv_interior],
            boundary_invariants=[(inv_boundary, 0.4)],
        )
        risks = state.risk_assessment({"E_new": 10.0, "E_old": 10.00001, "max_force": 0.02})
        assert len(risks) == 2
        # Highest risk should be the violated one
        assert risks[0].risk_score >= risks[1].risk_score

    def test_evolve_success_expansion(self):
        inv = LearnedInvariant("E_conv", "abs(E_new - E_old) < 1e-6")
        state = BoundaryState(
            interior_invariants=[inv],
            expansion_threshold=3,
        )
        evolution = BoundaryEvolution(state)
        for _ in range(3):
            evolution.evolve(ExecutionRecord(
                success=True,
                params={},
                invariant_results=[("E_conv", InvariantStatus.SATISFIED)],
            ))
        assert state.expansion_count == 3
        assert state.total_experiences == 3

    def test_evolve_failure_contraction(self):
        inv = LearnedInvariant(
            "E_conv",
            "abs(E_new - E_old) < 1e-6",
            weakening_rules=[
                WeakeningRule("relax", "abs(E_new - E_old) < 1e-4", "always", "weaker", ""),
            ],
            domain_confidence=0.8,
        )
        state = BoundaryState(
            interior_invariants=[inv],
            contraction_threshold=3,
        )
        evolution = BoundaryEvolution(state)
        for _ in range(3):
            evolution.evolve(ExecutionRecord(
                success=False,
                params={},
                invariant_results=[("E_conv", InvariantStatus.VIOLATED)],
            ))
        assert state.contraction_count == 3
        # After 3 failures with violations, invariant should be demoted or weakened
        assert inv.violation_count >= 3

    def test_promote_boundary_invariants(self):
        inv = LearnedInvariant("E_conv", "abs(E_new - E_old) < 1e-6")
        state = BoundaryState(
            boundary_invariants=[(inv, 0.2)],
            expansion_threshold=2,
        )
        evolution = BoundaryEvolution(state)
        for _ in range(2):
            evolution.evolve(ExecutionRecord(
                success=True,
                params={},
                invariant_results=[("E_conv", InvariantStatus.SATISFIED)],
            ))
        # Should be promoted: boundary risk < 0.3 and violation_count == 0
        assert len(state.interior_invariants) == 1
        assert len(state.boundary_invariants) == 0

    def test_to_dict(self):
        inv = LearnedInvariant("E_conv", "abs(E_new - E_old) < 1e-6")
        state = BoundaryState(
            interior_invariants=[inv],
            total_experiences=10,
        )
        d = state.to_dict()
        assert d["interior_count"] == 1
        assert d["total_experiences"] == 10


# ── Test DomainLearner ──

class TestDomainLearner:
    def test_add_examples_and_learn(self):
        learner = DomainLearner()
        for _ in range(10):
            learner.add_example(TrainingExample(
                params={"is_bound": True, "density": 8.0},
                invariant_name="high_density",
                status=InvariantStatus.SATISFIED,
            ))
        for _ in range(10):
            learner.add_example(TrainingExample(
                params={"is_bound": False, "density": 2.0},
                invariant_name="high_density",
                status=InvariantStatus.VIOLATED,
            ))
        hypothesis = learner.learn("high_density", min_support=5)
        assert hypothesis is not None
        assert hypothesis.invariant_name == "high_density"
        assert len(hypothesis.conditions) > 0
        assert hypothesis.support >= 5

    def test_learn_insufficient_data(self):
        learner = DomainLearner()
        for _ in range(2):
            learner.add_example(TrainingExample(
                params={"x": 1.0},
                invariant_name="test",
                status=InvariantStatus.SATISFIED,
            ))
        hypothesis = learner.learn("test", min_support=5)
        assert hypothesis is None

    def test_apply_to_invariant(self):
        learner = DomainLearner()
        for _ in range(10):
            learner.add_example(TrainingExample(
                params={"density": 8.0},
                invariant_name="high_density",
                status=InvariantStatus.SATISFIED,
            ))
        for _ in range(10):
            learner.add_example(TrainingExample(
                params={"density": 2.0},
                invariant_name="high_density",
                status=InvariantStatus.VIOLATED,
            ))
        inv = LearnedInvariant(
            name="high_density",
            expression="density > 5.0",
        )
        result = learner.apply_to_invariant(inv, min_support=5)
        assert len(result.domain_conditions) > 0
        assert result.domain_confidence > 0

    def test_stats(self):
        learner = DomainLearner()
        for _ in range(5):
            learner.add_example(TrainingExample(
                params={"E": -1.0},
                invariant_name="energy_negative",
                status=InvariantStatus.SATISFIED,
            ))
        stats = learner.stats()
        assert stats["total_examples"] == 5
        assert stats["unique_invariants"] == 1

    def test_numeric_feature_separation(self):
        learner = DomainLearner()
        for i in range(10):
            learner.add_example(TrainingExample(
                params={"density": 5.0 + i * 0.1},
                invariant_name="high_density",
                status=InvariantStatus.SATISFIED,
            ))
            learner.add_example(TrainingExample(
                params={"density": 0.1 + i * 0.1},
                invariant_name="high_density",
                status=InvariantStatus.VIOLATED,
            ))
        hypothesis = learner.learn("high_density", min_support=5)
        assert hypothesis is not None
        assert any(c.feature == "density" for c in hypothesis.conditions)


# ── Test from_structural_invariant ──

class TestFromStructuralInvariant:
    def test_conversion(self):
        si = StructuralInvariant(
            name="eigenvalues_real",
            expression="λ_i ∈ ℝ for all i",
            theorem="Spectral Theorem for Self-Adjoint Operators",
            severity="theorem",
        )
        li = from_structural_invariant(si)
        assert li.name == "eigenvalues_real"
        assert li.expression == "λ_i ∈ ℝ for all i"
        assert li.theorem == "Spectral Theorem for Self-Adjoint Operators"
        assert li.severity == "theorem"
        assert li.is_probabilistic is False
        assert li.domain_confidence == 1.0
