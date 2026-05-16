"""Rigorous mathematical validation toolkit inspired by cross-method triangulation.

Provides three complementary validation frameworks:

1. CrossValidationMatrix - Method x Conclusion verification grid
2. FalsifiablePredictionTable - Prediction-verification paradigm
3. DualPerspectiveAnalyzer - Geometric + Analytic parallel review

These tools upgrade mathematical claims from "looks correct" to "survived
independent cross-validation", following the epistemic principle that a
conclusion is only as reliable as the number of independent methods that
confirm it.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    CONFIRMED = "confirmed"
    PARTIALLY_CONFIRMED = "partially_confirmed"
    UNCONFIRMED = "unconfirmed"
    CONTRADICTED = "contradicted"
    NOT_TESTED = "not_tested"


class PredictionStatus(Enum):
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"
    PENDING = "pending"


@dataclass
class MethodConclusionCell:
    method: str
    conclusion: str
    status: ValidationStatus = ValidationStatus.NOT_TESTED
    evidence: str = ""
    confidence: float = 0.0
    notes: str = ""


@dataclass
class CrossValidationMatrix:
    """Method x Conclusion verification grid.

    Each cell records whether a specific method confirms a specific conclusion.
    The overall reliability of a conclusion is proportional to the number of
    independent methods that confirm it.

    Usage:
        matrix = CrossValidationMatrix(
            methods=["method_A", "method_B", "method_C"],
            conclusions=["conclusion_1", "conclusion_2"],
        )
        matrix.set("method_A", "conclusion_1", ValidationStatus.CONFIRMED,
                    evidence="metric = 3.2")
        report = matrix.report()
    """

    methods: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    cells: Dict[str, MethodConclusionCell] = field(default_factory=dict)

    def _key(self, method: str, conclusion: str) -> str:
        return f"{method}::{conclusion}"

    def set(
        self,
        method: str,
        conclusion: str,
        status: ValidationStatus,
        evidence: str = "",
        confidence: float = 0.0,
        notes: str = "",
    ):
        if method not in self.methods:
            self.methods.append(method)
        if conclusion not in self.conclusions:
            self.conclusions.append(conclusion)

        self.cells[self._key(method, conclusion)] = MethodConclusionCell(
            method=method,
            conclusion=conclusion,
            status=status,
            evidence=evidence,
            confidence=confidence,
            notes=notes,
        )

    def get(self, method: str, conclusion: str) -> Optional[MethodConclusionCell]:
        return self.cells.get(self._key(method, conclusion))

    def conclusion_reliability(self, conclusion: str) -> Dict[str, Any]:
        if conclusion not in self.conclusions:
            return {"conclusion": conclusion, "reliability": 0.0, "n_confirmed": 0}

        confirmed = 0
        total = 0
        confidences = []

        for method in self.methods:
            cell = self.get(method, conclusion)
            if cell and cell.status != ValidationStatus.NOT_TESTED:
                total += 1
                if cell.status == ValidationStatus.CONFIRMED:
                    confirmed += 1
                    confidences.append(cell.confidence)
                elif cell.status == ValidationStatus.PARTIALLY_CONFIRMED:
                    confirmed += 0.5
                    confidences.append(cell.confidence * 0.5)
                elif cell.status == ValidationStatus.CONTRADICTED:
                    confidences.append(-cell.confidence)

        reliability = confirmed / total if total > 0 else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "conclusion": conclusion,
            "reliability": reliability,
            "n_confirmed": confirmed,
            "n_tested": total,
            "avg_confidence": avg_conf,
        }

    def report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("CROSS-VALIDATION MATRIX")
        lines.append("=" * 72)

        col_w = max(len(c) for c in self.conclusions) + 2 if self.conclusions else 20
        method_w = max(len(m) for m in self.methods) + 2 if self.methods else 15

        header = "Method".ljust(method_w)
        for c in self.conclusions:
            header += c[:col_w].ljust(col_w)
        lines.append(header)
        lines.append("-" * len(header))

        status_symbols = {
            ValidationStatus.CONFIRMED: "✓",
            ValidationStatus.PARTIALLY_CONFIRMED: "~",
            ValidationStatus.UNCONFIRMED: "?",
            ValidationStatus.CONTRADICTED: "✗",
            ValidationStatus.NOT_TESTED: "·",
        }

        for method in self.methods:
            row = method.ljust(method_w)
            for conclusion in self.conclusions:
                cell = self.get(method, conclusion)
                symbol = status_symbols.get(cell.status, "·") if cell else "·"
                row += symbol.center(col_w)
            lines.append(row)

        lines.append("-" * len(header))

        for conclusion in self.conclusions:
            rel = self.conclusion_reliability(conclusion)
            lines.append(
                f"  {conclusion}: reliability={rel['reliability']:.0%} "
                f"({rel['n_confirmed']:.1f}/{rel['n_tested']} methods)"
            )

        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "methods": self.methods,
            "conclusions": self.conclusions,
            "cells": {
                k: {
                    "method": v.method,
                    "conclusion": v.conclusion,
                    "status": v.status.value,
                    "evidence": v.evidence,
                    "confidence": v.confidence,
                }
                for k, v in self.cells.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class FalsifiablePrediction:
    id: str
    statement: str
    mathematical_condition: str
    test_method: str
    status: PredictionStatus = PredictionStatus.PENDING
    test_result: str = ""
    evidence: str = ""
    confidence: float = 0.0


@dataclass
class FalsifiablePredictionTable:
    """Prediction-verification paradigm for mathematical claims.

    Instead of post-hoc rationalization, this framework requires stating
    precise, falsifiable mathematical conditions BEFORE testing. Each
    prediction has a clear mathematical condition that, if violated,
    would falsify the claim.

    Inspired by cross-method triangulation practices in computational science.

    Usage:
        table = FalsifiablePredictionTable()
        table.add("P1", "Scale separation holds",
                  "epsilon < 0.1", "VDOS frequency ratio")
        table.verify("P1", PredictionStatus.VERIFIED,
                     "epsilon = 0.03", evidence="VDOS ratio = 0.03")
        print(table.report())
    """

    predictions: Dict[str, FalsifiablePrediction] = field(default_factory=dict)

    def add(
        self,
        prediction_id: str,
        statement: str,
        mathematical_condition: str,
        test_method: str,
    ):
        self.predictions[prediction_id] = FalsifiablePrediction(
            id=prediction_id,
            statement=statement,
            mathematical_condition=mathematical_condition,
            test_method=test_method,
        )

    def verify(
        self,
        prediction_id: str,
        status: PredictionStatus,
        test_result: str = "",
        evidence: str = "",
        confidence: float = 0.0,
    ):
        if prediction_id in self.predictions:
            p = self.predictions[prediction_id]
            p.status = status
            p.test_result = test_result
            p.evidence = evidence
            p.confidence = confidence

    def n_verified(self) -> int:
        return sum(
            1
            for p in self.predictions.values()
            if p.status == PredictionStatus.VERIFIED
        )

    def n_falsified(self) -> int:
        return sum(
            1
            for p in self.predictions.values()
            if p.status == PredictionStatus.FALSIFIED
        )

    def overall_verdict(self) -> str:
        total = len(self.predictions)
        if total == 0:
            return "NO_PREDICTIONS"
        verified = self.n_verified()
        falsified = self.n_falsified()
        if falsified > 0:
            return "PARTIALLY_FALSIFIED"
        if verified == total:
            return "ALL_VERIFIED"
        if verified > total / 2:
            return "MOSTLY_VERIFIED"
        if verified > 0:
            return "PARTIALLY_VERIFIED"
        return "UNVERIFIED"

    def report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("FALSIFIABLE PREDICTION TABLE")
        lines.append("=" * 72)

        status_symbols = {
            PredictionStatus.VERIFIED: "✓ VERIFIED",
            PredictionStatus.FALSIFIED: "✗ FALSIFIED",
            PredictionStatus.INCONCLUSIVE: "? INCONCLUSIVE",
            PredictionStatus.PENDING: "· PENDING",
        }

        for pid, p in self.predictions.items():
            symbol = status_symbols.get(p.status, "· UNKNOWN")
            lines.append(f"\n  [{pid}] {p.statement}")
            lines.append(f"      Condition: {p.mathematical_condition}")
            lines.append(f"      Test:      {p.test_method}")
            lines.append(f"      Status:    {symbol}")
            if p.test_result:
                lines.append(f"      Result:    {p.test_result}")
            if p.evidence:
                lines.append(f"      Evidence:  {p.evidence}")

        lines.append("")
        lines.append("-" * 72)
        total = len(self.predictions)
        lines.append(
            f"  Verdict: {self.overall_verdict()} "
            f"({self.n_verified()}/{total} verified, "
            f"{self.n_falsified()}/{total} falsified)"
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": {
                k: {
                    "id": v.id,
                    "statement": v.statement,
                    "mathematical_condition": v.mathematical_condition,
                    "test_method": v.test_method,
                    "status": v.status.value,
                    "test_result": v.test_result,
                    "evidence": v.evidence,
                    "confidence": v.confidence,
                }
                for k, v in self.predictions.items()
            },
            "verdict": self.overall_verdict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class Perspective(Enum):
    GEOMETRIC = "geometric"
    ANALYTIC = "analytic"


@dataclass
class PerspectiveChecklist:
    perspective: Perspective
    name: str
    description: str
    computations: List[str] = field(default_factory=list)
    results: Dict[str, str] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    completed: bool = False


@dataclass
class DualPerspectiveResult:
    conclusion: str
    geometric_verdict: str = "pending"
    analytic_verdict: str = "pending"
    agreement: Optional[bool] = None
    geometric_evidence: List[str] = field(default_factory=list)
    analytic_evidence: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DualPerspectiveAnalyzer:
    """Geometric + Analytic parallel review framework.

    Applies two independent mathematical traditions to the same question:
    - Geometric perspective: "What geometric structure exists?"
      Focuses on curvature, minimal submanifolds, isoperimetric properties.
    - Analytic perspective: "Is the statistical signal real?"
      Focuses on random matrix tests, finite-sample bias, condition numbers.

    A conclusion is only accepted when BOTH perspectives agree.

    Inspired by cross-method triangulation in computational science.

    Usage:
        analyzer = DualPerspectiveAnalyzer("Mathematical structure exists in the simulation")

        analyzer.set_geometric_checklist([
            "Compute sectional curvature K",
            "Check if K < 0 (hyperbolic structure)",
            "Find minimal submanifold dimension",
        ])
        analyzer.set_analytic_checklist([
            "Marchenko-Pastur test on covariance eigenvalues",
            "Compute condition number of Hessian",
            "Finite-sample bias correction for PCA",
        ])

        analyzer.add_geometric_evidence("K = -0.02 (negative curvature)")
        analyzer.add_analytic_evidence("Signal eigenvalues > MP upper bound")

        result = analyzer.evaluate()
        print(result.agreement)  # True if both perspectives agree
    """

    conclusion: str
    geometric_checklist: PerspectiveChecklist = field(
        default_factory=lambda: PerspectiveChecklist(
            perspective=Perspective.GEOMETRIC,
            name="Geometric Perspective (Differential Geometry)",
            description="What geometric structure exists?",
        )
    )
    analytic_checklist: PerspectiveChecklist = field(
        default_factory=lambda: PerspectiveChecklist(
            perspective=Perspective.ANALYTIC,
            name="Analytic Perspective (Probability + Harmonic Analysis)",
            description="Is the statistical signal real?",
        )
    )

    def set_geometric_checklist(self, computations: List[str]):
        self.geometric_checklist.computations = computations

    def set_analytic_checklist(self, computations: List[str]):
        self.analytic_checklist.computations = computations

    def add_geometric_evidence(
        self, evidence: str, computation_idx: Optional[int] = None
    ):
        self.geometric_checklist.evidence.append(evidence)
        if computation_idx is not None and computation_idx < len(
            self.geometric_checklist.computations
        ):
            key = self.geometric_checklist.computations[computation_idx]
            self.geometric_checklist.results[key] = evidence

    def add_analytic_evidence(
        self, evidence: str, computation_idx: Optional[int] = None
    ):
        self.analytic_checklist.evidence.append(evidence)
        if computation_idx is not None and computation_idx < len(
            self.analytic_checklist.computations
        ):
            key = self.analytic_checklist.computations[computation_idx]
            self.analytic_checklist.results[key] = evidence

    def set_geometric_verdict(self, verdict: str):
        self.geometric_checklist.completed = True
        self.geometric_checklist.description = verdict

    def set_analytic_verdict(self, verdict: str):
        self.analytic_checklist.completed = True
        self.analytic_checklist.description = verdict

    def evaluate(self) -> DualPerspectiveResult:
        geo_positive = any(
            kw in " ".join(self.geometric_checklist.evidence).lower()
            for kw in ["confirmed", "exists", "negative", "stable", "yes", "true"]
        )
        ana_positive = any(
            kw in " ".join(self.analytic_checklist.evidence).lower()
            for kw in ["significant", "above", "reject", "confirmed", "yes", "true"]
        )

        geo_negative = any(
            kw in " ".join(self.geometric_checklist.evidence).lower()
            for kw in ["flat", "positive curvature", "no structure", "trivial", "false"]
        )
        ana_negative = any(
            kw in " ".join(self.analytic_checklist.evidence).lower()
            for kw in [
                "below",
                "not significant",
                "noise",
                "consistent with null",
                "false",
            ]
        )

        if geo_positive and not geo_negative:
            geo_verdict = "SUPPORTS"
        elif geo_negative and not geo_positive:
            geo_verdict = "CONTRADICTS"
        elif geo_positive and geo_negative:
            geo_verdict = "MIXED"
        else:
            geo_verdict = "INCONCLUSIVE"

        if ana_positive and not ana_negative:
            ana_verdict = "SUPPORTS"
        elif ana_negative and not ana_positive:
            ana_verdict = "CONTRADICTS"
        elif ana_positive and ana_negative:
            ana_verdict = "MIXED"
        else:
            ana_verdict = "INCONCLUSIVE"

        agreement = None
        if geo_verdict == ana_verdict and geo_verdict != "INCONCLUSIVE":
            agreement = True
        elif geo_verdict != ana_verdict and "INCONCLUSIVE" not in (
            geo_verdict,
            ana_verdict,
        ):
            agreement = False

        return DualPerspectiveResult(
            conclusion=self.conclusion,
            geometric_verdict=geo_verdict,
            analytic_verdict=ana_verdict,
            agreement=agreement,
            geometric_evidence=self.geometric_checklist.evidence,
            analytic_evidence=self.analytic_checklist.evidence,
        )

    def report(self) -> str:
        result = self.evaluate()
        lines = []
        lines.append("=" * 72)
        lines.append(f"DUAL-PERSPECTIVE ANALYSIS: {self.conclusion}")
        lines.append("=" * 72)

        lines.append(f"\n  {self.geometric_checklist.name}")
        lines.append(f"  Question: {self.geometric_checklist.description}")
        lines.append("  Computations:")
        for comp in self.geometric_checklist.computations:
            res = self.geometric_checklist.results.get(comp, "(not done)")
            mark = "✓" if comp in self.geometric_checklist.results else "·"
            lines.append(f"    {mark} {comp}: {res}")
        lines.append(f"  Verdict: {result.geometric_verdict}")

        lines.append(f"\n  {self.analytic_checklist.name}")
        lines.append(f"  Question: {self.analytic_checklist.description}")
        lines.append("  Computations:")
        for comp in self.analytic_checklist.computations:
            res = self.analytic_checklist.results.get(comp, "(not done)")
            mark = "✓" if comp in self.analytic_checklist.results else "·"
            lines.append(f"    {mark} {comp}: {res}")
        lines.append(f"  Verdict: {result.analytic_verdict}")

        lines.append("")
        lines.append("-" * 72)
        if result.agreement is True:
            lines.append(
                "  AGREEMENT: Both perspectives converge on the same conclusion ✓"
            )
        elif result.agreement is False:
            lines.append(
                "  DISAGREEMENT: Perspectives diverge -- further investigation needed ✗"
            )
        else:
            lines.append(
                "  INCONCLUSIVE: At least one perspective lacks sufficient evidence ?"
            )
        lines.append("=" * 72)
        return "\n".join(lines)


def create_cross_validation_from_schema(schema: Any) -> CrossValidationMatrix:
    """Auto-populate a cross-validation matrix from a MathSchema.

    Extracts conclusions from governing equations and symbolic constraints,
    and sets up verification methods from the schema's numerical method info.
    """
    matrix = CrossValidationMatrix()

    methods = ["symbolic_verification", "dimensional_analysis", "conservation_check"]
    solver = getattr(
        getattr(getattr(schema, "numerical_method", None), "solver", None),
        "type",
        "",
    )
    if solver:
        methods.append(f"solver_consistency({solver})")

    model = getattr(schema, "mathematical_model", None)
    if model:
        for eq in getattr(model, "governing_equations", []):
            eq_name = getattr(eq, "name", "unnamed_eq")
            for method in methods:
                matrix.set(
                    method=method,
                    conclusion=f"eq:{eq_name}",
                    status=ValidationStatus.NOT_TESTED,
                )

    for constraint in getattr(schema, "symbolic_constraints", []):
        expr = getattr(constraint, "expression", "unnamed_constraint")
        for method in methods:
            matrix.set(
                method=method,
                conclusion=f"constraint:{expr[:40]}",
                status=ValidationStatus.NOT_TESTED,
            )

    return matrix


def create_prediction_table_from_schema(schema: Any) -> FalsifiablePredictionTable:
    """Auto-populate a falsifiable prediction table from a MathSchema.

    Converts symbolic constraints into testable predictions.
    """
    table = FalsifiablePredictionTable()

    for i, constraint in enumerate(getattr(schema, "symbolic_constraints", [])):
        expr = getattr(constraint, "expression", "")
        constraint_type = getattr(constraint, "constraint_type", "")
        table.add(
            prediction_id=f"P{i+1}",
            statement=f"Constraint '{expr[:50]}' is satisfied",
            mathematical_condition=expr,
            test_method=f"symbolic_verification({constraint_type})",
        )

    model = getattr(schema, "mathematical_model", None)
    if model:
        for i, eq in enumerate(getattr(model, "governing_equations", [])):
            math_form = getattr(eq, "mathematical_form", "")
            if math_form:
                table.add(
                    prediction_id=f"EQ{i+1}",
                    statement=f"Governing equation {i+1} is well-posed",
                    mathematical_condition=math_form,
                    test_method="existence_uniqueness_check",
                )

    return table
