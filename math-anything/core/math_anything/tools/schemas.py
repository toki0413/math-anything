"""Pydantic input/output schemas for all math tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractInput(BaseModel):
    engine: str = Field(
        description="Engine name: vasp, lammps, abaqus, quantum_espresso, gromacs, multiwfn, ansys"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine parameters (e.g. {'ENCUT': 520, 'SIGMA': 0.05})",
    )
    filepath: str = Field(
        default="", description="Path to input file (alternative to params)"
    )


class ExtractOutput(BaseModel):
    math_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    approximations: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ValidateInput(BaseModel):
    math_schema: dict[str, Any] = Field(
        description="Extracted mathematical schema to validate", alias="schema"
    )

    model_config = {"populate_by_name": True}


class ValidateOutput(BaseModel):
    valid: bool = False
    constraint_results: list[dict[str, Any]] = Field(default_factory=list)
    dimensional_issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    total_issues: int = 0


class CompareInput(BaseModel):
    schema_a: dict[str, Any] = Field(description="First mathematical schema")
    schema_b: dict[str, Any] = Field(description="Second mathematical schema")


class CompareOutput(BaseModel):
    equations_changed: bool = False
    problem_type_a: str = ""
    problem_type_b: str = ""
    shared_structure: dict[str, Any] = Field(default_factory=dict)
    unique_to_a: dict[str, Any] = Field(default_factory=dict)
    unique_to_b: dict[str, Any] = Field(default_factory=dict)


class VerifyInput(BaseModel):
    statement: str = Field(description="Mathematical statement to verify")
    proof_text: str = Field(
        default="", description="Proof text to verify against the statement"
    )
    task_type: str = Field(
        default="proof", description="Task type: proof, validation, consistency"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="List of assumptions"
    )
    goals: list[str] = Field(default_factory=list, description="List of goals")
    engine: str = Field(default="", description="Engine for geometric context")
    with_geometry: bool = Field(default=False, description="Include geometric context")


class VerifyOutput(BaseModel):
    formal_status: str = ""
    overall_confidence: float = 0.0
    layer_results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PropositionInput(BaseModel):
    math_schema: dict[str, Any] = Field(
        description="Extracted mathematical schema", alias="schema"
    )
    engine: str = Field(default="", description="Engine name for context")

    model_config = {"populate_by_name": True}


class PropositionOutput(BaseModel):
    core_problem: str = ""
    proof_tasks: list[dict[str, Any]] = Field(default_factory=list)
    validation_tasks: list[dict[str, Any]] = Field(default_factory=list)
    consistency_checks: list[dict[str, Any]] = Field(default_factory=list)
    total_tasks: int = 0


class CrossValidateInput(BaseModel):
    methods: list[str] = Field(description="List of validation methods")
    conclusions: list[str] = Field(description="List of conclusions to validate")


class CrossValidateOutput(BaseModel):
    matrix: dict[str, Any] = Field(default_factory=dict)
    report: str = ""


class DualPerspectiveInput(BaseModel):
    conclusion: str = Field(description="Conclusion to analyze from dual perspectives")
    geometric_checks: list[str] = Field(
        default_factory=list, description="Geometric perspective checklist"
    )
    analytic_checks: list[str] = Field(
        default_factory=list, description="Analytic perspective checklist"
    )


class DualPerspectiveOutput(BaseModel):
    result: dict[str, Any] = Field(default_factory=dict)
    report: str = ""


class EmergenceInput(BaseModel):
    engine: str = Field(description="Engine name")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Engine parameters"
    )
    filepath: str = Field(default="", description="Path to input file")


class EmergenceOutput(BaseModel):
    emergence: dict[str, Any] = Field(default_factory=dict)
    math_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    warnings: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class GeometryInput(BaseModel):
    engine: str = Field(description="Engine name")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Engine parameters"
    )
    lattice_vectors: dict[str, list[float]] | None = Field(
        default=None, description="Lattice vectors for crystal structures"
    )
    space_group: str | None = Field(
        default=None, description="Space group for crystal structures"
    )


class GeometryOutput(BaseModel):
    manifold: dict[str, Any] = Field(default_factory=dict)
    metric_tensor: dict[str, Any] = Field(default_factory=dict)
    curvature: dict[str, Any] = Field(default_factory=dict)
    fiber_bundle: dict[str, Any] | None = None
