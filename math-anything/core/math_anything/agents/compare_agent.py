"""Compare Agent - Cross-engine and cross-model comparison."""

from typing import Any, Dict, List

from .base import AgentResult, AgentStatus, BaseAgent


class CompareAgent(BaseAgent):
    """Compare mathematical structures across engines or models.

    Supports:
    1. Cross-engine mapping - VASP ENCUT <-> QE ecutwfc
    2. Cross-model comparison - different parameter sets
    3. Mathematical diff - what changed mathematically

    Input context:
        - 'schema_a': First mathematical schema (required)
        - 'schema_b': Second mathematical schema (required)

    Output data:
        - 'equations_changed': Whether governing equations differ
        - 'shared_structure': Common mathematical structures
        - 'unique_to_a': Structures only in schema A
        - 'unique_to_b': Structures only in schema B
        - 'parameter_mapping': Cross-engine parameter mapping
    """

    @property
    def name(self) -> str:
        return "compare"

    @property
    def description(self) -> str:
        return "Compare mathematical structures across engines or models"

    def validate_inputs(self, context: Dict[str, Any]) -> list:
        missing = []
        if "schema_a" not in context:
            missing.append("schema_a")
        if "schema_b" not in context:
            missing.append("schema_b")
        return missing

    def run(self, context: Dict[str, Any]) -> AgentResult:
        schema_a = context["schema_a"]
        schema_b = context["schema_b"]

        if not isinstance(schema_a, dict):
            schema_a = schema_a.to_dict() if hasattr(schema_a, "to_dict") else {}
        if not isinstance(schema_b, dict):
            schema_b = schema_b.to_dict() if hasattr(schema_b, "to_dict") else {}

        math_a = schema_a.get("mathematical_structure", {})
        math_b = schema_b.get("mathematical_structure", {})

        equations_changed = math_a.get("canonical_form") != math_b.get("canonical_form")

        approx_a = {
            a.get("name", "")
            for a in schema_a.get("approximations", [])
            if isinstance(a, dict)
        }
        approx_b = {
            a.get("name", "")
            for a in schema_b.get("approximations", [])
            if isinstance(a, dict)
        }

        shared = approx_a & approx_b
        unique_a = approx_a - approx_b
        unique_b = approx_b - approx_a

        constraint_a = {
            c.get("expression", "")
            for c in schema_a.get("constraints", [])
            if isinstance(c, dict)
        }
        constraint_b = {
            c.get("expression", "")
            for c in schema_b.get("constraints", [])
            if isinstance(c, dict)
        }

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "equations_changed": equations_changed,
                "problem_type_a": math_a.get("problem_type", "unknown"),
                "problem_type_b": math_b.get("problem_type", "unknown"),
                "shared_structure": {
                    "shared_approximations": list(shared),
                    "shared_constraints": list(constraint_a & constraint_b),
                },
                "unique_to_a": {
                    "approximations": list(unique_a),
                    "constraints": list(constraint_a - constraint_b),
                },
                "unique_to_b": {
                    "approximations": list(unique_b),
                    "constraints": list(constraint_b - constraint_a),
                },
            },
        )
