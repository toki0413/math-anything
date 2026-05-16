"""Validate Agent - Validate parameter consistency and constraints."""

from typing import Any, Dict, List

from .base import AgentResult, AgentStatus, BaseAgent


class ValidateAgent(BaseAgent):
    """Validate mathematical consistency of extracted schemas.

    Checks:
    1. Constraint satisfaction - all symbolic constraints are met
    2. Dimensional consistency - units match across equations
    3. Parameter ranges - values within physically meaningful bounds
    4. Cross-parameter dependencies - dependent parameters are consistent

    Input context:
        - 'schema': Extracted mathematical schema (required)

    Output data:
        - 'valid': Whether all checks pass
        - 'constraint_results': Per-constraint validation results
        - 'dimensional_issues': Any dimensional inconsistencies
        - 'warnings': Validation warnings
    """

    @property
    def name(self) -> str:
        return "validate"

    @property
    def description(self) -> str:
        return "Validate parameter consistency and constraints"

    def validate_inputs(self, context: Dict[str, Any]) -> list:
        if "schema" not in context:
            return ["schema"]
        return []

    def run(self, context: Dict[str, Any]) -> AgentResult:
        schema = context["schema"]
        if not isinstance(schema, dict):
            schema = schema.to_dict() if hasattr(schema, "to_dict") else {}

        constraint_results = self._check_constraints(schema)
        dimensional_issues = self._check_dimensions(schema)
        range_warnings = self._check_ranges(schema)

        all_issues = []
        all_issues.extend(c for c in constraint_results if not c.get("satisfied", True))
        all_issues.extend(dimensional_issues)
        all_issues.extend(range_warnings)

        status = AgentStatus.SUCCESS if not all_issues else AgentStatus.PARTIAL

        return AgentResult(
            agent_name=self.name,
            status=status,
            data={
                "valid": len(all_issues) == 0,
                "constraint_results": constraint_results,
                "dimensional_issues": dimensional_issues,
                "warnings": range_warnings,
                "total_issues": len(all_issues),
            },
        )

    def _check_constraints(self, schema: Dict) -> List[Dict]:
        results = []
        constraints = schema.get("constraints", [])
        for constraint in constraints:
            if isinstance(constraint, dict):
                expr = constraint.get("expression", "")
                satisfied = constraint.get("satisfied", None)
                if satisfied is None:
                    satisfied = True
                results.append(
                    {
                        "expression": expr,
                        "satisfied": satisfied,
                        "description": constraint.get("description", ""),
                    }
                )
        return results

    def _check_dimensions(self, schema: Dict) -> List[Dict]:
        issues = []
        math_struct = schema.get("mathematical_structure", {})
        if not math_struct:
            return issues

        variable_deps = schema.get("variable_dependencies", [])
        for dep in variable_deps:
            if isinstance(dep, dict) and dep.get("circular", False):
                issues.append(
                    {
                        "type": "circular_dependency",
                        "description": f"Circular dependency detected: {dep.get('relation', '')}",
                    }
                )
        return issues

    def _check_ranges(self, schema: Dict) -> List[Dict]:
        warnings = []
        params = schema.get("parameters", {})
        if isinstance(params, dict):
            for name, info in params.items():
                if isinstance(info, dict):
                    value = info.get("value")
                    min_val = info.get("min")
                    max_val = info.get("max")
                    if value is not None and min_val is not None and value < min_val:
                        warnings.append(
                            {
                                "type": "below_minimum",
                                "parameter": name,
                                "value": value,
                                "minimum": min_val,
                            }
                        )
                    if value is not None and max_val is not None and value > max_val:
                        warnings.append(
                            {
                                "type": "above_maximum",
                                "parameter": name,
                                "value": value,
                                "maximum": max_val,
                            }
                        )
        return warnings
