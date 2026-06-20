"""ANSYS check engine.

Leverages FEM domain template; adds ANSYS-specific checks.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from ..templates import FEMCheckTemplate, FEMParamExtractor
from .base import CheckEngine, CheckResult


class AnsysCheckEngine(CheckEngine):
    """Validate ANSYS APDL input parameters."""

    @property
    def engine_name(self) -> str:
        return "ansys"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        params = FEMParamExtractor.extract(schema, engine="ansys")
        tpl = FEMCheckTemplate(params)
        results = tpl.to_check_results()
        results.extend(self._check_ansys_specific(params))
        return results

    def _check_ansys_specific(self, params: Dict[str, Any]) -> List[CheckResult]:
        results = []
        materials = params.get("materials", [])
        elements = params.get("elements", [])
        bcs = params.get("boundary_conditions", [])
        esize = params.get("element_size")
        constraints = params.get("constraints", [])

        # Check for material definition
        if not materials:
            results.append(
                CheckResult(
                    rule="No material defined",
                    severity="error",
                    message="No MP commands found in the APDL script.",
                    suggestion="Define material properties with MP,EX,1,... and MP,PRXY,1,...",
                )
            )

        # Check for element type
        if not elements:
            results.append(
                CheckResult(
                    rule="No element type",
                    severity="error",
                    message="No ET command found.",
                    suggestion="Define element types with ET,1,SOLID185 (or appropriate element).",
                )
            )

        # Check for boundary conditions - count DOFs, not command lines
        fixed_dof_count = 0
        for bc in bcs:
            if bc.get("bc_type") == "displacement":
                dof = str(bc.get("dof", "")).upper()
                if dof == "ALL":
                    fixed_dof_count += 6
                else:
                    fixed_dof_count += 1

        if not bcs:
            results.append(
                CheckResult(
                    rule="No boundary conditions",
                    severity="warning",
                    message="No D or F commands found.",
                    suggestion="Apply constraints (D) and loads (F) before solving.",
                )
            )
        elif fixed_dof_count < 6:
            results.append(
                CheckResult(
                    rule="Possibly insufficient constraints",
                    severity="warning",
                    message=f"Only {fixed_dof_count} constrained DOF(s) detected.",
                    suggestion="In 3D, at least 6 DOF constraints are typically needed to prevent rigid body motion.",
                )
            )

        # Element size check
        if esize is not None and esize <= 0:
            results.append(
                CheckResult(
                    rule="Invalid element size",
                    severity="error",
                    message=f"ESIZE = {esize} is not positive.",
                    suggestion="Set a positive element size with ESIZE command.",
                )
            )

        # APDL symbolic constraints from parser
        for c in constraints:
            if not c.get("satisfied", True):
                severity = "error" if "critical" in c.get("description", "") else "warning"
                results.append(
                    CheckResult(
                        rule=f"APDL constraint: {c.get('property', 'unknown')}",
                        severity=severity,
                        message=f"{c.get('property', '')} = {c.get('value', '')} violates {c.get('constraint', '')}. {c.get('description', '')}",
                        suggestion="Adjust parameter to satisfy the constraint.",
                    )
                )

        return results
