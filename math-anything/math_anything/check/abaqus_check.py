"""Abaqus pre-flight parameter consistency checks.

Rules are derived from mathematical/physical requirements of FEM.
"""

from typing import Any, Dict, List

from ..schemas import MathSchema
from .base import CheckEngine, CheckResult


class AbaqusCheckEngine(CheckEngine):
    """Validate Abaqus input parameters before submission."""

    @property
    def engine_name(self) -> str:
        return "abaqus"

    def check(self, schema: MathSchema) -> List[CheckResult]:
        raw: Dict[str, Any] = schema.raw_symbols or {}
        materials = raw.get("materials", [])
        steps = raw.get("steps", [])
        bcs = raw.get("boundary_conditions", [])
        elements = raw.get("elements", [])
        nlgeom = raw.get("nlgeom", False)
        nodes = raw.get("nodes", 0)
        cards = raw.get("cards", [])

        results: List[CheckResult] = []
        results.extend(self._check_materials(materials))
        results.extend(self._check_boundary_conditions(bcs, steps))
        results.extend(self._check_elements(elements, steps))
        results.extend(self._check_steps(steps, nlgeom))
        results.extend(self._check_mesh(nodes, elements))
        results.extend(self._check_cards(cards))
        return results

    def _check_materials(self, materials: List[Dict]) -> List[CheckResult]:
        results = []
        if not materials:
            results.append(
                CheckResult(
                    rule="No material defined",
                    severity="error",
                    message="No *Material card found in the input file.",
                    suggestion="Every element must reference a material. Add at least one *Material definition.",
                )
            )
            return results

        for mat in materials:
            E = mat.get("youngs_modulus")
            nu = mat.get("poisson_ratio")
            rho = mat.get("density")
            mtype = mat.get("model_type", "unknown")

            if E is not None and E <= 0:
                results.append(
                    CheckResult(
                        rule="Non-positive Young's modulus",
                        severity="error",
                        message=f"Material '{mat.get('name')}' has E = {E} <= 0.",
                        suggestion="Young's modulus must be strictly positive for a stable material.",
                    )
                )

            if nu is not None and not (-1 < nu < 0.5):
                results.append(
                    CheckResult(
                        rule="Poisson's ratio out of range",
                        severity="error",
                        message=f"Material '{mat.get('name')}' has nu = {nu} outside (-1, 0.5).",
                        suggestion="For positive definite stiffness, -1 < nu < 0.5. Most solids have 0 <= nu <= 0.49.",
                    )
                )

            if E is not None and E > 0 and nu is not None and -1 < nu < 0.5:
                if nu > 0.499:
                    results.append(
                        CheckResult(
                            rule="Near-incompressibility (nu -> 0.5)",
                            severity="warning",
                            message=f"nu = {nu} is very close to 0.5 (incompressible limit).",
                            suggestion="Use hybrid elements (e.g., C3D8H) or mixed formulation to avoid volumetric locking.",
                        )
                    )

            if mtype == "elastic" and (E is None or nu is None):
                results.append(
                    CheckResult(
                        rule="Incomplete elastic material",
                        severity="warning",
                        message=f"Material '{mat.get('name')}' is elastic but E or nu is missing.",
                        suggestion="*Elastic requires two values: Young's modulus and Poisson's ratio.",
                    )
                )

            if rho is None and any(s.get("analysis_type") in ("dynamic", "modal") for s in []):
                results.append(
                    CheckResult(
                        rule="Density missing for dynamic analysis",
                        severity="error",
                        message=f"Material '{mat.get('name')}' has no *Density.",
                        suggestion="Dynamic/modal analyses require mass density. Add *Density under *Material.",
                    )
                )

        return results

    def _check_boundary_conditions(self, bcs: List[Dict], steps: List[Dict]) -> List[CheckResult]:
        results = []
        if not bcs:
            analysis = steps[0].get("analysis_type", "static") if steps else "static"
            if analysis == "static":
                results.append(
                    CheckResult(
                        rule="No boundary conditions",
                        severity="error",
                        message="No *Boundary card found.",
                        suggestion="Static FEM requires boundary conditions to prevent rigid body motion and ensure K is invertible.",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        rule="No boundary conditions",
                        severity="warning",
                        message="No *Boundary card found.",
                        suggestion="Verify that the model is properly constrained for the intended physics.",
                    )
                )
            return results

        fixed_dofs = sum(1 for b in bcs if b.get("value", 0.0) == 0.0)
        if fixed_dofs < 3:
            results.append(
                CheckResult(
                    rule="Possibly insufficient constraints",
                    severity="warning",
                    message=f"Only {fixed_dofs} zero-value boundary condition(s) detected.",
                    suggestion="In 3D, at least 6 constraints (3 translations + 3 rotations) are typically needed to prevent rigid body motion. Ensure constraints are on independent nodes.",
                )
            )

        return results

    def _check_elements(self, elements: List[str], steps: List[Dict]) -> List[CheckResult]:
        results = []
        if not elements:
            results.append(
                CheckResult(
                    rule="No element type specified",
                    severity="error",
                    message="No *Element card with TYPE= found.",
                    suggestion="Define elements using *Element, type=... in the input file.",
                )
            )
            return results

        elem = elements[0].upper()
        analysis = steps[0].get("analysis_type", "static") if steps else "static"

        # Check element type vs analysis compatibility
        if analysis == "heat" and any(e.upper().startswith("C3D") for e in elements):
            results.append(
                CheckResult(
                    rule="Solid elements for heat transfer",
                    severity="info",
                    message="Continuum elements (C3D*) are used for heat transfer.",
                    suggestion="Ensure DC3D8/DC3D4 (diffusion elements) are used if temperature is the only DOF.",
                )
            )

        if "C3D8R" in elem and analysis == "static":
            results.append(
                CheckResult(
                    rule="Reduced integration elements",
                    severity="info",
                    message="C3D8R (reduced integration) is selected.",
                    suggestion="Check that hourglass energy is < 5% of total strain energy. Use C3D8I for bending-dominated problems.",
                )
            )

        if "C3D4" in elem:
            results.append(
                CheckResult(
                    rule="Linear tetrahedron elements",
                    severity="warning",
                    message="C3D4 (linear tet) is very stiff and locks in bending/incompressibility.",
                    suggestion="Use C3D10 (quadratic tet) for accuracy, or C3D8R for efficiency.",
                )
            )

        return results

    def _check_steps(self, steps: List[Dict], nlgeom: bool) -> List[CheckResult]:
        results = []
        if not steps:
            results.append(
                CheckResult(
                    rule="No analysis step",
                    severity="error",
                    message="No *Step card found.",
                    suggestion="At least one *Step ... *End Step block is required.",
                )
            )
            return results

        for step in steps:
            max_inc = step.get("max_increments", 100)
            if max_inc > 10000:
                results.append(
                    CheckResult(
                        rule="Very large max increments",
                        severity="info",
                        message=f"Max increments = {max_inc} is very large.",
                        suggestion="This is acceptable but may produce large output files. Monitor .sta file for convergence.",
                    )
                )

            initial = step.get("initial_inc")
            total = step.get("total_time")
            if initial is not None and total is not None:
                if initial > total:
                    results.append(
                        CheckResult(
                            rule="Initial increment exceeds total time",
                            severity="error",
                            message=f"Initial increment ({initial}) > total step time ({total}).",
                            suggestion="Reduce initial increment to be <= total step time.",
                        )
                    )
                elif total / initial > 1000:
                    results.append(
                        CheckResult(
                            rule="Many increments expected",
                            severity="info",
                            message=f"Total time / initial increment = {total / initial:.0f} increments possible.",
                            suggestion="Ensure max_increments is large enough to cover the full step.",
                        )
                    )

            if step.get("analysis_type") == "static" and nlgeom and step.get("initial_inc", 1.0) > 0.1:
                results.append(
                    CheckResult(
                        rule="Large initial increment with NLGEOM",
                        severity="warning",
                        message="NLGEOM is ON but initial increment is relatively large.",
                        suggestion="For strongly nonlinear problems, start with a smaller initial increment (e.g., 0.01) to improve convergence.",
                    )
                )

        return results

    def _check_mesh(self, nodes: int, elements: List[str]) -> List[CheckResult]:
        results = []
        if nodes == 0:
            results.append(
                CheckResult(
                    rule="No nodes defined",
                    severity="error",
                    message="No *Node card found or node count is zero.",
                    suggestion="Define nodes using *Node in the input file before elements.",
                )
            )
        elif nodes < 10:
            results.append(
                CheckResult(
                    rule="Very small mesh",
                    severity="info",
                    message=f"Only {nodes} node(s) detected.",
                    suggestion="Verify this is a simplified test model; production meshes usually need > 1000 nodes for mesh-independent results.",
                )
            )

        return results

    def _check_cards(self, cards: List[str]) -> List[CheckResult]:
        results = []
        card_set = {c.upper() for c in cards}

        if "PART" in card_set and "END PART" not in card_set:
            results.append(
                CheckResult(
                    rule="Unclosed *Part block",
                    severity="warning",
                    message="*Part found but *End Part is missing.",
                    suggestion="Ensure every *Part is closed with *End Part.",
                )
            )

        if "ASSEMBLY" in card_set and "END ASSEMBLY" not in card_set:
            results.append(
                CheckResult(
                    rule="Unclosed *Assembly block",
                    severity="warning",
                    message="*Assembly found but *End Assembly is missing.",
                    suggestion="Ensure every *Assembly is closed with *End Assembly.",
                )
            )

        if "STEP" in card_set and "END STEP" not in card_set:
            results.append(
                CheckResult(
                    rule="Unclosed *Step block",
                    severity="error",
                    message="*Step found but *End Step is missing.",
                    suggestion="Every *Step must be terminated with *End Step.",
                )
            )

        return results
